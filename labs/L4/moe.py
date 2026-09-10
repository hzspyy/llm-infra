#!/usr/bin/env python3
"""L4.4 —— MoE：稀疏激活的系统代价。

用一个真实的 MoE 模型（OLMoE-1B-7B：64 个专家、top-8、1B 激活 / 7B 总量）：

  [A] 参数量 vs 激活量：稀疏到底稀疏了什么
  [B] 路由：真实文本上每个专家被选了多少次（负载均衡）
  [C] grouped GEMM：把 token 按专家分组的代价
  [D] 显存与带宽的账：MoE 为什么难服务

用法：
    python moe.py
    python moe.py B
"""

import glob
import json
import os
import sys
from collections import Counter

import torch

# 模型都已在本地缓存；强制离线，避免 vLLM/transformers 每次去连 Hub
# （连不上时会直接抛 httpx.ConnectError，即使文件就在本地）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MB = 1024 * 1024
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = os.environ.get("L44_MODEL", "allenai/OLMoE-1B-7B-0924-Instruct")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def snap(repo):
    d = f"{HUB}/models--{repo.replace('/', '--')}/snapshots"
    g = sorted(glob.glob(d + "/*"))
    return g[0] if g else None


def timeit(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 参数量 vs 激活量")

    d = snap(REPO)
    if not d:
        print(f"  {REPO} 未下载"); return
    c = json.load(open(d + "/config.json"))
    for k in ["hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads",
              "num_experts", "num_experts_per_tok", "vocab_size",
              "norm_topk_prob", "router_aux_loss_coef"]:
        if k in c:
            print(f"    {k:<26} {c[k]}")

    H, I, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    E, K = c["num_experts"], c["num_experts_per_tok"]
    V = c["vocab_size"]

    p_expert = 3 * H * I                  # SwiGLU 三个矩阵
    p_router = H * E
    p_moe_layer = E * p_expert + p_router
    nq = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nq)
    hd = H // nq
    p_attn = H * nq * hd + 2 * H * nkv * hd + nq * hd * H
    p_layer = p_moe_layer + p_attn
    total = L * p_layer + V * H

    act_moe = K * p_expert + p_router     # 每 token 只走 K 个专家
    act_layer = act_moe + p_attn
    act_total = L * act_layer + V * H

    print(f"\n  一个专家 = 3×{H}×{I} = {p_expert:,} 参数")
    print(f"  一层有 {E} 个专家 + 1 个 router({H}×{E})")
    print(f"\n  {'':<24} {'总参数':>16} {'每 token 激活':>16} {'比值':>8}")
    print(f"  {'MoE 部分/层':<24} {p_moe_layer:>16,} {act_moe:>16,} "
          f"{p_moe_layer / act_moe:>7.1f}×")
    print(f"  {'attention/层':<24} {p_attn:>16,} {p_attn:>16,} {1.0:>7.1f}×")
    print(f"  {'每层合计':<24} {p_layer:>16,} {act_layer:>16,} "
          f"{p_layer / act_layer:>7.1f}×")
    print(f"  {f'{L} 层 + embedding':<24} {total:>16,} {act_total:>16,} "
          f"{total / act_total:>7.1f}×")
    print(f"\n  bf16 权重显存：全部 {total * 2 / MB / 1024:.2f} GiB，"
          f"而每 token 只用到 {act_total * 2 / MB / 1024:.2f} GiB 的量")
    print(f"  **但那 {total * 2 / MB / 1024:.2f} GiB 必须全部驻留显存** —— "
          f"你不知道下一个 token 会路由到谁。")
    print("  这就是 MoE 的核心系统代价：**按总参数量买显存，按激活量拿算力。**")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 路由：真实文本上专家被选了多少次")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config
    E, K = cfg.num_experts, cfg.num_experts_per_tok

    text = ("The transformer architecture uses self-attention to model "
            "dependencies between tokens. In a mixture-of-experts layer, a router "
            "assigns each token to a small subset of expert networks. This makes "
            "the total parameter count much larger than the number of parameters "
            "actually used for any single token. 混合专家模型的路由决定了每个 token "
            "走哪几个专家。def forward(self, x): return self.mlp(self.norm(x)) "
            "1 + 1 = 2. The quick brown fox jumps over the lazy dog. ") * 8
    ids = tok(text, return_tensors="pt").input_ids.cuda()[:, :1024]
    S = ids.shape[1]
    print(f"  {REPO}")
    print(f"  {E} 个专家，每 token 选 top-{K}，输入 {S} 个 token")

    # 挂钩子抓每一层 router 的选择
    picks = {}

    def mk_hook(li):
        def hook(mod, inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            if logits.dim() == 3:
                logits = logits.view(-1, logits.shape[-1])
            top = logits.topk(K, dim=-1).indices        # [tokens, K]
            picks.setdefault(li, []).append(top.detach().cpu())
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            handles.append(gate.register_forward_hook(mk_hook(li)))
    print(f"  在 {len(handles)} 层的 router 上挂了钩子")

    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    sub("总体：所有层所有 token 加起来")
    allpick = torch.cat([torch.cat(v) for v in picks.values()]).flatten()
    cnt = torch.bincount(allpick, minlength=E).float()
    ideal = cnt.sum().item() / E
    print(f"  总选择次数 {int(cnt.sum().item()):,}，理想每专家 {ideal:,.0f}")
    print(f"  实际 min={int(cnt.min())}  max={int(cnt.max())}  "
          f"中位数={int(cnt.median())}")
    print(f"  **最热/最冷 = {cnt.max().item() / max(cnt.min().item(), 1):.1f}×**")
    print(f"  最热的专家拿了 {cnt.max().item() / cnt.sum().item():.2%}"
          f"（均匀应为 {1 / E:.2%}）")
    # 变异系数
    cv = (cnt.std() / cnt.mean()).item()
    print(f"  变异系数 CV = {cv:.3f}   （0 = 完全均衡）")

    sub("逐层看：不均衡程度一样吗")
    print(f"  {'层':>4} {'最热占比':>10} {'最冷占比':>10} {'最热/最冷':>10} {'CV':>8}")
    for li in sorted(picks)[:4] + sorted(picks)[-3:]:
        p = torch.cat(picks[li]).flatten()
        c = torch.bincount(p, minlength=E).float()
        print(f"  {li:>4} {c.max().item() / c.sum().item():>9.2%} "
              f"{c.min().item() / c.sum().item():>9.2%} "
              f"{c.max().item() / max(c.min().item(), 1):>9.1f}× "
              f"{(c.std() / c.mean()).item():>8.3f}")

    sub("⚠ 跨层求和会把不均衡抹掉")
    print(f"  总体那张表看着还行（最热/最冷 {cnt.max().item() / max(cnt.min().item(), 1):.1f}×，"
          f"CV {cv:.3f}），")
    print("  但逐层表里最热/最冷动辄几百倍、CV 在 1.0 以上，而且**每层都有专家拿到 0.00%**。")
    print("  原因：不同层的热专家不是同一批，加起来互相填平了。")
    print("  **而系统关心的是每一层**——每一层都要做一次 grouped GEMM，")
    print("  每一层的最热专家各自决定那一层的关键路径。所以看总体是看错了对象。")

    sub("对系统的直接含义（用逐层的数）")
    per_layer_tokens = S * K                     # 每层每 token 选 K 个
    ideal_layer = per_layer_tokens / E
    hot_frac = max(
        (torch.bincount(torch.cat(picks[li]).flatten(), minlength=E).float().max()
         / per_layer_tokens).item() for li in picks)
    print(f"  每层 {S} 个 token × top-{K} = {per_layer_tokens} 次选择，"
          f"均匀应为每专家 {ideal_layer:.0f} 个")
    print(f"  实测最热的专家拿到该层的 {hot_frac:.1%}，"
          f"即 {hot_frac * per_layer_tokens:.0f} 个 token —— "
          f"是均匀值的 {hot_frac * E:.1f} 倍")
    print("  grouped GEMM 若各段并行执行，总时间由最大的那一段决定，")
    print(f"  所以并行效率的上界就是 1/{hot_frac * E:.1f} = {1 / (hot_frac * E):.0%}。")
    print("  在专家并行（EP）里，这直接变成**某张卡在算、其它卡在等**（L6.3）。")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C():
    title("[C] grouped GEMM：分组本身要多少代价")

    if not torch.cuda.is_available():
        print("  需要 CUDA"); return
    d = snap(REPO)
    c = json.load(open(d + "/config.json")) if d else {
        "hidden_size": 2048, "intermediate_size": 1024,
        "num_experts": 64, "num_experts_per_tok": 8}
    H, I = c["hidden_size"], c["intermediate_size"]
    E, K = c["num_experts"], c["num_experts_per_tok"]
    T = 4096                                  # 一个 step 的 token 数

    print(f"  H={H} I={I} 专家数={E} top-{K}  一步 {T} 个 token")
    w = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    # 均匀路由
    torch.manual_seed(0)
    assign = torch.randint(0, E, (T * K,), device="cuda")
    xs = x.repeat_interleave(K, dim=0)

    def dense_all():
        """朴素：每个 token 都过所有专家（正确但浪费 E/K 倍）"""
        return torch.einsum("th,ehi->tei", x, w)

    def grouped():
        """按专家排序后分段做 GEMM —— 这是真实实现的形态"""
        order = assign.argsort()
        xs_sorted = xs[order]
        counts = torch.bincount(assign, minlength=E)
        outs = []
        off = 0
        cpu_counts = counts.tolist()
        for e in range(E):
            n = cpu_counts[e]
            if n:
                outs.append(xs_sorted[off:off + n] @ w[e])
            off += n
        return torch.cat(outs) if outs else None

    def sort_only():
        order = assign.argsort()
        return xs[order]

    t_grp = timeit(grouped, n=10)
    t_sort = timeit(sort_only, n=10)
    flops = 2 * T * K * H * I
    print(f"\n  {'做法':<34} {'ms':>9} {'TFLOP/s':>10}")
    print(f"  {'grouped GEMM（含排序）':<30} {t_grp:>9.3f} "
          f"{flops / t_grp / 1e9:>10.1f}")
    print(f"  {'其中只排序':<32} {t_sort:>9.3f} "
          f"{'—':>10}   占 {t_sort / t_grp:.1%}")
    ref = torch.randn(T * K, H, device="cuda", dtype=torch.bfloat16)
    w0 = w[0]
    t_one = timeit(lambda: ref @ w0, n=10)
    print(f"  {'同样 FLOP 的单个大 GEMM':<30} {t_one:>9.3f} "
          f"{flops / t_one / 1e9:>10.1f}   <- 上界")
    print(f"\n  分成 {E} 个小 GEMM 相对一个大 GEMM 慢 {t_grp / t_one:.1f}×。")
    print("  原因：每个小 GEMM 的 M 只有 ~{:.0f}，远小于 tensor core 喜欢的形状"
          .format(T * K / E))
    print("  （2.7 §六 测过：瘦长 GEMM 的效率远低于方阵）。")
    print("  真实实现用专门的 grouped GEMM kernel（一次 launch 处理所有分段），")
    print("  比这里的 python 循环好得多 —— 这里量的是**上界**。")

    sub("不均衡会让情况更糟")
    for name, probs in [("均匀", None), ("倾斜（一个专家占 20%）", "skew")]:
        if probs is None:
            a = torch.randint(0, E, (T * K,), device="cuda")
        else:
            a = torch.randint(0, E, (T * K,), device="cuda")
            m = torch.rand(T * K, device="cuda") < 0.2
            a[m] = 0
        cnt = torch.bincount(a, minlength=E)
        print(f"  {name:<22} max={cnt.max().item():>6} "
              f"min={cnt.min().item():>5} max/mean={cnt.max().item() / cnt.float().mean().item():>5.2f}×")
    print("  分段 GEMM 的总时间由最大的那一段决定（若并行执行），")
    print("  所以 max/mean 就是并行效率的直接上界。")
    del w, x
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] 显存与带宽的账")

    d = snap(REPO)
    if not d:
        print("  模型未下载"); return
    c = json.load(open(d + "/config.json"))
    H, I, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    E, K = c["num_experts"], c["num_experts_per_tok"]

    p_expert = 3 * H * I
    print(f"  一个专家 {p_expert * 2 / MB:.2f} MB (bf16)")
    print(f"  一层 {E} 个专家 = {E * p_expert * 2 / MB:.1f} MB")
    print(f"  {L} 层 = {L * E * p_expert * 2 / MB / 1024:.2f} GiB 的专家权重")

    sub("decode 时到底要读多少权重")
    print("  batch=1 时每 token 只碰 K 个专家：")
    print(f"    读 {K}×{L} = {K * L} 个专家 = "
          f"{K * L * p_expert * 2 / MB:.1f} MB")
    print("  但 batch 一大，不同 token 路由到不同专家，读取量迅速逼近全部：")
    print(f"  {'batch':>7} {'期望被碰到的专家数/层':>22} {'读取量 MB':>12} "
          f"{'占全部':>8}")
    import math
    for B in [1, 4, 16, 64, 256, 1024]:
        # 每 token 选 K 个；B 个 token 共 B*K 次抽取，期望覆盖的专家数
        expected = E * (1 - (1 - K / E) ** B)
        mb = expected * L * p_expert * 2 / MB
        print(f"  {B:>7} {expected:>22.1f} {mb:>12.1f} "
              f"{expected / E:>7.1%}")
    print(f"\n  batch≥64 时基本每层的所有 {E} 个专家都要读一遍 ——")
    print("  **MoE 的稀疏性在大 batch 下从带宽角度基本消失**，")
    print("  只在算力上仍然稀疏（每 token 仍只算 K 个）。")
    print("  这就是为什么 MoE 适合「算力受限」的场景，")
    print("  而在小 batch decode（带宽受限，3.3）上优势有限。")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}  model {REPO}")
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
    os._exit(0)
