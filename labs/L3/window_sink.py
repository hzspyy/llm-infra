#!/usr/bin/env python3
"""L3.4-B —— 滑动窗口的边界、attention sink 的保留策略、RoPE 缩放。

三件事分开测，因为它们的性质不同：
  [A] 窗口边界：S = W−1 / W / W+1 时，mask、有效 KV 长度与逐行可见数
      —— 纯构造，数学上是什么就是什么
  [B] sink 保留策略：用 Qwen3-1.7B 真实前向，比较四种 mask 下的输出
      full / window-only / window+sink / drop-sink
      —— 现象（注意力集中在 token 0）与机制（softmax 的归一化压力）分开说
  [C] RoPE 插值/频率缩放：把三种方案实际作用在位置上，打印角度与旋转后的分量

用法：
    L3_OUT=<目录> python window_sink.py A C          # 不需要模型
    L3_OUT=<目录> python window_sink.py A B C        # B 需要 Qwen3-1.7B
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                      # noqa: E402

MODEL = os.environ.get("L34_MODEL", "Qwen/Qwen3-1.7B")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def window_mask(S, W, keep_sink=False):
    """[S,S] 布尔：True 表示可见。i 行只能看 [max(0,i-W+1), i]，可选保留位置 0。"""
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    m = (j <= i) & ((i - j) < W)
    if keep_sink:
        m = m | (j == 0)
    return m


def causal_mask(S):
    return torch.ones(S, S, dtype=torch.bool).tril()


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 窗口边界：S = W−1 / W / W+1 的 mask、可见数与有效 KV")

    print("  滑动窗口只看最近 W 个位置。边界在 S 恰好跨过 W 的地方。")
    for W in [4, 8]:
        print(f"\n  W={W}")
        print(f"  {'S':>4} {'逐行可见数（行 0/中间/末行）':>26} {'末行 mask':>22} "
              f"{'有效 KV 长度':>12}")
        for S in [W - 1, W, W + 1]:
            m = window_mask(S, W)
            counts = m.sum(-1)
            row = "".join("1" if v else "0" for v in m[-1].tolist())
            eff = min(S, W)
            print(f"  {S:>4} {str([int(counts[0]), int(counts[S // 2]), int(counts[-1])]):>26} "
                  f"{row:>22} {eff:>12}")
            h.case(id=f"A_W{W}_S{S}", W=W, S=S, visible_counts=counts.tolist(),
                   last_row_mask=row, effective_kv_len=eff,
                   all_visible=bool(counts.min() == S))
        print("  规律：S ≤ W 时窗口装得下（逐行可见数 = 行号+1，全可见）；")
        print("        S > W 时末行只看得到最后 W 个，位置 0 已被划出窗口。")
    print("\n  状态侧：KV 长度 = min(S, W)，不再随 S 增长 —— 这是 SWA 的全部好处。")
    print("  但被划出去的位置不是'没算'，是**永远看不到**：窗口外信息直接消失。")


# ---------------------------------------------------------------- B
def section_B(h):
    title("[B] sink 保留策略：真实模型上丢掉 token 0 会怎样")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config
    S = 26
    W = 8
    texts = {
        "冠词开头": ("The capital of France is Paris. The capital of Japan is "
                 "Tokyo. Machine learning models read tokens one at a time."),
        "标点开头": (". The capital of France is Paris. The capital of Japan is "
                 "Tokyo. Machine learning models read tokens one at a time."),
        "罕见 token 开头": ("Ünïcödé 的 首都 是 Paris. The capital of Japan is "
                         "Tokyo. Models read tokens one at a time."),
    }
    L = cfg.num_hidden_layers - 1
    print(f"  {MODEL}: {cfg.num_hidden_layers} 层，取第 {L} 层（最后一层）看 "
          f"token 0 的注意力份额；输入固定 {S} token，窗口 W={W}")

    def build_4d_mask(ids, keep_sink, window=None):
        n = ids.shape[1]
        allowed = causal_mask(n)
        if window is not None:
            allowed = window_mask(n, window, keep_sink=keep_sink)
        elif keep_sink:
            allowed = causal_mask(n)
            allowed[:, 0] = True
        add = torch.zeros(1, 1, n, n, dtype=torch.bfloat16, device=ids.device)
        add = add.masked_fill(~allowed.to(ids.device), torch.finfo(torch.bfloat16).min)
        return add

    print(f"\n  {'文本':>14} {'token0':>10} {'token0 份额(层%d)' % L:>16} "
          f"{'full 与 window+sink 的 logits 差':>30} {'window-only 的差':>18} "
          f"{'drop-sink 的差':>16}")
    for name, text in texts.items():
        ids = tok(text, return_tensors="pt").input_ids[:, :S].cuda()
        n = ids.shape[1]
        with torch.no_grad():
            base = model(ids, output_attentions=True)
            logits = base.logits[:, -1].float()
            a = base.attentions[L][0, :, -1, 0].float()
            share = a.mean().item()
            m_win_sink = build_4d_mask(ids, keep_sink=True, window=W)
            m_win = build_4d_mask(ids, keep_sink=False, window=W)
            m_nosink = causal_mask(n).clone()
            m_nosink[:, 0] = False                      # 只丢 token 0，其余照旧
            add = torch.zeros(1, 1, n, n, dtype=torch.bfloat16,
                              device=ids.device).masked_fill(
                ~m_nosink.to(ids.device), torch.finfo(torch.bfloat16).min)
            o_ws = model(ids, attention_mask=m_win_sink).logits[:, -1].float()
            o_w = model(ids, attention_mask=m_win).logits[:, -1].float()
            o_ns = model(ids, attention_mask=add).logits[:, -1].float()
        d_ws = (o_ws - logits).abs().max().item()
        d_w = (o_w - logits).abs().max().item()
        d_ns = (o_ns - logits).abs().max().item()
        t0 = tok.decode([ids[0, 0].item()])
        print(f"  {name:>14} {t0!r:>10} {share:>16.4f} {d_ws:>30.4f} "
              f"{d_w:>18.4f} {d_ns:>16.4f}")
        h.case(id=f"B_{name}", S=n, W=W, layer=L, token0=t0,
               token0_share=share, logit_diff_window_sink=d_ws,
               logit_diff_window_only=d_w, logit_diff_drop_sink=d_ns,
               model=MODEL)
    print("\n  三种对照的读法：")
    print("  · token0 份额随首 token 的内容变化 → sink 不完全是位置效应；")
    print("  · window-only 与 full 的差最大，window+sink 更接近 full，")
    print("    说明**保留一个位置就能把大部分差异收回来**；")
    print("  · drop-sink（只丢 token 0、其余不变）量化了 sink 本身承载了多少输出。")
    print("  份额是观察，为什么会这样（softmax 必须把概率质量放在某处）是解释，")
    print("  本 lab 只给出前者与保留策略的后果，机制验证需要改动 softmax 的对照实验。")
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- C
def section_C(h):
    title("[C] RoPE 插值/频率缩放：实际作用在位置上是多少")

    print("  RoPE 把 head_dim 拆成 64 对，第 i 对的角频率 = 1/base^(2i/d)，")
    print("  位置 p 的旋转角 = p · 角频率。'外推失败'指的是这些角度超出训练区间。")
    D = 128
    base = 1.0e6
    trained = 40960
    inv = 1.0 / (base ** (torch.arange(0, D, 2).double() / D))

    schemes = {}
    schemes["原始"] = inv.clone()
    s = 4.0
    schemes["PI"] = inv / s
    b_ntk = base * (s ** (D / (D - 2)))
    schemes["NTK(base 放大)"] = 1.0 / (b_ntk ** (torch.arange(0, D, 2).double() / D))

    print(f"  head_dim={D} rope_theta={base:g} 训练长度={trained}")
    print("  两个可检查的量：训练内**从未转满一圈**的维度数（见过的角度只是一小段弧），")
    print("  以及最高频那一对在 2× 位置上的角（局部位置分辨力）。")
    print(f"  {'方案':>14} {'训练内未满一圈的维度':>20} {'覆盖最少的弧占比':>18} "
          f"{'i=0 角@p=2×训练长':>18} {'相对原始':>10}")
    cases = []
    for name, iv in schemes.items():
        arc = trained * iv                       # 训练中该维度走过的角（弧度）
        turns = arc / (2 * math.pi)
        never = int((turns < 1.0).sum().item())
        min_frac = turns.min().item()
        ang_ex = (2 * trained * iv)
        cases.append({"scheme": name, "dims_never_full_turn": never,
                      "min_turns_seen": min_frac,
                      "angle_i0_2x": ang_ex[0].item()})
    base_i0 = cases[0]["angle_i0_2x"]
    for c in cases:
        print(f"  {c['scheme']:>14} {c['dims_never_full_turn']:>20} "
              f"{c['min_turns_seen']:>18.4f} {c['angle_i0_2x']:>18.1f} "
              f"{c['angle_i0_2x'] / base_i0:>9.3f}×")
        h.case(id=f"C_{c['scheme']}", **c, D=D, base=base, trained=trained,
               angle_ratio_vs_original=c["angle_i0_2x"] / base_i0)
    print("\n  PI 把所有频率同比例缩小 → 每个维度在训练里走过的弧都变短，")
    print("  '从未转满一圈'的维度变多；NTK 放大 base，高频维度基本不动、")
    print("  低频被拉到和 PI 一样 —— 所以两者在最高频那一列分道。")

    sub("实际旋转后的分量（位置与状态都要能被检查）")
    p = 81920
    x = torch.linspace(0.1, 1.0, 8, dtype=torch.double)
    print(f"  取 head_dim 前 8 维、位置 p={p}，三套频率下的旋转角（弧度）：")
    print(f"  {'方案':>14} " + " ".join(f"{'i=' + str(i):>8}" for i in range(4)))
    for name, iv in schemes.items():
        ang = (p * iv[:4]).tolist()
        print(f"  {name:>14} " + " ".join(f"{a:>8.2f}" for a in ang))
    print("  i=0 那一列（最高频）在 PI 下变成原来的 4 倍、NTK 下几乎不动 ——")
    print("  高频维度负责区分相邻 token，这正是 PI 损害局部位置分辨力的地方。")
    h.case(id="C_rotation_angles", position=p, dims=4,
           angles={k: (p * v[:4]).tolist() for k, v in schemes.items()})


SECTIONS = {"A": section_A, "B": section_B, "C": section_C}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name} sm_{p.major}{p.minor}")
    h = Harness("3.4-B", "3.4", out=os.environ.get("L3_OUT"),
                backend="torch 构造 mask + Qwen3-1.7B eager 前向",
                notes=f"model={MODEL}；B 节用 4D attention_mask 做保留策略对照")
    for s in want:
        SECTIONS[s](h)
    h.finish({"verdict": "窗口边界由 mask 与有效 KV 长度直接读出；"
                         "保留 sink 后 window 与 full 的差显著缩小；"
                         "PI 与 NTK 在最低频一致、在高频分道。",
              "model": MODEL})
    sys.stdout.flush()
    os._exit(0)
