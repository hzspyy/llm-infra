#!/usr/bin/env python3
"""L4.0 —— checkpoint 的其余部分：分片、命名、mmap、权重绑定、量化格式。

字节布局在 `inspect_safetensors.py` 里已经逐字节拆过。这里补另外五件事：

  [A] 分片与 index：一个张量怎么在 5 个文件里被找到
  [B] 命名约定：磁盘上的名字 与 引擎内部的名字 不是一回事
  [C] mmap 与零拷贝：get_tensor 到底做了什么
  [D] tie_word_embeddings：配置说绑定了，文件里却存了两份（实测）
  [E] 量化格式：AWQ / GPTQ 的 checkpoint 长什么样，多存了哪些张量

用法：
    python checkpoint_formats.py
    python checkpoint_formats.py D E
"""

import glob
import json
import os
import struct
import sys

MB = 1024 * 1024
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def snap(repo):
    """repo 名 -> 本地 snapshot 目录。"""
    d = f"{HUB}/models--{repo.replace('/', '--')}/snapshots"
    got = sorted(glob.glob(d + "/*"))
    return got[0] if got else None


def st_header(path):
    """自己读 safetensors 的头，不用库。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n


# ---------------------------------------------------------------- A
def section_A():
    title("[A] 分片与 index：一个张量怎么在 5 个文件里被找到")

    d = snap("Qwen/Qwen3-8B")
    if not d:
        print("  没有 Qwen3-8B，跳过"); return
    files = sorted(glob.glob(d + "/model-*.safetensors"))
    print(f"  {len(files)} 个分片：")
    total = 0
    for f in files:
        sz = os.path.getsize(f)
        total += sz
        hdr, hn = st_header(f)
        ntensor = len([k for k in hdr if k != "__metadata__"])
        print(f"    {os.path.basename(f):<32} {sz / MB:>8.1f} MB  "
              f"头 {hn:>6} B  {ntensor:>4} 个张量")
    print(f"  合计 {total / MB / 1024:.2f} GiB")

    idx = d + "/model.safetensors.index.json"
    if os.path.exists(idx):
        j = json.load(open(idx))
        wm = j["weight_map"]
        print(f"\n  index.json: metadata={j.get('metadata')}")
        print(f"  weight_map 有 {len(wm)} 条，形如 张量名 -> 哪个文件")
        for k in list(wm)[:4]:
            print(f"    {k:<50} -> {wm[k]}")
        print("    ...")
        # 同一层的权重是不是在同一个文件里
        from collections import defaultdict
        by_file = defaultdict(list)
        for k, v in wm.items():
            by_file[v].append(k)
        print(f"\n  每个分片里的层号范围：")
        for f in sorted(by_file):
            layers = sorted({int(k.split(".")[2]) for k in by_file[f]
                             if k.startswith("model.layers.")})
            other = [k for k in by_file[f] if not k.startswith("model.layers.")]
            rng = f"layers {layers[0]}..{layers[-1]}" if layers else "无 layer"
            print(f"    {f:<34} {rng:<20} 另有 {len(other)} 个非 layer 张量: "
                  f"{other[:2]}")
        print("\n  注意层号范围**是重叠的**（0..7 和 7..17）——")
        print("  分片纯粹按字节大小切，**不保证一层的张量在同一个文件里**。")
        l7 = {k: v for k, v in wm.items() if k.startswith("model.layers.7.")}
        from collections import Counter as _C
        print(f"  第 7 层的 {len(l7)} 个张量分布：{dict(_C(v[-24:] for v in l7.values()))}")
        for k in sorted(l7):
            print(f"    {k.split('model.layers.')[1]:<40} {l7[k][-24:]}")
        print("  q/k/v_proj 在第 1 个文件，其余在第 2 个 —— 同一层被切开了。")
        print("  所以「加载完一层就能开始算」需要跨文件协调，不是免费的。")

    sub("找一个具体张量：它在哪个文件的哪些字节")
    name = "model.layers.20.self_attn.q_proj.weight"
    fn = json.load(open(idx))["weight_map"][name]
    hdr, hn = st_header(d + "/" + fn)
    info = hdr[name]
    a, b = info["data_offsets"]
    print(f"  {name}")
    print(f"    文件      {fn}")
    print(f"    dtype     {info['dtype']}   shape {info['shape']}")
    print(f"    数据区偏移 [{a}, {b})  共 {b - a} 字节")
    print(f"    文件内绝对偏移 = 8 + {hn} + {a} = {8 + hn + a}")
    import math
    n_elem = math.prod(info["shape"])
    print(f"    校验: {n_elem} 元素 × 2 字节(bf16) = {n_elem * 2} "
          f"{'✓' if n_elem * 2 == b - a else '✗'}")


# ---------------------------------------------------------------- B
def section_B():
    title("[B] 命名：磁盘上的名字不是引擎里的名字")

    d = snap("Qwen/Qwen3-1.7B")
    f = sorted(glob.glob(d + "/*.safetensors"))[0]
    hdr, _ = st_header(f)
    names = [k for k in hdr if k != "__metadata__"]
    layer0 = sorted(n for n in names if n.startswith("model.layers.0."))
    print(f"  Qwen3-1.7B 第 0 层在磁盘上的 {len(layer0)} 个张量：")
    for n in layer0:
        print(f"    {n:<52} {str(hdr[n]['shape']):<16} {hdr[n]['dtype']}")

    print("\n  注意 q_proj / k_proj / v_proj 是**三个独立张量**。")
    print("  但 0.0 里我们看到 vLLM 用的是一个融合的 QKVParallelLinear ——")
    print("  加载时会把这三个拼成一个大矩阵，好处是一次 GEMM 代替三次。")
    print("  所以「权重名」在磁盘、HF 实现、推理引擎里是三套，加载器负责翻译。")

    sub("非 layer 的张量")
    other = sorted(n for n in names if not n.startswith("model.layers."))
    for n in other:
        print(f"    {n:<52} {str(hdr[n]['shape']):<16} {hdr[n]['dtype']}")

    sub("q_norm / k_norm：Qwen3 比 Qwen2 多出来的东西")
    qn = [n for n in layer0 if "q_norm" in n or "k_norm" in n]
    if qn:
        for n in qn:
            print(f"    {n:<52} {str(hdr[n]['shape'])}")
        print("    这是 QK-Norm：对 Q 和 K 在 head_dim 上做 RMSNorm 之后再算 attention。")
        print("    shape 是 [head_dim]，不是 [hidden]——说明它作用在每个头内部。")
        print("    实现里漏掉它，模型能跑但输出是错的（4.1 会对齐每一层）。")
    else:
        print("    这个模型没有 q_norm/k_norm")


# ---------------------------------------------------------------- C
def section_C():
    title("[C] mmap 与零拷贝：get_tensor 到底做了什么")

    import time
    import torch
    from safetensors import safe_open
    d = snap("Qwen/Qwen3-1.7B")
    f = sorted(glob.glob(d + "/*.safetensors"))[0]
    print(f"  文件 {os.path.basename(f)}  {os.path.getsize(f) / MB:.1f} MB")

    with safe_open(f, framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        name = "model.layers.0.mlp.down_proj.weight"
        t0 = time.perf_counter()
        t = sf.get_tensor(name)
        dt_get = (time.perf_counter() - t0) * 1000
        nbytes = t.numel() * t.element_size()
        t0 = time.perf_counter()
        s = float(t.float().abs().sum())          # 强制把每一页都碰一遍
        dt_touch = (time.perf_counter() - t0) * 1000

    print(f"  张量 {name}")
    print(f"    shape {tuple(t.shape)}  {nbytes / MB:.1f} MB")
    print(f"    get_tensor()          {dt_get:>8.3f} ms  -> "
          f"{nbytes / (dt_get / 1000) / 1e9:>8.1f} GB/s")
    print(f"    之后完整读一遍        {dt_touch:>8.3f} ms  -> "
          f"{nbytes / (dt_touch / 1000) / 1e9:>8.1f} GB/s")
    print("\n  get_tensor 快得不真实，因为它只建了一个 mmap 视图，**没有读盘**。")
    print("  真正的读发生在你第一次碰那些页的时候（缺页中断）。")
    print("  L1.5 踩过这个坑：给 get_tensor 计时得到「104 GB/s 冷加载」。")

    sub("加载到 GPU 才是真的搬")
    t0 = time.perf_counter()
    g = t.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    dt_h2d = (time.perf_counter() - t0) * 1000
    print(f"    .to('cuda')           {dt_h2d:>8.3f} ms  -> "
          f"{nbytes / (dt_h2d / 1000) / 1e9:>8.1f} GB/s   （PCIe，见 L1.3）")
    del g
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D():
    title("[D] tie_word_embeddings：配置说绑定了，文件里却存了两份")

    import torch
    from safetensors import safe_open

    for repo in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"]:
        d = snap(repo)
        if not d:
            continue
        cfg = json.load(open(d + "/config.json"))
        tie = cfg.get("tie_word_embeddings")
        files = sorted(glob.glob(d + "/*.safetensors"))
        loc = {}
        for f in files:
            for k in st_header(f)[0]:
                if k in ("lm_head.weight", "model.embed_tokens.weight"):
                    loc[k] = f
        print(f"\n  {repo}")
        print(f"    config: tie_word_embeddings = {tie}")
        print(f"    分片数 {len(files)}，合计 "
              f"{sum(os.path.getsize(x) for x in files) / MB:.1f} MB")
        for k, f in sorted(loc.items()):
            print(f"    {k:<32} 在 {os.path.basename(f)}")
        if len(loc) == 2:
            with safe_open(loc["model.embed_tokens.weight"], framework="pt",
                           device="cpu") as sf:
                e = sf.get_tensor("model.embed_tokens.weight")
            with safe_open(loc["lm_head.weight"], framework="pt",
                           device="cpu") as sf:
                l = sf.get_tensor("lm_head.weight")
            same = torch.equal(e, l)
            nb = l.numel() * l.element_size()
            tot = sum(os.path.getsize(x) for x in files)
            print(f"    两者 shape {tuple(e.shape)}，逐位完全相同: {same}")
            print(f"    max|diff| = {(e.float() - l.float()).abs().max().item()}")
            print(f"    lm_head 那份 {nb / MB:.1f} MB，占 checkpoint 的 {nb / tot:.1%}")

    print("\n  读出来的事实：Qwen3-1.7B 的 config 写着 tie=True，")
    print("  但 checkpoint **确实存了一份独立的 lm_head.weight**，")
    print("  而且和 embed_tokens **逐位相同**。它单独占了一个分片。")
    print()
    print("  所以「绑定就不存第二份」这个说法对这个 checkpoint 不成立。")
    print("  这 593.5 MB 是**冗余**的：加载器按 tie=True 走绑定路径时不会用它。")
    print("  教训：config 的字段描述的是**模型结构**，不是**文件内容**。")
    print("  想知道文件里有什么，只能去读文件。")

    sub("这一份权重有多大")
    d = snap("Qwen/Qwen3-1.7B")
    cfg = json.load(open(d + "/config.json"))
    V, H = cfg["vocab_size"], cfg["hidden_size"]
    print(f"    vocab={V}  hidden={H}  ->  {V * H} 个参数 = "
          f"{V * H * 2 / MB:.1f} MB (bf16)")
    print(f"    占 1.7B 模型参数量的 {V * H / 1.7e9:.1%}")
    print("    小模型上 embedding 占比特别高（词表大、hidden 小），")
    print("    这也是小模型更倾向于用 tie 的原因。")


# ---------------------------------------------------------------- E
def section_E():
    title("[E] 量化 checkpoint：多存了什么，少存了什么")

    import math
    repos = ["Qwen/Qwen2.5-1.5B-Instruct-AWQ",
             "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"]
    for repo in repos:
        d = snap(repo)
        if not d:
            print(f"  {repo} 未下载，跳过"); continue
        cfg = json.load(open(d + "/config.json"))
        qc = cfg.get("quantization_config", {})
        print(f"\n  === {repo} ===")
        print(f"  quantization_config: {json.dumps(qc, ensure_ascii=False)}")
        files = sorted(glob.glob(d + "/*.safetensors"))
        size = sum(os.path.getsize(f) for f in files)
        hdr, _ = st_header(files[0])
        names = [k for k in hdr if k != "__metadata__"]
        print(f"  文件合计 {size / MB:.1f} MB，张量 {len(names)} 个")
        # 第 0 层的 q_proj 相关的所有张量
        rel = sorted(n for n in names if "layers.0.self_attn.q_proj" in n)
        print(f"  第 0 层 q_proj 相关的张量：")
        for n in rel:
            info = hdr[n]
            nb = info["data_offsets"][1] - info["data_offsets"][0]
            print(f"    {n:<48} {str(info['shape']):<16} "
                  f"{info['dtype']:<6} {nb / 1024:>8.1f} KB")

    sub("对照：未量化的同一个位置")
    d = snap("Qwen/Qwen3-1.7B")
    hdr, _ = st_header(sorted(glob.glob(d + "/*.safetensors"))[0])
    for n in sorted(k for k in hdr if "layers.0.self_attn.q_proj" in k):
        info = hdr[n]
        nb = info["data_offsets"][1] - info["data_offsets"][0]
        print(f"    {n:<48} {str(info['shape']):<16} "
              f"{info['dtype']:<6} {nb / 1024:>8.1f} KB")

    print("\n  读法：量化格式把一个 [out, in] 的 bf16 矩阵拆成三样东西 ——")
    print("    qweight  打包好的低位整数（多个权重塞进一个 int32）")
    print("    scales   每组一个缩放系数（组大小就是 group_size）")
    print("    qzeros   每组一个零点（非对称量化才有）")
    print("  所以「4 bit 量化」的实际每权重位数要把 scales/qzeros 算进去，")
    print("  下一节按 group_size 算一下真实的等效位宽。")

    sub("两个文件差 443 MB，差在哪")
    from collections import Counter
    for repo in repos:
        d = snap(repo)
        if not d:
            continue
        cfg = json.load(open(d + "/config.json"))
        h = {}
        for f in sorted(glob.glob(d + "/*.safetensors")):
            h.update({k: v for k, v in st_header(f)[0].items() if k != "__metadata__"})
        def nb(k):
            return h[k]["data_offsets"][1] - h[k]["data_offsets"][0]
        tot = sum(nb(k) for k in h)
        print(f"\n  {repo.split('/')[-1]}  tie={cfg.get('tie_word_embeddings')}  "
              f"总 {tot / MB:.1f} MB")
        print(f"    dtype 分布 {dict(Counter(v['dtype'] for v in h.values()))}")
        for k in sorted(h, key=nb, reverse=True)[:3]:
            print(f"    {k:<42} {str(h[k]['shape']):<18} "
                  f"{h[k]['dtype']:<5} {nb(k) / MB:>7.1f} MB")
    print("\n  两处：")
    print("  1. **AWQ 那份多存了一个 lm_head.weight（445.1 MB）**，尽管 tie=True；")
    print("     GPTQ 那份没有。443 MB 的差额基本就是这一项。")
    print("     这和 [D] 节 Qwen3-1.7B 是同一个模式 —— 不是个例。")
    print("  2. **embedding 表根本没有被量化**，两边都是 F16 的 445.1 MB。")
    print("     GPTQ 文件 1096.5 MB 里它占 41%。所以「4 bit 模型」这个说法")
    print("     只描述了线性层；词表大、模型小的时候，未量化的部分才是大头。")

    sub("真实等效位宽")
    for repo in repos:
        d = snap(repo)
        if not d:
            continue
        cfg = json.load(open(d + "/config.json"))
        qc = cfg.get("quantization_config", {})
        bits = qc.get("bits", 4)
        gs = qc.get("group_size", 128)
        # 每 gs 个权重：gs*bits 位 qweight + 1 个 fp16 scale + gs*bits/gs 位零点
        per_group = gs * bits + 16 + bits      # scale fp16 + zero (bits 位)
        print(f"  {repo.split('/')[-1]:<36} bits={bits} group_size={gs} "
              f"-> 等效 {per_group / gs:.2f} bit/权重 "
              f"(压缩比 {16 / (per_group / gs):.2f}× vs bf16)")


SECTIONS = {"A": section_A, "B": section_B, "C": section_C,
            "D": section_D, "E": section_E}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SECTIONS)
    for s in want:
        SECTIONS[s]()
    sys.stdout.flush()
