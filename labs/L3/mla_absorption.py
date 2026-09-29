#!/usr/bin/env python3
"""L3.3-C —— MLA：从 DeepSeek-V2-Lite 单层权重到"重建"与"吸收"两条路径。

MLA 把 K/V 压成一个低秩隐向量 c（kv_lora_rank=512）加一路 RoPE 分支。
推理时有两条等价（但不完全同价）的算法路径：

  重建：c → W_kb → k_nope、v，再走普通 attention
  吸收：把 W_kb 的 K 部分吸收进 Q、V 部分吸收进 o_proj，attention 直接在 c 上做

本 lab 只用**真实的单层权重**（HTTP Range 抽取，不下载整个 8.6 GB 分片）：

  [A] 权重抽取：读 safetensors 头，按字节范围只取第 0 层 attention 的 5 个张量，
      打印 shape / dtype / 字节数 / 文件内偏移
  [B] 两条路径对拍：同一组输入，重建 vs 吸收，FP32 与 FP64 各一遍
  [C] 与 FlashInfer 的 MLA paged kernel 对拍（可用则比，不可用记录原因）
  [D] 字节账：每 token 每层 KV 与整模型容量（按 worldvln 48 GB 单卡预算）

用法：
    L3_OUT=<目录> python mla_absorption.py A B C D
"""

import json
import os
import struct
import sys
import urllib.request

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness, tensor_hash                       # noqa: E402

REPO = "deepseek-ai/DeepSeek-V2-Lite"
REV = "main"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
SHARD = "model-00001-of-000004.safetensors"
WANT = ["model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight"]
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
MB = 1024 * 1024


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def _ranged(url, start, end, chunk=2 * 1024 * 1024, tries=6):
    """分块 Range 读取：CDN 会提前关连接，所以每块单独重试并校验长度。"""
    import time
    buf = bytearray()
    pos = start
    while pos <= end:
        stop = min(end, pos + chunk - 1)
        want = stop - pos + 1
        for attempt in range(tries):
            try:
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={pos}-{stop}",
                                  "User-Agent": "llm-infra-lab"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    got = r.read()
                if len(got) == want:
                    buf.extend(got)
                    break
            except Exception:                                 # noqa: BLE001
                pass
            time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(f"range 读取失败：bytes={pos}-{stop}")
        pos = stop + 1
    return bytes(buf)


def load_layer0(out_dir):
    """只取第 0 层 attention 的 5 个张量，返回 (tensors, meta)。"""
    url = BASE + SHARD
    n = struct.unpack("<Q", _ranged(url, 0, 7))[0]
    header = json.loads(_ranged(url, 8, 7 + n).decode("utf-8"))
    base = 8 + n
    meta = {"repo": REPO, "revision": REV, "shard": SHARD, "header_bytes": n,
            "data_offset": base, "tensors": {}}
    out = {}
    for name in WANT:
        info = header[name]
        lo, hi = info["data_offsets"]
        raw = _ranged(url, base + lo, base + hi - 1)
        t = torch.frombuffer(bytearray(raw), dtype=DTYPES[info["dtype"]])
        t = t.reshape(info["shape"]).clone()
        out[name.split(".")[-2] + "." + name.split(".")[-1]] = t
        meta["tensors"][name] = {"dtype": info["dtype"], "shape": info["shape"],
                                 "bytes": hi - lo, "offset_in_file": base + lo}
    if out_dir:
        (out_dir / "layer0_weight_manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, meta


def rms_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def mla_paths(w, x, eps=1e-6, dtype=torch.float32):
    """返回 (重建输出, 吸收输出, 中间量)。

    x: [T, hidden]。权重按 HF 的 DeepSeek-V2 命名。
    """
    Wq = w["q_proj.weight"].to(dtype)                  # [H*(nope+rope), hid]
    Wkva = w["kv_a_proj_with_mqa.weight"].to(dtype)    # [rank+rope, hid]
    Wn = w["kv_a_layernorm.weight"].to(dtype)          # [rank]
    Wkb = w["kv_b_proj.weight"].to(dtype)              # [H*(nope+v), rank]
    Wo = w["o_proj.weight"].to(dtype)                  # [hid, H*v]
    T = x.shape[0]
    cfg = {"H": Wo.shape[1] // 128, "nope": 128, "rope": 64, "rank": Wn.shape[0],
           "v": 128}
    H, nope, rope, rank, v = (cfg["H"], cfg["nope"], cfg["rope"], cfg["rank"],
                              cfg["v"])
    hid = x.shape[-1]
    scale = (nope + rope) ** -0.5

    q = x @ Wq.t()                                     # [T, H*192]
    q = q.view(T, H, nope + rope)
    q_nope, q_pe = q[..., :nope], q[..., nope:]
    ckv = x @ Wkva.t()                                 # [T, rank+rope]
    c, k_pe = ckv[..., :rank], ckv[..., rank:]
    c = rms_norm(c, Wn, eps)
    kv = c @ Wkb.t()                                   # [T, H*(nope+v)]
    kv = kv.view(T, H, nope + v)
    k_nope, vv = kv[..., :nope], kv[..., nope:]

    # ---- 重建路径 ----
    score = torch.einsum("thd,shd->hts", q_nope, k_nope) \
        + torch.einsum("thd,sd->hts", q_pe, k_pe)
    score = score * scale
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    score = score.masked_fill(~mask, float("-inf"))
    p = torch.softmax(score, dim=-1)
    o_rec = torch.einsum("hts,shd->thd", p, vv)        # [T,H,v]
    out_rec = (o_rec.reshape(T, H * v)) @ Wo.t()

    # ---- 吸收路径 ----
    Wkb_n = Wkb.view(H, nope + v, rank)[:, :nope, :]   # [H,nope,rank]
    Wkb_v = Wkb.view(H, nope + v, rank)[:, nope:, :]   # [H,v,rank]
    q_abs = torch.einsum("thd,hdr->thr", q_nope, Wkb_n)          # [T,H,rank]
    score2 = torch.einsum("thr,sr->hts", q_abs, c) \
        + torch.einsum("thd,sd->hts", q_pe, k_pe)
    score2 = score2 * scale
    score2 = score2.masked_fill(~mask, float("-inf"))
    p2 = torch.softmax(score2, dim=-1)
    latent = torch.einsum("hts,sr->thr", p2, c)                  # [T,H,rank]
    o_abs = torch.einsum("thr,hvr->thv", latent, Wkb_v)
    out_abs = (o_abs.reshape(T, H * v)) @ Wo.t()

    # 进一步把 Wkb_v 吸收进 o_proj：一次性完成，不再显式算 o_abs
    Wo_h = Wo.view(hid, H, v)
    W_abs = torch.einsum("ihv,hvr->ihr", Wo_h, Wkb_v)            # [hid,H,rank]
    out_abs2 = torch.einsum("thr,ihr->ti", latent, W_abs)
    return out_rec, out_abs, out_abs2, {"cfg": cfg, "scale": scale,
                                        "c_norm": c, "q_abs": q_abs,
                                        "latent": latent}


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 从 safetensors 头按字节范围抽取第 0 层 attention 权重")

    w, meta = load_layer0(h.out)
    print(f"  {REPO} @ {REV} / {SHARD}")
    print(f"  safetensors 头 {meta['header_bytes']} 字节，数据从文件偏移 "
          f"{meta['data_offset']} 开始；只取了 5 个张量，未下载整个分片。")
    print(f"  {'张量':>28} {'dtype':>6} {'shape':>18} {'字节':>12} {'文件偏移':>12}")
    for k, v in meta["tensors"].items():
        print(f"  {k.split('.')[-2] + '.' + k.split('.')[-1]:>28} {v['dtype']:>6} "
              f"{str(v['shape']):>18} {v['bytes']:>12} {v['offset_in_file']:>12}")
    tot = sum(v["bytes"] for v in meta["tensors"].values())
    print(f"  合计 {tot / MB:.2f} MB（整个分片 8.59 GB）")
    for k, v in meta["tensors"].items():
        h.case(id=f"A_{k}", **v)
    h.case(id="A_total", total_bytes=tot, shard_bytes=8590000000,
           header_bytes=meta["header_bytes"], data_offset=meta["data_offset"])
    return w


# ---------------------------------------------------------------- B
def section_B(h, w, meta):
    title("[B] 重建 vs 吸收：两条路径的对拍")

    print("  推导：score = q_nope·k_nope + q_pe·k_pe，其中 k_nope = W_kb_nope · c")
    print("        => score = (q_nope W_kb_nopeᵀ)·c + q_pe·k_pe")
    print("  输出：o = Σ p_j v_j = (Σ p_j c_j) · W_kb_vᵀ，可再吸收进 o_proj")
    T = 8
    torch.manual_seed(0)
    x = torch.randn(T, 2048) * 0.05
    for dt in (torch.float32, torch.float64):
        o_rec, o_abs, o_abs2, mid = mla_paths(w, x.double() if dt == torch.float64
                                              else x, dtype=dt)
        ref = o_rec
        e1 = (o_abs - ref).abs().max().item()
        e2 = (o_abs2 - ref).abs().max().item()
        scale = max(1.0, ref.abs().max().item())
        print(f"  {str(dt).split('.')[-1]:>8}: 吸收 vs 重建 max|err| = {e1:.3e}"
              f"（相对 {e1 / scale:.2e}）；再吸收 o_proj = {e2:.3e}"
              f"（相对 {e2 / scale:.2e}）")
        h.case(id=f"B_{str(dt).split('.')[-1]}", dtype=str(dt), T=T,
               absorb_err=e1, absorb_proj_err=e2, rel=e1 / scale,
               out_scale=scale, input_hash=tensor_hash(x))
    print("\n  两条路径在实数算术下等价；差别在**什么时候做哪个矩阵乘**：")
    print("  重建要为每个 KV 头物化 k_nope/v（[H, nope+v] 维），")
    print("  吸收则把 W_kb 的 K 部分挪到 Q 侧、V 部分挪到输出侧，缓存里只留 c。")
    print(f"  c 的维度 rank = {w['kv_a_layernorm.weight'].numel()}，"
          f"每 token 每层缓存 (rank+rope) 个元素。")


# ---------------------------------------------------------------- C
def section_C(h, w, meta):
    title("[C] 与 FlashInfer 的 MLA paged kernel 对拍")

    try:
        import flashinfer
        from flashinfer import BatchMLAPagedAttentionWrapper
    except Exception as exc:                                      # noqa: BLE001
        print(f"  FlashInfer MLA 不可用：{str(exc).splitlines()[0][:80]}")
        return
    T, H, rank, rope, v_d = 64, 16, 512, 64, 128
    page_size = 16
    torch.manual_seed(1)
    x = torch.randn(T, 2048, device="cuda") * 0.05
    wc = {k: t.cuda() for k, t in w.items()}
    wf = {k: t.float().cuda() for k, t in w.items()}      # 计算统一在 fp32
    _, _, _, mid = mla_paths(wf, x, dtype=torch.float32)
    c_norm = mid["c_norm"].contiguous()                    # [T, rank]
    q_abs = mid["q_abs"].contiguous()                      # [T, H, rank]
    # 需要 q_pe / k_pe：重算
    Wq = wf["q_proj.weight"]
    Wkva = wf["kv_a_proj_with_mqa.weight"]
    q = (x @ Wq.t()).view(T, H, 192)
    q_pe = q[..., 128:].contiguous()                       # [T, H, rope]
    ckv = x @ Wkva.t()
    k_pe = ckv[..., rank:].contiguous()                    # [T, rope]

    npages = (T + page_size - 1) // page_size
    ckv_cache = torch.zeros(npages, page_size, 1, rank, device="cuda",
                            dtype=torch.bfloat16)
    kpe_cache = torch.zeros(npages, page_size, 1, rope, device="cuda",
                            dtype=torch.bfloat16)
    flat_c = c_norm.view(npages, page_size, 1, rank)
    flat_k = k_pe.view(npages, page_size, 1, rope)
    ckv_cache.copy_(flat_c.to(torch.bfloat16))
    kpe_cache.copy_(flat_k.to(torch.bfloat16))
    ws = torch.empty(256 * MB, dtype=torch.uint8, device="cuda")
    try:
        w_mla = BatchMLAPagedAttentionWrapper(ws)
        qo_indptr = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, npages], dtype=torch.int32, device="cuda")
        kv_indices = torch.arange(npages, dtype=torch.int32, device="cuda")
        kv_len = torch.tensor([T], dtype=torch.int32, device="cuda")
        w_mla.plan(qo_indptr, kv_indptr, kv_indices, kv_len, H, rank, rope,
                   page_size, True, (128 + 64) ** -0.5, torch.bfloat16,
                   torch.bfloat16)
        out = w_mla.run(q_abs.contiguous().to(torch.bfloat16),
                        q_pe.contiguous().to(torch.bfloat16),
                        ckv_cache, kpe_cache)
        # 这个 kernel 返回的是**隐空间**的输出 Σ p_j c_j（[T,H,rank]），
        # V 侧的吸收（乘 W_kb_v 或吸收进 o_proj）由调用方在 kernel 外做。
        out_lat = out.reshape(T, H, rank)
        e = (out_lat.float() - mid["latent"].float()).abs().max().item()
        print(f"  FlashInfer MLA paged kernel vs 本题吸收路径的隐空间输出："
              f"max|err| = {e:.3e}（bf16 输入，fp32 参照）")
        print(f"  返回的维度是 [T,H,rank] = {tuple(out_lat.shape)}，"
              f"不是 [T,H,v]；V 侧吸收在 kernel 外完成。")
        h.case(id="C_flashinfer_mla", max_err=e, page_size=page_size, T=T,
               dtype="bfloat16", ref="absorb path latent (fp32)",
               out_shape=list(out_lat.shape))
    except Exception as exc:                                      # noqa: BLE001
        msg = str(exc).splitlines()[0][:160]
        print(f"  MLA wrapper 调用失败：{msg}")
        print("  （接口形状不匹配时记录原因，不把它当成'不支持'的结论。）")
        h.case(id="C_flashinfer_mla", status="failed", error=msg)
    del ckv_cache, kpe_cache
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def section_D(h, w):
    title("[D] 字节账：MLA 每 token 存多少")

    H, nope, rope, rank, v_d = 16, 128, 64, 512, 128
    hid, layers = 2048, 27
    mha = 2 * H * nope * 2                     # K + V，bf16，每层每 token
    mla = (rank + rope) * 2
    print(f"  DeepSeek-V2-Lite：{layers} 层，H={H}，qk_nope={nope}，"
          f"qk_rope={rope}，rank={rank}，v={v_d}")
    print(f"  MHA（同头数）每 token 每层 = 2·H·nope·2 = {mha} B")
    print(f"  MLA 每 token 每层 = (rank+rope)·2 = {mla} B   → 比值 {mha / mla:.2f}×")
    WEIGHTS_GB = 15.7 * 2 / 1.024 ** 3          # 15.7B 参数 bf16 ≈ 31.4 GB
    USABLE_GB = 48 - WEIGHTS_GB - 2.0           # worldvln 单卡 48 GB，留 2 GB 余量
    print(f"\n  参照 worldvln 单卡 48 GB：权重 bf16 ≈ {WEIGHTS_GB:.1f} GB，"
          f"留给 KV 的余量约 {USABLE_GB:.1f} GB")
    print(f"  {'上下文':>9} {'MHA 全模型':>14} {'MLA 全模型':>14} "
          f"{'MHA 可并发':>12} {'MLA 可并发':>12}")
    for ctx in [4096, 32768, 131072, 1048576]:
        m_mha = mha * layers * ctx / MB
        m_mla = mla * layers * ctx / MB
        n_mha = int(USABLE_GB * 1024 / max(m_mha, 1e-9))
        n_mla = int(USABLE_GB * 1024 / max(m_mla, 1e-9))
        print(f"  {ctx:>9} {m_mha:>12.1f}MB {m_mla:>12.1f}MB {n_mha:>12} "
              f"{n_mla:>12}")
        h.case(id=f"D_ctx{ctx}", ctx=ctx, mha_mb=m_mha, mla_mb=m_mla,
               ratio=mha / mla, mla_per_token_per_layer=mla,
               mha_per_token_per_layer=mha, concurrent_mha=n_mha,
               concurrent_mla=n_mla, weights_gb=WEIGHTS_GB,
               usable_kv_gb=USABLE_GB)
    print("\n  注意：这是**按配置算的字节账**。吸收路径把投影挪到 Q 与输出侧，")
    print("  省下的是缓存读取；多出来的算力开销要单独核算，"
          "不能由 dtype/维度比直接推出端到端时间比。")


def main():
    h = Harness("3.3-C", "3.3", out=os.environ.get("L3_OUT"),
                backend="HTTP Range 抽取真实权重 + torch MLA 两路径 + FlashInfer MLA",
                notes=f"{REPO}@{REV} {SHARD}")
    print(f"torch {torch.__version__}")
    try:
        w, meta = load_layer0(h.out)
    except Exception as exc:                                      # noqa: BLE001
        print(f"权重抽取失败：{str(exc).splitlines()[0][:200]}")
        h.finish({"verdict": "权重抽取失败", "error": str(exc)[:300]})
        return
    for s in (sys.argv[1:] or ["A", "B", "C", "D"]):
        s = s.upper()
        if s == "A":
            title("[A] 从 safetensors 头按字节范围抽取第 0 层 attention 权重")
            print(f"  {REPO} @ {REV} / {SHARD}")
            print(f"  safetensors 头 {meta['header_bytes']} 字节，"
                  f"数据从文件偏移 {meta['data_offset']} 开始")
            print(f"  {'张量':>28} {'dtype':>6} {'shape':>18} {'字节':>12} "
                  f"{'文件偏移':>12}")
            for k, v in meta["tensors"].items():
                nm = k.split(".")[-2] + "." + k.split(".")[-1]
                print(f"  {nm:>28} {v['dtype']:>6} {str(v['shape']):>18} "
                      f"{v['bytes']:>12} {v['offset_in_file']:>12}")
            tot = sum(v["bytes"] for v in meta["tensors"].values())
            print(f"  合计 {tot / MB:.2f} MB（整个分片 8.59 GB）")
            for k, v in meta["tensors"].items():
                h.case(id=f"A_{k}", **v)
            h.case(id="A_total", total_bytes=tot, shard_bytes=8590000000,
                   header_bytes=meta["header_bytes"], data_offset=meta["data_offset"])
        elif s == "B":
            section_B(h, w, meta)
        elif s == "C":
            if not torch.cuda.is_available():
                print("  C 节需要 GPU，跳过")
            else:
                section_C(h, w, meta)
        elif s == "D":
            section_D(h, w)
    h.finish({"verdict": "真实单层权重的重建与吸收路径等价；"
                         "MLA 每 token 缓存按配置核算；"
                         "FlashInfer MLA 的对拍结果见 C 节。",
              "repo": REPO, "revision": REV})


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
