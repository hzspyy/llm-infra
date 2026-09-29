#!/usr/bin/env python3
"""L2.7-C lab · 用一个真实 Qwen3-1.7B block 比较三条路：eager / compile / export+AOTInductor。

- 形状与权重都取真实 checkpoint：`Qwen/Qwen3-1.7B` 的 config 与第 0 层权重
  （从本机 HF 缓存里的 safetensors 直接读，不整模型加载）。
- 三条路各自分开记：**编译时间 / 加载时间 / 稳态**，以及显存峰值。
- 同时比较两种 AOT 接口：
    `torch._inductor.aoti_compile_and_package` + `aoti_load_package`（.pt2 包）
    `torch._export.aot_compile` + `aot_load`（当前版本已标 deprecated，返回 .so + Python callable）
  以及 `torch.compile` 自己（返回 Python callable，编译在首次调用时发生）。

    CPATH=$CUDA_HOME/include LIBRARY_PATH=$CUDA_HOME/lib \
      python labs/L2/qwen3_block_aoti.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import torch

SNAP = "/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots"
B, S = 4, 512


def find_snapshot() -> pathlib.Path:
    snaps = sorted(p for p in pathlib.Path(SNAP).glob("*") if p.is_dir())
    if not snaps:
        raise SystemExit(f"找不到 Qwen3-1.7B 快照：{SNAP}")
    return snaps[-1]


def load_layer0(snap: pathlib.Path):
    """从 safetensors 索引里挑出第 0 层需要的分片并只读该层权重。"""
    import json as _json
    from safetensors import safe_open

    index = _json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if k.startswith("model.layers.0.")]
    shards = sorted({index[k] for k in keys})
    sd = {}
    for shard in shards:
        with safe_open(snap / shard, framework="pt") as f:
            for k in keys:
                if index[k] == shard:
                    sd[k[len("model.layers.0."):]] = f.get_tensor(k)
    return sd, keys, shards


def compare(y, ref, atol=2e-2, rtol=1e-2):
    """bf16 的 block 不能要求位级一致：报 max|d|、相对 RMS 与 allclose 判定。"""
    d = (y.float() - ref.float()).abs()
    refn = ref.float()
    return {"max_abs_diff": float(d.max()),
            "rel_rms": float(d.norm() / (refn.norm() + 1e-30)),
            "allclose_atol_rtol": (atol, rtol),
            "allclose": bool(torch.allclose(y.float(), refn, atol=atol, rtol=rtol)),
            "has_nan": bool(torch.isnan(y).any())}


def bench(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir).resolve()   # output_path 必须是绝对路径，否则 ld 在别的 cwd 下失败
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    snap = find_snapshot()
    cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
    layer_sd, keys, shards = load_layer0(snap)
    layer = Qwen3DecoderLayer(cfg, layer_idx=0)
    missing, unexpected = layer.load_state_dict(layer_sd, strict=False)

    dev, dt = "cuda", torch.bfloat16
    layer = layer.eval().to(device=dev, dtype=dt)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    torch.manual_seed(0)
    hid = torch.randn(B, S, cfg.hidden_size, device=dev, dtype=dt)
    # 真实的 RoPE 频率（cos/sin 的最后一维必须等于 head_dim）
    pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=dev).float() / head_dim))
    ang = pos.float().unsqueeze(-1) * inv
    ang = torch.cat((ang, ang), dim=-1)
    cos = torch.cos(ang).to(dt)
    sin = torch.sin(ang).to(dt)

    payload: dict = {
        "snapshot": str(snap), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0),
        "config": {"hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size,
                    "num_attention_heads": cfg.num_attention_heads,
                    "num_key_value_heads": cfg.num_key_value_heads, "head_dim": head_dim},
        "input": {"batch": B, "seq": S, "dtype": str(dt)},
        "layer0_weights": {"tensors": len(keys), "shards": shards,
                            "missing": list(missing), "unexpected": list(unexpected)[:5]},
        "paths": {},
    }
    print(f"=== Qwen3-1.7B layer0 · {payload['device']} · torch {torch.__version__} ===")
    print(f"    hidden={cfg.hidden_size} inter={cfg.intermediate_size} "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} head_dim={head_dim}")
    print(f"    第 0 层权重 {len(keys)} 个张量，来自 {len(shards)} 个分片；"
          f"missing={list(missing)}")

    def run_eager():
        with torch.no_grad():
            return layer(hid, position_embeddings=(cos, sin), use_cache=False)

    # ---- 1. eager ----
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ref = run_eager()
    torch.cuda.synchronize()
    payload["paths"]["eager"] = {
        **bench(run_eager),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "out_shape": list(ref.shape),
    }
    print(f"    eager       稳态 {payload['paths']['eager']['median_ms']:.3f} ms  "
          f"峰值 {payload['paths']['eager']['peak_reserved_mb']:.0f} MB")

    # ---- 2. torch.compile（默认模式；Python callable，首次调用编译）----
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        comp = torch.compile(run_eager, dynamic=False)
        y = comp()
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        cmp = compare(y, ref)
        payload["paths"]["compile"] = {
            **bench(comp), "first_call_s": compile_s, **cmp,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "returns": "Python callable",
        }
        print(f"    compile     首次 {compile_s:7.2f} s（含编译）  稳态 "
              f"{payload['paths']['compile']['median_ms']:.3f} ms  "
              f"max|d|={cmp['max_abs_diff']:.3e} rel_rms={cmp['rel_rms']:.2e}")
    except Exception as exc:                                   # noqa: BLE001
        payload["paths"]["compile"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"    compile     [失败] {payload['paths']['compile']['error'][:110]}")

    # ---- 3. export + AOTInductor 包（.pt2）----
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        pkg = out / "qwen3_layer0.pt2"

        class Wrap(torch.nn.Module):
            def forward(self, h, c, s):
                with torch.no_grad():
                    return layer(h, position_embeddings=(c, s), use_cache=False)

        w = Wrap().eval()
        t0 = time.perf_counter()
        ep = torch.export.export(w, (hid, cos, sin))
        export_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        package = torch._inductor.aoti_compile_and_package(ep, package_path=str(pkg))
        aoti_compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fn = torch._inductor.aoti_load_package(package)
        load_s = time.perf_counter() - t0
        y = fn(hid, cos, sin)
        cmp = compare(y, ref)
        payload["paths"]["aoti_package"] = {
            **bench(lambda: fn(hid, cos, sin)),
            "export_s": export_s, "aoti_compile_s": aoti_compile_s, "load_s": load_s,
            "artifact_mb": pkg.stat().st_size / 2**20 if pkg.exists() else None,
            **cmp, "returns": "非 Python 制品（.pt2 + .so）",
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        print(f"    AOTI 包      export {export_s:.2f} s + 编译 {aoti_compile_s:.2f} s，"
              f"加载 {load_s:.4f} s，制品 "
              f"{payload['paths']['aoti_package']['artifact_mb']:.2f} MB，"
              f"稳态 {payload['paths']['aoti_package']['median_ms']:.3f} ms，"
              f"max|d|={cmp['max_abs_diff']:.3e}")
    except Exception as exc:                                   # noqa: BLE001
        payload["paths"]["aoti_package"] = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        print(f"    AOTI 包     [失败] {payload['paths']['aoti_package']['error'][:140]}")

    # ---- 4. 已弃用的 export.aot_compile + aot_load（对照两种 AOT 接口）----
    try:
        so = out / "qwen3_layer0_export.so"

        class Wrap2(torch.nn.Module):
            def forward(self, h, c, s):
                with torch.no_grad():
                    return layer(h, position_embeddings=(c, s), use_cache=False)

        t0 = time.perf_counter()
        path = torch._export.aot_compile(Wrap2().eval(), (hid, cos, sin),
                                          options={"aot_inductor.output_path": str(so)})
        c_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fn2 = torch._export.aot_load(path, dev)
        l_s = time.perf_counter() - t0
        y2 = fn2(hid, cos, sin)
        cmp2 = compare(y2, ref)
        payload["paths"]["export_aot"] = {
            **bench(lambda: fn2(hid, cos, sin)),
            "compile_s": c_s, "load_s": l_s, **cmp2,
            "returns": "Python callable（加载 .so）",
            "artifact_mb": pathlib.Path(path).stat().st_size / 2**20
            if pathlib.Path(path).exists() else None,
        }
        print(f"    export.aot  编译 {c_s:.2f} s，加载 {l_s:.4f} s，稳态 "
              f"{payload['paths']['export_aot']['median_ms']:.3f} ms，"
              f"max|d|={cmp2['max_abs_diff']:.3e}")
    except Exception as exc:                                   # noqa: BLE001
        payload["paths"]["export_aot"] = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        print(f"    export.aot  [失败] {payload['paths']['export_aot']['error'][:140]}")

    (out / "qwen3_block_aoti.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out / 'qwen3_block_aoti.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
