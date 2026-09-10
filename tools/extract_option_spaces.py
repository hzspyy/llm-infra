#!/usr/bin/env python3
"""把已装库里**可枚举的选项空间**导出成 JSON，供 coverage_check.py 查漏。

思路：很多"我漏了 X"的情况，X 其实就写在某个 Literal / Enum / 注册表里。
与其等人指出来，不如把这些集合抽出来，和正文对一遍。

在 crater 上跑：
    python tools/extract_option_spaces.py > results/crater/option_spaces.json
"""

import inspect
import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

out = {}


def lit(mod_src, name):
    """从源码里抠一个 Literal[...] 的成员。"""
    m = re.search(rf"\b{name}\s*=\s*Literal\[(.*?)\]", mod_src, re.S)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def add(key, values, note=""):
    vals = sorted(set(v for v in values if v))
    if vals:
        out[key] = {"values": vals, "n": len(vals), "note": note}


# ---- vLLM: 投机解码方法 ----
try:
    import vllm.config.speculative as sp
    src = inspect.getsource(sp)
    methods = lit(src, "SpeculativeMethod")
    for extra in ["MTPModelTypes", "EagleModelTypes", "NgramGPUTypes",
                  "DSparkModelTypes", "DFlashModelTypes"]:
        add(f"vllm.speculative.{extra}", lit(src, extra),
            "投机解码的一类方法")
    add("vllm.speculative.SpeculativeMethod", methods, "投机解码顶层方法")
except Exception as e:                                        # noqa: BLE001
    out["_err_speculative"] = str(e)[:200]

# ---- vLLM: CUDA Graph 模式 ----
try:
    from vllm.config.compilation import CUDAGraphMode
    add("vllm.CUDAGraphMode", [m.name for m in CUDAGraphMode], "图执行模式")
except Exception as e:                                        # noqa: BLE001
    out["_err_cudagraph"] = str(e)[:200]

# ---- vLLM: 量化方法 ----
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    add("vllm.QUANTIZATION_METHODS", list(QUANTIZATION_METHODS), "支持的量化格式")
except Exception as e:                                        # noqa: BLE001
    out["_err_quant"] = str(e)[:200]

# ---- vLLM: attention 后端 ----
try:
    from vllm.attention.backends.registry import AttentionBackendEnum
    add("vllm.AttentionBackendEnum", [m.name for m in AttentionBackendEnum],
        "attention 后端")
except Exception as e:                                        # noqa: BLE001
    try:
        from vllm.platforms.interface import _Backend
        add("vllm._Backend", [m.name for m in _Backend], "attention 后端")
    except Exception as e2:                                   # noqa: BLE001
        out["_err_attnbackend"] = f"{e} | {e2}"[:200]

# ---- vLLM: KV cache dtype ----
try:
    import vllm.config.cache as cc
    add("vllm.CacheDType", lit(inspect.getsource(cc), "CacheDType"),
        "KV cache 的 dtype")
    add("vllm.PrefixCachingHashAlgo",
        lit(inspect.getsource(cc), "PrefixCachingHashAlgo"), "前缀缓存哈希")
except Exception as e:                                        # noqa: BLE001
    out["_err_cache"] = str(e)[:200]

# ---- torch: SDPA 后端 ----
try:
    from torch.nn.attention import SDPBackend
    add("torch.SDPBackend", [m.name for m in SDPBackend], "SDPA 后端")
except Exception as e:                                        # noqa: BLE001
    out["_err_sdpa"] = str(e)[:200]

# ---- torch: dynamo 后端 ----
try:
    import torch._dynamo
    add("torch.dynamo_backends", torch._dynamo.list_backends(), "torch.compile 后端")
except Exception as e:                                        # noqa: BLE001
    out["_err_dynamo"] = str(e)[:200]

# ---- torch: 浮点 dtype ----
try:
    import torch
    dts = [n for n in dir(torch)
           if n.startswith(("float", "bfloat")) and isinstance(
               getattr(torch, n, None), torch.dtype)]
    add("torch.float_dtypes", dts, "浮点格式")
except Exception as e:                                        # noqa: BLE001
    out["_err_dtypes"] = str(e)[:200]

# ---- vLLM: 支持的模型架构（只取 MoE / 多模态两类的名字）----
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    add("vllm.moe_archs", [a for a in archs if "Moe" in a or "MoE" in a],
        "vLLM 支持的 MoE 架构")
except Exception as e:                                        # noqa: BLE001
    out["_err_archs"] = str(e)[:200]

json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
