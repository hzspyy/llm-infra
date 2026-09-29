#!/usr/bin/env python3
"""L5.2 任务 C · 前缀缓存的**缓存键身份**：到底哪些字段进了块哈希。

修订计划问的是：

    固定 input_ids，改变模型/adapter revision、位置规则与多模态预处理身份；
    分析哪些字段进入真实缓存键；显式展示正确失效及不被系统自动识别的更新。

这里不靠猜。直接调用 vLLM 0.29.0 自己的三个入口：

    get_hash_fn_by_name(...)       -> 引擎实际用的哈希函数
    init_none_hash(...)            -> 首块的哨兵父哈希
    get_request_block_hasher(...)  -> 引擎实际用的逐块哈希器

再用真实 `Request` / `LoRARequest` 对象构造请求，对同一串 token 改变**一个字段**，
比较块哈希序列。凡是进了键的字段，哈希必然不同；没进的，哈希逐位相同——
后者就是「引擎不会自动失效」的那一类更新。

多模态部分（``(mm identifier, 块内偏移)``）在本机没有可用的 processor，
只从源码确认字段来源，不在这里伪造张量；实测入口见 4.5/4.8。

用法（在装了 vLLM 的环境里，CPU 即可）：
    python cache_key_identity.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

BLOCK = 16
TOKENS = [1000 + (i * 37) % 50000 for i in range(96)]      # 6 个满块


def engine_hasher(block_size: int = BLOCK):
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher, init_none_hash)
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)
    return get_request_block_hasher(block_size, fn)


def make_request(rid, tokens, lora=None, salt=None, sampling=None, parent_hashes=None):
    from vllm import SamplingParams
    from vllm.v1.request import Request
    sp = sampling or SamplingParams(max_tokens=8, temperature=0.0)
    r = Request(request_id=rid, prompt_token_ids=list(tokens),
                sampling_params=sp, pooling_params=None,
                lora_request=lora, cache_salt=salt)
    if parent_hashes:
        r.block_hashes = list(parent_hashes)
    return r


def hashes(hasher, **kw) -> list[bytes]:
    return [bytes(h) for h in hasher(make_request(kw.pop("rid", "r"), TOKENS, **kw))]


def first_diff(a: list[bytes], b: list[bytes]) -> int | None:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def prefix_share(a: list[bytes], b: list[bytes]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from vllm.lora.request import LoRARequest

    hasher = engine_hasher()
    out, rep = [], {}

    base = hashes(hasher, rid="base")
    out.append(f"L5.2-C 缓存键身份 · block_size={BLOCK} · {len(base)} 个满块 · "
               f"哈希算法 sha256（引擎实际入口 get_request_block_hasher）")
    out.append(f"  {'用例':<34}{'首个不同块':>11}{'共有前缀块':>11}{'结论':>28}")
    rep["base_n_blocks"] = len(base)

    def row(name, h, verdict_when_same, verdict_when_diff):
        fd = first_diff(base, h)
        ps = prefix_share(base, h)
        verdict = verdict_when_same if fd is None else verdict_when_diff
        out.append(f"  {name:<34}{(fd if fd is not None else '—'):>11}{ps:>11}"
                   f"{verdict:>28}")
        return dict(first_diff=fd, prefix_blocks=ps, verdict=verdict)

    rep["cases"] = {}
    rep["cases"]["lora_name_A"] = row(
        "LoRA 名字 A（其余不变）",
        hashes(hasher, rid="loraA", lora=LoRARequest("A", 1, "/tmp/a")),
        "不失效（异常）", "进键：第 0 块即不同")
    rep["cases"]["lora_name_B_vs_A"] = None
    hA = hashes(hasher, rid="loraA2", lora=LoRARequest("A", 1, "/tmp/a"))
    hB = hashes(hasher, rid="loraB", lora=LoRARequest("B", 2, "/tmp/b"))
    fdAB = first_diff(hA, hB)
    out.append(f"  {'LoRA 名字 A vs B':<34}{fdAB if fdAB is not None else '—':>11}"
               f"{prefix_share(hA, hB):>11}{'进键：名字不同即不同' if fdAB == 0 else '异常':>28}")
    rep["cases"]["lora_name_B_vs_A"] = dict(first_diff=fdAB,
                                            prefix_blocks=prefix_share(hA, hB))

    # 同名不同权重：键里只有名字
    hA2 = hashes(hasher, rid="loraA_same_name",
                 lora=LoRARequest("A", 7, "/tmp/a-different-weights"))
    fd_same = first_diff(hA, hA2)
    out.append(f"  {'同名 A、不同路径/权重':<34}"
               f"{(fd_same if fd_same is not None else '—'):>11}"
               f"{prefix_share(hA, hA2):>11}"
               f"{'不进键：同名即同键' if fd_same is None else '异常':>28}")
    rep["cases"]["lora_same_name_diff_weights"] = dict(
        first_diff=fd_same, prefix_blocks=prefix_share(hA, hA2))

    # cache_salt：源码里只加在第 0 块的 extra_keys 上，但父链会把差异带下去
    hs = hashes(hasher, rid="salt", salt="tenant-1")
    fd_s = first_diff(base, hs)
    tail_same = sum(1 for x, y in zip(base[1:], hs[1:]) if x == y)
    out.append(f"  {'cache_salt = tenant-1':<34}"
               f"{(fd_s if fd_s is not None else '—'):>11}"
               f"{prefix_share(base, hs):>11}"
               f"{f'进键：salt 只加在第 0 块，尾 {len(base) - 1} 块相同 {tail_same}':>28}")
    rep["cases"]["cache_salt"] = dict(first_diff=fd_s,
                                      prefix_blocks=prefix_share(base, hs),
                                      tail_same=tail_same)

    # 采样参数不进键
    from vllm import SamplingParams
    sampled = hashes(hasher, rid="sampling",
                     sampling=SamplingParams(max_tokens=64, temperature=0.9,
                                             top_p=0.5, seed=1234))
    fd_sm = first_diff(base, sampled)
    out.append(f"  {'temperature/top_p/seed 改变':<34}"
               f"{(fd_sm if fd_sm is not None else '—'):>11}"
               f"{prefix_share(base, sampled):>11}"
               f"{'不进键' if fd_sm is None else '异常':>28}")
    rep["cases"]["sampling_params"] = dict(first_diff=fd_sm,
                                           prefix_blocks=prefix_share(base, sampled))

    # 父块链：同样的块 token 跟在不同父块后面
    parent_of_second_block = [base[0]]
    tail_tokens = TOKENS[BLOCK:2 * BLOCK]
    h_tail = hashes(hasher, rid="tail", parent_hashes=parent_of_second_block)
    h_tail_alt = hashes(hasher, rid="tail2", parent_hashes=[hA[0]])
    fd_p = first_diff(h_tail, h_tail_alt)
    out.append(f"  {'同一块 token、不同父块哈希':<34}"
               f"{(fd_p if fd_p is not None else '—'):>11}"
               f"{prefix_share(h_tail, h_tail_alt):>11}"
               f"{'进键：滚动哈希' if fd_p == 0 else '异常':>28}")
    rep["cases"]["parent_chain"] = dict(first_diff=fd_p,
                                        prefix_blocks=prefix_share(h_tail, h_tail_alt))

    out.append("")
    out.append("多模态（源码确认，未在本机跑真实张量）：extra_keys 里带 "
               "`(mm_feature.identifier, offset - start_token_idx)`，见 "
               "v1/core/kv_cache_utils.py `_gen_mm_extra_hash_keys`；"
               "prompt_embeds 带每块 sha256，见 `_gen_prompt_embeds_extra_hash_keys`。")
    out.append("模型权重/revision 不在键里：换模型要换引擎，进程内的块表本来就是空的，"
               "所以「同一引擎内改权重」不是它承诺支持的更新方式。")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "cache_key_identity.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "cache_key_identity.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"\n写入 {args.out}/cache_key_identity.txt 与 cache_key_identity.json")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
