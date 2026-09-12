#!/usr/bin/env python3
"""最小池化批处理：打包寻址 vs padding 到矩形。

只处理一件事——一个 batch 里长度不一的序列，怎么凑成一次前向，
以及两种凑法在**结果相同**的前提下差多少计算量。

打包（packed）：所有 token 首尾相接成一条扁平数组，序列之间用
`query_start_loc` 记录起点。前向算完拿到的是 [total_tokens, d] 的扁平张量，
池化层按 cursor 把每一段取出来。vLLM 走的就是这条路。

padding：把每条序列补到批内最长，凑成 [B, L_max, d]。池化层按 mask 求平均。

两者的结果一样，代价不一样：padding 多算的部分是 (L_max - L_i) 这些位置。

    python mini_pooling.py
"""
from __future__ import annotations

import json
import random


def pack(seqs):
    """把变长序列拼成扁平数组，返回 (ids, query_start_loc)。"""
    flat, starts = [], [0]
    for s in seqs:
        flat.extend(s)
        starts.append(len(flat))
    return flat, starts


def pad(seqs):
    """补齐到批内最长，返回 (ids_2d, mask)。"""
    L = max(len(s) for s in seqs)
    ids = [[0] * L for _ in seqs]
    mask = [[0.0] * L for _ in seqs]
    for i, s in enumerate(seqs):
        for j, t in enumerate(s):
            ids[i][j] = t
            mask[i][j] = 1.0
    return ids, mask


def pool_packed(hidden, starts, method):
    """hidden: [total_tokens, d] 的扁平列表。"""
    out = []
    for i in range(len(starts) - 1):
        seg = hidden[starts[i]:starts[i + 1]]
        if method == "MEAN":
            d = len(seg[0])
            out.append([sum(row[k] for row in seg) / len(seg) for k in range(d)])
        elif method == "CLS":
            out.append(list(seg[0]))
        elif method == "LAST":
            out.append(list(seg[-1]))
        else:
            raise ValueError(method)
    return out


def pool_padded(hidden, mask, method):
    """hidden: [B, L_max, d]；mask: [B, L_max]。padding 位置不参与。"""
    out = []
    for row, m in zip(hidden, mask):
        if method == "MEAN":
            n = sum(m)
            d = len(row[0])
            out.append([sum(row[j][k] * m[j] for j in range(len(row))) / n for k in range(d)])
        elif method == "CLS":
            out.append(list(row[0]))
        elif method == "LAST":
            out.append(list(row[int(sum(m)) - 1]))
        else:
            raise ValueError(method)
    return out


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def self_test(seed=512):
    rng = random.Random(seed)
    d = 4
    report = {"equivalence": [], "waste": []}

    for trial in range(200):
        lengths = [rng.randint(1, 40) for _ in range(rng.randint(1, 12))]
        seqs = [[rng.randint(0, 99) for _ in range(n)] for n in lengths]
        total = sum(lengths)
        # 同一个 token 位置给同一个 hidden 向量，保证两种批法的输入一致
        hidden_flat = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(total)]

        _, starts = pack(seqs)
        ids_2d, mask = pad(seqs)
        hidden_2d = []
        pos = 0
        for i, n in enumerate(lengths):
            row = hidden_flat[pos:pos + n] + [[0.0] * d] * (len(ids_2d[0]) - n)
            hidden_2d.append(row)
            pos += n

        for method in ("MEAN", "CLS", "LAST"):
            a = pool_packed(hidden_flat, starts, method)
            b = pool_padded(hidden_2d, mask, method)
            assert close(a, b), (method, lengths, a, b)

        L_max = max(lengths)
        report["waste"].append({
            "n_seqs": len(lengths), "real_tokens": total,
            "padded_tokens": len(lengths) * L_max,
            "ratio": round(len(lengths) * L_max / total, 4)})

    report["equivalence"] = {"methods": ["MEAN", "CLS", "LAST"], "trials": 200,
                             "result": "打包与 padding 的输出逐元素相同（1e-9 内）"}
    ratios = sorted(r["ratio"] for r in report["waste"])
    report["waste_summary"] = {
        "min": ratios[0], "median": ratios[len(ratios) // 2], "max": ratios[-1],
        "note": "随机长度 1–40、每批 1–12 条；padding 让算的 token 数变成中位数 "
                f"{ratios[len(ratios) // 2]:.2f} 倍"}
    return report


def fixed_length_buckets(lengths, bucket=8):
    """按长度分桶后再 padding，桶内长度接近，浪费小得多。"""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    total_padded, real = 0, sum(lengths)
    for i in range(0, len(order), bucket):
        grp = order[i:i + bucket]
        total_padded += len(grp) * max(lengths[j] for j in grp)
    return {"buckets": (len(order) + bucket - 1) // bucket,
            "padded_tokens": total_padded, "real_tokens": real,
            "ratio": round(total_padded / real, 4)}


if __name__ == "__main__":
    r = self_test()
    rng = random.Random(7)
    real = [rng.randint(4, 512) for _ in range(256)]
    r["bucketing_demo"] = {
        "no_bucket": {"padded_tokens": len(real) * max(real), "real_tokens": sum(real),
                      "ratio": round(len(real) * max(real) / sum(real), 3)},
        "bucket_8": fixed_length_buckets(real, 8),
        "bucket_32": fixed_length_buckets(real, 32),
        "note": "长度 4–512 均匀随机、256 条；分桶让 pad 比例从「按批内最长」降到接近 1",
    }
    print(json.dumps(r, ensure_ascii=False, indent=2))
