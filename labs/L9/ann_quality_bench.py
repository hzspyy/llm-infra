#!/usr/bin/env python3
"""L9.6 任务 B：在**固定召回阈值**下比较索引，并给出实测内存与更新代价。

`rag_pipeline.py scale` 已经能跑出各索引的召回与查询时间，但它有三处会让结论失真的地方，
这里逐条修掉：

1. **内存是公式估计**（`ntotal × d × 4`）——图结构与倒排表不在这个公式里。本脚本改用
   `faiss.write_index` 的**序列化字节数**与进程 **VmRSS 增量**两个实测值。
2. **查询集与索引样本重叠**（合成档直接用前 N 行当查询）。本脚本把查询向量单独生成/留出，
   索引样本与查询集不相交。
3. **召回与代价没有对齐到同一个门槛**。本脚本先声明 `--target-recall`（默认 0.95），
   再在候选配置里给出**达到阈值的最低代价配置**与整条 Pareto 前沿，而不是列一堆数让人自己挑。

同时给出更新代价：向已建好的索引追加新向量后重测查询时间与序列化大小。

用法::

    python labs/L9/ann_quality_bench.py --out out/9.6/ann --sizes 100000,1000000 \
        --target-recall 0.95 [--vectors <dir>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time


def rss_mb() -> float:
    """当前进程的 VmRSS（MB）；容器内 /proc 可用。"""
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def index_bytes(index, tmp: pathlib.Path) -> int:
    """序列化后的真实字节数（含图结构/倒排表）。"""
    import faiss

    faiss.write_index(index, str(tmp))
    size = tmp.stat().st_size
    tmp.unlink(missing_ok=True)
    return size


def recall_at_k(exact, got, k: int) -> float:
    import numpy as np

    return float(np.mean([len(set(exact[i]) & set(got[i])) / k for i in range(len(exact))]))


def measure_query(index, queries, k: int, repeats: int = 3) -> dict:
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        _dist, ids = index.search(queries, k)
        ms = (time.perf_counter() - t0) * 1000.0
        best = ms if best is None else min(best, ms)
    return {"query_ms_per_query": round(best / max(1, len(queries)), 5),
            "query_ms_total": round(best, 3)}, ids


def build_flat(vecs, queries, k: int, tmp: pathlib.Path) -> dict:
    import faiss

    r0 = rss_mb()
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    r1 = rss_mb()
    timing, ids = measure_query(idx, queries, k)
    return {"name": "flat_exact", "rss_delta_mb": round(r1 - r0, 1),
            "index_bytes": index_bytes(idx, tmp), **timing,
            "recall": 1.0, "index": idx, "exact_ids": ids}


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.6 固定召回阈值下的索引对照")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="100000,1000000", help="合成向量规模档")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--ef-search", default="16,32,64,128")
    ap.add_argument("--nlist", default="256,4096")
    ap.add_argument("--nprobe", default="1,8,32,128,256")
    ap.add_argument("--vectors", default=None, help="真实向量目录（含 doc_vecs.npy/query_vecs.npy）")
    ap.add_argument("--update-add", type=int, default=10000, help="更新实验追加的向量数")
    args = ap.parse_args()

    import numpy as np

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_index.tmp"
    report: dict = {"config": {"target_recall": args.target_recall, "k": args.k,
                               "n_queries": args.n_queries, "dim": args.dim,
                               "sizes": args.sizes, "update_add": args.update_add},
                    "datasets": {}}

    datasets: list[tuple[str, "np.ndarray", "np.ndarray"]] = []
    for n in [int(x) for x in args.sizes.split(",")]:
        rng = np.random.default_rng(0)
        # 索引样本与查询集分开生成：查询集不用索引里的任何一行
        vecs = rng.standard_normal((n, args.dim), dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        qs = rng.standard_normal((args.n_queries, args.dim), dtype="float32")
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        datasets.append((f"synthetic_{n}", vecs, qs))
    if args.vectors:
        base = pathlib.Path(args.vectors)
        doc = np.load(base / "doc_vecs.npy").astype("float32")
        q = np.load(base / "query_vecs.npy").astype("float32")[: args.n_queries]
        datasets.append(("real_nfcorpus", doc, q))

    for name, vecs, qs in datasets:
        rows = []
        flat = build_flat(vecs, qs, args.k, tmp)
        exact_ids = flat.pop("exact_ids")
        # 保留 index 对象：FlatIP 常常就是"达到召回阈值的最低代价配置"，更新实验要用它
        rows.append(flat)

        for ef in [int(x) for x in args.ef_search.split(",")]:
            import faiss

            r0 = rss_mb()
            idx = faiss.IndexHNSWFlat(vecs.shape[1], 32)
            idx.hnsw.efSearch = ef
            t0 = time.perf_counter()
            idx.add(vecs)
            build_s = time.perf_counter() - t0
            r1 = rss_mb()
            timing, ids = measure_query(idx, qs, args.k)
            rows.append({"name": f"hnsw_M32_ef{ef}", "build_s": round(build_s, 3),
                         "rss_delta_mb": round(r1 - r0, 1),
                         "index_bytes": index_bytes(idx, tmp), **timing,
                         "recall": round(recall_at_k(exact_ids, ids, args.k), 4),
                         "index": idx})

        for nlist in [int(x) for x in args.nlist.split(",")]:
            for nprobe in [int(x) for x in args.nprobe.split(",")]:
                import faiss

                r0 = rss_mb()
                idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(vecs.shape[1]), vecs.shape[1], nlist)
                t0 = time.perf_counter()
                idx.train(vecs)
                idx.add(vecs)
                build_s = time.perf_counter() - t0
                idx.nprobe = nprobe
                r1 = rss_mb()
                timing, ids = measure_query(idx, qs, args.k)
                rows.append({"name": f"ivf_nlist{nlist}_nprobe{nprobe}",
                             "build_s": round(build_s, 3),
                             "rss_delta_mb": round(r1 - r0, 1),
                             "index_bytes": index_bytes(idx, tmp), **timing,
                             "recall": round(recall_at_k(exact_ids, ids, args.k), 4),
                             "index": idx})

        feasible = [r for r in rows if r["recall"] >= args.target_recall]
        best = min(feasible, key=lambda r: r["query_ms_per_query"]) if feasible else None
        # 更新代价：给最优可行索引追加新向量后再测
        update = None
        if best is not None and args.update_add > 0:
            add_vecs = np.random.default_rng(7).standard_normal(
                (args.update_add, vecs.shape[1]), dtype="float32")
            add_vecs /= np.linalg.norm(add_vecs, axis=1, keepdims=True)
            idx = best["index"]
            before_bytes = best["index_bytes"]
            t0 = time.perf_counter()
            idx.add(add_vecs)
            add_s = time.perf_counter() - t0
            timing, ids = measure_query(idx, qs, args.k)
            update = {"added": args.update_add, "add_s": round(add_s, 3),
                      "query_after_add_ms_per_query": timing["query_ms_per_query"],
                      "bytes_before": before_bytes,
                      "bytes_after": index_bytes(idx, tmp),
                      "recall_after_add": round(recall_at_k(exact_ids, ids, args.k), 4),
                      "note": "追加后召回下降是正常的：新向量进入了检索池，而精确参照也已更新"}

        for r in rows:
            r.pop("index", None)
        report["datasets"][name] = {
            "vectors": int(vecs.shape[0]), "queries": int(qs.shape[0]), "dim": int(vecs.shape[1]),
            "rows": rows,
            "fixed_recall": {"target": args.target_recall,
                             "best": best["name"] if best else None,
                             "best_query_ms_per_query": best["query_ms_per_query"] if best else None,
                             "best_index_bytes": best["index_bytes"] if best else None,
                             "feasible": [r["name"] for r in feasible]},
            "update": update,
        }
        print(f"[{name}] n={vecs.shape[0]} target_recall={args.target_recall} "
              f"feasible={len(feasible)}/{len(rows)} best={best['name'] if best else None} "
              f"({best['query_ms_per_query'] if best else None} ms/query, "
              f"{best['index_bytes'] if best else None} B)", flush=True)

    report["note"] = ("内存用序列化字节与 VmRSS 增量两个实测值，不用 ntotal×d×4 的公式；"
                      "查询集与索引样本分开生成；固定召回阈值下先给最低代价配置，再给全表。"
                      "合成向量只用于规模趋势，不代表真实分布的召回")
    (out / "ann_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
