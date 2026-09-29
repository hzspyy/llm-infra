#!/usr/bin/env python3
"""L9.6 任务 A/B/C：检索、ANN 取舍与「检索到生成」的完整成本。

三个模式：

``retrieve``（任务 A）固定语料 revision、chunk 规则、query 指令与归一化方式，用 FlatIP 生成
  精确 top-k 参照，并给出 NFCorpus 的 Recall@k / nDCG@10。相关性标注用 qrels，**不冒充生成
  答案标注**。

``scale``（任务 B）在固定 Recall@k 阈值下比较 HNSW（efSearch=16/32/64/128）与 IVF
  （nlist/nprobe 按数据量选）的构建、查询时间与内存；另在 10 万/100 万合成向量上做规模实验，
  与真实语料分开报告。

``e2e``（任务 C）对 top-k=5/20/100 接 rerank 与 prompt 注入，测各阶段与端到端；固定
  HotpotQA 200 题，报答案 EM/F1 与引用有效性。较低 recall 或更短上下文带来的加速不得
  直接称系统改进。

用法::

    python labs/L9/rag_pipeline.py retrieve --out DIR
    python labs/L9/rag_pipeline.py scale --out DIR
    python labs/L9/rag_pipeline.py e2e --out DIR --n 200
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402
from session_kv_bench import tok_len  # noqa: E402  （同一套模板渲染口径）

CHUNK_CHARS = 900
CHUNK_STRIDE = 700

QUERY_INSTRUCT = ("Instruct: Given a medical question, retrieve relevant passages\nQuery: ")


# --------------------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------------------

def load_nfcorpus():
    return T.load_nfcorpus()


def chunk_docs(docs: list[dict]) -> list[dict]:
    chunks = []
    for d in docs:
        text = (d.get("title") or "") + "\n" + d["text"]
        if len(text) <= CHUNK_CHARS:
            chunks.append({"chunk_id": f"{d['_id']}#0", "doc_id": d["_id"], "text": text})
            continue
        i = 0
        while i < len(text):
            piece = text[i:i + CHUNK_CHARS]
            chunks.append({"chunk_id": f"{d['_id']}#{i // CHUNK_STRIDE}", "doc_id": d["_id"], "text": piece})
            if i + CHUNK_CHARS >= len(text):
                break
            i += CHUNK_STRIDE
    return chunks


def load_hotpotqa(n: int, seed: int = 0) -> list[dict]:
    import pandas as pd

    files = glob.glob("/scratch/learn/models/hf/hub/datasets--hotpotqa--hotpot_qa/snapshots/*/distractor/validation-*.parquet")
    if not files:
        files = glob.glob("/scratch/learn/models/hf/hub/datasets--hotpotqa--hotpot_qa/snapshots/*/distractor/*.parquet")
    df = pd.read_parquet(files[0])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "qid": r["id"], "question": r["question"], "answer": r["answer"],
            "context_titles": list(r["context"]["title"]),
            "context_sentences": [list(s) for s in r["context"]["sentences"]],
            "supporting_titles": list(r["supporting_facts"]["title"]),
        })
    random.Random(seed).shuffle(rows)
    return rows[:n]


# --------------------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------------------

def recall_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranking[:k]) & gold) / len(gold)


def ndcg_at_k(ranking: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def em_f1(pred: str, gold: str) -> tuple[float, float]:
    p, g = normalize_answer(pred), normalize_answer(gold)
    em = 1.0 if p == g else 0.0
    pt, gt = p.split(), g.split()
    common = {}
    for t in pt:
        if t in gt:
            common[t] = min(pt.count(t), gt.count(t))
    overlap = sum(common.values())
    if not overlap:
        return em, 0.0
    prec, rec = overlap / len(pt), overlap / len(gt)
    return em, 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------------------
# 嵌入与检索
# --------------------------------------------------------------------------------------

async def embed_all(args, texts: list[str], batch: int = 32) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = await client.embeddings.create(model=args.embed_model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    await client.close()
    return out


def l2norm(vecs):
    import numpy as np

    a = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, 1e-12)


def flat_topk(index_vecs, query_vecs, k: int):
    import numpy as np

    sims = query_vecs @ index_vecs.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return order, sims


async def cmd_retrieve_async(args) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_nfcorpus()
    docs, qrels = data["docs"], data["qrels"]
    chunks = chunk_docs(docs)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
        keep = {c["doc_id"] for c in chunks}
        qrels = {k: {d: v for d, v in rel.items() if d in keep} for k, rel in qrels.items()}
        qrels = {k: v for k, v in qrels.items() if v}
    queries = [q for q in data["queries"] if q["_id"] in qrels]
    queries.sort(key=lambda q: q["_id"])
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    t_embed = time.perf_counter()
    doc_vecs = l2norm(await embed_all(args, [c["text"] for c in chunks]))
    query_vecs = l2norm(await embed_all(args, [QUERY_INSTRUCT + q["text"] for q in queries]))
    embed_s = time.perf_counter() - t_embed
    order, sims = flat_topk(doc_vecs, query_vecs, args.k)
    ks = [1, 5, 10, 20, 100]
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"ndcg@{k}": 0.0 for k in ks})
    per_query = []
    for qi, q in enumerate(queries):
        rel = qrels[q["_id"]]
        ranking = [chunks[j]["doc_id"] for j in order[qi]]
        # 同一文档的多个 chunk 去重后计算指标
        dedup = []
        for d in ranking:
            if d not in dedup:
                dedup.append(d)
        row = {"query_id": q["_id"], "num_relevant": len(rel)}
        for k in ks:
            row[f"recall@{k}"] = round(recall_at_k(dedup, set(rel), k), 4)
            row[f"ndcg@{k}"] = round(ndcg_at_k(dedup, rel, k), 4)
            metrics[f"recall@{k}"] += row[f"recall@{k}"]
            metrics[f"ndcg@{k}"] += row[f"ndcg@{k}"]
        per_query.append(row)
    n = len(queries)
    metrics = {k: round(v / n, 4) for k, v in metrics.items()}
    summary = {
        "config": {
            "corpus_docs": len(docs), "chunks": len(chunks), "queries": n,
            "chunk_chars": CHUNK_CHARS, "chunk_stride": CHUNK_STRIDE,
            "embed_model": args.embed_model, "similarity": "cosine (L2-normalized + inner product)",
            "query_instruct": QUERY_INSTRUCT,
            "revision": "BeIR/nfcorpus snapshot b5026a0e",
        },
        "embed_seconds": round(embed_s, 2),
        "metrics": metrics,
        "per_query": per_query,
    }
    (out / "retrieve.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    import numpy as np

    np.save(out / "doc_vecs.npy", doc_vecs)
    np.save(out / "query_vecs.npy", query_vecs)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({k: v for k, v in c.items()}, ensure_ascii=False) + "\n")
    with open(out / "queries.jsonl", "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_query"}, ensure_ascii=False, indent=1))
    return 0


def cmd_scale(args) -> int:
    import faiss
    import numpy as np

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"config": {"k": args.k, "queries": args.n_queries}, "real": {}, "synthetic": {}}

    doc_vecs = np.load(pathlib.Path(args.vectors) / "doc_vecs.npy")
    query_vecs = np.load(pathlib.Path(args.vectors) / "query_vecs.npy")[: args.n_queries]
    d = doc_vecs.shape[1]

    def memory_mb(index) -> float:
        try:
            return index.ntotal * d * 4 / 1e6 + (index.size() if hasattr(index, "size") else 0) / 1e6
        except Exception:  # noqa: BLE001
            return float("nan")

    def measure(index, queries, k, *, train=False):
        t0 = time.perf_counter()
        if train:
            index.train(doc_vecs)
        index.add(doc_vecs)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, exact = index.search(query_vecs, k)
        query_ms = (time.perf_counter() - t0) * 1000.0
        return {"build_s": round(build_s, 3), "query_ms_total": round(query_ms, 3),
                "query_ms_per_query": round(query_ms / max(1, len(query_vecs)), 4),
                "memory_mb_estimate": round(memory_mb(index), 2)}, exact

    flat = faiss.IndexFlatIP(d)
    flat_meta, exact = measure(flat, query_vecs, args.k)
    result["real"]["flat_exact"] = {**flat_meta, "recall@k": 1.0}

    for ef in (16, 32, 64, 128, 256):
        hnsw = faiss.IndexHNSWFlat(d, 32)
        hnsw.hnsw.efSearch = ef
        meta, got = measure(hnsw, query_vecs, args.k)
        recall = float(np.mean([len(set(exact[i]) & set(got[i])) / args.k for i in range(len(query_vecs))]))
        result["real"][f"hnsw_ef{ef}"] = {**meta, "recall@k": round(recall, 4)}

    for nlist in (64, 256):
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist)
        for nprobe in (1, 4, 16, 64):
            ivf.nprobe = nprobe
            idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist)
            idx.nprobe = nprobe
            meta, got = measure(idx, query_vecs, args.k, train=True)
            recall = float(np.mean([len(set(exact[i]) & set(got[i])) / args.k for i in range(len(query_vecs))]))
            result["real"][f"ivf_nlist{nlist}_nprobe{nprobe}"] = {**meta, "recall@k": round(recall, 4)}

    # 合成规模实验：10 万 / 100 万，与真实任务分开
    for n in (100_000, 1_000_000):
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((n, 128), dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        qs = vecs[: args.n_queries]
        exact_idx = faiss.IndexFlatIP(128)
        exact_idx.add(vecs)
        t0 = time.perf_counter()
        _, exact_syn = exact_idx.search(qs, args.k)
        exact_ms = (time.perf_counter() - t0) * 1000.0
        rows = {"flat_exact": {"query_ms_per_query": round(exact_ms / len(qs), 4), "recall@k": 1.0,
                               "memory_mb_estimate": round(n * 128 * 4 / 1e6, 1)}}
        for ef in (32, 128):
            h = faiss.IndexHNSWFlat(128, 32)
            h.hnsw.efSearch = ef
            t0 = time.perf_counter()
            h.add(vecs)
            build = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, got = h.search(qs, args.k)
            qms = (time.perf_counter() - t0) * 1000.0
            recall = float(np.mean([len(set(exact_syn[i]) & set(got[i])) / args.k for i in range(len(qs))]))
            rows[f"hnsw_ef{ef}"] = {"build_s": round(build, 2), "query_ms_per_query": round(qms / len(qs), 4),
                                    "recall@k": round(recall, 4), "memory_mb_estimate": round(n * 128 * 4 / 1e6, 1)}
        for nlist in (256, 4096):
            ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(128), 128, nlist)
            ivf.nprobe = max(1, nlist // 16)
            t0 = time.perf_counter()
            ivf.train(vecs)
            ivf.add(vecs)
            build = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, got = ivf.search(qs, args.k)
            qms = (time.perf_counter() - t0) * 1000.0
            recall = float(np.mean([len(set(exact_syn[i]) & set(got[i])) / args.k for i in range(len(qs))]))
            rows[f"ivf_nlist{nlist}_nprobe{ivf.nprobe}"] = {
                "build_s": round(build, 2), "query_ms_per_query": round(qms / len(qs), 4),
                "recall@k": round(recall, 4), "memory_mb_estimate": round(n * 128 * 4 / 1e6, 1)}
        result["synthetic"][str(n)] = rows
        del vecs
        print(f"[synthetic] n={n} done", flush=True)

    (out / "scale.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1)[:3000])
    return 0


def cmd_retrieve(args) -> int:
    """同步入口：把整段检索实验放进同一个事件循环（异步 client 不能跨 asyncio.run 复用）。"""
    return asyncio.run(cmd_retrieve_async(args))


async def cmd_prepare_async(args) -> int:
    """离线准备 HotpotQA 检索语料与问题向量（embedding 服务单独一次启动）。"""
    import numpy as np

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hotpot = load_hotpotqa(args.n, args.seed)
    corpus, gold_per_q = hotpot_corpus(hotpot)
    doc_vecs = l2norm(await embed_all(args, [c["text"] for c in corpus]))
    q_vecs = l2norm(await embed_all(args, [QUERY_INSTRUCT + h["question"] for h in hotpot]))
    np.save(out / "corpus_vecs.npy", doc_vecs)
    np.save(out / "question_vecs.npy", q_vecs)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for c in corpus:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fh:
        for h in hotpot:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    (out / "corpus_meta.json").write_text(
        json.dumps({"corpus_paragraphs": len(corpus), "questions": len(hotpot),
                    "gold_per_q": [sorted(g) for g in gold_per_q]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps({"corpus_paragraphs": len(corpus), "questions": len(hotpot)}, ensure_ascii=False))
    return 0


def cmd_prepare(args) -> int:
    return asyncio.run(cmd_prepare_async(args))


def cmd_e2e(args) -> int:
    return asyncio.run(cmd_e2e_async(args))


async def _ask(client, model, prompt, max_tokens=64):
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


_RERANKER: dict = {}


def get_reranker(args):
    """只加载一次 reranker：每个问题重新 from_pretrained 会把 0.6B 权重反复搬上 GPU。"""
    if "model" not in _RERANKER:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.rerank_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.rerank_model, torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
        model.eval()
        _RERANKER.update(tok=tok, model=model,
                         yes_id=tok.convert_tokens_to_ids("yes"),
                         no_id=tok.convert_tokens_to_ids("no"))
    return _RERANKER


def rerank_pairs(args, pairs: list[tuple[str, str]]) -> list[float]:
    """用 Qwen3-Reranker-0.6B 的 yes/no 概率给 (query, passage) 打分（HF 路径，同 5.12）。"""
    import torch

    rk = get_reranker(args)
    tok, model = rk["tok"], rk["model"]
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
              "the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
              "<|im_end|>\n<|im_start|>user\n")
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), args.rerank_batch):
            batch = pairs[i:i + args.rerank_batch]
            texts = [f"{prefix}<Instruct>: Given a web search query, retrieve relevant passages\n"
                     f"<Query>: {q}\n<Document>: {p}<|im_end|>\n<|im_start|>assistant\n" for q, p in batch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to("cuda")
            logits = model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, rk["no_id"]], logits[:, rk["yes_id"]]], dim=1)
            prob = torch.softmax(pair.float(), dim=1)[:, 1]
            scores.extend(prob.cpu().tolist())
    return scores


def build_context(picked: list[dict], budget_chars: int) -> tuple[str, list[dict]]:
    """按字符预算拼上下文：超出预算的片段被丢弃，返回实际使用的片段。"""
    used, parts, total = [], [], 0
    for c in picked:
        text = c["text"][:800]
        if total + len(text) > budget_chars and used:
            break
        used.append(c)
        parts.append(f"[{len(used)}] {text}")
        total += len(text)
    return "\n\n".join(parts), used


def hotpot_corpus(hotpot: list[dict]) -> tuple[list[dict], list[set[str]]]:
    """把固定题集的 distractor 段落并集当作检索语料；返回语料与每题的支持段落 id。"""
    corpus: list[dict] = []
    seen: set[str] = set()
    gold_per_q: list[set[str]] = []
    for h in hotpot:
        gold = set()
        for title, sentences in zip(h["context_titles"], h["context_sentences"]):
            pid = f"hp::{title}"
            if pid not in seen:
                seen.add(pid)
                corpus.append({"chunk_id": pid, "doc_id": pid, "title": title,
                               "text": title + "\n" + " ".join(sentences)})
            if title in set(h["supporting_titles"]):
                gold.add(pid)
        gold_per_q.append(gold)
    return corpus, gold_per_q


_reranker = None


async def cmd_e2e_async(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.embeddings:
        # 离线阶段已经算好：生成服务与 embedding 服务是两次启动，不能在同一进程里互相调用
        import numpy as np

        emb = pathlib.Path(args.embeddings)
        hotpot = [json.loads(l) for l in open(emb / "questions.jsonl", encoding="utf-8")]
        corpus = [json.loads(l) for l in open(emb / "corpus.jsonl", encoding="utf-8")]
        with open(emb / "corpus_meta.json", encoding="utf-8") as fh:
            gold_per_q = [set(g) for g in json.load(fh)["gold_per_q"]]
        doc_vecs = np.load(emb / "corpus_vecs.npy")
        q_vecs = np.load(emb / "question_vecs.npy")
    else:
        hotpot = load_hotpotqa(args.n, args.seed)
        corpus, gold_per_q = hotpot_corpus(hotpot)
        questions = [h["question"] for h in hotpot]
        doc_vecs = l2norm(await embed_all(args, [c["text"] for c in corpus]))
        q_vecs = l2norm(await embed_all(args, [QUERY_INSTRUCT + q for q in questions]))
    order, _sims = flat_topk(doc_vecs, q_vecs, max(args.topk))

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    progress = {"done": 0, "total": len(hotpot) * len(args.topk)}

    async def one(qi: int, h: dict, topk: int):
        async with sem:
            gold = gold_per_q[qi]
            picked = [corpus[j] for j in order[qi][:topk]]
            head_scores = None
            if args.rerank:
                scores = rerank_pairs(args, [(h["question"], c["text"]) for c in picked])
                picked = [c for _, c in sorted(zip(scores, picked), key=lambda t: -t[0])]
                head_scores = sorted(scores, reverse=True)[:3]
            context, used = build_context(picked, args.context_chars)
            retrieved_ids = [c["chunk_id"] for c in used]
            prompt = (f"Answer the question using only the numbered passages. Cite the passages you use "
                      f"as [n]. End with `Answer: <short answer>`.\n\n{context}\n\nQuestion: {h['question']}")
            t0 = time.perf_counter()
            error = None
            try:
                text = await _ask(client, args.gen_model, prompt)
            except Exception as exc:  # noqa: BLE001 - 失败请求也要进分母
                text, error = "", f"{type(exc).__name__}: {exc}"
            gen_ms = (time.perf_counter() - t0) * 1000.0
            if "Answer:" in text:
                tail = text.split("Answer:")[-1].strip()
                ans = tail.splitlines()[0] if tail.splitlines() else ""
            else:
                ans = text.strip()
            cites = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
            cited = {retrieved_ids[i - 1] for i in cites if 1 <= i <= len(retrieved_ids)}
            em, f1 = em_f1(ans, h["answer"])
            covered = len(cited & gold)
            rows.append({
                "qid": h["qid"], "topk": topk, "rerank": args.rerank,
                "prompt_chars": len(prompt), "context_chunks": len(used),
                "support_recall": round(len(set(retrieved_ids) & gold) / max(1, len(gold)), 4),
                "gen_ms": round(gen_ms, 2), "em": em, "f1": f1,
                "cited": cites[:8],
                "invalid_citation": sum(1 for i in cites if not (1 <= i <= len(retrieved_ids))),
                "has_citation": bool(cites),
                "citation_precision": round(covered / max(1, len(cited)), 4) if cited else 0.0,
                "gold_covered_by_citation": round(covered / max(1, len(gold)), 4),
                "answer": ans[:80], "rerank_scores_head": head_scores, "error": error,
            })
            progress["done"] += 1
            if progress["done"] % 50 == 0:
                print(f"[e2e] rerank={args.rerank} {progress['done']}/{progress['total']}", flush=True)

    await asyncio.gather(*(one(qi, h, topk) for qi, h in enumerate(hotpot) for topk in args.topk))
    await client.close()

    agg = {}
    for r in rows:
        key = f"topk{r['topk']}|rerank{int(r['rerank'])}"
        a = agg.setdefault(key, {"n": 0, "em": 0.0, "f1": 0.0, "gen_ms": [], "has_cite": 0, "invalid": 0,
                                 "support_recall": 0.0, "citation_precision": 0.0, "prompt_chars": []})
        a["n"] += 1
        a["em"] += r["em"]
        a["f1"] += r["f1"]
        a["gen_ms"].append(r["gen_ms"])
        a["has_cite"] += int(r["has_citation"])
        a["invalid"] += r["invalid_citation"]
        a["support_recall"] += r["support_recall"]
        a["citation_precision"] += r["citation_precision"]
        a["prompt_chars"].append(r["prompt_chars"])
    table = {k: {"n": v["n"], "em": round(v["em"] / v["n"], 4), "f1": round(v["f1"] / v["n"], 4),
                 "support_recall": round(v["support_recall"] / v["n"], 4),
                 "citation_precision": round(v["citation_precision"] / v["n"], 4),
                 "gen_ms_mean": round(statistics.fmean(v["gen_ms"]), 2),
                 "prompt_chars_mean": round(statistics.fmean(v["prompt_chars"]), 1),
                 "citation_rate": round(v["has_cite"] / v["n"], 4),
                 "errors": sum(1 for r in rows if r.get("error") and r["topk"] == int(k.split("|")[0][4:])
                               and r["rerank"] == bool(int(k.split("|")[1][6:]))),
                 "invalid_citations": v["invalid"]}
             for k, v in sorted(agg.items())}
    summary = {"config": {"n": len(hotpot), "topk": args.topk, "rerank": args.rerank,
                          "gen_model": args.gen_model, "rerank_model": args.rerank_model,
                          "corpus_paragraphs": len(corpus), "context_chars": args.context_chars,
                          "concurrency": args.concurrency,
                          "dataset": "hotpot_qa distractor/validation, fixed by seed"},
               "table": table, "rows": rows}
    (out / f"e2e-rerank{int(args.rerank)}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.6 RAG pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(base_url="http://127.0.0.1:8016/v1", embed_model="Qwen/Qwen3-Embedding-0.6B",
                  gen_model="Qwen/Qwen3-4B")

    p = sub.add_parser("retrieve")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--limit-chunks", type=int, default=0, help="冒烟用：只取前 N 个片段")
    p.add_argument("--limit-queries", type=int, default=0, help="冒烟用：只取前 N 条查询")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("scale")
    p.add_argument("--out", required=True)
    p.add_argument("--vectors", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=100)
    p.set_defaults(func=cmd_scale)

    p = sub.add_parser("e2e")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--vectors", default=None,
                   help="旧路径的片段向量目录；当前 e2e 用 --embeddings 的 HotpotQA 语料")
    p.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--topk", type=int, nargs="+", default=[5, 20, 100])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-batch", type=int, default=16)
    p.add_argument("--context-chars", type=int, default=24000,
                   help="最终上下文的最大字符数（约 6k token），超出部分丢弃并记录实际片段数")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--embeddings", default=None,
                   help="离线准备好的 HotpotQA 向量目录（生成服务不能同时提供 embeddings 接口）")
    p.set_defaults(func=cmd_e2e)

    p = sub.add_parser("prepare")
    for k, v in common.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
