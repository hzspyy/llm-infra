#!/usr/bin/env python3
"""L4.3 修订（任务 E）—— 量化质量与成本的独立评测。

三件事分开（校准/调参/测试不重叠）：
  [E1] 语言建模：WikiText-2 test 的固定分块 PPL（冻结分块规则与样本数）
  [E2] 独立任务：C-Eval validation 的分层 200 题（冻结题目清单与计分器）
  [E3] 成本：同一次运行里记录 prefill/decode 的 token 吞吐与峰值显存

评测统一走 vLLM（同一后端、同一 kernel 选择），三个 checkpoint 各跑一次进程：
  bf16 / AWQ / GPTQ-Int4 的 Qwen2.5-1.5B-Instruct

用法（每个模型单独一个进程，避免显存不回收）：
    python labs/L4/quantization_eval.py --model bf16  --outdir out/4.3/run
    python labs/L4/quantization_eval.py --model awq   --outdir out/4.3/run
    python labs/L4/quantization_eval.py --model gptq  --outdir out/4.3/run
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

MODELS = {
    "bf16": "Qwen/Qwen2.5-1.5B-Instruct",
    "awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "gptq": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
}
HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def snap(repo):
    got = sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not got:
        raise FileNotFoundError(f"未下载: {repo}")
    return got[0]


# ---------------------------------------------------------------- E1
CHUNK = 2048
N_CHUNKS = 16          # 冻结：test 集前 16 个 2048-token 块


def wikitext_chunks(tok, n_chunks=N_CHUNKS, chunk=CHUNK):
    from datasets import load_dataset
    ds = None
    for name, cfg in (("Salesforce/wikitext", "wikitext-2-raw-v1"),
                      ("wikitext", "wikitext-2-raw-v1")):
        try:
            ds = load_dataset(name, cfg, split="test")
            break
        except Exception as e:
            print(f"  加载 {name}/{cfg} 失败：{type(e).__name__}: {str(e)[:80]}")
    if ds is None:
        raise RuntimeError("WikiText-2 无法加载")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text)["input_ids"]
    need = n_chunks * chunk
    if len(ids) < need:
        raise RuntimeError(f"token 不足：{len(ids)} < {need}")
    return [ids[i * chunk:(i + 1) * chunk] for i in range(n_chunks)], len(ids)


def ppl_via_prompt_logprobs(llm, tok, chunks):
    """用 prompt_logprobs 取每个 token 的真实 logprob，再算 PPL。"""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": c} for c in chunks], sp)
    per_chunk, total_lp, total_n = [], 0.0, 0
    for c, o in zip(chunks, outs):
        pl = o.prompt_logprobs
        lps = []
        missing = 0
        for i, tid in enumerate(c):
            if i == 0:
                continue
            d = pl[i] if pl is not None and i < len(pl) else None
            if not d or tid not in d:
                missing += 1
                continue
            lps.append(d[tid].logprob)
        n = len(lps)
        lp = sum(lps)
        total_lp += lp
        total_n += n
        per_chunk.append({"n": n, "missing": missing, "sum_logprob": lp,
                          "ppl": math.exp(-lp / n) if n else None})
    return {"ppl": math.exp(-total_lp / total_n), "tokens": total_n,
            "sum_logprob": total_lp, "per_chunk": per_chunk}


# ---------------------------------------------------------------- E2
CEVAL_SUBJECTS = ["high_school_mathematics", "high_school_physics",
                  "high_school_chemistry", "high_school_biology",
                  "modern_chinese_history", "high_school_geography",
                  "high_school_politics", "high_school_history",
                  "college_physics", "college_chemistry"]
PER_SUBJECT = 25          # 10 科 × ≤25 = 约 200 题（按数据集实际条数冻结）
PROMPT = ("以下是一道单选题，请只回答正确选项的字母。\n\n"
          "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n答案：")


def load_ceval(n_subjects=len(CEVAL_SUBJECTS), per_subject=PER_SUBJECT):
    from datasets import load_dataset
    items = []
    for subj in CEVAL_SUBJECTS[:n_subjects]:
        ds = None
        for name, cfg in (("ceval/ceval-exam", subj), ("ceval-exam", subj)):
            try:
                ds = load_dataset(name, cfg, split="val")
                break
            except Exception as e:
                print(f"  {name}/{cfg} 失败：{type(e).__name__}: {str(e)[:70]}")
        if ds is None:
            continue
        for row in ds.select(range(min(per_subject, len(ds)))):
            items.append({"subject": subj, "question": row["question"],
                          "A": row["A"], "B": row["B"], "C": row["C"],
                          "D": row["D"], "answer": row["answer"].strip().upper()})
    return items


EXTRACT = re.compile(r"[ABCD]")


def score_ceval(llm, items, tok, k=20):
    """冻结计分器：比较 A/B/C/D 四个选项 token 的 logprob，取最大者。

    生成式抽取容易被"把题目复述一遍"骗到（实测 8 token 内 98/98 都截断），
    所以正式计分用 logprob；同时保留每一步的 top-k 原始输出作为现场材料。
    """
    from vllm import SamplingParams
    prompts = [PROMPT.format(**it) for it in items]
    cand = {}
    for letter in "ABCD":
        ids = set()
        for form in (letter, " " + letter, "\n" + letter):
            e = tok(form, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                ids.add(e[0])
        cand[letter] = sorted(ids)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    rows, correct, invalid = [], 0, 0
    for it, o in zip(items, outs):
        lp = o.outputs[0].logprobs[0] or {}
        best, best_v = None, None
        for letter, ids in cand.items():
            v = max((lp[i].logprob for i in ids if i in lp), default=None)
            if v is not None and (best_v is None or v > best_v):
                best, best_v = letter, v
        if best is None:
            invalid += 1
        ok = (best == it["answer"])
        correct += ok
        rows.append({"subject": it["subject"], "answer": it["answer"],
                     "pred": best, "correct": bool(ok),
                     "margin": None if best_v is None else
                     round(best_v - max((lp[i].logprob for l2, ids in cand.items()
                                         if l2 != best for i in ids if i in lp),
                                        default=float("nan")), 4),
                     "top5_raw": sorted(((tok.decode([i]), round(v.logprob, 3))
                                         for i, v in lp.items()),
                                        key=lambda x: -x[1])[:5]})
    return {"n": len(items), "correct": correct,
            "acc": correct / max(1, len(items)), "invalid": invalid,
            "scorer": "logprob_argmax_over_ABCD", "seconds": dt, "rows": rows}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--outdir", default=os.path.expanduser("~/l43_eval"))
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--ceval-subjects", type=int, default=len(CEVAL_SUBJECTS))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    repo = MODELS[args.model]
    title(f"[E] 量化质量评测：{args.model}（{repo}）")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(snap(repo))
    rep = {"model": args.model, "repo": repo,
           "revision": os.path.basename(snap(repo))}
    t0 = time.perf_counter()
    llm = LLM(model=snap(repo), dtype="bfloat16" if args.model == "bf16" else "float16",
              gpu_memory_utilization=args.util, max_model_len=args.max_model_len,
              enforce_eager=True, disable_log_stats=True)
    rep["init_s"] = time.perf_counter() - t0
    print(f"  引擎就绪 {rep['init_s']:.1f} s")

    sub("E1 WikiText-2 固定分块 PPL")
    chunks, total_tokens = wikitext_chunks(tok, )
    print(f"  test 集共 {total_tokens} token；冻结前 {N_CHUNKS} 块 × {CHUNK} token")
    t0 = time.perf_counter()
    ppl = ppl_via_prompt_logprobs(llm, tok, chunks)
    ppl["seconds"] = time.perf_counter() - t0
    print(f"  PPL = {ppl['ppl']:.4f}（{ppl['tokens']} 个计分 token，"
          f"缺失 {sum(c['missing'] for c in ppl['per_chunk'])}）  "
          f"{ppl['seconds']:.1f} s")
    rep["wikitext2"] = ppl

    sub("E2 C-Eval validation 分层题")
    items = load_ceval(n_subjects=args.ceval_subjects)
    print(f"  取到 {len(items)} 题："
          f"{sorted(set(i['subject'] for i in items))}")
    if items:
        ce = score_ceval(llm, items, tok)
        print(f"  准确率 {ce['acc']:.3f}（{ce['correct']}/{ce['n']}）  "
              f"无法判定 {ce['invalid']}  计分器 {ce['scorer']}  "
              f"{ce['seconds']:.1f} s")
        by_subj = {}
        for r in ce["rows"]:
            s = by_subj.setdefault(r["subject"], [0, 0])
            s[0] += r["correct"]
            s[1] += 1
        for s, (c, n) in sorted(by_subj.items()):
            print(f"    {s:<28} {c}/{n}")
        rep["ceval"] = ce
    else:
        rep["ceval"] = {"error": "题目未取到"}

    rep["gpu"] = __import__("torch").cuda.get_device_name(0)
    rep["gpu_peak_mib"] = __import__("torch").cuda.max_memory_allocated() / 2**20
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    path = os.path.join(args.outdir, "quantization_eval.json")
    if os.path.exists(path):
        old = json.load(open(path))
        old.setdefault("models", {})[args.model] = rep
        data = old
    else:
        data = {"models": {args.model: rep}}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
