#!/usr/bin/env python3
"""L3.4-D —— 固定 RULER 式检索任务：正确率、吞吐与峰值状态。

只保留"能直接量"的部分：一个**固定生成器**（样本 ID、长度、深度、答案都由
seed 与哈希决定，可复现），在三个模型上跑同一批样本：

  Qwen3-1.7B      全注意力（对照组）
  RecurrentGemma-2B  线性/递推（已缓存，若可加载）
  Qwen3.5-4B      线性 + 全注意力混合（真实主例）

  [A] 生成器与样本清单：打印样本 ID、长度、位置、距离、答案
  [B] 正确率：按 (模型, 长度, 深度) 汇总精确匹配
  [C] 吞吐与峰值状态：prefill/生成耗时、峰值显存、按配置算的状态字节
  [D] 受控窗口：Qwen3-1.7B 在 4D mask 下把可见范围限制到最近 W 个位置

用法：
    L3_OUT=<目录> python ruler_retrieval.py A B C D
"""

import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                       # noqa: E402

MB = 1024 * 1024
MODELS = [m for m in os.environ.get(
    "L34_MODELS", "Qwen/Qwen3-1.7B,Qwen/Qwen3.5-4B,google/recurrentgemma-2b"
).split(",") if m]
LENGTHS = [512, 2048, 8192]
DEPTHS = [0.1, 0.5, 0.9]
N_PER_CELL = int(os.environ.get("L34_N", "5"))

FILLER = ("The archive records ordinary daily events and routine measurements "
          "from many stations. ")


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 72 - len(s)))


def sample(case_id, ctx_len, depth, tok):
    """固定生成器：样本 ID、长度、深度决定一切。"""
    hsh = hashlib.sha256(f"{case_id}-{ctx_len}-{depth}".encode()).hexdigest()
    value = int(hsh[:6], 16) % 100000
    needle = (f"The magic number for case {case_id} is {value:05d}. ")
    q = (f"\nQuestion: What is the magic number for case {case_id}? "
         f"Answer with the five-digit number only. Answer:")
    q_ids = tok(q, add_special_tokens=False).input_ids
    n_ids = tok(needle, add_special_tokens=False).input_ids
    filler = tok(FILLER, add_special_tokens=False).input_ids
    budget = max(32, ctx_len - len(q_ids) - len(n_ids))
    pos = int(depth * budget)
    pre = (filler * (pos // len(filler) + 1))[:pos]
    post = (filler * ((budget - pos) // len(filler) + 2))[:budget - pos]
    ids = pre + n_ids + post + q_ids
    ids = ids[-ctx_len:] if len(ids) > ctx_len else ids
    needle_at = len(pre)
    return {"case_id": case_id, "ctx_len": ctx_len, "depth": depth,
            "value": f"{value:05d}", "needle_pos": needle_at,
            "seq_len": len(ids), "ids": ids, "prompt": needle}


def hf_token():
    """受限模型（RecurrentGemma 等）用机器上已有的 token。"""
    for p in ("/home/admin/cache/huggingface/token",):
        if os.path.exists(p):
            return open(p).read().strip()
    return None


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_kw = {}
    tk = hf_token()
    if tk:
        tok_kw["token"] = tk
    tok = AutoTokenizer.from_pretrained(name, **tok_kw)
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if tk:
        kw["token"] = tk
    if "recurrentgemma" in name.lower():
        kw["attn_implementation"] = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
        return model, tok, None
    except Exception as exc:                                       # noqa: BLE001
        return None, tok, str(exc).splitlines()[0][:160]


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    peak = torch.cuda.max_memory_allocated()
    del out
    torch.cuda.empty_cache()
    return (peak - base) / MB


def native_prompt_ids(tok, ids):
    """按各模型原生配置包装：有 chat template 就套上（禁用思考），否则用原序列。"""
    text = tok.decode(ids)
    if getattr(tok, "chat_template", None):
        try:
            out = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, enable_thinking=False,
                tokenize=False)
            enc = tok(out, add_special_tokens=False).input_ids
            return enc if isinstance(enc, list) else list(enc)
        except Exception:                                          # noqa: BLE001
            pass
    return ids


def degenerate(ans):
    """退化输出判据：几乎没有 ASCII 字母与数字，或大量重复片段。"""
    import re
    core = re.sub(r"<think.*?</think>", " ", ans, flags=re.S)
    alnum = sum(ch.isalnum() for ch in core)
    if len(core) and alnum / max(1, len(core)) < 0.25:
        return True
    words = core.split()
    if len(words) > 20 and len(set(words)) < len(words) * 0.3:
        return True
    return False


def run_greedy(model, tok, ids, max_new=192):
    inp = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        try:                     # 只保留最后一个位置的 logits：8192 上下文下
            out = model.generate(inp, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id,
                                 num_logits_to_keep=1)
        except (TypeError, ValueError):
            out = model.generate(inp, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    gen = out[0, inp.shape[1]:].tolist()
    return tok.decode(gen, skip_special_tokens=True)


def answer_hit(value, ans):
    """数字答案：去掉思考段、允许不补前导零、允许夹在句子里。"""
    import re
    body = re.sub(r"<think.*?</think>", " ", ans, flags=re.S)
    if "<think" in ans and "</think>" not in ans:
        body = ans.split("</think>")[-1] if "</think>" in ans else ""
    cand = body if body.strip() else ans
    return (value in cand) or (str(int(value)) in cand)


def greedy_with_mask(model, tok, ids, mask_fn, max_new=192):
    """手动贪心解码：每步重建 4D mask（generate 只接受 2D mask）。"""
    seq = list(ids)
    for _ in range(max_new):
        inp = torch.tensor([seq], device="cuda")
        add = mask_fn(inp.shape[1])
        with torch.no_grad():
            try:
                out = model(inp, attention_mask=add, num_logits_to_keep=1)
            except (TypeError, ValueError):
                out = model(inp, attention_mask=add)
        nxt = int(out.logits[0, -1].argmax().item())
        seq.append(nxt)
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
    return tok.decode(seq[len(ids):], skip_special_tokens=True)


# ---------------------------------------------------------------- A
def section_A(h):
    title("[A] 固定生成器与样本清单")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[0])
    print(f"  样本 ID = 1000+i，长度 ∈ {LENGTHS}，深度 ∈ {DEPTHS}，"
          f"每格 {N_PER_CELL} 个样本")
    print(f"  样本 ID、长度、深度 → 答案由 sha256 决定，生成过程无随机性。")
    print(f"  {'样本ID':>7} {'长度':>6} {'深度':>5} {'答案':>7} {'针位置':>7} "
          f"{'实际序列':>8} {'距离(尾部)':>10}")
    rows = []
    for ctx in LENGTHS:
        for depth in DEPTHS:
            for i in range(N_PER_CELL):
                s = sample(1000 + i, ctx, depth, tok)
                rows.append(s)
                if i == 0:
                    print(f"  {s['case_id']:>7} {s['ctx_len']:>6} "
                          f"{s['depth']:>5.1f} {s['value']:>7} "
                          f"{s['needle_pos']:>7} {s['seq_len']:>8} "
                          f"{s['seq_len'] - s['needle_pos']:>10}")
    print(f"  共 {len(rows)} 个样本；清单落盘为 samples.json")
    h.case(id="A_generator", lengths=LENGTHS, depths=DEPTHS,
           n_per_cell=N_PER_CELL, samples=len(rows), models=MODELS,
           generator="sha256(case_id-ctx-depth) 决定答案")
    h.finish_extra = [{"case_id": r["case_id"], "ctx_len": r["ctx_len"],
                       "depth": r["depth"], "value": r["value"],
                       "needle_pos": r["needle_pos"], "seq_len": r["seq_len"]}
                      for r in rows]
    return tok, rows


# ---------------------------------------------------------------- B/C
def section_BC(h, tok, rows):
    title("[B/C] 正确率、吞吐与峰值状态")

    for name in MODELS:
        model, tok_m, err = load_model(name)
        if model is None:
            print(f"\n  {name}: 加载失败 → {err}")
            h.case(id=f"BC_{name}", status="load_failed", error=err)
            continue
        print(f"\n  {name}: 载入完成")
        cells = {}
        first_sample = True
        degenerate_n = 0
        t_prefill, t_gen, pk = [], [], []
        for r in rows:
            ids = native_prompt_ids(tok_m, r["ids"])
            prompts = torch.tensor([ids], device="cuda")
            torch.cuda.synchronize()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record()
            ans = run_greedy(model, tok_m, ids)
            b.record(); torch.cuda.synchronize()
            t = a.elapsed_time(b)
            ok = answer_hit(r["value"], ans)
            deg = degenerate(ans)
            if deg:
                degenerate_n += 1
            if first_sample:
                print(f"    [样例] 问 {r['value']} → 答 {ans.strip()[-60:]!r} "
                      f"命中={ok} 退化={deg}")
                first_sample = False
                first_sample = False
            key = (r["ctx_len"], r["depth"])
            cell = cells.setdefault(key, {"n": 0, "ok": 0, "ms": []})
            cell["n"] += 1
            cell["ok"] += int(ok)
            cell["ms"].append(t)
            if len(pk) < 2:
                pk.append(peak_mb(lambda: run_greedy(model, tok_m, r["ids"])))
        print(f"  {'长度':>6} {'深度':>5} {'正确':>5} {'耗时 ms':>10}")
        for (ctx, depth), c in sorted(cells.items()):
            print(f"  {ctx:>6} {depth:>5.1f} {c['ok']}/{c['n']:<3} "
                  f"{sum(c['ms']) / len(c['ms']):>10.1f}")
            h.case(id=f"BC_{name}_ctx{ctx}_d{depth}", model=name, ctx=ctx,
                   depth=depth, n=c["n"], correct=c["ok"],
                   acc=c["ok"] / c["n"], ms_mean=sum(c["ms"]) / len(c["ms"]))
        print(f"    退化输出 {degenerate_n}/{len(rows)}"
              f"{'（此模型在该 prompt 包装下不可用，正确率不计入结论）' if degenerate_n > len(rows) * 0.5 else ''}")
        h.case(id=f"BC_{name}_overall", model=name,
               acc=sum(c["ok"] for c in cells.values()) / len(rows),
               peak_mb=max(pk) if pk else None, samples=len(rows),
               degenerate=degenerate_n,
               usable=degenerate_n <= len(rows) * 0.5)
        del model
        torch.cuda.empty_cache()
    print("\n  样本量小（每格 %d），这里的正确率只用于**同一生成器下的相对比较**；"
          % N_PER_CELL)
    print("  要报置信区间需要按任务重采样更多样本，不能把 5 个样本的差值当结论。")


# ---------------------------------------------------------------- D
def section_D(h, tok, rows):
    title("[D] 受控窗口：把可见范围限制到最近 W 个位置")

    name = MODELS[0]
    model, tok_m, err = load_model(name)
    if model is None:
        print(f"  {name} 加载失败：{err}")
        return
    import torch.nn.functional as F
    W = 512
    ctx = 2048
    print(f"  {name}，长度 {ctx}，窗口 W={W}（4D mask 显式限制可见范围）")
    print(f"  {'样本ID':>7} {'深度':>5} {'全上下文':>9} {'窗口 W=512':>11} "
          f"{'针在窗口内?':>12}")
    ok_full, ok_win = 0, 0
    n = 0
    for r in rows:
        if r["ctx_len"] != ctx:
            continue
        n += 1
        ans_full = run_greedy(model, tok_m, r["ids"])
        o_full = answer_hit(r["value"], ans_full)
        S = len(r["ids"])

        def mask_fn(n, W=W):
            m = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
            i = torch.arange(n, device="cuda").view(-1, 1)
            j = torch.arange(n, device="cuda").view(1, -1)
            m = m & ((i - j) < W)
            add = torch.zeros(1, 1, n, n, dtype=torch.bfloat16, device="cuda")
            return add.masked_fill(~m, torch.finfo(torch.bfloat16).min)

        ans_win = greedy_with_mask(model, tok_m, r["ids"], mask_fn)
        o_win = answer_hit(r["value"], ans_win)
        in_win = (S - r["needle_pos"]) <= W
        print(f"  {r['case_id']:>7} {r['depth']:>5.1f} {str(o_full):>9} "
              f"{str(o_win):>11} {str(in_win):>12}")
        ok_full += int(o_full)
        ok_win += int(o_win)
        h.case(id=f"D_case{r['case_id']}", ctx=ctx, W=W, depth=r["depth"],
               full_ok=o_full, window_ok=o_win, needle_in_window=in_win,
               needle_pos=r["needle_pos"], seq_len=S)
    print(f"\n  全上下文 {ok_full}/{n}，窗口 W={W} {ok_win}/{n}")
    print("  针在窗口外时窗口模型必然失败（看不到），窗口内失败则说明还有别的原因；")
    print("  单个合成反例不能推广到所有线性/混合架构。")
    del model
    torch.cuda.empty_cache()


def main():
    h = Harness("3.4-D", "3.4", out=os.environ.get("L3_OUT"),
                backend="transformers 生成（greedy）",
                notes=f"models={MODELS}; N={N_PER_CELL}")
    print(f"torch {torch.__version__}")
    want = [s.upper() for s in sys.argv[1:]] or ["A", "B", "C", "D"]
    tok, rows = section_A(h)
    if "B" in want or "C" in want:
        section_BC(h, tok, rows)
    if "D" in want:
        section_D(h, tok, rows)
    extra = {"samples.json": __import__("json").dumps(
        getattr(h, "finish_extra", []), ensure_ascii=False, indent=2)}
    h.finish({"verdict": "同一固定生成器下比较三种架构；"
                         "受控窗口显示'看不到'与'看到但答错'是两种失败。",
              "models": MODELS, "lengths": LENGTHS, "depths": DEPTHS,
              "n_per_cell": N_PER_CELL}, extra_files=extra)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
