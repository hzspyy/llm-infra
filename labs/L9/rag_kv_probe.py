#!/usr/bin/env python3
"""L9.6 任务 D：片段 KV 的可复用条件与错拼接反例。

前缀缓存只认「从头逐字相同」。把同一段文档放进不同位置，即使内容一模一样也不会命中；
同一位置换一个问句，前缀仍然命中。本脚本用四种构造验证这条规则，并给出错拼接的原始读数：

* ``repeat``：同一 prompt 连发两次（完全相同的上下文与位置与问句）；
* ``same_chunk_other_prefix``：同一片段前插一段不同的前缀（片段位置改变）；
* ``same_chunk_other_question``：片段与位置不变，只换问句（尾部改变）；
* ``reordered_chunks``：同一批片段换顺序（多片段场景的错拼接）。

近似方案（CacheBlend 一类"片段 KV 拼接 + 重算"）不在本脚本内实现，标记为 `UNVERIFIED`：
它需要引擎侧的片段级 KV 接口，本章没有该接口的实测条件。

用法::
    python labs/L9/rag_kv_probe.py --base-url ... --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent_tasks as T  # noqa: E402
from session_kv_bench import reset_cache  # noqa: E402

SYSTEM = "你是一个严谨的问答助手。只根据提供的资料回答，回答末尾写 `Answer: <短答案>`。"


def doc_texts(n: int = 6, chars: int = 2400) -> list[str]:
    data = T.load_nfcorpus()
    out = []
    for d in data["docs"]:
        text = ((d.get("title") or "") + "\n" + d["text"]).strip()
        if len(text) >= chars:
            out.append(text[:chars])
        if len(out) >= n:
            break
    return out


async def ask(client, model, messages, max_tokens=24):
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None)
    return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
            "text": (resp.choices[0].message.content or "")[:60]}


async def amain(args) -> int:
    from openai import AsyncOpenAI

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    docs = doc_texts(args.docs, args.chars)
    q1 = "他汀类药物对心血管风险的主要结论是什么？"
    q2 = "这些研究里提到的样本量有多大？"

    rows = []

    async def ask_pair(case: str, first_msgs, second_msgs, note: str, reset_before=True):
        """先发第一个 prompt 建立缓存，再发第二个 prompt 看能命中多少（跨 prompt 对照）。"""
        if reset_before:
            await reset_cache(client)
        first = await ask(client, args.model, first_msgs)
        second = await ask(client, args.model, second_msgs)
        rows.append({"case": case, "note": note, "first": first, "second": second,
                     "second_hit_ratio": round((second["cached_tokens"] or 0) / max(1, second["prompt_tokens"]), 4)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    def msg(body: str, question: str):
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"资料：\n{body}\n\n问题：{question}"}]

    # 1) 完全相同的 prompt（同一预填 + 同一问句）：第二次应当几乎全命中
    await ask_pair("repeat", msg(docs[0], q1), msg(docs[0], q1),
                   "同一上下文、同一位置、同一问句")

    # 2) 同一片段、前面插了另一段：片段位置改变，共同前缀从 system 之后就断
    await ask_pair("same_chunk_other_prefix", msg(docs[0], q1), msg(docs[1] + "\n\n" + docs[0], q1),
                   "目标片段相同但前面插了另一段，位置改变")

    # 3) 同一片段、同一位置、换问句：只有尾部不同
    await ask_pair("same_chunk_other_question", msg(docs[0], q1), msg(docs[0], q2),
                   "上下文与位置不变，只换问句")

    # 4) 多片段换顺序：共同前缀在第一个被换的片段处断开
    await ask_pair("reordered_chunks", msg(docs[0] + "\n\n" + docs[2], q1),
                   msg(docs[2] + "\n\n" + docs[0], q1), "同一批片段换顺序")

    # 5) 前缀相同、只是把新片段追加到末尾：应当命中到追加点
    await ask_pair("append_new_chunk", msg(docs[0], q1), msg(docs[0] + "\n\n" + docs[3], q1),
                   "只在末尾追加新片段")

    await client.close()
    summary = {
        "config": {"model": args.model, "docs": len(docs), "chars_per_doc": args.chars},
        "cases": rows,
        "rule": "可复用条件 = 相同上下文 + 相同位置 + 相同模型身份；三者的任一改变都使命中退化到改变点之前",
        "approximate_scheme": {
            "name": "CacheBlend 一类片段级拼接 + 选择性重算",
            "status": "UNVERIFIED",
            "reason": "需要引擎侧片段级 KV 读写接口；本章运行环境没有该接口的实测条件",
        },
    }
    (out / "rag_kv_probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({r["case"]: {"first_cached": r["first"]["cached_tokens"],
                                  "second_cached": r["second"]["cached_tokens"],
                                  "second_hit_ratio": r["second_hit_ratio"]} for r in rows},
                     ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.6 片段 KV 复用探针")
    ap.add_argument("--base-url", default="http://127.0.0.1:8016/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=6)
    ap.add_argument("--chars", type=int, default=2400)
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
