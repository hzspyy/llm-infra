#!/usr/bin/env python3
"""L9.1 任务 C 的跨引擎核对：同一条 BFCL 题在两引擎上的模板、支持模式与实际请求。

BFCL 的整题对照由 `bfcl_multi_turn.py` 给出（两引擎各跑同一批 50 题）。本脚本回答的是
任务书里那句「先核对同一题在两引擎的模板、支持模式与实际请求」——同一段 messages + tools
交给两个引擎，逐项落盘：

* **模板**：同一 `chat_template_kwargs`（`enable_thinking` true/false）下引擎实际渲染出的
  prompt token 数，与本地 HF tokenizer 的 `apply_chat_template` 参照比对，判断哪个引擎
  接受了该参数、哪个忽略了它。vLLM 另有 `/tokenize` 端点，直接取回渲染后的 token 串，可
  与参照逐 token 对拍；SGLang 0.5.19 没有该端点，只有 `usage.prompt_tokens`。
* **支持模式**：`tool_choice` = none/auto/required/named 四种取值在同一引擎上的接受/拒绝
  路径与 `finish_reason`，附原始错误文本（引擎把它变成 400 还是静默降级）。
* **实际请求**：一次 `auto` 调用的原始响应片段（tool_call id/name/arguments、finish_reason、
  usage），用于对比两引擎的工具调用编码。

依赖 9.1-C 的 BFCL 环境（`bfcl-eval==2025.12.17`）只用于 `dump-entry` 子命令；`probe`
子命令只需要 `openai`、`httpx`、`transformers`，可用远程机器的 `envs/serve` 运行。

用法::

    # 1) 取一条真实 BFCL 题（messages + tools）落成 JSON
    /scratch/learn/envs/bfcl/bin/python labs/L9/cross_engine_template.py dump-entry \\
        --entry-id multi_turn_base_0 --out out/9.1/cross-engine/entry.json

    # 2) 对某一引擎跑模板/支持模式/实际请求三组探针
    /scratch/learn/envs/serve/bin/python labs/L9/cross_engine_template.py probe \\
        --entry out/9.1/cross-engine/entry.json --engine vllm \\
        --base-url http://127.0.0.1:8061/v1 --model Qwen/Qwen3-4B \\
        --tokenizer Qwen/Qwen3-4B --out out/9.1/cross-engine/probe-vllm.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time


# --------------------------------------------------------------------------- dump
def bfcl_root() -> pathlib.Path:
    import bfcl_eval

    return pathlib.Path(bfcl_eval.__file__).resolve().parent


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_tools_for(classes: list[str], root: pathlib.Path) -> list[dict]:
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

    tools: list[dict] = []
    for cls in classes:
        fname = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls)
        if fname is None:
            raise KeyError(f"BFCL 映射里没有 {cls}")
        for row in _load_jsonl(root / "data" / "multi_turn_func_doc" / fname):
            tools.append({"type": "function",
                          "function": {"name": row["name"], "description": row["description"],
                                       "parameters": row["parameters"]}})
    return tools


def cmd_dump_entry(args) -> int:
    root = bfcl_root()
    rows = _load_jsonl(root / "data" / "BFCL_v4_multi_turn_base.json")
    entry = next((r for r in rows if r["id"] == args.entry_id), None)
    if entry is None:
        raise SystemExit(f"题目不存在：{args.entry_id}")
    messages = [{"role": m["role"], "content": m["content"]} for m in entry["question"][0]]
    payload = {"id": entry["id"], "category": entry["id"].rsplit("_", 1)[0],
               "involved_classes": entry["involved_classes"],
               "turn0_messages": messages,
               "turns": len(entry["question"]),
               "tools": load_tools_for(entry["involved_classes"], root)}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"id": payload["id"], "tools": len(payload["tools"]),
                      "messages": len(messages), "turns": payload["turns"]}, ensure_ascii=False))
    return 0


# -------------------------------------------------------------------------- probe
def _reference_tokens(tokenizer, messages: list[dict], tools: list[dict], thinking: bool) -> dict:
    """本地 HF 参照：同一份模板 + 同一组 kwargs 应当渲染出的 token。

    注意 transformers 5.x 在 `tokenize=True` 时返回 `BatchEncoding` 而不是裸 list，
    直接 `len()` 得到的是字段个数（2）而不是 token 数；这里显式取 `input_ids`。
    """
    enc = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                        tokenize=True, enable_thinking=thinking)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                         tokenize=False, enable_thinking=thinking)
    return {"count": len(ids), "tokens": list(ids), "text": text}


def _render(client_cfg: dict, messages: list[dict], tools: list[dict], thinking: bool) -> dict:
    """向引擎发一次 max_tokens=1 的调用，只取 prompt token 账。"""
    from openai import OpenAI

    client = OpenAI(base_url=client_cfg["base_url"], api_key="EMPTY", timeout=120.0)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=client_cfg["model"], messages=messages, tools=tools or None,
            tool_choice="auto" if tools else None, temperature=0.0, max_tokens=1,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        usage = resp.usage
        return {"ok": True, "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "prompt_tokens": None, "completion_tokens": None,
                "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()


def _engine_tokenize(engine: str, base_url: str, model: str, messages: list[dict],
                     tools: list[dict], thinking: bool) -> dict:
    """取引擎实际渲染出的 prompt token：vLLM 用 `/tokenize`，SGLang 用 `/v1/tokenize`。

    两个端点的请求体不同：vLLM 收 `add_generation_prompt`/`return_token_strs`，SGLang 收
    `tool_choice` 且把 chat 渲染交给它自己的编码路径。返回的 `tokens` 与本地 HF 参照可逐
    id 对拍，因此「模板差异」不是靠 prompt token 总数猜出来的。
    """
    import httpx

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    if engine == "vllm":
        path = "/tokenize"
        body = {"model": model, "messages": messages, "tools": tools or None,
                "add_generation_prompt": True, "return_token_strs": True,
                "chat_template_kwargs": {"enable_thinking": thinking}}
    else:
        path = "/v1/tokenize"
        body = {"model": model, "messages": messages, "tools": tools or None,
                "tool_choice": "auto" if tools else None,
                "chat_template_kwargs": {"enable_thinking": thinking}}
    try:
        r = httpx.post(root + path, json=body, timeout=120.0)
        if r.status_code != 200:
            return {"ok": False, "path": path, "status": r.status_code, "error": r.text[:400]}
        d = r.json()
        tokens = d.get("tokens")
        out = {"ok": True, "path": path, "status": 200, "count": d.get("count"),
               "tokens": tokens if isinstance(tokens, list) else None,
               "text": "".join(d.get("token_strs") or []) or None}
        if out["text"] is None and out["tokens"]:
            # SGLang 不回 token 串：用 /detokenize 还原同一段 prompt 文本
            try:
                rd = httpx.post(root + "/v1/detokenize",
                                json={"model": model, "tokens": out["tokens"],
                                      "skip_special_tokens": False}, timeout=60.0)
                if rd.status_code == 200:
                    out["text"] = rd.json().get("text")
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": path, "status": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _tool_choice_matrix(cfg: dict, messages: list[dict], tools: list[dict]) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(base_url=cfg["base_url"], api_key="EMPTY", timeout=120.0)
    out: list[dict] = []
    named = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
    for tc in (["none", "auto", "required", named] if tools else ["none"]):
        label = tc if isinstance(tc, str) else "named"
        row = {"tool_choice": label, "supported": None, "error": None, "finish_reason": None,
               "tool_names": [], "content_len": None, "prompt_tokens": None}
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=cfg["model"], messages=messages, tools=tools or None,
                tool_choice=tc if tools else None, temperature=0.0, max_tokens=64,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = resp.choices[0]
            row["supported"] = True
            row["finish_reason"] = msg.finish_reason
            row["tool_names"] = [c.function.name for c in (msg.message.tool_calls or [])]
            row["content_len"] = len(msg.message.content or "")
            row["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", None)
        except Exception as exc:  # noqa: BLE001
            row["supported"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        out.append(row)
    client.close()
    return out


def _raw_auto_call(cfg: dict, messages: list[dict], tools: list[dict]) -> dict:
    """保留一次 auto 调用的原始字段：tool_call 编码、finish_reason、usage。"""
    from openai import OpenAI

    client = OpenAI(base_url=cfg["base_url"], api_key="EMPTY", timeout=120.0)
    try:
        resp = client.chat.completions.create(
            model=cfg["model"], messages=messages, tools=tools, tool_choice="auto",
            temperature=0.0, max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        c = resp.choices[0]
        raw = {
            "response_id": resp.id, "object": resp.object, "model": resp.model,
            "finish_reason": c.finish_reason,
            "message_role": c.message.role,
            "content": (c.message.content or "")[:400],
            "reasoning_len": len(getattr(c.message, "reasoning_content", None) or ""),
            "tool_calls": [{"id": t.id, "type": t.type, "function_name": t.function.name,
                            "arguments": t.function.arguments,
                            "arguments_json_valid": _json_ok(t.function.arguments)}
                           for t in (c.message.tool_calls or [])],
            "usage": {"prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                      "completion_tokens": getattr(resp.usage, "completion_tokens", None)},
        }
    except Exception as exc:  # noqa: BLE001
        raw = {"error": f"{type(exc).__name__}: {exc}"}
    client.close()
    return raw


def _json_ok(s: str | None) -> bool:
    try:
        json.loads(s or "")
        return True
    except Exception:  # noqa: BLE001
        return False


def _engine_identity(cfg: dict) -> dict:
    import httpx

    base = cfg["base_url"].rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    ident: dict = {"base_url": cfg["base_url"], "model": cfg["model"]}
    for path in ("/version", "/server_info", "/v1/models"):
        try:
            r = httpx.get(root + path, timeout=30.0)
            if r.status_code == 200:
                body = r.json()
                if path == "/server_info":
                    body = {k: body.get(k) for k in
                            ("version", "model_path", "served_model_name", "tool_call_parser",
                             "reasoning_parser", "context_length", "mem_fraction_static",
                             "dtype", "chat_template")}
                ident[path] = body
        except Exception as exc:  # noqa: BLE001
            ident[path] = f"{type(exc).__name__}: {exc}"
    return ident


def cmd_probe(args) -> int:
    entry = json.loads(pathlib.Path(args.entry).read_text(encoding="utf-8"))
    messages, tools = entry["turn0_messages"], entry["tools"]
    cfg = {"base_url": args.base_url, "model": args.model}

    reference: dict = {}
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for thinking in (False, True):
            reference[str(thinking)] = _reference_tokens(tok, messages, tools, thinking)

    renders: dict = {}
    for thinking in (False, True):
        row = _render(cfg, messages, tools, thinking)
        ref = reference.get(str(thinking)) or {}
        row["reference_count"] = ref.get("count")
        row["tokenize"] = _engine_tokenize(args.engine, args.base_url, args.model,
                                           messages, tools, thinking)
        tz = row["tokenize"]
        if tz.get("tokens") and ref.get("tokens"):
            row["token_ids_equal_reference"] = tz["tokens"] == ref["tokens"]
            row["first_divergence"] = next(
                (i for i, (a, b) in enumerate(zip(tz["tokens"], ref["tokens"])) if a != b), None)
        if tz.get("text") and ref.get("text"):
            row["token_text_equal_reference"] = tz["text"] == ref["text"]
            row["engine_text_chars"] = len(tz["text"])
            row["reference_text_chars"] = len(ref["text"])
        renders[str(thinking)] = row

    summary = {
        "engine": args.engine,
        "entry_id": entry["id"],
        "n_tools": len(tools),
        "n_messages": len(messages),
        "identity": _engine_identity(cfg),
        "render_by_thinking": renders,
        "reference": reference,
        "tool_choice_matrix": _tool_choice_matrix(cfg, messages, tools),
        "raw_auto_call": _raw_auto_call(cfg, messages, tools),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("engine", "entry_id", "n_tools",
                                              "render_by_thinking", "tool_choice_matrix")},
                     ensure_ascii=False, indent=1)[:4000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1-C 跨引擎模板/支持模式/实际请求核对")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump-entry", help="从 BFCL 取一条题（messages + tools）")
    d.add_argument("--entry-id", default="multi_turn_base_0")
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_dump_entry)

    p = sub.add_parser("probe", help="对单个引擎跑三组探针")
    p.add_argument("--entry", required=True)
    p.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer 路径或名字，用于本地参照渲染")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
