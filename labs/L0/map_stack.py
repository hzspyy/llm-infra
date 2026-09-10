#!/usr/bin/env python3
"""L0 lab · 把「一次 /v1/chat/completions」的调用栈从安装好的引擎里挖出来。

**双引擎并置**：vLLM 与 SGLang 走的是同一条链路，但每一层的工程选择不同。
把两边对齐着看，你学到的就不是「vLLM 是怎么写的」，而是「这一层必须解决什么问题、
有哪几种解法」——后者才是能带走的东西。

不要抄别人博客里的行号——那是某个版本的快照，几周后就错了。
这个脚本对**你自己装的这个版本**做符号定位，输出带真实 file:line 的分层地图。
vLLM 0.29 就把 entrypoints 整个重组过，SGLang 也一直在动，但这条链路的
**结构**没变：协议层 → 渲染层 → 前端 → IPC → 引擎核心 → 调度器 →
显存/前缀缓存 → 执行器 → 模型 → attention 后端 → 采样 → 回传 → 流式输出。

用法：
    python map_stack.py --engine vllm --format md
    python map_stack.py --engine sglang --sglang-path /path/to/sglang
    python map_stack.py --engine both --format md --out results/stack_map.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

# 十四个「必须有人做」的阶段。两个引擎在每个阶段的做法可能不同，
# 但没有哪个阶段可以被跳过——这就是并置的价值。
STAGE_NAMES = [
    "HTTP 路由", "协议层", "模板渲染/分词", "前端入口", "输入处理", "跨进程 IPC",
    "引擎主循环", "一步 = 一次前向", "调度器", "前缀复用查询", "显存分配",
    "执行/模型前向", "采样", "结果回传与流式输出",
]

# (阶段索引, 进程/线程, 文件相对路径, 符号正则, 这一层做的事)
VLLM_STAGES: list[tuple[int, str, str, str, str]] = [
    (0, "API 进程 · asyncio", "entrypoints/openai/chat_completion/api_router.py",
     r"^async def create_chat_completion",
     "FastAPI 路由。到这里为止都还是普通 web 服务：反序列化 JSON、校验 pydantic 模型。"),
    (1, "API 进程 · asyncio", "entrypoints/openai/chat_completion/serving.py",
     r"^\s+async def create_chat_completion",
     "OpenAI 协议语义：解析 messages、工具定义、采样参数，决定流式还是一次性返回。"),
    (2, "API 进程 · asyncio", "renderers/hf.py", r"^\s+def render_messages",
     "套 chat template → 字符串 → token id；多模态输入拆成占位符 + 图像/视频数据。"
     "0.29 把 renderer 提成了顶层可插拔包（renderers/registry.py），"
     "因为 DeepSeek / Kimi / Mistral 的模板语义塞不进一个通用函数。"),
    (3, "API 进程 · asyncio", "v1/engine/async_llm.py", r"^\s+async def generate",
     "AsyncLLM.generate：给请求建一个输出队列（RequestOutputCollector）然后 await。"
     "HTTP 协程从此挂起，直到有 token 被推回来。"),
    (4, "API 进程 · asyncio", "v1/engine/input_processor.py", r"^\s+def process_inputs",
     "构造 EngineCoreRequest：token id、采样参数、多模态数据的哈希与缓存键、LoRA 请求。"),
    (5, "API 进程 → 引擎进程", "v1/engine/core_client.py", r"^\s+async def add_request_async",
     "序列化后经 ZeroMQ 送到独立的 EngineCore 进程。拆进程是为了绕开 GIL："
     "HTTP / 分词 / detokenize 不能和 GPU 调度抢同一个解释器。"),
    (6, "引擎进程 · busy loop", "v1/engine/core.py", r"^\s+def run_busy_loop",
     "EngineCore 的 while True：收新请求 → step() → 把输出塞回 IPC。整个系统的心跳。"),
    (7, "引擎进程", "v1/engine/core.py", r"^\s+def step\(",
     "step()：调度 → 执行 → 收结果。continuous batching 的「iteration-level」"
     "就是这个 step 的粒度：每一步都能换一批请求，不必等某个请求生成完。"),
    (8, "引擎进程", "v1/core/sched/scheduler.py", r"^\s+def schedule",
     "本步跑哪些请求、每个跑几个 token。chunked prefill、抢占、优先级、token 预算都在这里。"
     "输出 SchedulerOutput——一份给 GPU 的施工图。"),
    (9, "引擎进程", "v1/core/kv_cache_manager.py", r"^\s+def get_computed_blocks",
     "按**块哈希**查前缀有多少已算过。vLLM 用哈希表（O(1) 查一个块），"
     "命中的块直接引用，对应 token 不再进入 prefill。"),
    (10, "引擎进程", "v1/core/block_pool.py", r"^\s+def get_new_blocks",
     "物理块分配器：free list + 块哈希表。PagedAttention 的「页」就是这里的 KVCacheBlock。"),
    (11, "worker · GPU", "v1/worker/gpu_model_runner.py", r"^\s+def execute_model",
     "组装输入张量（_prepare_inputs，同文件）→ 选 CUDA Graph 或 eager → 模型 forward。"),
    (12, "worker · GPU", "v1/sample/sampler.py", r"^\s+def forward",
     "logits → 惩罚 → 温度 → top-k/top-p → 采样。全在 GPU 上做，避免把 logits 拷回 CPU。"),
    (13, "API 进程 · asyncio", "v1/engine/output_processor.py", r"^\s+def process_outputs",
     "增量 detokenize、处理 stop string、推进每个请求的输出队列；"
     "再由 serving.py 的 chat_completion_stream_generator 包成 SSE 写回 socket。"),
]

SGLANG_STAGES: list[tuple[int, str, str, str, str]] = [
    (0, "HTTP 进程 · asyncio", "entrypoints/http_server.py",
     r'@app\.post\("/v1/chat/completions"',
     "同样是 FastAPI，但 SGLang 把路由集中在**一个** http_server.py 里，"
     "而不是像 vLLM 那样每个能力一个 api_router 包。"),
    (1, "HTTP 进程 · asyncio", "entrypoints/openai/serving_chat.py", r"^class OpenAIServingChat",
     "OpenAI 协议适配。注意 SGLang 把 thinking mode（推理模型的 <think> 段）"
     "做成了协议层的一等公民（同文件 ThinkingMode）。"),
    (2, "HTTP 进程 · asyncio", "managers/tokenizer_manager.py", r"^class TokenizerManager",
     "**这是 SGLang 与 vLLM 最大的结构差异**：分词不是一个函数，而是一个独立的 "
     "TokenizerManager 进程/组件，同时负责分词、请求登记（ReqState）、"
     "以及把结果路由回对应的 HTTP 协程。"),
    (3, "HTTP 进程 · asyncio", "managers/tokenizer_manager.py", r"^\s+async def generate_request",
     "前端入口。与 vLLM 的 AsyncLLM.generate 对应：登记 ReqState，await 事件。"),
    (4, "HTTP 进程 · asyncio", "managers/io_struct.py", r"^class TokenizedGenerateReqInput|^class GenerateReqInput",
     "SGLang 的请求结构体集中在 io_struct.py。对照 vLLM 的 EngineCoreRequest 看"
     "两边各自认为「一个请求」必须携带哪些字段，很能看出设计取向。"),
    (5, "Tokenizer 进程 → Scheduler 进程", "managers/tokenizer_manager.py",
     r"^\s+async def _handle_batch_request",
     "SGLang 是**三进程**结构：TokenizerManager → Scheduler → DetokenizerManager，"
     "三者用 ZeroMQ 串起来。vLLM 是两进程（API + EngineCore），detokenize 在 API 进程。"
     "多一次进程边界换来 detokenize 不阻塞调度——这是一个真实的取舍。"),
    (6, "Scheduler 进程 · busy loop", "managers/scheduler.py", r"^\s+def event_loop_normal",
     "Scheduler 的主循环。同文件还有 event_loop_overlap 变体："
     "把 CPU 侧调度与 GPU 执行重叠起来（vLLM 的对应物是异步调度）。"),
    (7, "Scheduler 进程", "managers/scheduler.py", r"^\s+def run_batch",
     "跑一个 batch。与 vLLM 的 step() 对应。"),
    (8, "Scheduler 进程", "managers/scheduler.py", r"^\s+def get_next_batch_to_run",
     "选批策略。SGLang 把策略单独放在 managers/schedule_policy.py，"
     "支持 LPM（最长前缀优先）等对 radix cache 友好的排序——这是它和 vLLM 的关键差异之一。"),
    (9, "Scheduler 进程", "mem_cache/radix_cache.py", r"^\s+def match_prefix",
     "**RadixAttention**：前缀不是哈希表里的独立块，而是一棵基数树（radix tree）。"
     "match_prefix 沿树走最长公共前缀，天然支持共享分支与子树级淘汰。"
     "对照 vLLM 的哈希表方案：树能表达前缀间的包含关系，哈希表不能。"),
    (10, "Scheduler 进程", "mem_cache/radix_cache.py", r"^\s+def evict\(",
     "按 LRU 从叶子往上淘汰整棵子树。同文件的 cache_finished_req / "
     "cache_unfinished_req 决定一个请求的 KV 何时进入共享树。"),
    (11, "TP worker · GPU", "model_executor/model_runner.py", r"^\s+def forward\(",
     "ModelRunner.forward。输入批信息在 model_executor/forward_batch_info.py 的 "
     "ForwardBatch 里（对照 vLLM 的 _prepare_inputs）。"),
    (12, "TP worker · GPU", "layers/sampler.py", r"^\s+def forward",
     "采样。SGLang 默认走 FlashInfer 的采样 kernel。"),
    (13, "Detokenizer 进程", "managers/detokenizer_manager.py", r"^\s+def event_loop",
     "**独立进程**做增量 detokenize，再经 ZeroMQ 回到 TokenizerManager，"
     "由它唤醒对应的 HTTP 协程写 SSE。vLLM 把这一步放在 API 进程内。"),
]

ENGINES = {
    "vllm": ("vllm", VLLM_STAGES, "srt_prefix_none"),
    "sglang": ("sglang", SGLANG_STAGES, "srt"),
}


def find_root(module: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    spec = importlib.util.find_spec(module)
    if not spec or not spec.origin:
        return None
    root = Path(spec.origin).parent
    if module == "sglang":
        root = root / "srt"
    return root


def version_of(module: str, root: Path | None = None) -> str:
    """先试导入；失败就从 root 旁边的 dist-info 目录名里读版本。

    这是必要的：两个引擎装在不同 venv 里（依赖打架），
    所以跑本脚本的解释器通常只能 import 其中一个。"""
    try:
        mod = __import__(module)
        v = getattr(mod, "__version__", None)
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    if root:
        for parent in (root, root.parent, root.parent.parent):
            for d in parent.glob(f"{module}*.dist-info"):
                m = re.search(r"-(\d[^-]*)\.dist-info$", d.name)
                if m:
                    return m.group(1)
    return "未知"


def locate(root: Path, rel: str, pattern: str) -> tuple[int | None, str]:
    path = root / rel
    if not path.exists():
        return None, "文件不存在（版本差异）"
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if rx.search(line):
            return i, line.strip()
    return None, "符号未找到（可能已改名）"


def scan(engine: str, explicit: str | None) -> dict:
    module, stages, _ = ENGINES[engine]
    root = find_root(module, explicit)
    if root is None or not root.exists():
        return {"engine": engine, "version": version_of(module), "root": None, "rows": []}
    rows = []
    for stage_idx, where, rel, pat, note in stages:
        line, code = locate(root, rel, pat)
        rows.append({"stage_idx": stage_idx, "stage": STAGE_NAMES[stage_idx],
                     "where": where, "file": rel, "line": line, "code": code, "note": note})
    return {"engine": engine, "version": version_of(module, root), "root": str(root), "rows": rows}


def render_md(scans: list[dict]) -> str:
    out = []
    for s in scans:
        out.append(f"**{s['engine']}** `{s['version']}` · `{s['root']}`")
    out.append("")

    if len(scans) == 1:
        s = scans[0]
        out += ["| # | 阶段 | 在哪个进程 | 代码位置 | 这一层做的事 |", "|---|---|---|---|---|"]
        for r in s["rows"]:
            loc = f"`{r['file']}:{r['line']}`" if r["line"] else f"`{r['file']}` ⚠️"
            out.append(f"| {r['stage_idx'] + 1} | **{r['stage']}** | {r['where']} | {loc} | {r['note']} |")
        return "\n".join(out)

    # 并置：按阶段对齐两个引擎
    by_engine = {s["engine"]: {r["stage_idx"]: r for r in s["rows"]} for s in scans}
    names = [s["engine"] for s in scans]
    out += ["| # | 阶段 | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names)]
    for i, stage in enumerate(STAGE_NAMES):
        cells = []
        for e in names:
            r = by_engine[e].get(i)
            if not r:
                cells.append("—")
                continue
            loc = f"`{r['file']}:{r['line']}`" if r["line"] else f"`{r['file']}` ⚠️"
            cells.append(f"{loc}<br><small>{r['where']}</small>")
        out.append(f"| {i + 1} | **{stage}** | " + " | ".join(cells) + " |")

    out += ["", "### 每一层各自在做什么", ""]
    for i, stage in enumerate(STAGE_NAMES):
        out.append(f"**{i + 1}. {stage}**")
        for e in names:
            r = by_engine[e].get(i)
            if r:
                out.append(f"- `{e}`：{r['note']}")
        out.append("")
    return "\n".join(out)


def render_text(scans: list[dict]) -> str:
    out = []
    for s in scans:
        out += [f"{s['engine']} {s['version']}  ·  {s['root']}", "=" * 78]
        for r in s["rows"]:
            loc = f"{r['file']}:{r['line']}" if r["line"] else f"{r['file']} [未找到]"
            out += [f"\n{r['stage_idx'] + 1:>2}. {r['stage']}   [{r['where']}]",
                    f"    {loc}", f"    > {r['code']}", f"    {r['note']}"]
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang", "both"], default="vllm")
    ap.add_argument("--vllm-path", default=None)
    ap.add_argument("--sglang-path", default=None)
    ap.add_argument("--format", choices=["text", "md", "json"], default="text")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    engines = ["vllm", "sglang"] if args.engine == "both" else [args.engine]
    scans = [scan(e, args.vllm_path if e == "vllm" else args.sglang_path) for e in engines]
    scans = [s for s in scans if s["rows"]]
    if not scans:
        raise SystemExit("没找到任何引擎；用 --vllm-path / --sglang-path 指定 site-packages 里的路径")

    if args.format == "json":
        text = json.dumps(scans, indent=2, ensure_ascii=False)
    elif args.format == "md":
        text = render_md(scans)
    else:
        text = render_text(scans)

    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")

    for s in scans:
        missing = [r["file"] for r in s["rows"] if r["line"] is None]
        if missing:
            print(f"\n[!] {s['engine']}: {len(missing)} 个符号未定位：" + ", ".join(missing))


if __name__ == "__main__":
    main()
