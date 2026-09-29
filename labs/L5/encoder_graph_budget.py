#!/usr/bin/env python3
"""L5.4 补测 · vision encoder 的独立 CUDA Graph 与它的 token 预算。

decode 图与 encoder 图是两套管理（`vllm/v1/worker/encoder_cudagraph.py`）。
encoder 侧按 **token 预算档**捕获：一张图对应一个固定的 token 容量，
运行时把图像贪心装进"最小的够用档"再回放。本脚本实测三件事：

  1. 预算档是怎么定的：`_generate_budgets`（`:192`）从模型给的
     `get_encoder_cudagraph_budget_range` 取 min/max，生成 2 的幂次档；
     `max_batch_size <= min_token_budget` 是捕获前的不变量（`:78-95`）。
  2. 打开 `cudagraph_mm_encoder` 后，不同图像规模落在哪一档、图是否真的被回放。
  3. 显式给档位（少给几档）与自动推断相比，延迟与输出是否一致。

用法（crater）：
    python encoder_graph_budget.py --out <dir> [--budgets 2048,4096,8192]
"""

import argparse
import json
import os
import statistics
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("L54_VLM", "Qwen/Qwen3-VL-4B-Instruct")


def safe_util(reserve_gib=6.0, cap=0.55):
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return min(cap, max(free / gib - reserve_gib, 1.0) / (total / gib))


def find_manager(obj, depth=0, seen=None):
    """在 runner 对象树里找 EncoderCudaGraphManager（不依赖私有属性名）。"""
    seen = seen or set()
    if id(obj) in seen or depth > 3:
        return None
    seen.add(id(obj))
    if type(obj).__name__ == "EncoderCudaGraphManager":
        return obj
    for name, val in vars(obj).items() if hasattr(obj, "__dict__") else []:
        if type(val).__name__ == "EncoderCudaGraphManager":
            return val
        if hasattr(val, "__dict__") and type(val).__module__.startswith("vllm"):
            got = find_manager(val, depth + 1, seen)
            if got is not None:
                return got
    return None


def make_llm(util, budgets=None, cudagraph_mm=True):
    from vllm import LLM
    cc = {"cudagraph_mm_encoder": cudagraph_mm}
    if budgets:
        cc["encoder_cudagraph_token_budgets"] = budgets
    return LLM(model=MODEL, gpu_memory_utilization=util, max_model_len=8192,
               enforce_eager=False, enable_prefix_caching=False,
               disable_log_stats=False, max_num_batched_tokens=8192,
               limit_mm_per_prompt={"image": 8}, compilation_config=cc)


def make_image(size, color):
    from PIL import Image
    return Image.new("RGB", size, color)


def build_prompt_ids(proc, question="用一句话描述这张图。"):
    """Qwen3-VL 的 chat 模板要自己插入视觉占位符，不能只给纯文本。"""
    convs = [{"role": "user",
              "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    ins = proc.apply_chat_template(convs, add_generation_prompt=True,
                                   tokenize=True, return_dict=True,
                                   return_tensors="pt")
    return ins["input_ids"][0].tolist()


def run_case(llm, proc, images, prompt_ids, out=16):
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(max_tokens=out, temperature=0.0, ignore_eos=True)
    prompts = [TokensPrompt(prompt_token_ids=list(prompt_ids),
                            multi_modal_data={"image": im}) for im in images]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = (time.perf_counter() - t0) * 1000
    return dict(n_images=len(images), wall_ms=wall,
                texts=[o.outputs[0].text for o in outs],
                tokens=[list(o.outputs[0].token_ids) for o in outs])


def manager_facts(mgr):
    if mgr is None:
        return dict(found=False)
    facts = dict(found=True, token_budgets=list(mgr.token_budgets),
                 max_batch_size=mgr.max_batch_size,
                 max_frames_per_batch=getattr(mgr, "max_frames_per_batch", None),
                 captured=mgr.is_captured(),
                 n_graphs=mgr.get_num_graphs_to_capture())
    stats = None
    try:
        stats = mgr.get_cumulative_stats()
    except Exception:                                           # noqa: BLE001
        pass
    facts["cumulative_stats"] = stats
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default=None, help="逗号分隔；不给则自动推断")
    ap.add_argument("--util", type=float, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    budgets = [int(x) for x in args.budgets.split(",")] if args.budgets else None
    util = args.util or safe_util()
    tag = "auto" if not budgets else "user_" + "_".join(str(b) for b in budgets)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    prompt_ids = build_prompt_ids(proc)
    llm = make_llm(util, budgets)
    engine = llm.llm_engine.engine_core
    core = getattr(engine, "engine_core", engine)
    runner = core.model_executor.driver_worker.worker.model_runner
    mgr = find_manager(runner)
    facts = manager_facts(mgr)

    # 图像规模覆盖若干档：小图、刚好一档、跨档
    sizes = [(224, 224), (448, 448), (896, 896), (1344, 1344)]
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    cases = []
    for _ in range(1):                       # 预热
        run_case(llm, proc, [make_image(sizes[0], colors[0])], prompt_ids)
    plan = [(list(sz), [make_image(sz, col)]) for sz, col in zip(sizes, colors)]
    plan += [(f"{n}x224", [make_image(sizes[0], colors[i % 4]) for i in range(n)])
             for n in (2, 4)]
    for label, imgs in plan:
        runs = [run_case(llm, proc, imgs, prompt_ids) for _ in range(args.repeats)]
        walls = sorted(r["wall_ms"] for r in runs)
        cases.append(dict(size=label, n_images=len(imgs),
                          wall_median_ms=statistics.median(walls),
                          wall_range=[walls[0], walls[-1]],
                          texts=runs[0]["texts"],
                          tokens=[t for r in runs for t in r["tokens"]]))
    facts_after = manager_facts(mgr)

    meta = dict(model=MODEL, budgets=budgets, util=util,
                torch=torch.__version__, vllm=__import__("vllm").__version__,
                gpu=torch.cuda.get_device_name(0))
    lines = [f"encoder CUDA Graph 预算 · {MODEL} · budgets={budgets or '自动'} · "
             f"prompt token 数 {len(prompt_ids)}",
             f"捕获前：{json.dumps(facts, ensure_ascii=False)}",
             f"跑完后：{json.dumps(facts_after, ensure_ascii=False)}", ""]
    lines.append(f"  {'输入':>10}{'中位 ms':>9}{'极差 ms':>16}{'首条输出前 24 字符':>40}")
    for c in cases:
        first = (c["texts"][0][:24] if c["texts"] else "")
        rng_ = f"[{c['wall_range'][0]:.0f}, {c['wall_range'][1]:.0f}]"
        lines.append(f"  {str(c['size']):>10}{c['wall_median_ms']:>9.1f}{rng_:>16}"
                     f"{first:>40}")
    print("\n".join(lines))

    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(dict(meta=meta, before=facts, after=facts_after, cases=cases),
                  f, indent=1)
    with open(os.path.join(args.out, f"{tag}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    print(f"\n写入 {args.out}/{tag}.{{txt,json}}")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    os._exit(0)
