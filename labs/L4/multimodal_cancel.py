#!/usr/bin/env python3
"""L4.8 修订（C 段的取消部分）—— 多模态请求的取消与资源回收。

在三个位置发起取消，并验证资源回收与引擎健康：
  [C1] 慢模态：一次请求塞 4 张 1280×720 大图，处理中途 abort
  [C2] 单图请求：处理中途 abort
  [C3] 错误图像：损坏的 PNG 字节（走异常路径，而不是取消）
每个用例之后：再发一个纯文本请求，测首 token 延迟（引擎是否仍然可用），
并记录显存（torch 分配器）与已接收 chunk 数、abort 之后是否还有输出。

用 VLLM_ENABLE_V1_MULTIPROCESSING=0 让引擎在进程内运行，便于观察显存与取消语义。

用法：
    python labs/L4/multimodal_cancel.py --outdir out/4.8/20260913-serving
"""

import argparse
import asyncio
import glob
import io
import json
import os
import sys
import time

import torch

HUB = os.environ.get("HF_HOME", "/scratch/learn/models/hf") + "/hub"
REPO = "Qwen/Qwen3-VL-4B-Instruct"
SUMMARY = {}


def snap(repo=REPO):
    return sorted(glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*"))[0]


def make_images():
    from PIL import Image, ImageDraw
    imgs = []
    for i in range(4):
        im = Image.new("RGB", (1280, 720), (18 + i * 10, 22, 34))
        ImageDraw.Draw(im).text((60, 360), f"FRAME-{i}", fill=(255, 255, 255))
        imgs.append(im)
    return imgs


def corrupt_png(nbytes=4096):
    """伪造一个 PNG 头 + 随机字节：processor 打开时会失败。"""
    return b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * (nbytes // 256)


def mem_mib():
    return torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None


async def main_async(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.inputs.llm import TokensPrompt

    d = snap()
    proc = AutoProcessor.from_pretrained(d)
    imgs = make_images()
    args_ = AsyncEngineArgs(model=d, dtype="bfloat16", gpu_memory_utilization=0.5,
                            max_model_len=8192, enforce_eager=True,
                            limit_mm_per_prompt={"image": 4, "video": 1},
                            disable_log_stats=True)
    t0 = time.perf_counter()
    engine = AsyncLLM.from_engine_args(args_)
    print(f"  引擎就绪 {time.perf_counter()-t0:.1f} s；显存 {mem_mib():.0f} MiB")

    text_ids = proc.tokenizer("用一句话解释 MoE。")["input_ids"]
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)

    async def health(tag):
        """取消之后：纯文本请求是否正常 + 显存是否回落。"""
        t0 = time.perf_counter()
        n = 0
        async for out in engine.generate(TokensPrompt(prompt_token_ids=text_ids),
                                         sp1, request_id=f"health-{tag}-{time.time()}"):
            n += 1
        return {"ttft_s": time.perf_counter() - t0, "chunks": n, "mem_mib": mem_mib()}

    base = await health("base")
    print(f"  基线：纯文本首 token {base['ttft_s']*1e3:.1f} ms，显存 {base['mem_mib']:.0f} MiB")

    def mm_token_ids(convs):
        ins = proc.apply_chat_template(convs, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt")
        return ins["input_ids"][0].tolist()

    async def case(tag, mm_data, abort_after, sp, prompt_ids=None):
        """用独立任务在 abort_after 秒后触发 abort（不能依赖"收到 chunk 才检查"）。"""
        rid = f"{tag}-{time.time()}"
        box = {"t": None}

        async def abort_later():
            await asyncio.sleep(abort_after)
            box["t"] = time.perf_counter()
            await engine.abort(rid)

        got, after, err = 0, 0, None
        t0 = time.perf_counter()
        task = asyncio.create_task(abort_later())
        ids_ = prompt_ids if prompt_ids is not None else text_ids
        gen = engine.generate(TokensPrompt(prompt_token_ids=ids_,
                                           multi_modal_data=mm_data),
                              sp, request_id=rid)
        try:
            async for out in gen:
                got += 1
                if box["t"] is not None:
                    after += 1
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:140]}"
        t_end = time.perf_counter()
        if not task.done():
            task.cancel()
        return {"case": tag, "chunks_before_abort": got - after,
                "chunks_after_abort": after,
                "abort_fired": box["t"] is not None,
                "abort_to_stop_s": (t_end - box["t"]) if box["t"] else None,
                "error": err, "wall_s": t_end - t0,
                "mem_mib": mem_mib()}

    sp32 = SamplingParams(max_tokens=32, temperature=0.0)
    results = []
    # 文本长生成：一定能产生 token，用于验证「取消后不再有输出」
    r = await case("text_long_abort", None, 0.15,
                   SamplingParams(max_tokens=512, temperature=0.0))
    results.append(r)
    print(f"  text_long_abort  abort 已触发={r['abort_fired']}，取消前 "
          f"{r['chunks_before_abort']} chunk，取消后 {r['chunks_after_abort']} chunk，"
          f"abort→生成器结束 {((r['abort_to_stop_s'] or 0)*1e3):.0f} ms，"
          f"总墙钟 {r['wall_s']*1e3:.0f} ms")
    h = await health("text_long")
    r["health"] = h
    print(f"               随后纯文本首 token {h['ttft_s']*1e3:.1f} ms"
          f"（基线 {base['ttft_s']*1e3:.1f} ms）")
    conv4 = [[{"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
               + [{"type": "text", "text": "描述这些图"}]}]]
    conv1 = [[{"role": "user", "content": [{"type": "image", "image": imgs[0]},
                                          {"type": "text", "text": "描述这张图"}]}]]
    for tag, data, delay, conv in (("slow_4images", {"image": imgs}, 0.20, conv4),
                                   ("single_image", {"image": [imgs[0]]}, 0.10, conv1)):
        try:
            pids = mm_token_ids(conv)
        except Exception as e:
            pids = None
            print(f"  {tag}: chat 模板构造失败 {type(e).__name__}: {str(e)[:80]}")
        r = await case(tag, data, delay, sp32, prompt_ids=pids)
        results.append(r)
        print(f"  {tag:<13} abort 已触发={r['abort_fired']}，取消前收到 "
              f"{r['chunks_before_abort']} chunk，取消后 {r['chunks_after_abort']} chunk"
              f"（应为 0），abort→生成器结束 "
              f"{((r['abort_to_stop_s'] or 0)*1e3):.0f} ms，总墙钟 {r['wall_s']*1e3:.0f} ms"
              f"{'  错误 ' + r['error'] if r['error'] else ''}")
        h = await health(tag)
        r["health"] = h
        print(f"               随后纯文本首 token {h['ttft_s']*1e3:.1f} ms"
              f"（基线 {base['ttft_s']*1e3:.1f} ms），显存 {h['mem_mib']:.0f} MiB")

    sub("C3 错误图像：损坏的 PNG 字节")
    bad = corrupt_png()
    from PIL import Image
    try:
        Image.open(io.BytesIO(bad)).convert("RGB")
        err = None
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
    print(f"  客户端解码即失败：{err}")
    results.append({"case": "corrupt_png", "client_decode_error": err})
    h = await health("after_error")
    print(f"  错误路径之后纯文本首 token {h['ttft_s']*1e3:.1f} ms，"
          f"显存 {h['mem_mib']:.0f} MiB")
    results[-1]["health"] = h

    if args.scan:
        sub("C5 取消时机扫描：验证「abort 是否只在解码阶段有效」")
        scan_rows = []
        for n_img in (1, 2, 4):
            data = {"image": imgs[:n_img]}
            conv = [[{"role": "user", "content":
                      [{"type": "image", "image": im} for im in imgs[:n_img]]
                      + [{"type": "text", "text": "描述这些图"}]}]]
            try:
                pids = mm_token_ids(conv)
            except Exception:
                pids = None
            for delay in (0.02, 0.1, 0.3, 0.8):
                r = await case(f"{n_img}img@{int(delay*1000)}ms", data, delay, sp32,
                               prompt_ids=pids)
                scan_rows.append(r)
                print(f"  {n_img} 图 / {delay*1e3:4.0f} ms abort：取消前 "
                      f"{r['chunks_before_abort']:>3} chunk，取消后 "
                      f"{r['chunks_after_abort']:>3} chunk，abort→结束 "
                      f"{((r['abort_to_stop_s'] or 0)*1e3):6.0f} ms，"
                      f"墙钟 {r['wall_s']*1e3:6.0f} ms")
                h = await health(f"scan{r['case']}")
                r["health"] = h
        SUMMARY["C_scan"] = {"rows": scan_rows,
                             "hypothesis": "取消在解码阶段有效、在预处理/编码阶段无效"}
        ok = [r for r in scan_rows if r["chunks_before_abort"] > 0]
        bad = [r for r in scan_rows if r["chunks_after_abort"] > 1]
        print(f"  汇总：{len(scan_rows)} 组里，abort 前已有输出的 {len(ok)} 组"
              f"（其中取消后仍有多于 1 个 chunk 的 {len([r for r in ok if r['chunks_after_abort']>1])} 组）；"
              f"abort 前尚无输出的组 {len(scan_rows)-len(ok)} 组，"
              f"其中取消后仍有输出的 {len([r for r in scan_rows if r['chunks_before_abort']==0 and r['chunks_after_abort']>0])} 组")

    if args.reclaim:
        sub("C6 资源回收：一批被取消的重请求之后，显存是否回到基线")
        def dev_free():
            free, total = torch.cuda.mem_get_info()
            return free / 2**20, total / 2**20

        def pool_state():
            for path in ("engine_core.engine_core.scheduler.kv_cache_manager.block_pool",
                         "engine_core.scheduler.kv_cache_manager.block_pool"):
                obj = engine
                try:
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    return {"free_blocks": int(obj.get_num_free_blocks()),
                            "total_blocks": int(obj.num_gpu_blocks)}
                except Exception:
                    continue
            return None

        f0, tot = dev_free()
        p0 = pool_state()
        print(f"  基线：设备空闲 {f0:.0f} / {tot:.0f} MiB；块池 {p0}")
        n_req = 8
        heavy = [{"image": imgs} for _ in range(n_req)]
        tasks = []
        for i, data in enumerate(heavy):
            conv = [[{"role": "user", "content":
                      [{"type": "image", "image": im} for im in imgs]
                      + [{"type": "text", "text": "描述"}]}]]
            pids = mm_token_ids(conv)
            tasks.append(case(f"reclaim{i}", data, 0.05, sp32, prompt_ids=pids))
        rs = await asyncio.gather(*tasks)
        after_immediate = dev_free()[0]
        await asyncio.sleep(3.0)
        h = await health("after_reclaim")
        f1, _ = dev_free()
        p1 = pool_state()
        print(f"  {n_req} 个重请求全部在 50 ms 时取消；立即空闲 {after_immediate:.0f} MiB，"
              f"等待 3 s 后 {f1:.0f} MiB（基线 {f0:.0f}，差 {f0-f1:+.0f} MiB）")
        print(f"  取消后块池：{p1}")
        print(f"  健康检查首 token {h['ttft_s']*1e3:.1f} ms（基线 "
              f"{base['ttft_s']*1e3:.1f} ms），取消后各请求收到 chunk "
              f"{sorted(set(r['chunks_after_abort'] for r in rs))}")
        SUMMARY["C_reclaim"] = {"baseline_free_mib": f0, "after_immediate_mib": after_immediate,
                               "after_3s_mib": f1, "delta_mib": f0 - f1,
                               "pool_before": p0, "pool_after": p1,
                               "health_after": h,
                               "chunks_after_abort": [r["chunks_after_abort"] for r in rs],
                               "n_requests": n_req}

    sub("C4 取消时机与资源回收汇总")
    for r in results:
        if "health" in r and "ttft_s" in r["health"]:
            print(f"  {r['case']:<13} 取消后首 token "
                  f"{r['health']['ttft_s']*1e3:7.1f} ms（基线 {base['ttft_s']*1e3:.1f} ms）"
                  f"  显存 {r['health']['mem_mib']:7.0f} MiB")
    SUMMARY["C_cancel"] = {"baseline": base, "cases": results,
                           "env": {"torch": torch.__version__,
                                   "multiprocessing": "disabled（进程内引擎）"}}
    try:
        engine.shutdown()
    except Exception:
        pass
    return SUMMARY


def sub(s):
    print("\n--- " + s + " " + "-" * max(0, 70 - len(s)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.expanduser("~/l48cancel"))
    ap.add_argument("--scan", action="store_true", help="跑取消时机扫描")
    ap.add_argument("--reclaim", action="store_true", help="跑资源回收观测")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    title = "[C] 多模态请求的取消与资源回收"
    print("=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)
    asyncio.run(main_async(args))
    path = os.path.join(args.outdir, "multimodal_cancel.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(SUMMARY)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写出 {path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
