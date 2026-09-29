#!/usr/bin/env python3
"""L5.8 —— worker 崩溃后的请求恢复：四种策略的实际结果。

本章此前没有覆盖 worker 崩溃。真实引擎里 worker 是独立进程（vLLM 的 EngineCore），
而 nanoserve 的 server.py 把引擎放在**同一个事件循环**里：

    async def engine_loop(self):
        while True:
            ...
            trace = self.engine.step()      # 没有任何 try/except
            ...
            await asyncio.sleep(0)

所以 `step()` 抛异常时，这个任务会带着未处理异常结束，而 **HTTP 服务本身还在跑**：
客户端照样能连上，只是永远等不到 token。这一条要用实验确认，不能只读代码。

四种策略各跑一次（都发同一条流式请求）：

  A baseline          不注入故障，作为输出参照
  B 崩溃，不处理       现状：engine_loop 死去，HTTP 仍在接受连接
  C 崩溃，捕获后重启循环  在**同一个 engine** 上继续（"捕获异常"不等于"恢复"）
  D 独立进程重启       杀掉服务进程再起一个，验证输出与参照一致、块池干净

记录：客户端是否收到 EOF/错误、后续请求能否完成、泄漏块数、输出是否与参照一致。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PORT = 8151
REPO = os.environ.get("NANOSERVE_MODEL", "Qwen/Qwen3-1.7B")
HUB = os.environ.get("HF_HUB_CACHE", "/scratch/learn/models/hf/hub")
BLOCKS = 256


def build_engine():
    from engine import Engine
    from model import PagedModel
    model = PagedModel(REPO, HUB, num_blocks=BLOCKS, block_size=16)
    return Engine(model, block_size=16, max_batched_tokens=256,
                  max_num_seqs=4, enable_prefix_cache=False, eos_ids=[])


def prompt_ids(tok, text):
    return tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))


async def client_stream(text, tok, timeout=6.0, max_tokens=24):
    """发一条流式请求，返回 (事件数, 是否收到 EOF, 是否超时, 文本)。"""
    # server.py 的 /generate 收的是 {prompt: 文本}，不是 input_ids
    body = json.dumps({"prompt": text, "max_tokens": max_tokens}).encode()
    t0 = time.perf_counter()
    reader, writer = await asyncio.open_connection("127.0.0.1", PORT)
    writer.write(b"POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                 b"Content-Type: application/json\r\n"
                 b"Content-Length: %d\r\n\r\n" % len(body) + body)
    await writer.drain()
    events, raw = 0, b""
    timed_out = False
    eof = False
    try:
        while True:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=timeout)
            if not chunk:
                eof = True
                break
            raw += chunk
            events += chunk.count(b"data:")
            if time.perf_counter() - t0 > 30:
                break
    except asyncio.TimeoutError:
        timed_out = True
    writer.close()
    return dict(events=events, eof=eof, timed_out=timed_out,
                seconds=round(time.perf_counter() - t0, 3),
                text=raw.decode("utf-8", "replace")[-200:])


async def case(engine, tok, srv, label, crash_after=None, restart_loop=False,
         second_request=True, crash_in="step"):
    """跑一次；crash_after 指定第几次调用抛异常。

    crash_in="step" 在 step() 之前抛：引擎状态**完全没动**；
    crash_in="run"  在 _run() 里抛：此时 _schedule 已经分好块、扩过块表，
    所以能看出"半写状态"下的恢复与泄漏差异。
    """
    from server import Server
    server = Server(engine, tok)
    orig_step = engine.step
    orig_run = engine._run
    state = {"n": 0}

    def step_wrapper():
        if crash_in == "run":
            return orig_step()
        state["n"] += 1
        if crash_after is not None and state["n"] == crash_after:
            raise RuntimeError("injected worker crash (before step)")
        return orig_step()

    def run_wrapper(groups):
        if crash_in != "run":
            return orig_run(groups)
        state["n"] += 1
        if crash_after is not None and state["n"] == crash_after:
            raise RuntimeError("injected worker crash (inside _run)")
        return orig_run(groups)

    engine.step = step_wrapper
    engine._run = run_wrapper
    srv_handle = await asyncio.start_server(server.handle, "127.0.0.1", PORT)

    loop_task = asyncio.create_task(server.engine_loop())
    await asyncio.sleep(0.2)

    first = await client_stream("Continue this story about a lighthouse keeper. " * 3,
                                tok)
    crashed = loop_task.done()
    exc = None
    if crashed:
        exc = repr(loop_task.exception())

    second = None
    if second_request:
        second = await client_stream("Summarize the theory of relativity.", tok,
                                     timeout=4.0)

    # 恢复动作
    resumed = None
    if restart_loop:
        # 只重启循环任务，引擎对象还是崩过的那一个
        try:
            loop_task2 = asyncio.create_task(server.engine_loop())
            await asyncio.sleep(0.1)
            resumed = await client_stream("Write a Python function that adds two numbers.",
                                          tok, timeout=4.0)
            if not loop_task2.done():
                loop_task2.cancel()
        except Exception as e:                                # pragma: no cover
            resumed = dict(error=f"{type(e).__name__}: {e}")

    if not loop_task.done():
        loop_task.cancel()
    srv_handle.close()
    await srv_handle.wait_closed()
    engine.step = orig_step
    engine._run = orig_run
    return dict(case=label, crash_at=crash_after, crash_in=crash_in,
                running_left=len(engine.running),
                waiting_left=len(engine.waiting),
                loop_task_dead=crashed, loop_task_exception=exc,
                first_request=first, second_request=second,
                after_restart_loop=resumed,
                steps_run=state["n"], leak=engine.leak_check())


_LOG_PATH = None


def args_log():
    return _LOG_PATH or "/dev/null"


def run_isolated(args):
    """在一个独立进程里起服务，用于 D：进程被杀掉再重启。"""
    env = dict(os.environ, NANOSERVE_MODEL=REPO, HF_HUB_CACHE=HUB)
    py = "/scratch/learn/envs/serve/bin/python"
    log = open(args_log(), "ab")
    proc = subprocess.Popen(
        [py, "-u", str(pathlib.Path(__file__).resolve().parent / "server.py"),
         "--port", str(PORT), "--blocks", str(BLOCKS)],
        stdout=log, stderr=subprocess.STDOUT, env=env,
        cwd=str(pathlib.Path(__file__).resolve().parent))
    for _ in range(600):
        if proc.poll() is not None:
            return proc          # 已经退出，日志里有原因
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.5)
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global _LOG_PATH
    _LOG_PATH = str(args.out / "iso-server.log")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, local_files_only=True)
    report = dict(model=REPO, blocks=BLOCKS, port=PORT, cases=[])

    def free(tag):
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    async def async_cases():
        # A：不注入
        eng = build_engine()
        report["cases"].append(await case(eng, tok, None, "A_baseline"))
        del eng; free("A")

        # B：崩溃，不处理（server.py 的现状）
        eng = build_engine()
        report["cases"].append(await case(eng, tok, None, "B_crash_no_handler",
                                          crash_after=12))
        report["B_leak_after_crash"] = report["cases"][-1]["leak"]
        del eng; free("B")

        # C：崩溃后只重启循环任务，仍在同一个 engine 上
        eng = build_engine()
        report["cases"].append(await case(eng, tok, None,
                                          "C_catch_and_restart_loop",
                                          crash_after=12, restart_loop=True))
        del eng; free("C")

        # C2：在 _run 里崩（已经分过块、扩过块表），看半写状态
        eng = build_engine()
        report["cases"].append(await case(eng, tok, None,
                                          "C2_crash_inside_run",
                                          crash_after=12, restart_loop=True,
                                          crash_in="run"))
        del eng; free("C2")

    # D 先跑：它要另起一个进程再加载一份权重，我这边一旦占着显存它就会 OOM
    # D：独立进程被杀 + 重启
    proc = run_isolated(args)
    time.sleep(1.0)
    d = dict(case="D_independent_process_restart", killed=False)
    try:
        d["first_request"] = asyncio.run(client_stream("hello", tok, timeout=4.0))
        proc.kill()
        d["killed"] = True
        proc.wait(timeout=30)
        time.sleep(2.0)
        proc2 = run_isolated(args)
        time.sleep(1.0)
        d["after_restart"] = asyncio.run(client_stream("hello", tok, timeout=30.0))
        proc2.terminate()
        try:
            proc2.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc2.kill()
    except Exception as e:                                    # pragma: no cover
        d["error"] = f"{type(e).__name__}: {e}"
    report["cases"].append(d)

    asyncio.run(async_cases())

    (args.out / "crash_recovery.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in report["cases"]:
        print(f"--- {c['case']}")
        print(f"    loop 任务已死 {c.get('loop_task_dead')} "
              f"{c.get('loop_task_exception') or ''}")
        for key in ("first_request", "second_request", "after_restart_loop",
                    "after_restart"):
            if c.get(key):
                r = c[key]
                print(f"    {key:<18} 事件 {r.get('events')} EOF {r.get('eof')} "
                      f"超时 {r.get('timed_out')} {r.get('seconds')}s")
        if "leak" in c:
            print(f"    泄漏块 {c['leak']['leaked']}")
        if c.get("error"):
            print(f"    error {c['error']}")


if __name__ == "__main__":
    main()
