#!/usr/bin/env python3
"""8.2-C: Ray Serve 映射（schema 级对照，本机未安装 Ray，未在真实 Ray 集群运行）。

为什么用 subprocess 而不是 Ray 的 vLLM 集成：把一个 OpenAI 兼容服务端作为 replica
跑，能直接对齐 8.2-A 的六阶段；Ray 只负责"副本数、就绪、优雅退出"这三件事。
字段含义对照固定 commit 的 Ray 源码（`ray-project/ray` 03aa07f）：
    python/ray/serve/_private/deployment_state.py:172-174
        ReplicaState.STARTING / RECOVERING / RUNNING
    以及同文件的 PENDING_ALLOCATION / PENDING_INITIALIZATION（:541-544）——
    对应 8.2-D 的"拿到了 GPU 但模型还没就绪"。

运行方式（需先 `pip install "ray[serve]"`）：
    ray start --head --num-gpus=2
    serve run labs/L8/orchestration/ray_serve_vllm.py:app
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

from ray import serve

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-1.7B")
BASE_PORT = int(os.environ.get("BASE_PORT", "8001"))


@serve.deployment(
    num_replicas=2,
    # 每个副本独占 1 张 GPU：Ray 只在分配成功后才会启动副本，
    # 这与 K8s 的 nvidia.com/gpu 限量和 Slurm 的 --gres 是同一层语义。
    ray_actor_options={"num_gpus": 1},
    # 就绪判据必须比"进程起来"更强：health_check 通过才算 RUNNING，
    # 对应 8.2-A 里 serving 探针（真实补全）而非 pid_alive。
    health_check_period_s=5,
    health_check_timeout_s=10,
    # 优雅退出窗口要覆盖最长的在途请求；超时后 Ray 会强杀，在途请求丢失。
    graceful_shutdown_timeout_s=120,
    max_ongoing_requests=64,
)
class VLLMReplica:
    def __init__(self) -> None:
        self.port = BASE_PORT + serve.get_replica_context().replica_tag_hash % 100
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        env = dict(os.environ)
        # Ray 已经把本副本可见的 GPU 通过 CUDA_VISIBLE_DEVICES 限好，
        # 这里再选 0 是相对可见集合的索引，不是物理编号。
        env["CUDA_VISIBLE_DEVICES"] = "0"
        self.proc = subprocess.Popen(
            ["vllm", "serve", MODEL, "--port", str(self.port),
             "--max-model-len", "8192", "--gpu-memory-utilization", "0.85"],
            env=env, start_new_session=True,
        )
        # 阻塞到真正可服务；否则 Ray 会把这个副本标成 RUNNING 但请求全失败。
        self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        import urllib.request
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        raise RuntimeError("replica never became ready")

    def __call__(self, request):
        # 真实路由由 Ray Serve 的 ingress 完成；这里只暴露健康与转发意图。
        return {"port": self.port, "model": MODEL}

    def __exit__(self, *exc):
        if self.proc is not None:
            # 先 SIGTERM 让引擎自己收尾（排空在途请求），超时才强杀。
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()


app = VLLMReplica.bind()
