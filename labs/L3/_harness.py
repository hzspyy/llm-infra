#!/usr/bin/env python3
"""L3 实验工件记录：manifest.json / cases.json / summary.json。

执行计划要求每轮实验固定输入哈希、模型与软件 pin、实际后端、dtype/shape、
参数、seed、预热、重复、同步、计时边界、容差与输出路径。逐个 lab 手写这些字段
容易漏项，所以集中在这里：lab 只登记每个 case 的关键字段，收尾时一次落盘。

约定：
    L3_OUT=<目录>  工件落盘位置；默认 ./out/<chapter>/<run_id> 不存在时退回脚本同级 out/
用法：
    from _harness import Harness
    h = Harness("3.1-B", "3.1", out=os.environ.get("L3_OUT"))
    h.case(id="tiled_S4096_D128", dtype="bfloat16", shape=[1, 8, 4096, 128],
           params={"block": 64}, ms=0.51)
    h.finish({"verdict": "..."})
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tensor_hash(*tensors) -> str:
    """对张量内容取一个稳定指纹（不依赖对象地址）。"""
    try:
        import torch
    except Exception:                                        # noqa: BLE001
        return "no-torch"
    h = hashlib.sha256()
    for t in tensors:
        tt = t.detach().to("cpu").contiguous().reshape(-1)
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        try:
            raw = tt.numpy().tobytes()
        except TypeError:                                    # bf16/fp8 无 numpy 映射
            raw = tt.float().numpy().tobytes()
        h.update(raw)
    return h.hexdigest()[:16]


def soft_env() -> dict:
    """记录能影响数值与性能的软件版本。"""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
    }
    for mod in ("torch", "triton", "flashinfer", "vllm", "transformers"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:                                    # noqa: BLE001
            env[mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = p.name
            env["sm"] = f"{p.major}{p.minor}"
            env["sm_count"] = p.multi_processor_count
            env["l2_mib"] = round(p.L2_cache_size / 1024 / 1024, 1)
            env["cuda"] = torch.version.cuda
    except Exception:                                        # noqa: BLE001
        pass
    return env


class Harness:
    def __init__(self, task: str, chapter: str, out: str | None = None,
                 backend: str | None = None, notes: str = ""):
        self.task = task
        self.chapter = chapter
        self.backend = backend
        self.notes = notes
        self.cases: list[dict] = []
        self.t0 = time.time()
        self.out = Path(out) if out else Path("out") / chapter / task
        self.out.mkdir(parents=True, exist_ok=True)
        self.meta = soft_env()

    def case(self, **fields):
        fields.setdefault("task", self.task)
        self.cases.append(fields)
        return fields

    def path(self, name: str) -> Path:
        return self.out / name

    def finish(self, summary: dict | None = None, extra_files: dict | None = None) -> dict:
        manifest = {
            "task": self.task,
            "chapter": self.chapter,
            "backend": self.backend,
            "notes": self.notes,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0)),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": round(time.time() - self.t0, 2),
            "env": self.meta,
            "case_count": len(self.cases),
            "source": {
                "argv": sys.argv,
                "script": os.path.basename(sys.argv[0]) if sys.argv else "",
                "script_sha256_16": _sha256_text(Path(sys.argv[0]).read_text(encoding="utf-8"))
                if sys.argv and Path(sys.argv[0]).exists() else None,
            },
            "outputs": [p.name for p in sorted(self.out.glob("*"))],
        }
        (self.out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out / "cases.json").write_text(
            json.dumps(self.cases, ensure_ascii=False, indent=2), encoding="utf-8")
        if summary is not None:
            (self.out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if extra_files:
            for name, text in extra_files.items():
                (self.out / name).write_text(text, encoding="utf-8")
        print(f"\n[harness] 工件写入 {self.out}  ({len(self.cases)} 个 case)")
        return manifest
