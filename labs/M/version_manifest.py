#!/usr/bin/env python3
"""
采集完整的可复现环境快照

Usage:
    python3 version_manifest.py > ../../results/local/M1/env_snapshot.json
"""
import torch
import subprocess
import json
import sys
import platform
from pathlib import Path


def capture_environment():
    """采集完整的可复现环境快照"""
    manifest = {
        "pytorch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "source_locations": {
            "torch_package": str(Path(torch.__file__).parent),
            "torch_includes": str(Path(torch.__file__).parent / "include"),
            "torch_lib": str(Path(torch.__file__).parent / "lib"),
        },
    }

    # 使用构建记录，避免把包目录的祖先仓库提交误当 PyTorch 提交。
    manifest["pytorch"]["source_commit"] = torch.version.git_version

    # CUDA 设备信息（如果可用）
    if torch.cuda.is_available():
        manifest["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            manifest["cuda_devices"].append({
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1024**3,
            })

    return manifest


if __name__ == "__main__":
    env = capture_environment()
    print(json.dumps(env, indent=2), file=sys.stdout)
