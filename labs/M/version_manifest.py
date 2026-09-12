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
            "version": subprocess.check_output(
                ["python3", "--version"], text=True
            ).strip(),
            "executable": subprocess.check_output(
                ["which", "python3"], text=True
            ).strip(),
        },
        "source_locations": {
            "torch_package": str(Path(torch.__file__).parent),
            "torch_includes": str(Path(torch.__file__).parent / "include"),
            "torch_lib": str(Path(torch.__file__).parent / "lib"),
        },
    }

    # 尝试获取 git 提交（如果是从源码安装）
    try:
        torch_path = Path(torch.__file__).parent.parent
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=torch_path,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        manifest["pytorch"]["source_commit"] = git_hash
    except:
        manifest["pytorch"]["source_commit"] = "N/A (wheel install)"

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
