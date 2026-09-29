#!/usr/bin/env python3
"""
labs/L7/training_job_manifest.py
训练作业清单与状态机流转（Job Manifest & State Machine）。

覆盖计划任务的四件事：
1. 作业声明：节点规模、GPU 拓扑、rendezvous 后端、恢复点、重试策略；
2. 环境契约：由资源声明推导每个 rank 的 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT，
   并检查 torchrun 命令与资源声明是否一致；
3. 生命周期：待调度 → 准入 → 初始化 → 运行 → 失败 → 重启 → 恢复 → 完成；
4. 非法配置：零节点、非标准 GPU 数、未知 rendezvous 后端、缺恢复目录、命令与资源不一致。

Usage:
    python labs/L7/training_job_manifest.py --outdir "$RUN_DIR/job-manifest"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class JobResourceSpec:
    num_nodes: int
    gpus_per_node: int
    cpus_per_task: int
    memory_per_node_gb: int
    gpu_type: str = "H100-SXM5-80GB"
    interconnect: str = "InfiniBand-400Gbps"


@dataclass
class JobRendezvousSpec:
    backend: str = "c10d"  # "c10d" 或 "etcd-v2"
    master_addr: str = "coordinator-node-0.cluster.local"
    master_port: int = 29500
    timeout_seconds: int = 1800


@dataclass
class JobRestartPolicy:
    max_restarts: int = 3
    restart_delay_seconds: int = 10
    auto_resume: bool = True
    checkpoint_dir: str = "/scratch/learn/checkpoints/smollm3-3b"


@dataclass
class JobManifest:
    job_id: str
    job_name: str
    model_name: str
    resources: JobResourceSpec
    rendezvous: JobRendezvousSpec
    restart_policy: JobRestartPolicy
    environment_variables: Dict[str, str] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)


class JobLifecycleSimulator:
    """作业生命周期状态机，以及由资源声明推导的环境契约。"""

    VALID_STATES = [
        "PENDING",
        "ADMITTED",
        "INITIALIZING",
        "RUNNING",
        "FAILED_TRANSIENT",
        "RESTARTING",
        "RESUMING",
        "COMPLETED",
        "FAILED_TERMINAL",
    ]

    def __init__(self, manifest: JobManifest):
        self.manifest = manifest
        self.current_state = "PENDING"
        self.restart_count = 0
        self.history: List[Dict[str, Any]] = []

    def _transition(self, new_state: str, reason: str = ""):
        assert new_state in self.VALID_STATES, f"Invalid state: {new_state}"
        self.history.append({
            "from_state": self.current_state,
            "to_state": new_state,
            "restart_count": self.restart_count,
            "reason": reason,
        })
        self.current_state = new_state

    def validate_manifest(self) -> List[str]:
        errors = []
        r = self.manifest.resources
        if r.num_nodes <= 0:
            errors.append("num_nodes must be positive")
        if r.gpus_per_node not in (1, 2, 4, 8):
            errors.append(f"Invalid gpus_per_node: {r.gpus_per_node}, "
                          "standard topologies require 1, 2, 4, or 8")
        if self.manifest.rendezvous.backend not in ("c10d", "etcd-v2"):
            errors.append(f"Unsupported rendezvous backend: {self.manifest.rendezvous.backend}")
        if not self.manifest.restart_policy.checkpoint_dir:
            errors.append("checkpoint_dir is required for fault-tolerant auto_resume")
        errors.extend(self.validate_command())
        return errors

    def validate_command(self) -> List[str]:
        """检查 torchrun 启动参数是否与资源声明一致。"""
        command = self.manifest.command
        errors = []
        if not command or command[0] != "torchrun":
            return errors
        options = {item.split("=", 1)[0]: item.split("=", 1)[1]
                   for item in command[1:] if item.startswith("--") and "=" in item}
        r = self.manifest.resources
        if options.get("--nproc_per_node") not in (None, str(r.gpus_per_node)):
            errors.append("--nproc_per_node does not match gpus_per_node")
        if options.get("--nnodes") not in (None, str(r.num_nodes)):
            errors.append("--nnodes does not match num_nodes")
        return errors

    def derive_rank_env(self) -> List[Dict[str, Any]]:
        """由资源声明推导 rank → 环境变量映射。"""
        world_size = self.manifest.resources.num_nodes * self.manifest.resources.gpus_per_node
        rows = []
        for rank in range(world_size):
            rows.append({
                "rank": rank,
                "node_index": rank // self.manifest.resources.gpus_per_node,
                "local_rank": rank % self.manifest.resources.gpus_per_node,
                "env": {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank % self.manifest.resources.gpus_per_node),
                    "WORLD_SIZE": str(world_size),
                    "MASTER_ADDR": self.manifest.rendezvous.master_addr,
                    "MASTER_PORT": str(self.manifest.rendezvous.master_port),
                },
            })
        return rows

    def simulate_lifecycle_with_failure(self, inject_failure_step: int = 1500) -> List[Dict[str, Any]]:
        self.history.clear()
        errors = self.validate_manifest()
        if errors:
            self._transition("FAILED_TERMINAL", f"Config validation errors: {errors}")
            return self.history

        self._transition("ADMITTED", "Gang admission granted; all requested nodes and GPUs available")
        world_size = self.manifest.resources.num_nodes * self.manifest.resources.gpus_per_node
        self._transition(
            "INITIALIZING",
            f"Rendezvous joined by {world_size} ranks via "
            f"{self.manifest.rendezvous.backend}:{self.manifest.rendezvous.master_port}",
        )
        self._transition("RUNNING", "Distributed communication established; training loop started at step 0")
        self._transition(
            "FAILED_TRANSIENT",
            f"worker-node-1 encountered NCCL watchdog timeout at step {inject_failure_step}",
        )
        if self.restart_count < self.manifest.restart_policy.max_restarts:
            self.restart_count += 1
            self._transition("RESTARTING",
                             f"Restart policy invoked (attempt {self.restart_count}/"
                             f"{self.manifest.restart_policy.max_restarts})")
            self._transition(
                "RESUMING",
                f"Loading latest valid checkpoint from {self.manifest.restart_policy.checkpoint_dir} "
                f"at step {inject_failure_step - 50}",
            )
            self._transition("RUNNING", "Resumed execution from checkpoint; trained to final step 3000")
            self._transition("COMPLETED", "Target token budget reached; final checkpoint committed")
        else:
            self._transition("FAILED_TERMINAL", "Max restart limit exceeded")
        return self.history


def valid_manifest() -> JobManifest:
    return JobManifest(
        job_id="job-smollm3-3b-pretrain-001",
        job_name="smollm3-3b-stage1",
        model_name="SmolLM3-3B",
        resources=JobResourceSpec(num_nodes=6, gpus_per_node=8, cpus_per_task=12,
                                  memory_per_node_gb=512),
        rendezvous=JobRendezvousSpec(backend="c10d",
                                     master_addr="coordinator-node-0.cluster.local",
                                     master_port=29500),
        restart_policy=JobRestartPolicy(max_restarts=3, restart_delay_seconds=10,
                                        auto_resume=True,
                                        checkpoint_dir="/scratch/learn/checkpoints/smollm3-3b"),
        environment_variables={
            "NCCL_DEBUG": "INFO",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
        command=["torchrun", "--nproc_per_node=8", "--nnodes=6", "train.py",
                 "--config=recipes/smollm3.yaml"],
    )


def illegal_manifests() -> Dict[str, JobManifest]:
    """五个反例：每个只改一处。"""
    cases = {}
    base = valid_manifest()

    zero_nodes = valid_manifest()
    zero_nodes.resources.num_nodes = 0
    zero_nodes.command = ["torchrun", "--nproc_per_node=8", "--nnodes=0", "train.py"]
    cases["zero_nodes"] = zero_nodes

    bad_gpu_count = valid_manifest()
    bad_gpu_count.resources.gpus_per_node = 3
    bad_gpu_count.command = ["torchrun", "--nproc_per_node=3", "--nnodes=6", "train.py"]
    cases["non_standard_gpus_per_node"] = bad_gpu_count

    bad_backend = valid_manifest()
    bad_backend.rendezvous.backend = "mpi"
    cases["unknown_rendezvous_backend"] = bad_backend

    no_ckpt_dir = valid_manifest()
    no_ckpt_dir.restart_policy.checkpoint_dir = ""
    cases["missing_checkpoint_dir"] = no_ckpt_dir

    command_mismatch = valid_manifest()
    command_mismatch.command = ["torchrun", "--nproc_per_node=4", "--nnodes=6", "train.py"]
    cases["command_resource_mismatch"] = command_mismatch

    cases["_baseline_for_comparison"] = base
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="训练作业清单与生命周期状态机")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    manifest = valid_manifest()
    simulator = JobLifecycleSimulator(manifest)
    history = simulator.simulate_lifecycle_with_failure(inject_failure_step=1500)
    rank_env = simulator.derive_rank_env()

    illegal = {}
    for name, case in illegal_manifests().items():
        if name.startswith("_"):
            continue
        sim = JobLifecycleSimulator(case)
        illegal[name] = {"errors": sim.validate_manifest(),
                         "history": sim.simulate_lifecycle_with_failure()}

    result = {
        "manifest": asdict(manifest),
        "world_size": len(rank_env),
        "rank_env_head": rank_env[:2],
        "rank_env_tail": rank_env[-2:],
        "state_machine_history": history,
        "illegal_configs": illegal,
        "source_kind": "declared job spec + local state-machine simulation",
    }
    (args.outdir / "job_manifest_simulation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    cases = {
        "world_size": len(rank_env),
        "transitions": len(history),
        "illegal_cases": {name: value["errors"] for name, value in illegal.items()},
        "scope": "declarative manifest and state machine only; no cluster submission",
    }
    (args.outdir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")

    print(f"作业状态机 {len(history)} 次跃迁；world_size={len(rank_env)}")
    for record in history:
        print(f"  [{record['from_state']}] -> [{record['to_state']}] | {record['reason']}")
    print(f"rank0 环境: {rank_env[0]['env']}")
    print(f"rank{len(rank_env)-1} 环境: {rank_env[-1]['env']}")
    for name, value in illegal.items():
        print(f"  非法配置 {name}: {value['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())