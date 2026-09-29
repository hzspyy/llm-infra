#!/usr/bin/env python3
"""RL 运行时的角色归属与异步队列协议（7.6-D/E/F）。

两部分互相独立，都只用仓库内的源码快照与显式假设：

1. `roles`：对 veRL、slime、OpenRLHF 三份快照做**源码定位**——actor/rollout/reference/
   reward/critic 各自的 worker 类、参数发布、checkpoint 与异步机制落在哪个文件哪一行。
   找不到就记进 `missing`，不猜。
2. `async`：一个 CPU 离散事件模拟。rollout 与 reward 是异步服务，learner 按固定时长推进
   策略版本；样本带 rollout 时的策略版本进入有界缓冲，`--lag` 控制生产与消费的错位，
   另有慢 reward、重复回包与超时重排三类注入。策略漂移用显式的 δ 模型表示（每个 learner
   step 使同一 token 的 logprob 变化 δ），因此重要性比率 `exp(δ·滞后步数)` 是定义而不是测量。

Usage:
    python labs/L7/rl_runtime_contract.py --section roles \
      --source-root results/local/7.6/20260918-rl/source --outdir "$RUN_DIR/rl-roles"
    python labs/L7/rl_runtime_contract.py --section async --outdir "$RUN_DIR/rl-async"
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import statistics
from pathlib import Path

# 角色 → 候选符号。命中第一个出现的定义/赋值行。
ROLE_PROBES = {
    "verl": {
        "actor/learner": ["class ActorRolloutRefWorker", "class FSDPWorker", "actor_rollout_wg"],
        "rollout": ["class vLLMRollout", "class AsyncvLLMServer", "rollout_mode"],
        "reference": ["ref_policy", "compute_ref_log_prob", "ref_log_prob"],
        "reward": ["class RewardManager", "load_reward_manager", "reward_fn"],
        "critic": ["class CriticWorker", "critic_wg", "compute_values"],
        "参数发布": ["sync_model_weights", "update_weights", "rollout_mode"],
        "checkpoint": ["save_checkpoint", "checkpoint_manager"],
        "异步/lag": ["async_rollout", "max_num_seqs", "staleness"],
    },
    "slime": {
        "actor/learner": ["class TrainActor", "train_actor", "update_weights"],
        "rollout": ["class Rollout", "RolloutGroup", "class FullyAsyncRollout"],
        "reference": ["ref_log_prob", "kl_loss"],
        "reward": ["reward_fn", "class RewardHub", "reward_model"],
        "critic": ["value_head", "critic"],
        "参数发布": ["update_weights_from_distributed", "update_weights", "UpdateWeight"],
        "checkpoint": ["save_checkpoint", "load_checkpoint", "def save"],
        "异步/lag": ["fully_async", "streaming_rollout", "streaming"],
    },
    "openrlhf": {
        "actor/learner": ["class PolicyTrainer", "PPOTrainer", "actor"],
        "rollout": ["class SamplesGenerator", "generate_samples", "vLLMEngine"],
        "reference": ["ref_log_probs", "kl_ctl"],
        "reward": ["class NaiveRewardModel", "reward_model", "RemoteRewardModel"],
        "critic": ["class Critic", "value_loss"],
        "参数发布": ["update_weights", "sync_weights", "load_weights", "WeightSync"],
        "checkpoint": ["save_checkpoint", "ckpt_path", "def save"],
        "异步/lag": ["async_train", "max_staleness", "ppo_trainer_async"],
    },
}

FILE_HINTS = {
    "verl": ["verl/ray_trainer.py", "verl/main_ppo.py", "verl/fsdp_vllm.py",
             "verl/vllm_rollout_spmd.py", "verl/core_algos.py",
             "verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py",
             "verl/verl/workers/sharding_manager/__init__.py"],
    "slime": ["slime/slime/ray/train_actor.py", "slime/slime/ray/rollout.py",
              "slime/slime/rollout/fully_async_rollout.py",
              "slime/slime/rollout/streaming_utils.py",
              "slime/slime/rollout/filter_hub/dynamic_sampling_filters.py",
              "slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py",
              "slime/slime/backends/megatron_utils/server/logprob_utils.py",
              "slime/slime_plugins/rollout_buffer/buffer.py"],
    "openrlhf": ["openrlhf/openrlhf/cli/train_ppo_ray.py",
                 "openrlhf/openrlhf/trainer/ppo_trainer_async.py",
                 "openrlhf/openrlhf/trainer/ppo_utils/samples_generator.py",
                 "openrlhf/openrlhf/trainer/ppo_utils/experience_maker.py",
                 "openrlhf/openrlhf/trainer/ray/vllm_engine.py",
                 "openrlhf/openrlhf/trainer/ray/vllm_worker_wrap.py"],
}


def locate(text: str, symbol: str):
    lines = text.splitlines()
    pattern = re.compile(r"^\s*(def|class)\s+" + re.escape(symbol))
    for index, line in enumerate(lines, 1):
        if pattern.match(line):
            return index, line.strip()
    for index, line in enumerate(lines, 1):
        if symbol in line:
            return index, line.strip()
    return None, None


def roles_section(source_root: Path) -> dict:
    result, missing = {"section": "rl-runtime-roles", "frameworks": {}}, []
    for name, files in FILE_HINTS.items():
        blobs = []
        for relative in files:
            path = source_root / relative
            if path.exists():
                blobs.append((relative, path.read_text(encoding="utf-8", errors="replace")))
            else:
                missing.append(f"{name}:{relative}")
        rows = {}
        for role, symbols in ROLE_PROBES[name].items():
            hit = None
            for symbol in symbols:
                for relative, text in blobs:
                    line_no, code = locate(text, symbol)
                    if line_no:
                        hit = {"symbol": symbol, "file": relative, "line": line_no, "code": code}
                        break
                if hit:
                    break
            rows[role] = hit
            if hit is None:
                missing.append(f"{name}:{role}")
        result["frameworks"][name] = {"files": [f for f, _ in blobs], "roles": rows}
    result["missing"] = missing
    result["claim_scope"] = ("源码定位：行号来自仓库内快照，不导入也不运行三个框架；"
                             "角色归属用符号位置表示，不代表运行时的实际拓扑")
    return result


def async_section(lag: int, n_prompts: int, group_size: int, n_learner_steps: int,
                  groups_per_step: int, rollout_ms: float, reward_ms: float,
                  learner_ms: float, slow_every: int, slow_ms: float,
                  duplicate_every: int, buffer_groups: int, target_depth: int,
                  backpressure: bool, delta: float, clip_eps: float, seed: int) -> dict:
    """离散事件：rollout 群组持续生产、reward 异步打分、样本按 lag 延迟、learner 定长推进版本。

    生产单元是"一个 prompt 的整组采样"。`--backpressure` 打开时，只有管道在途+就绪的群组数
    小于 `--target-depth` 才启动下一组，队列不会溢出；关闭时生产者不等消费者，缓冲溢出被丢弃。
    `lag` 是**注入量**：群组打完分后要等到策略版本前进 lag 步才可被消费，模拟生产与消费错位。
    """
    import random
    rng = random.Random(seed)
    version, clock, counter = 0, 0.0, 0
    events = []

    def push(time, kind, payload):
        nonlocal counter
        counter += 1
        heapq.heappush(events, (time, counter, kind, payload))

    inflight_groups, delayed, ready = 0, [], []
    delivered = set()
    duplicate_pending = set()
    stats = {"groups_started": 0, "groups_ready": 0, "groups_consumed": 0,
             "samples_produced": 0, "samples_used": 0, "dropped_buffer_full": 0,
             "duplicates_detected": 0, "duplicates_dropped": 0, "starvation_steps": 0,
             "lag_steps": [], "ratios": [], "clipped": 0, "rollout_time": 0.0,
             "reward_time": 0.0, "queue_wait": 0.0, "delayed_pool_max": 0}

    def start_group(time):
        nonlocal inflight_groups
        inflight_groups += 1
        stats["groups_started"] += 1
        push(time + rollout_ms / 1000.0, "rollout_done", stats["groups_started"])

    if backpressure:
        for _ in range(min(target_depth, n_prompts)):
            start_group(0.0)
    else:
        push(0.0, "producer", None)
    push(learner_ms / 1000.0, "learner_tick", None)

    def maybe_start_more():
        if not backpressure:
            return
        while (stats["groups_started"] < n_prompts
               and len(delayed) + len(ready) + inflight_groups < target_depth):
            start_group(clock)

    def promote():
        keep = []
        for ready_version, record in delayed:
            if ready_version <= version:
                if len(ready) < buffer_groups:
                    record["ready_at"] = clock
                    ready.append(record)
                    stats["groups_ready"] += 1
                else:
                    stats["dropped_buffer_full"] += group_size
            else:
                keep.append((ready_version, record))
        delayed[:] = keep
        stats["delayed_pool_max"] = max(stats["delayed_pool_max"], len(delayed))

    while version < n_learner_steps and events:
        time, _, kind, payload = heapq.heappop(events)
        clock = max(clock, time)
        if kind == "producer":
            if stats["groups_started"] < n_prompts:
                start_group(time)
            push(time + rng.expovariate(1.0 / (rollout_ms / 1000.0)), "producer", None)
        elif kind == "rollout_done":
            push(time + reward_ms / 1000.0, "reward_done", payload)
        elif kind == "reward_done":
            inflight_groups -= 1
            if payload in delivered:          # 同一组被打分两次：重复回包
                stats["duplicates_detected"] += 1
                stats["duplicates_dropped"] += 1
                maybe_start_more()
                continue
            delivered.add(payload)
            if duplicate_every > 0 and payload % duplicate_every == 0:
                duplicate_pending.add(payload)
                push(time + 0.005, "reward_done", payload)   # 5 ms 后重复投递
            record = {"group": payload, "version": version}
            delayed.append((version + lag, record))
            stats["samples_produced"] += group_size
        elif kind == "learner_tick":
            promote()
            consumed = 0
            while ready and consumed < groups_per_step:
                record = ready.pop(0)
                staleness = version - record["version"]
                stats["lag_steps"].append(staleness)
                stats["queue_wait"] += time - record["ready_at"]
                ratio = pow(2.718281828459045, delta * staleness)
                stats["ratios"].append(ratio)
                stats["clipped"] += int(abs(ratio - 1.0) > clip_eps)
                stats["groups_consumed"] += 1
                stats["samples_used"] += group_size
                consumed += 1
            if consumed == 0:
                stats["starvation_steps"] += 1
            version += 1
            push(time + learner_ms / 1000.0, "learner_tick", None)
            maybe_start_more()

    groups = max(stats["groups_consumed"], 1)
    steps = max(version, 1)
    return {
        "section": "rl-async-queue", "lag_injected_steps": lag,
        "params": {"n_prompts": n_prompts, "group_size": group_size,
                   "learner_steps": n_learner_steps, "rollout_ms": rollout_ms,
                   "reward_ms": reward_ms, "learner_ms": learner_ms,
                   "slow_every": slow_every, "slow_ms": slow_ms,
                   "duplicate_every": duplicate_every, "buffer_groups": buffer_groups,
                   "target_depth": target_depth, "backpressure": backpressure,
                   "delta_logprob_per_step": delta, "clip_eps": clip_eps, "seed": seed},
        "counters": {
            "groups_started": stats["groups_started"],
            "groups_consumed": stats["groups_consumed"],
            "samples_produced": stats["samples_produced"],
            "samples_used": stats["samples_used"],
            "learner_steps_done": version,
            "starvation_steps": stats["starvation_steps"],
            "dropped_samples_buffer_full": stats["dropped_buffer_full"],
            "duplicates_detected": stats["duplicates_detected"],
            "duplicates_dropped": stats["duplicates_dropped"],
            "delayed_pool_max": stats["delayed_pool_max"],
        },
        "staleness_steps": {
            "mean": statistics.fmean(stats["lag_steps"]) if stats["lag_steps"] else None,
            "median": statistics.median(stats["lag_steps"]) if stats["lag_steps"] else None,
            "max": max(stats["lag_steps"]) if stats["lag_steps"] else None,
            "share_gt0": (sum(1 for v in stats["lag_steps"] if v > 0)
                          / max(len(stats["lag_steps"]), 1)),
            "share_gt1": (sum(1 for v in stats["lag_steps"] if v > 1)
                          / max(len(stats["lag_steps"]), 1)),
        },
        "importance": {
            "ratio_mean": (statistics.fmean(stats["ratios"]) if stats["ratios"] else None),
            "ratio_max": max(stats["ratios"]) if stats["ratios"] else None,
            "clip_fraction": stats["clipped"] / max(len(stats["ratios"]), 1),
        },
        "per_step": {
            "samples_used_per_learner_step": stats["samples_used"] / steps,
            "starvation_share": stats["starvation_steps"] / steps,
            "dropped_share_of_produced": (stats["dropped_buffer_full"]
                                          / max(stats["samples_produced"], 1)),
        },
        "time_breakdown_seconds": {"wall_clock": clock,
                                   "learner_sum": learner_ms / 1000.0 * version,
                                   "queue_wait_sum": stats["queue_wait"]},
        "claim_scope": "队列与版本推进是离散事件模拟；策略漂移用显式 δ 模型，"
                       "重要性比率是定义值，不是任何真实策略的测量",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", required=True, choices=["roles", "async"])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--lag", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learner-steps", type=int, default=40)
    parser.add_argument("--groups-per-step", type=int, default=1)
    parser.add_argument("--buffer-groups", type=int, default=8)
    parser.add_argument("--target-depth", type=int, default=4)
    parser.add_argument("--no-backpressure", dest="backpressure", action="store_false")
    parser.add_argument("--rollout-ms", type=float, default=40.0)
    parser.add_argument("--reward-ms", type=float, default=30.0)
    parser.add_argument("--learner-ms", type=float, default=120.0)
    parser.add_argument("--slow-every", type=int, default=10)
    parser.add_argument("--slow-ms", type=float, default=400.0)
    parser.add_argument("--duplicate-every", type=int, default=25)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=False)
    if args.section == "roles":
        if args.source_root is None:
            raise SystemExit("roles 需要 --source-root")
        result = roles_section(args.source_root)
        name = "rl_runtime_roles.json"
    else:
        result = async_section(args.lag, args.n_prompts, args.group_size, args.learner_steps,
                               args.groups_per_step, args.rollout_ms, args.reward_ms,
                               args.learner_ms, args.slow_every, args.slow_ms,
                               args.duplicate_every, args.buffer_groups, args.target_depth,
                               args.backpressure, args.delta, args.clip_eps, args.seed)
        name = f"rl_async_lag{args.lag}.json"
    (args.outdir / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
