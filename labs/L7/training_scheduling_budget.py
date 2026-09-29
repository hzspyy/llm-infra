#!/usr/bin/env python3
"""训练作业的准入策略对照与公开流程预算账（7.11-B/G）。

两部分都用显式假设，输出原始事件表与派生指标：

1. `admission`：在固定的卡数与作业集合上比较 FIFO 顺序准入、整组准入 + 保守回填、
   带优先级抢占（按 checkpoint 间隔计重放损失）的 makespan、排队、空闲卡时与重放卡时。
2. `budget`：读取 SmolLM3 的三个 Nanotron YAML，按 YAML 字段与声明的启动规模推导
   每阶段 token 预算、6P FLOPs、GPU 小时、checkpoint 次数与字节、故障重放代价。
3. `public-flows`：只读仓库内的配置快照（Cosmos Edge SFT 的 TOML、SpecForge 的
   Eagle3/DFlash 配方与草稿 config、CosyVoice2 GRPO 的 run.sh），复算 token 预算、
   可训练参数组、checkpoint 次数、奖励服务占比与特征复用节省，并列出公开程度。

Usage:
    python labs/L7/training_scheduling_budget.py --section admission --outdir "$RUN_DIR/sched"
    python labs/L7/training_scheduling_budget.py --section budget \
        --yaml-dir results/local/7.11/<run>/source/smollm3 --outdir "$RUN_DIR/budget"
    python labs/L7/training_scheduling_budget.py --section public-flows \
        --source-root results/local/7.11/<run>/source --outdir "$RUN_DIR/public-flows"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# 准入策略对照
# ---------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    cards: int
    duration_s: float
    priority: int
    arrival_s: float
    ckpt_interval_s: float


@dataclass
class Placement:
    job_id: str
    attempt: int
    start_s: float
    end_s: float
    preempted_at: Optional[float] = None
    lost_work_s: float = 0.0


@dataclass
class SimResult:
    policy: str
    makespan_s: float
    gpu_seconds_busy: float
    gpu_seconds_useful: float
    gpu_seconds_rework: float
    gpu_seconds_idle: float
    utilization: float
    useful_fraction: float
    queue_time_s: dict = field(default_factory=dict)
    placements: list = field(default_factory=list)
    notes: str = ""


def default_jobs() -> list[Job]:
    """显式作业集合：8 卡机器上四个作业，含一次后到的高优先级到达。

    J1 占用 6 卡且时间最长；J2 需要 4 卡，在 J1 运行期间放不下；J3 只需要 2 卡，
    可以回填进 J1 留下的空闲卡；J4 在 t=200 s 以更高优先级到达，用来观察抢占代价。
    """
    return [
        Job("J1-long-6cards", cards=6, duration_s=600.0, priority=1, arrival_s=0.0, ckpt_interval_s=300.0),
        Job("J2-low-4cards", cards=4, duration_s=300.0, priority=0, arrival_s=0.0, ckpt_interval_s=100.0),
        Job("J3-small-2cards", cards=2, duration_s=150.0, priority=0, arrival_s=10.0, ckpt_interval_s=50.0),
        Job("J4-high-4cards", cards=4, duration_s=300.0, priority=2, arrival_s=200.0, ckpt_interval_s=100.0),
    ]


def simulate(cards: int, jobs: list[Job], policy: str, horizon_s: float = 6000.0) -> SimResult:
    """离散事件模拟；1 秒步长，只在到达/完成/抢占处改变状态。"""
    spec = {job.job_id: job for job in jobs}
    remaining = {job.job_id: job.duration_s for job in jobs}
    pending: list[str] = []
    running: list[dict] = []
    placements: list[Placement] = []
    attempts = {job.job_id: 0 for job in jobs}
    first_start: dict[str, float] = {}
    busy_card_seconds = 0.0
    rework_card_seconds = 0.0
    step = 1.0

    def free(t: float) -> int:
        return cards - sum(r["cards"] for r in running if r["start_s"] <= t < r["end_s"])

    def attempt_start(jid: str, t: float) -> None:
        attempts[jid] += 1
        first_start.setdefault(jid, t)
        placements.append(Placement(jid, attempts[jid], t, t + remaining[jid]))
        running.append({"job_id": jid, "cards": spec[jid].cards, "start_s": t,
                        "end_s": t + remaining[jid], "attempt": attempts[jid]})

    def head_wait_start(t: float, jid: str) -> Optional[float]:
        """估算更早到达的等待作业最早可能的启动时刻（按当前运行作业的结束时间）。"""
        have = free(t)
        if have >= spec[jid].cards:
            return t
        cursor = t
        for end_s, cards_freed in sorted((r["end_s"], r["cards"]) for r in running):
            have += cards_freed
            cursor = max(cursor, end_s)
            if have >= spec[jid].cards:
                return cursor
        return None

    t = 0.0
    while t < horizon_s:
        for job in jobs:
            if abs(job.arrival_s - t) < step / 2 and job.job_id not in pending and \
                    attempts[job.job_id] == 0 and not any(r["job_id"] == job.job_id for r in running):
                pending.append(job.job_id)

        for r in list(running):
            if r["end_s"] <= t:
                running.remove(r)
                remaining[r["job_id"]] -= r["end_s"] - r["start_s"]
                next(p for p in placements
                     if p.job_id == r["job_id"] and p.attempt == r["attempt"]).end_s = t

        if policy == "preemptive":
            for jid in sorted(pending, key=lambda j: (-spec[j].priority, spec[j].arrival_s)):
                need = spec[jid].cards - free(t)
                if need <= 0:
                    continue
                victims, freed = [], 0
                for victim in sorted(running, key=lambda r: (spec[r["job_id"]].priority, r["start_s"])):
                    if spec[victim["job_id"]].priority >= spec[jid].priority:
                        break
                    victims.append(victim)
                    freed += victim["cards"]
                    if freed >= need:
                        break
                if freed < need:
                    # 整组准入：凑不齐就不抢占，避免把低优先级作业反复抢占又放回
                    continue
                for victim in victims:
                    elapsed = t - victim["start_s"]
                    saved = (elapsed // spec[victim["job_id"]].ckpt_interval_s) * spec[victim["job_id"]].ckpt_interval_s
                    lost = elapsed - saved
                    placement = next(p for p in placements
                                     if p.job_id == victim["job_id"] and p.attempt == victim["attempt"])
                    placement.end_s = t
                    placement.preempted_at = t
                    placement.lost_work_s = lost
                    remaining[victim["job_id"]] -= saved
                    rework_card_seconds += lost * spec[victim["job_id"]].cards
                    running.remove(victim)
                    if victim["job_id"] not in pending:
                        pending.append(victim["job_id"])

        placed = True
        while placed:
            placed = False
            order = sorted(pending, key=lambda j: (-spec[j].priority, spec[j].arrival_s)) \
                if policy == "preemptive" else sorted(pending, key=lambda j: spec[j].arrival_s)
            for jid in order:
                if free(t) < spec[jid].cards:
                    continue
                earlier = [j for j in pending if spec[j].arrival_s < spec[jid].arrival_s]
                if policy == "fifo" and earlier:
                    continue                      # 顺序准入：不允许绕过更早的等待作业
                if policy == "gang_backfill" and earlier:
                    head_start = head_wait_start(t, min(earlier, key=lambda j: spec[j].arrival_s))
                    if head_start is not None and t + remaining[jid] > head_start:
                        continue                  # 会推迟更早作业，不做回填
                attempt_start(jid, t)
                pending.remove(jid)
                placed = True
                break

        busy_card_seconds += (cards - free(t)) * step
        if not running and not pending and all(attempts[j] > 0 for j in remaining):
            break
        t += step

    makespan = max((p.end_s for p in placements), default=0.0)
    useful = sum(job.cards * job.duration_s for job in jobs)
    idle = cards * makespan - busy_card_seconds
    queue_time = {job.job_id: (round(first_start[job.job_id] - job.arrival_s, 2)
                               if job.job_id in first_start else None) for job in jobs}
    notes = ("卡数、作业大小/时长/到达时间/优先级/checkpoint 间隔均为声明的合成假设；"
             "重放损失按 checkpoint 间隔上限估算，不含重建、重排队与冷启动开销")
    return SimResult(policy=policy, makespan_s=float(makespan),
                     gpu_seconds_busy=float(busy_card_seconds),
                     gpu_seconds_useful=float(useful),
                     gpu_seconds_rework=float(rework_card_seconds),
                     gpu_seconds_idle=float(idle),
                     utilization=float(busy_card_seconds / (cards * makespan)) if makespan else 0.0,
                     useful_fraction=float(useful / (cards * makespan)) if makespan else 0.0,
                     queue_time_s=queue_time,
                     placements=[asdict(p) for p in placements], notes=notes)


def admission_section() -> dict:
    cards = 8
    jobs = default_jobs()
    results = {policy: asdict(simulate(cards, [Job(**asdict(j)) for j in jobs], policy))
               for policy in ("fifo", "gang_backfill", "preemptive")}
    return {
        "section": "admission", "cards": cards,
        "job_set": [asdict(j) for j in jobs],
        "assumptions": [
            "同步训练要求整组同时就绪，不接受部分 rank 先启动",
            "抢占后作业从最后一个 checkpoint 重启，损失按 elapsed 减去已保存的检查点进度计",
            "FIFO 不允许后到作业绕过等待中的更早作业",
            "保守回填只在能放下且不推迟更早等待作业的预计启动时刻时提前启动",
        ],
        "results": results,
    }


# ---------------------------------------------------------------------------
# 公开流程预算账（SmolLM3 Nanotron 配方）
# ---------------------------------------------------------------------------
def budget_section(yaml_dir: Path, params: float, nodes: int, gpus_per_node: int,
                   tp: int, peak_tflops: float, mfu: float, ckpt_bytes_per_param: float) -> dict:
    """按 YAML 字段推导预算；三份 YAML 描述同一次训练的不同阶段，不能相加。"""
    def wall_and_gpu_hours(token_count: float) -> tuple:
        flops = 6.0 * params * token_count
        num_gpus = nodes * gpus_per_node
        seconds = flops / (peak_tflops * 1e12 * mfu * num_gpus)
        return seconds / 3600.0, seconds / 3600.0 * num_gpus

    rows = []
    dp = nodes * gpus_per_node // tp
    for path in sorted(yaml_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        tok = cfg["tokens"]
        steps = int(tok["train_steps"])
        mbs = int(tok["micro_batch_size"])
        acc = int(tok["batch_accumulation_per_replica"])
        seq = int(tok["sequence_length"])
        gbs = mbs * acc * dp
        full_tokens = steps * gbs * seq
        ckpt_interval = int(cfg["checkpoints"]["checkpoint_interval"])

        starts = [int(stage["start_training_step"]) for stage in cfg["data_stages"]]
        names = [stage.get("name") for stage in cfg["data_stages"]]
        segments = []
        for index, start in enumerate(starts):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else steps
            seg_steps = end - start + 1
            seg_tokens = seg_steps * gbs * seq
            seg_wall, seg_gpu = wall_and_gpu_hours(seg_tokens)
            segments.append({
                "stage": names[index], "start_step": start, "end_step": end,
                "steps": seg_steps, "tokens": seg_tokens,
                "flops_6P": 6.0 * params * seg_tokens,
                "wall_clock_hours_at_declared_mfu": seg_wall,
                "gpu_hours_at_declared_mfu": seg_gpu,
            })

        ckpt_count = steps // ckpt_interval
        ckpt_bytes = params * ckpt_bytes_per_param
        lost_tokens = (ckpt_interval / 2.0) * gbs * seq
        full_wall, full_gpu = wall_and_gpu_hours(full_tokens)
        lost_wall, lost_gpu = wall_and_gpu_hours(lost_tokens)
        rows.append({
            "yaml": path.name, "train_steps": steps, "micro_batch": mbs, "accumulation": acc,
            "dp_from_launch": dp, "global_batch_samples": gbs, "sequence_length": seq,
            "full_run_tokens": full_tokens, "full_run_flops_6P": 6.0 * params * full_tokens,
            "full_run_wall_clock_hours_at_declared_mfu": full_wall,
            "full_run_gpu_hours_at_declared_mfu": full_gpu,
            "stage_segments": segments,
            "checkpoint_interval_steps": ckpt_interval, "checkpoint_count": ckpt_count,
            "checkpoint_bytes_each": ckpt_bytes, "checkpoint_total_bytes": ckpt_count * ckpt_bytes,
            "mean_lost_tokens_per_failure": lost_tokens,
            "mean_lost_wall_clock_hours_per_failure": lost_wall,
            "mean_lost_gpu_hours_per_failure": lost_gpu,
        })
    return {
        "section": "budget",
        "source": "SmolLM3 Nanotron YAML（text/pretraining/smollm3/*.yaml，固定 commit 快照）",
        "assumptions": {
            "params": params, "nodes": nodes, "gpus_per_node": gpus_per_node, "tp": tp,
            "peak_tflops_per_gpu": peak_tflops, "mfu": mfu,
            "checkpoint_bytes_per_param": ckpt_bytes_per_param,
            "note": "YAML 未写 DP/TP；DP 由 run 名的 48 节点 × 8 卡与 TP=2 推出；"
                    "峰值与 MFU 是声明的设备假设，不是本机实测利用率；"
                    "checkpoint 字节按 bf16 参数 + fp32 master + Adam m/v 估计；"
                    "三份 YAML 的 train_steps 都是 4,720,000，描述同一次训练的不同阶段配置，"
                    "阶段区间互相重叠，不能把各行相加。",
        },
        "stage_split_source": "stage3_9T_11T.yaml（三份配置里唯一同时给出 stage2 与 decay 起点）",
        "run_steps": sorted({row["train_steps"] for row in rows}),
        "run_full_tokens": sorted({row["full_run_tokens"] for row in rows}),
        "rows": rows,
    }


def public_flows_section(source_root: Path, gpu_hour_price: float) -> dict:
    """三条公开流程（Cosmos Edge SFT、SpecForge 草稿、CosyVoice GRPO）的字段与派生预算。

    只读仓库内的配置快照。能算的只有"配置里写死的量"（token 预算、checkpoint 次数、
    reward 服务设备数、草稿的词表与层数）；作者机器的实际墙钟与日志多数没有公开，
    因此不输出任何 GPU 小时估计，只列出可复算的派生量与公开程度。
    """
    import tomllib

    cosmos_configs = {}
    for name in ("vision_sft_edge", "videophy2_sft_edge"):
        path = source_root / "cosmos" / f"{name}.toml"
        cfg = tomllib.loads(path.read_text())
        rows = int(cfg["trainer"]["max_iter"])
        accum = int(cfg["trainer"]["grad_accum_iter"])
        loader = cfg.get("dataloader_train", {})
        # 生成分支（vfm）按打包 token 数给预算，Reasoner 分支（vlm）按每批样本数 × 序列长度
        if "max_num_tokens_after_packing" in cfg["model"]:
            tokens = int(cfg["model"]["max_num_tokens_after_packing"])
            batch_budget = f"{tokens} packed tokens/micro-batch"
        else:
            tokens = int(loader.get("max_samples_per_batch", 1)) * int(loader["max_sequence_length"])
            batch_budget = (f"{loader.get('max_samples_per_batch')} 样本 × "
                            f"{loader['max_sequence_length']} 序列长度/micro-batch")
        save_iter = int(cfg["checkpoint"]["save_iter"])
        cosmos_configs[name] = {
            "config": path.name, "task": cfg["job"]["task"],
            "max_iter": rows, "grad_accum_iter": accum, "batch_budget": batch_budget,
            "tokens_per_micro_batch_upper_bound": tokens,
            "tokens_per_iter_upper_bound": accum * tokens,
            "tokens_full_run_per_rank_upper_bound": rows * accum * tokens,
            "optimizer_trainable_key_groups": cfg["optimizer"].get(
                "keys_to_select", "未指定（optimizer 覆盖全部参数）"),
            "lr": cfg["optimizer"]["lr"], "weight_decay": cfg["optimizer"]["weight_decay"],
            "precision": cfg["model"]["precision"],
            "activation_checkpointing": cfg["model"]["activation_checkpointing"]["mode"],
            "compile": bool(cfg.get("model", {}).get("compile", {}).get("enabled", False)),
            "ema": bool(cfg.get("model", {}).get("ema", {}).get("enabled", False)),
            "checkpoint_count": -(-rows // save_iter) if save_iter else None,
            "public_evidence": ["SFT 配置与权重公开", "作者迭代墙钟与训练日志未公开"],
        }

    specforge = {}
    for name in ("qwen3-8b-eagle3-disaggregated", "qwen3-8b-dflash-disaggregated"):
        path = source_root / "specforge" / f"{name}.yaml"
        cfg = yaml.safe_load(path.read_text())
        train = cfg["training"]
        data = cfg["data"]
        draft = yaml.safe_load((source_root / "specforge" /
                                Path(cfg["model"]["draft_model_config"]).name).read_text())
        steps = int(train["max_steps"])
        seq = int(data["max_length"])
        batch = int(train["batch_size"])
        epochs = int(train["num_epochs"])
        deployment = cfg["deployment"]
        specforge[name] = {
            "config": path.name, "strategy": train["strategy"],
            "target_model": cfg["model"]["target_model_path"],
            "target_vocab": draft["vocab_size"],
            "draft_vocab": draft.get("draft_vocab_size", "全词表（未裁剪）"),
            "draft_layers": draft["num_hidden_layers"], "draft_hidden": draft["hidden_size"],
            "draft_block_size": draft.get("block_size"),
            "draft_target_layers": draft.get("num_target_layers"),
            "draft_architecture": draft["architectures"][0],
            "max_steps": steps, "num_epochs": epochs, "batch_size": batch,
            "max_length": seq, "save_interval": train["save_interval"],
            "target_tokens_full_run": steps * batch * seq,
            "implied_samples_per_epoch": steps * batch / max(epochs, 1),
            "target_forward_per_step": "每个 step 一次 target 前向（batch=1，长度 %d）" % seq,
            "offline_feature_saving": {
                "online_target_forwards": steps,
                "offline_target_forwards": steps // max(epochs, 1),
                "saved_fraction": 1.0 - 1.0 / max(epochs, 1),
            },
            "deployment_mode": deployment["mode"],
            "trainer_ranks": deployment["trainer"],
            "checkpoint_count": steps // int(train["save_interval"])
            if train.get("save_interval") else None,
            "public_evidence": ["训练配置与 Qwen3-8B 草稿权重公开",
                                "作者各阶段墙钟与完整日志未公开；特征缓存复用次数需自测"],
        }

    run_sh = (source_root / "cosyvoice" / "run.sh").read_text()
    def grab(pattern: str) -> str:
        match = re.search(pattern, run_sh)
        if match is None:
            raise SystemExit(f"run.sh 中未找到 {pattern}")
        return match.group(1)

    train_samples = int(grab(r"head -n (\d+) data/aishell-3.jsonl"))
    cosyvoice = {
        "config": "run.sh", "train_samples": train_samples,
        "gpus": int(grab(r"n_gpus_per_node=(\d+)")),
        "micro_batch_size": int(grab(r"micro_batch_size=(\d+)")),
        "train_batch_size": int(grab(r"train_batch_size=(\d+)")),
        "rollout_n": int(grab(r"rollout\.n=(\d+)")),
        "max_prompt_length": int(grab(r"data\.max_prompt_length=(\d+)")),
        "max_response_length": int(grab(r"data\.max_response_length=(\d+)")),
        "total_epochs": int(grab(r"trainer\.total_epochs=(\d+)")),
        "save_freq": int(grab(r"trainer\.save_freq=(\d+)")),
        "reward_server_devices": int(grab(r"number-of-devices (\d+)")),
        "reward_server": "Triton 上的 token2wav + SenseVoice ASR 串联（reward_tts.py 远程打分）",
        "reward_calls_per_prompt": int(grab(r"rollout\.n=(\d+)")),
        "generations_total": train_samples * int(grab(r"rollout\.n=(\d+)")) * int(grab(r"trainer\.total_epochs=(\d+)")),
        "reward_device_share_of_trainer": int(grab(r"number-of-devices (\d+)")) / int(grab(r"n_gpus_per_node=(\d+)")),
        "public_evidence": ["RL 配方、奖励实现与基础权重公开",
                            "作者奖励服务时延、训练墙钟与曲线未公开；奖励服务本身由第三方 ASR/codec 组成"],
    }

    decisions = [
        {"decision": "SpecForge 用离线特征替代在线捕获",
         "reason": "在线模式每个 step 触发一次 target 前向；同一份数据要训 num_epochs=%d 轮，"
                   "离线预生成一次即可复用，target 前向从 %d 次降到 %d 次（省 %.0f%%）"
                   % (specforge["qwen3-8b-eagle3-disaggregated"]["num_epochs"],
                      specforge["qwen3-8b-eagle3-disaggregated"]["max_steps"],
                      specforge["qwen3-8b-eagle3-disaggregated"]["offline_feature_saving"]["offline_target_forwards"],
                      100 * specforge["qwen3-8b-eagle3-disaggregated"]["offline_feature_saving"]["saved_fraction"]),
         "cost": "需要落盘特征与校本身份（模型 revision/模板/层）；缓存失效就要重跑 target"},
        {"decision": "CosyVoice 的奖励服务从 8 设备缩到与训练共置或按需扩缩",
         "reason": "配方里奖励服务单独占 %d 个设备，等于训练集群的 %.0f%%；奖励调用量是 "
                   "%d 条样本 × n=%d × %d epoch = %d 次"
                   % (cosyvoice["reward_server_devices"],
                      100 * cosyvoice["reward_device_share_of_trainer"],
                      cosyvoice["train_samples"], cosyvoice["rollout_n"],
                      cosyvoice["total_epochs"], cosyvoice["generations_total"]),
         "cost": "共置会挤占 rollout/learner 的显存与算力；缩容会拉长奖励等待，进而放大策略陈旧度"},
        {"decision": "Cosmos 生成分支保持 save_iter=100（整轮 %d 次）、Reasoner 分支接受整轮只存最后"
                     % cosmos_configs["vision_sft_edge"]["checkpoint_count"],
         "reason": "生成分支 max_iter=%d、EMA 打开、全量激活重算，checkpoint 里除参数还有 optimizer 与 EMA；"
                   "Reasoner 分支 max_iter=%d 小于它的 save_iter=%d，中途不落盘"
                   % (cosmos_configs["vision_sft_edge"]["max_iter"],
                      cosmos_configs["videophy2_sft_edge"]["max_iter"],
                      int((source_root / "cosmos" / "videophy2_sft_edge.toml").read_text()
                          .split("save_iter")[1].split("=")[1].split("\n")[0])),
         "cost": "生成分支故障最多丢 100 步；Reasoner 分支一旦中断要整轮重跑——只有作者给出单迭代时间才能"
                 "把这条与更密的保存策略比出代价"},
    ]
    return {
        "section": "public-flows",
        "source_root": str(source_root),
        "claim_scope": "只复算配置里写死的量；作者机器、墙钟与日志多数未公开，不输出 GPU 小时估计",
        "gpu_hour_price_used_for_teaching_only": gpu_hour_price,
        "cosmos_edge_sft": cosmos_configs,
        "specforge_draft": specforge,
        "cosyvoice_grpo": cosyvoice,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="训练作业准入策略对照与公开流程预算账")
    parser.add_argument("--section", choices=("admission", "budget", "public-flows"),
                        required=True)
    parser.add_argument("--yaml-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--gpu-hour-price", type=float, default=2.0)
    parser.add_argument("--params", type=float, default=3.0e9)
    parser.add_argument("--nodes", type=int, default=48)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--peak-tflops", type=float, default=989.0)
    parser.add_argument("--mfu", type=float, default=0.40)
    parser.add_argument("--ckpt-bytes-per-param", type=float, default=2 + 4 + 8)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    if args.section == "admission":
        result = admission_section()
        sources = [__file__]
    elif args.section == "budget":
        if args.yaml_dir is None:
            raise SystemExit("--yaml-dir is required for the budget section")
        result = budget_section(args.yaml_dir, args.params, args.nodes, args.gpus_per_node,
                                args.tp, args.peak_tflops, args.mfu, args.ckpt_bytes_per_param)
        sources = [__file__, *sorted(str(p) for p in args.yaml_dir.glob("*.yaml"))]
    else:
        if args.source_root is None:
            raise SystemExit("--source-root is required for the public-flows section")
        result = public_flows_section(args.source_root, args.gpu_hour_price)
        sources = [__file__, *sorted(str(p) for p in args.source_root.rglob("*") if p.is_file())]

    (args.outdir / "scheduling_budget.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "argv": [str(x) for x in sys.argv], "section": args.section, "sources": sources,
        "claim_scope": "准入策略是合成离散事件模拟；预算账是配方字段与声明设备假设的推演，不含本机训练测量；"
                       "公开流程只复算配置字段，不输出 GPU 小时估计",
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())