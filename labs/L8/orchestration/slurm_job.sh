#!/usr/bin/env bash
# 8.2-C: Slurm 映射（schema 级对照，本机未接入 Slurm 控制面，未在真实集群提交）。
#
# Slurm 的三层与本节的状态机一一对应：
#   allocation（sbatch 拿到节点与 GRES=GPU）  <-> ALLOCATED
#   step（srun 启动作业步里的进程）           <-> LOADING / WARMING
#   进程内就绪（模型加载完并能服务）           <-> READY
# 关键点：allocation 成功完全不等于 step 里的模型可用，中间隔着与 8.2-A 同量级的
# 加载与编译时间（本机实测 Qwen3-1.7B 冷启动 46.1 s）。
#
#SBATCH --job-name=vllm-qwen3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                 # 与 K8s 的 nvidia.com/gpu、Ray 的 num_gpus 同层
#SBATCH --exclusive                  # 独占节点：避免与他人共享同一张卡
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

echo "allocation_time=$(date -Is) job_id=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

# TP=2：这一步要求两张卡同时可用。若只分到一张，NCCL 初始化会在这里阻塞或报错，
# 而不是"半可用"——这正是要 gang scheduling（PodGroup / --exclusive）的原因。
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --tensor-parallel-size 2 \
  --port 8000 \
  --max-model-len 8192 &

VLLM_PID=$!
# 轮询 /health 才算就绪；不能把 srun 返回当成就绪。
for _ in $(seq 1 300); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    echo "ready_time=$(date -Is)"
    break
  fi
  sleep 2
done

wait "$VLLM_PID"
