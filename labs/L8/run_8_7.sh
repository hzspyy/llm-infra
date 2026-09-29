#!/usr/bin/env bash
# labs/L8/run_8_7.sh - 8.7-A/B 的三次启动测量 (在 crater2 上执行).
#
# 三次运行构成一个 2x2 分解, 用来把"编译/图缓存"和"权重页缓存"两笔账分开:
#   1 cold_compile_cold_data  全新 VLLM_CACHE_ROOT + 定点驱逐权重页缓存 (真实冷读)
#   2 warm_compile_warm_data  同一 CACHE_ROOT, 不驱逐 (两次缓存都热)
#   3 cold_compile_warm_data  换新的 CACHE_ROOT (编译冷), 不驱逐 (权重仍热)
#
# 三点口径说明:
#   * 冷读条件由 `posix_fadvise(POSIX_FADV_DONTNEED)` 定点驱逐得到, 只作用于该模型
#     的权重文件, **不 drop_caches**, 不改动共享机器的其它状态;
#   * 冷/热由每次运行的 /proc/<pid>/io.read_bytes 证明 (权重 3.78 GiB: 冷读应接近它);
#   * 显式传 --model-path 指向本地快照, 让三次运行只差"缓存状态"这一个变量,
#     避免 HF 解析/网络取回 (本机权重目录在 NFS4 上) 混进对比。
set -euo pipefail

# 自己初始化学习环境, 不依赖登录 shell 的 PATH: vLLM 运行时要找 `ninja`
# (crater2 上它只在 venv 的 bin 里), 少了它会以 FileNotFoundError 终止引擎初始化。
# shellcheck disable=SC1091
[ -f /scratch/learn/env.sh ] && source /scratch/learn/env.sh
[ -f /root/learn/env.sh ] && source /root/learn/env.sh

RUN=${1:?run_dir}
PY=${PY:-/scratch/learn/envs/serve/bin/python}
VLLM=${VLLM:-/scratch/learn/envs/serve/bin/vllm}
SNAP=${SNAP:-/scratch/learn/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
mkdir -p "$RUN"
rm -rf "$RUN/cache" "$RUN/cache2"

echo "=== 1/3 cold compile + cold data (evict weights) ==="
"$PY" labs/L8/startup_timeline.py --out-dir "$RUN" --cache-root "$RUN/cache" \
  --label cold_compile_cold_data --port 19300 --model-path "$SNAP" \
  --evict-weights --python "$PY" --vllm "$VLLM"

echo "=== 2/3 warm compile + warm data ==="
"$PY" labs/L8/startup_timeline.py --out-dir "$RUN" --cache-root "$RUN/cache" \
  --label warm_compile_warm_data --port 19301 --model-path "$SNAP" \
  --python "$PY" --vllm "$VLLM"

echo "=== 3/3 cold compile + warm data ==="
"$PY" labs/L8/startup_timeline.py --out-dir "$RUN" --cache-root "$RUN/cache2" \
  --label cold_compile_warm_data --port 19302 --model-path "$SNAP" \
  --python "$PY" --vllm "$VLLM"

# 三段都必须真的到过 READY, 否则这一轮数据无效 (本轮第一次运行就因为参数冲突
# 立刻退出, 若不校验会被当成"启动很快")。
"$PY" - "$RUN" <<'PY'
import json, sys, pathlib
run = pathlib.Path(sys.argv[1])
bad = []
for label in ("cold_compile_cold_data", "warm_compile_warm_data", "cold_compile_warm_data"):
    p = run / f"startup_{label}.json"
    d = json.loads(p.read_text()) if p.exists() else {"ready": False}
    print(f"{label}: ready={d.get('ready')} "
          f"read_bytes_MiB={d.get('read_bytes_mib')} phases={len(d.get('phases', []))}")
    if not d.get("ready"):
        bad.append(label)
if bad:
    raise SystemExit(f"NOT READY: {bad}")
PY

echo DONE_8_7 "$RUN"