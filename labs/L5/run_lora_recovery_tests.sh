#!/bin/bash
set -e

cd /scratch/learn
source env.sh
source envs/serve/bin/activate

TIMESTAMP=$(date +%Y%m%d-%H%M)
OUT_DIR="work/out/lora-recovery-${TIMESTAMP}"
ADAPTER_DIR="work/scratch/lora-recovery-adapters-${TIMESTAMP}"

mkdir -p "$OUT_DIR" "$ADAPTER_DIR"

echo "=== LoRA Recovery Tests ==="
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUT_DIR"

# GPU status before
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT_DIR/gpu-before.txt"

# Test 1: Recovery paths
echo ""
echo "Test 1: Recovery after rank limit violation"
python work/labs/L5/lora_recovery_test.py \
  --output "$OUT_DIR/recovery" \
  --adapters "$ADAPTER_DIR/recovery" \
  2>&1 | tee "$OUT_DIR/recovery.log"

EXIT_RECOVERY=$?
echo "$EXIT_RECOVERY" > "$OUT_DIR/recovery-exit.txt"

# Test 2: KV invalidation
echo ""
echo "Test 2: KV invalidation on same-name update"
python work/labs/L5/lora_kv_invalidation_test.py \
  --output "$OUT_DIR/kv" \
  --adapters "$ADAPTER_DIR/kv" \
  2>&1 | tee "$OUT_DIR/kv.log"

EXIT_KV=$?
echo "$EXIT_KV" > "$OUT_DIR/kv-exit.txt"

# GPU status after
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT_DIR/gpu-after.txt"

# Compute processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT_DIR/processes-after.txt" 2>&1 || echo "No compute processes" > "$OUT_DIR/processes-after.txt"

# Summary
cat > "$OUT_DIR/summary.json" << SUMMARY
{
  "timestamp": "$TIMESTAMP",
  "tests": {
    "recovery": {
      "exit_code": $EXIT_RECOVERY,
      "log": "recovery.log",
      "results": "recovery/"
    },
    "kv_invalidation": {
      "exit_code": $EXIT_KV,
      "log": "kv.log",
      "results": "kv/"
    }
  },
  "environment": {
    "gpu_before": "gpu-before.txt",
    "gpu_after": "gpu-after.txt",
    "processes": "processes-after.txt"
  }
}
SUMMARY

echo ""
echo "=== Tests Complete ==="
echo "Results in: $OUT_DIR"
echo "Recovery exit: $EXIT_RECOVERY"
echo "KV test exit: $EXIT_KV"
