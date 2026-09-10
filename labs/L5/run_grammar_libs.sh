#!/usr/bin/env bash
# L5.6: library-level comparison of xgrammar / llguidance / outlines_core.
# CPU only except the last step; each run writes to a fresh directory.
set -u
source /scratch/learn/env.sh
cd /scratch/learn/work/labs/L5
PY=/scratch/learn/envs/serve/bin/python
OUT=/scratch/learn/work/out
stamp=$(date +%Y%m%d-%H%M)

$PY grammar_libs_compare.py --out "$OUT/grammar-libs-$stamp"     > "$OUT/grammar-libs-$stamp.log" 2>&1;  echo "libs=$?"
$PY grammar_mask_diff.py --dir "$OUT/grammar-libs-$stamp" --schema loose --steps 0 1 2 13 16 \
    > "$OUT/grammar-libs-$stamp/mask_diff_loose.txt" 2>&1;                                             echo "diff_loose=$?"
$PY grammar_mask_diff.py --dir "$OUT/grammar-libs-$stamp" --schema tight --steps 3 8 13 \
    > "$OUT/grammar-libs-$stamp/mask_diff_tight.txt" 2>&1;                                             echo "diff_tight=$?"
$PY fill_per_step.py       --out "$OUT/grammar-fill-steps-$stamp" > "$OUT/grammar-fill-steps-$stamp.log" 2>&1; echo "fill=$?"
$PY grammar_accept_cost.py --out "$OUT/grammar-accept-$stamp"     > "$OUT/grammar-accept-$stamp.log" 2>&1;     echo "accept=$?"
$PY grammar_jump_forward.py --out "$OUT/grammar-jump-$stamp"      > "$OUT/grammar-jump-$stamp.log" 2>&1;       echo "jump=$?"
$PY structural_tag_probe.py --out "$OUT/structural-tag-$stamp"    > "$OUT/structural-tag-$stamp.log" 2>&1;     echo "tag=$?"

# The only GPU step: apply a captured mask to real logits.
mkdir -p "$OUT/grammar-apply-$stamp"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/grammar-apply-$stamp/gpu-before.txt"
for f in xgrammar-tight/mask-012.bin xgrammar-loose/mask-003.bin outlines_core-tight/mask-013.bin; do
  echo "# $f"
  $PY mask_apply_probe.py "$OUT/grammar-libs-$stamp/$f"
done > "$OUT/grammar-apply-$stamp/apply.log" 2>&1; echo "apply=$?"
