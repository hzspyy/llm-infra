#!/usr/bin/env bash
set -uo pipefail
source /scratch/learn/env.sh
export HF_HUB_CACHE="$HF_HOME/hub"
run_root="/scratch/learn/work/out/api-layer-20260912-0325"
PY=/scratch/learn/envs/serve/bin/python
for cfg in "tokenize 1000" "tokenize 4" "generate 1000"; do
  probe=${cfg% *}; reps=${cfg#* }
  for mode in inloop thread; do
    $PY -u /scratch/learn/work/labs/L5/sse_head_of_line.py --mode "$mode" --probe "$probe" \
        --b-reps "$reps" --out "$run_root/data" > "$run_root/hol-$probe-$mode-b$reps.log" 2>&1
    printf "%s\n" "$?" > "$run_root/exit-hol-$probe-$mode-b$reps.txt"
    echo "done $probe $mode b$reps exit=$(cat $run_root/exit-hol-$probe-$mode-b$reps.txt)"
  done
done
echo HOL_ALL_DONE
