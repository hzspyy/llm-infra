#!/bin/bash
# 7.0b 的 CPU 机制检查。真实模型的同一组对照用 training_step_smollm.py 单独运行。
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to a new experiment directory on the learning disk}"
: "${TMPDIR:?Set TMPDIR to the learning disk as documented in ENVIRONMENTS.md}"
PYTHON_BIN=${PYTHON_BIN:-python}
mkdir -p "$RUN_DIR"
set -o noclobber
"$PYTHON_BIN" labs/L7/supervision_contract.py     > "$RUN_DIR/supervision.txt"
"$PYTHON_BIN" labs/L7/one_update_reference.py     > "$RUN_DIR/one-update.txt"
"$PYTHON_BIN" labs/L7/accumulation_denominator.py > "$RUN_DIR/accumulation.txt"
"$PYTHON_BIN" labs/L7/start_modes_and_resume.py   > "$RUN_DIR/start-resume.txt"
"$PYTHON_BIN" labs/L7/recipe_state_ledger.py      > "$RUN_DIR/recipe-ledger.txt"
