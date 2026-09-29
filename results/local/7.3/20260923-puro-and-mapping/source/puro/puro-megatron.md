# Puro-Megatron extensions

Puro-Megatron is based on NVIDIA Megatron-LM tag `core_v0.16.0`, commit
`3bec9aa97dda898d16ff5a89bac0ed2b6682b172`. It is a curated commit series,
not a snapshot of an internal development branch.

## Installation

Use the CUDA, PyTorch, Transformer Engine, and Apex setup required by the
pinned Megatron-LM release:

```bash
pip install --no-build-isolation .[mlm,dev]
```

Muon and MuonHyperball use NVIDIA NeMo Emerging Optimizers, pinned in
`pyproject.toml` and `uv.lock`.

## PROM packed NPY data

`--train-data-path` and `--valid-data-path` may point at PROM NPY datasets.
The custom path broadcasts batches over tensor-parallel ranks and constructs
THD packed-sequence metadata for micro-batches larger than one. Dataset indices
remain stable across checkpoint resume and phase transitions.

The phase-resume controls used in production are:

```bash
--reset-opt-param-scheduler-progress \
--reset-train-dataloader-progress
```

An explicit `--ckpt-step` selects that persistent checkpoint even when a newer
non-persistent checkpoint exists. `--non-persistent-ckpt-num-to-keep` controls
global non-persistent checkpoint retention.

## MuonHyperball and effective LR

Select MuonHyperball with:

```bash
--optimizer muon_hyperball \
--muon-hyperball-scalar-optimizer adam \
--muon-hyperball-lr-mult 10
```

Fused QKV and SwiGLU FC1 parameters can be treated as their logical matrices
for both normalization and Newton-Schulz:

```bash
--muon-qkv-norm-mode separate \
--muon-swiglu-norm-mode separate \
--muon-qkv-ns-mode separate \
--muon-swiglu-ns-mode separate
```

`--muon-hyperball-rms VALUE` fixes each logical matrix's Frobenius radius to
`VALUE * sqrt(global_numel)`, where `global_numel` includes tensor-parallel
shards. If it is unset, each logical matrix keeps its initialization-time
radius. It is mutually exclusive with `--muon-hyperball-radius`.

Eligible two-dimensional attention and MLP matrices use MuonHyperball.
Embeddings, output weights, biases, normalization weights, and all non-matrix
parameters stay on AdamW. Weight decay is disabled for fixed-radius MuonH
matrices and remains active on the AdamW route.

`--layerwise-optimizer-memory-balance` assigns whole Muon-family matrices to
data-parallel owners and balances AdamW flat shards around that state. This
mode requires `--ckpt-format torch_dist`; parameter gathering can overlap with
the forward pass through `--overlap-param-gather`.

Ordinary Muon supports relative-update control with
`--muon-effective-lr-mult`. `--muon-strict-effective-lr` additionally solves
the update scale after decoupled weight decay so the normalized pre/post weight
distance matches the requested effective LR. `--diag-interval` controls
effective-LR diagnostics.

## Minimal Puro-2B recipes

[`examples/puro/run_puro_2b.sh`](../examples/puro/run_puro_2b.sh) contains the
three launcher-neutral entry points retained for release:

- `phase1-power`: phase-1 training with the open-ended power schedule;
- `phase2-transition`: phase-1 checkpoint transition with scheduler and data
  progress reset; and
- `phase2-constant-continuation`: constant-LR continuation from a selected
  phase-2 checkpoint.

Set `TRAIN_DATA_PATH`, `VALID_DATA_PATH`, `DATA_CACHE_PATH`, `LOAD_PATH`, and
`SAVE_PATH`, then invoke the recipe through the launcher used by your cluster.
`PYTHON`, `PURO_MEGATRON_ROOT`, `TENSORBOARD_DIR`, `CKPT_STEP`, and
`MUON_HYPERBALL_RMS` are optional overrides. Extra command-line arguments are
appended last so a smoke test can override sample counts and batch sizes.

## Power LR schedule

The open-ended power schedule is selected with:

```bash
--lr-decay-style power \
--lr-power-decay-samples 1536000 \
--lr-power-exponent 0.5
```

After warmup, with elapsed samples `t`, time scale `tau`, and exponent `p`, it
uses `min_lr + (max_lr - min_lr) * (1 + t/tau)^(-p)`. It is intentionally not
clamped at the ordinary LR decay horizon.

## Startup theoretical FLOPs

`pretrain()` emits a structured theoretical FLOPs report on rank zero after
argument parsing and before distributed initialization. It reports pipeline
stages, layer groups, operators, shapes, precision, and parallel multiplicity.
A report failure is non-fatal and does not stop training.

## Release scope

The repository focuses on the extensions described above. See
[`ORIGIN_COMMITS.md`](ORIGIN_COMMITS.md) for the exact source and scope audit.
