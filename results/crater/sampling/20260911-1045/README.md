# L5.9 evidence index

source/: installed code snapshots; manifest.json holds SHA-256.
run-initial/: CPU engine comparisons, engine logprobs benchmark and first GPU run.
The first GPU run failed at batch 64 on a support mismatch. Its log is preserved.
gpu-r2/: second run continues after recording mismatch details; paths with mismatches are not claimed equivalent.
contracts/: SGLang GPU sampling and vLLM raw/processed logprobs probes, with exit codes.
derived.json: medians and unit conversions rounded to 3 decimals, derived from raw JSON.
mini.txt: standard-library implementation run on the Mac.
*.snapshot: final harness versions. Only engine mode loads Qwen3 model weights.

SGLang GPU contract probes started after gpu-r2 exited; no overlap with timings.
All experiment artifacts are append-only.
