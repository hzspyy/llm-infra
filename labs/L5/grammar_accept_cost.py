#!/usr/bin/env python3
"""L5.6 diagnostic: per-step cost of advancing the matcher (accept), by library.

Companion to fill_per_step.py, which times the mask fill. Together they account
for the CPU work each backend adds to one decode step.
"""
import argparse, json, os, statistics, time
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from structured_output_audit import MODEL, LOOSE, TIGHT, TARGET, save, emit, versions
from grammar_libs_compare import BACKENDS, LIBS

p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
eos = gen.eos_token_id
eos = [eos] if isinstance(eos, int) else list(eos)
ids = tok.encode(json.dumps(TARGET, separators=(',', ':')), add_special_tokens=False)
rows = []
for name in LIBS:
    lib = BACKENDS[name](tok, cfg.vocab_size, eos)
    for label, schema in [('loose', LOOSE), ('tight', TIGHT)]:
        compiled, _ = lib.compile(schema)
        per = [[] for _ in ids]
        for rep in range(35):
            m = lib.matcher(compiled); mask = lib.allocate()
            for step, tid in enumerate(ids):
                lib.fill(m, mask)          # engines always fill before accepting
                t = time.perf_counter_ns()
                ok = lib.accept(m, tid)
                ns = time.perf_counter_ns() - t
                assert ok, (name, label, step)
                if rep >= 5:
                    per[step].append(ns)
        total = sum(statistics.median(s) for s in per) / 1000
        for step, samples in enumerate(per):
            row = {'lib': name, 'schema': label, 'step': step,
                   'piece': tok.decode([ids[step]]),
                   'accept_median_us': statistics.median(samples) / 1000,
                   'accept_max_us': max(samples) / 1000}
            rows.append(row); emit('accept_step', **row)
        emit('accept_total', lib=name, schema=label, sum_of_medians_us=total)
save(args.out / 'accept_per_step.json', rows)
save(args.out / 'metadata.json', {'versions': versions(), 'model': MODEL,
                                  'model_commit': cfg._commit_hash, 'trajectory': ids})
