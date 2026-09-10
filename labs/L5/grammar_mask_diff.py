#!/usr/bin/env python3
"""L5.6: decode the bitmasks written by grammar_libs_compare.py.

    python grammar_mask_diff.py --dir RUN_DIRECTORY --schema loose --steps 0 1 13 16

Prints, for each requested step, the allowed token pieces per library and the
pieces that only one library allows. Reads the .bin masks, so it can be re-run
against a stored run without a GPU.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
LIBS = ('xgrammar', 'llguidance', 'outlines_core')


def load(path, n_bits):
    blob = path.read_bytes()
    words = [int.from_bytes(blob[i:i + 4], 'little') for i in range(0, len(blob), 4)]
    ids = set()
    for w_index, word in enumerate(words):
        while word:
            bit = (word & -word).bit_length() - 1
            tid = w_index * 32 + bit
            if tid < n_bits:
                ids.add(tid)
            word &= word - 1
    return ids


def main(run_dir, schema, steps, limit):
    from transformers import AutoTokenizer
    meta = json.loads((run_dir / 'metadata.json').read_text())
    tok = AutoTokenizer.from_pretrained(meta['model'], local_files_only=True)
    n_bits = len(tok)
    rows = json.loads((run_dir / 'mask_steps.json').read_text())
    for step in steps:
        head = [r for r in rows if r['schema'] == schema and r['step'] == step]
        target = head[0] if head else None
        print(f"\n===== schema={schema} step={step} "
              f"target={target['piece']!r} (id {target['token_id']})" if target else
              f"\n===== schema={schema} step={step}")
        sets = {}
        for lib in LIBS:
            path = run_dir / f'{lib}-{schema}' / f'mask-{step:03d}.bin'
            sets[lib] = load(path, n_bits)
            print(f"  {lib:14} legal={len(sets[lib]):7d}")
        for lib in LIBS:
            others = set().union(*(sets[o] for o in LIBS if o != lib))
            only = sorted(sets[lib] - others)
            missing = sorted(set.intersection(*(sets[o] for o in LIBS if o != lib)) - sets[lib])
            print(f"  {lib:14} only_here={len(only):6d} "
                  f"{[tok.decode([i]) for i in only[:limit]]}")
            print(f"  {lib:14} missing  ={len(missing):6d} "
                  f"{[tok.decode([i]) for i in missing[:limit]]}")
        if max(len(s) for s in sets.values()) <= limit:
            for lib in LIBS:
                print(f"  {lib:14} all={[tok.decode([i]) for i in sorted(sets[lib])]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', type=Path, required=True)
    p.add_argument('--schema', default='loose')
    p.add_argument('--steps', type=int, nargs='+', required=True)
    p.add_argument('--limit', type=int, default=20)
    args = p.parse_args()
    main(args.dir, args.schema, args.steps, args.limit)
