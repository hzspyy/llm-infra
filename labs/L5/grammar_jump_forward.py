#!/usr/bin/env python3
"""L5.6: how many characters can be skipped by jump-forward, per step.

SGLang calls xgrammar's find_jump_forward_string() to emit forced text without
running the model (srt/constrained/xgrammar_backend.py: try_jump_forward).
vLLM's guidance backend has the same capability behind a TODO. This walks the
target trajectory and records what the forced continuation would be at each step.
"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from structured_output_audit import MODEL, LOOSE, TIGHT, TARGET, save, emit, versions

p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
import xgrammar as xg
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
eos = gen.eos_token_id
eos = [eos] if isinstance(eos, int) else list(eos)
info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=cfg.vocab_size, stop_token_ids=eos)
compiler = xg.GrammarCompiler(info)
ids = tok.encode(json.dumps(TARGET, separators=(',', ':')), add_special_tokens=False)
rows = []
for label, schema in [('loose', LOOSE), ('tight', TIGHT)]:
    compiled = compiler.compile_json_schema(json.dumps(schema))
    m = xg.GrammarMatcher(compiled)
    prefix = ''
    skipped = 0
    for step, tid in enumerate(ids):
        jump = m.find_jump_forward_string()
        row = {'schema': label, 'step': step, 'prefix': prefix,
               'next_piece': tok.decode([tid]), 'jump_forward': jump,
               'jump_chars': len(jump) if jump else 0}
        rows.append(row); emit('jump', **row)
        skipped += row['jump_chars']
        assert m.accept_token(tid)
        prefix += tok.decode([tid])
    emit('jump_total', schema=label, generated_chars=len(prefix),
         jump_chars_if_taken_greedily=skipped)
save(args.out / 'jump_forward.json', rows)
save(args.out / 'metadata.json', {'versions': versions(), 'model': MODEL,
                                  'model_commit': cfg._commit_hash, 'trajectory': ids})
