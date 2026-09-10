#!/usr/bin/env python3
"""L5.6: structural tag = free text outside the tag, strict grammar inside it.

This is the shape a tool call actually has: the model writes prose, then emits a
trigger, and only the region between begin and end is constrained. The probe
walks one such trajectory and records the legal-token count at every step, so the
switch between "anything" and "only JSON" is visible as a number.

    python structural_tag_probe.py --out NEW_DIRECTORY
"""
import argparse, hashlib, json, os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from structured_output_audit import save, emit, versions, MODEL

BEGIN, END, TRIGGER = '<tool_call>', '</tool_call>', '<tool_call>'
ARGS_SCHEMA = {'type': 'object',
               'properties': {'name': {'type': 'string', 'enum': ['get_weather']},
                              'arguments': {'type': 'object',
                                            'properties': {'city': {'type': 'string',
                                                                    'enum': ['Beijing', 'Shenzhen']}},
                                            'required': ['city'],
                                            'additionalProperties': False}},
               'required': ['name', 'arguments'], 'additionalProperties': False}
PROSE = 'Let me check the weather. '
CALL = json.dumps({'name': 'get_weather', 'arguments': {'city': 'Shenzhen'}},
                  separators=(',', ':'))

p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
save(args.out / 'run.json', {
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

import xgrammar as xg
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
eos = gen.eos_token_id
eos = [eos] if isinstance(eos, int) else list(eos)
V = cfg.vocab_size

tag = xg.StructuralTagItem(begin=BEGIN, schema=json.dumps(ARGS_SCHEMA), end=END)
grammar = xg.Grammar.from_structural_tag([tag], [TRIGGER])
(args.out / 'structural_tag.ebnf').write_text(str(grammar))
info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=V, stop_token_ids=eos)
compiled = xg.GrammarCompiler(info).compile_structural_tag([tag], [TRIGGER])

text = PROSE + BEGIN + CALL + END
ids = tok.encode(text, add_special_tokens=False)
save(args.out / 'spec.json', {'schema': ARGS_SCHEMA, 'begin': BEGIN, 'end': END,
                              'trigger': TRIGGER, 'text': text, 'token_ids': ids,
                              'versions': versions(), 'model': MODEL,
                              'model_commit': cfg._commit_hash})
emit('environment', versions=versions(), model=MODEL, model_commit=cfg._commit_hash,
     text=text, n_tokens=len(ids))

m = xg.GrammarMatcher(compiled)
mask = xg.allocate_token_bitmask(1, V)
rows, prefix = [], ''
for step, tid in enumerate(ids):
    m.fill_next_token_bitmask(mask)
    words = [int(x) & 0xffffffff for x in mask.flatten().tolist()]
    if V % 32:
        words[-1] &= (1 << (V % 32)) - 1
    row = {'step': step, 'token_id': tid, 'piece': tok.decode([tid]),
           'legal_count': sum(w.bit_count() for w in words),
           'allowed': bool((words[tid // 32] >> (tid % 32)) & 1),
           'region': 'inside' if BEGIN in prefix and END not in prefix else 'outside',
           'prefix_tail': prefix[-24:]}
    assert row['allowed'], row
    assert m.accept_token(tid), row
    prefix += tok.decode([tid])
    rows.append(row); emit('tag_step', **row)
save(args.out / 'tag_steps.json', rows)
outside = [r['legal_count'] for r in rows if r['region'] == 'outside']
inside = [r['legal_count'] for r in rows if r['region'] == 'inside']
emit('summary', n_tokens=len(ids), outside_steps=len(outside), inside_steps=len(inside),
     outside_min=min(outside), outside_max=max(outside),
     inside_min=min(inside), inside_max=max(inside), terminated=m.is_terminated())
