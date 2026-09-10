#!/usr/bin/env python3
"""L5.6: compare xgrammar / llguidance / outlines_core at library level.

No model weights are loaded. Every library gets the same tokenizer, the same two
JSON schemas and the same target token trajectory, so per-step legal-token counts
are directly comparable.

    python grammar_libs_compare.py --out NEW_DIRECTORY

Recorded per library: tokenizer preprocessing cost, grammar compile cost,
per-step bitmask (shape/stride/bytes/legal count/target bit), fill cost, the
state contract on a rejected token, and where each library stops an invalid
document.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')

from structured_output_audit import MODEL, LOOSE, TIGHT, TARGET, save, emit, versions

BAD_CASES = [
    ('valid', TARGET),
    ('age_type', {**TARGET, 'age': '31'}),
    ('age_range', {**TARGET, 'age': 200}),
    ('city_enum', {**TARGET, 'city': 'London'}),
    ('name_pattern', {**TARGET, 'name': 'zhang'}),
    ('extra_key', {**TARGET, 'extra': True}),
]
LIBS = ('xgrammar', 'llguidance', 'outlines_core')


def popcount(words):
    return sum(w.bit_count() for w in words)


def mask_words(mask, n_bits):
    """int32 tensor row -> list of unsigned 32-bit words, padding bits cleared."""
    words = [int(x) & 0xffffffff for x in mask.flatten().tolist()]
    if n_bits % 32:
        words[-1] &= (1 << (n_bits % 32)) - 1
    return words


class XGrammar:
    """xgrammar: byte-level pushdown matcher compiled from the JSON schema."""
    name = 'xgrammar'

    def __init__(self, tok, vocab_size, eos_ids):
        import xgrammar as xg
        self.xg = xg
        t = time.perf_counter()
        self.info = xg.TokenizerInfo.from_huggingface(
            tok, vocab_size=vocab_size, stop_token_ids=eos_ids)
        self.prep_ms = (time.perf_counter() - t) * 1000
        self.compiler = xg.GrammarCompiler(self.info)
        self.vocab_size = vocab_size

    def compile(self, schema):
        t = time.perf_counter()
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        return compiled, (time.perf_counter() - t) * 1000

    def artifact(self, schema):
        return ('ebnf', str(self.xg.Grammar.from_json_schema(json.dumps(schema))))

    def matcher(self, compiled):
        return self.xg.GrammarMatcher(compiled, max_rollback_tokens=4)

    def allocate(self):
        return self.xg.allocate_token_bitmask(1, self.vocab_size)

    def fill(self, m, mask):
        m.fill_next_token_bitmask(mask)

    def accept(self, m, tid):
        return bool(m.accept_token(tid))

    def terminated(self, m):
        return bool(m.is_terminated())


class LLGuidance:
    """llguidance: Earley parser over a lark grammar derived from the schema."""
    name = 'llguidance'

    def __init__(self, tok, vocab_size, eos_ids):
        import llguidance
        import llguidance.hf
        import llguidance.torch
        self.llg, self.torch_mod = llguidance, llguidance.torch
        t = time.perf_counter()
        self.lltok = llguidance.hf.from_tokenizer(tok, max(vocab_size, len(tok)))
        self.prep_ms = (time.perf_counter() - t) * 1000
        self.vocab_size = vocab_size

    def compile(self, schema):
        t = time.perf_counter()
        # vLLM's GuidanceBackend passes whitespace_flexible=True by default.
        grammar = self.llg.LLMatcher.grammar_from_json_schema(
            schema, defaults={'whitespace_flexible': True})
        return grammar, (time.perf_counter() - t) * 1000

    def artifact(self, schema):
        grammar = self.llg.LLMatcher.grammar_from_json_schema(
            schema, defaults={'whitespace_flexible': True})
        return ('lark.json', grammar)

    def matcher(self, compiled):
        m = self.llg.LLMatcher(self.lltok, compiled)
        assert not m.is_error(), m.get_error()
        return m

    def allocate(self):
        return self.torch_mod.allocate_token_bitmask(1, self.lltok.vocab_size)

    def fill(self, m, mask):
        self.torch_mod.fill_next_token_bitmask(m, mask, 0)

    def accept(self, m, tid):
        return m.try_consume_tokens([tid]) == 1

    def terminated(self, m):
        return bool(m.is_stopped())


class OutlinesCore:
    """outlines_core: schema -> regex -> DFA index over the token vocabulary."""
    name = 'outlines_core'

    def __init__(self, tok, vocab_size, eos_ids):
        import torch
        import outlines_core as oc
        from outlines_core import json_schema
        from vllm.v1.structured_output.utils import get_outlines_vocabulary
        self.torch, self.oc, self.json_schema = torch, oc, json_schema
        t = time.perf_counter()
        self.vocab = get_outlines_vocabulary(tok)
        self.prep_ms = (time.perf_counter() - t) * 1000
        self.vocab_size = vocab_size

    def compile(self, schema):
        t = time.perf_counter()
        regex = self.json_schema.build_regex_from_schema(json.dumps(schema))
        index = self.oc.Index(regex, self.vocab.inner)
        return index, (time.perf_counter() - t) * 1000

    def artifact(self, schema):
        return ('regex.txt', self.json_schema.build_regex_from_schema(json.dumps(schema)))

    def matcher(self, compiled):
        return self.oc.Guide(compiled)

    def allocate(self):
        return self.torch.zeros((1, (self.vocab_size + 31) // 32), dtype=self.torch.int32)

    def fill(self, m, mask):
        row = mask[0]
        row.zero_()
        m.write_mask_into(row.data_ptr(), row.numel(), row.element_size())

    def accept(self, m, tid):
        if not m.accepts_tokens([tid]):
            return False
        m.advance(tid)
        return True

    def terminated(self, m):
        return bool(m.is_finished())


BACKENDS = {'xgrammar': XGrammar, 'llguidance': LLGuidance, 'outlines_core': OutlinesCore}


def walk(lib, compiled, ids, eos, vocab_size, tok_len, label, out_dir, tok):
    """Walk the target trajectory, recording the bitmask at every step."""
    m = lib.matcher(compiled)
    mask = lib.allocate()
    rows = []
    directory = out_dir / f'{lib.name}-{label}'
    directory.mkdir(parents=True)
    for step, tid in enumerate(list(ids) + [eos]):
        lib.fill(m, mask)
        words = mask_words(mask, vocab_size)
        real = list(words)
        if tok_len % 32:
            real[tok_len // 32] &= (1 << (tok_len % 32)) - 1
        del real[tok_len // 32 + 1:]
        blob = b''.join(w.to_bytes(4, 'little') for w in words)
        (directory / f'mask-{step:03d}.bin').write_bytes(blob)
        allowed = bool((words[tid // 32] >> (tid % 32)) & 1)
        row = {'lib': lib.name, 'schema': label, 'step': step, 'token_id': tid,
               'piece': tok.decode([tid]), 'legal_count': popcount(words),
               'legal_count_real_vocab': popcount(real),
               'word_index': tid // 32, 'bit_index': tid % 32,
               'word_hex': f'{words[tid // 32]:08x}', 'allowed': allowed,
               'shape': list(mask.shape), 'stride': list(mask.stride()),
               'dtype': str(mask.dtype), 'bytes': len(blob)}
        accepted = lib.accept(m, tid)
        row['accepted'] = accepted
        row['terminated_after'] = lib.terminated(m)
        rows.append(row)
        emit('mask_step', **row)
        if not accepted:
            break
    return rows, m


def rejection_contract(lib, compiled, vocab_size):
    """A rejected token must not advance the state. Compare masks around it."""
    import torch
    m = lib.matcher(compiled)
    mask = lib.allocate()
    lib.fill(m, mask)
    before = mask.clone()
    words = mask_words(mask, vocab_size)
    bad = next(i for i in range(vocab_size) if not ((words[i // 32] >> (i % 32)) & 1))
    rejected = not lib.accept(m, bad)
    lib.fill(m, mask)
    return {'rejected_token_id': bad, 'rejected': rejected,
            'mask_unchanged': bool(torch.equal(before, mask))}


def prefix_stop(lib, compiled, tok, value):
    """Feed an invalid document token by token; report where it is stopped."""
    text = json.dumps(value, separators=(',', ':'))
    ids = tok.encode(text, add_special_tokens=False)
    m = lib.matcher(compiled)
    accepted = []
    for tid in ids:
        if not lib.accept(m, tid):
            return {'text': text, 'stopped_at_token': len(accepted),
                    'accepted_prefix': tok.decode(accepted),
                    'rejected_piece': tok.decode([tid]), 'fully_accepted': False}
        accepted.append(tid)
    return {'text': text, 'stopped_at_token': None,
            'accepted_prefix': tok.decode(accepted), 'rejected_piece': None,
            'fully_accepted': True}


def fill_cost(lib, compiled, ids, warmup=5, reps=30):
    samples = []
    for rep in range(warmup + reps):
        m = lib.matcher(compiled)
        mask = lib.allocate()
        for tid in ids:
            t = time.perf_counter_ns()
            lib.fill(m, mask)
            ns = time.perf_counter_ns() - t
            if not lib.accept(m, tid):
                break
            if rep >= warmup:
                samples.append(ns)
    return {'steps': len(samples), 'median_us': statistics.median(samples) / 1000,
            'mean_us': statistics.mean(samples) / 1000,
            'max_us': max(samples) / 1000}


def main(out):
    import jsonschema
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    eos_ids = gen.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    V = cfg.vocab_size
    meta = {'versions': versions(), 'model': MODEL, 'model_commit': cfg._commit_hash,
            'tokenizer_len': len(tok), 'model_vocab_size': V, 'eos_ids': eos_ids,
            'libs': list(LIBS)}
    save(out / 'metadata.json', meta)
    save(out / 'schemas.json', {'loose': LOOSE, 'tight': TIGHT})
    emit('environment', **meta)

    ids = tok.encode(json.dumps(TARGET, separators=(',', ':')), add_special_tokens=False)
    emit('trajectory', tokens=ids, pieces=[tok.decode([i]) for i in ids], eos=eos_ids[0])

    all_rows, report = [], []
    for name in LIBS:
        lib = BACKENDS[name](tok, V, eos_ids)
        emit('prepare', lib=name, prep_ms=lib.prep_ms)
        for label, schema in [('loose', LOOSE), ('tight', TIGHT)]:
            kind, text = lib.artifact(schema)
            (out / f'{name}-{label}.{kind}').write_text(text)
            timings = []
            for _ in range(5):
                compiled, ms = lib.compile(schema)
                timings.append(ms)
            emit('compile', lib=name, schema=label, compile_ms=timings,
                 artifact_chars=len(text))
            rows, m = walk(lib, compiled, ids, eos_ids[0], V, len(tok), label, out, tok)
            all_rows.extend(rows)
            entry = {'lib': name, 'schema': label, 'prep_ms': lib.prep_ms,
                     'compile_ms': timings, 'artifact_kind': kind,
                     'artifact_chars': len(text),
                     'walk_completed': all(r['accepted'] for r in rows),
                     'terminated_after_eos': rows[-1]['terminated_after'],
                     'legal_counts': [r['legal_count'] for r in rows],
                     'legal_counts_real_vocab': [r['legal_count_real_vocab'] for r in rows]}
            entry['rejection'] = rejection_contract(lib, compiled, V)
            emit('rejection', lib=name, schema=label, **entry['rejection'])
            cases = []
            for key, value in BAD_CASES:
                stop = prefix_stop(lib, compiled, tok, value)
                errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
                stop.update(case=key, lib=name, schema=label,
                            schema_valid=not errors,
                            schema_errors=[e.message for e in errors])
                cases.append(stop)
                emit('case', **stop)
            entry['cases'] = cases
            entry['fill'] = fill_cost(lib, compiled, ids)
            emit('fill_timing', lib=name, schema=label, **entry['fill'])
            report.append(entry)
    save(out / 'mask_steps.json', all_rows)
    save(out / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'run.json', {
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    main(args.out)
