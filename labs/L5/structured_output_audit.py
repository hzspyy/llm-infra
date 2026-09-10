#!/usr/bin/env python3
"""L5.6 evidence harness. Run each backend in a separate process.

CPU: python structured_output_audit.py cpu --out NEW_DIRECTORY
GPU: python structured_output_audit.py bench --backend xgrammar --out NEW_DIRECTORY
Set L56_GPU_MEMORY_UTILIZATION explicitly for the selected GPU budget.
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
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
MODEL = os.environ.get('L56_MODEL', 'Qwen/Qwen3-1.7B')
LOOSE = {'type': 'object', 'properties': {'name': {'type': 'string'},
         'city': {'type': 'string'}, 'age': {'type': 'integer'}},
         'required': ['name', 'city', 'age'], 'additionalProperties': False}
TIGHT = {'type': 'object', 'properties': {
    'name': {'type': 'string', 'pattern': '^[A-Z][a-z]{2,9}$'},
    'city': {'type': 'string', 'enum': ['Beijing', 'Shanghai', 'Shenzhen']},
    'age': {'type': 'integer', 'minimum': 0, 'maximum': 120}},
    'required': ['name', 'city', 'age'], 'additionalProperties': False}
TARGET = {'name': 'Zhang', 'city': 'Shenzhen', 'age': 31}


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def emit(kind, **fields):
    print(json.dumps({'kind': kind, **fields}, ensure_ascii=False), flush=True)


def versions():
    names = ['torch', 'transformers', 'xgrammar', 'llguidance', 'outlines_core',
             'jsonschema', 'vllm', 'sglang']
    result = {}
    for name in names:
        try:
            result[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return result


def cpu(out):
    import torch
    import xgrammar as xg
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    stop = gen.eos_token_id
    stop = [stop] if isinstance(stop, int) else stop
    V = cfg.vocab_size
    t = time.perf_counter()
    info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=V, stop_token_ids=stop)
    info_ms = (time.perf_counter() - t) * 1000
    compiler = xg.GrammarCompiler(info)
    meta = {'versions': versions(), 'model': MODEL, 'model_commit': cfg._commit_hash,
            'tokenizer_len': len(tok), 'model_vocab_size': V, 'eos_ids': stop,
            'tokenizer_info_ms': info_ms}
    save(out / 'metadata.json', meta)
    save(out / 'model_config.json', cfg.to_dict())
    save(out / 'schemas.json', {'loose': LOOSE, 'tight': TIGHT})
    emit('environment', **meta)
    target = json.dumps(TARGET, separators=(',', ':'))
    ids = tok.encode(target, add_special_tokens=False)
    all_rows = []
    for label, schema in [('loose', LOOSE), ('tight', TIGHT)]:
        grammar = xg.Grammar.from_json_schema(json.dumps(schema))
        (out / f'{label}.ebnf').write_text(str(grammar))
        timings = []
        for _ in range(5):
            t = time.perf_counter()
            compiled = compiler.compile_json_schema(json.dumps(schema))
            timings.append((time.perf_counter() - t) * 1000)
        matcher = xg.GrammarMatcher(compiled, max_rollback_tokens=4)
        mask = xg.allocate_token_bitmask(1, V)
        directory = out / label
        directory.mkdir()
        for step, tid in enumerate(ids + [stop[0]]):
            matcher.fill_next_token_bitmask(mask)
            words = [int(x) & 0xffffffff for x in mask.flatten().tolist()]
            # Exclude padding bits beyond the model's logits dimension.
            if V % 32:
                words[-1] &= (1 << (V % 32)) - 1
            count = sum(x.bit_count() for x in words)
            legal = bool((words[tid // 32] >> (tid % 32)) & 1)
            blob = b''.join(w.to_bytes(4, 'little') for w in words)
            (directory / f'mask-{step:03d}.bin').write_bytes(blob)
            row = {'schema': label, 'step': step, 'token_id': tid,
                   'piece': tok.decode([tid]), 'legal_count': count,
                   'word_index': tid // 32, 'bit_index': tid % 32,
                   'word_hex': f'{words[tid // 32]:08x}', 'allowed': legal,
                   'shape': list(mask.shape), 'stride': list(mask.stride()),
                   'bytes': len(blob)}
            assert legal, row
            assert matcher.accept_token(tid), row
            row['terminated_after'] = matcher.is_terminated()
            all_rows.append(row)
            emit('mask_step', **row)
        assert matcher.is_terminated()
        matcher.rollback(1)
        matcher.fill_next_token_bitmask(mask)
        assert matcher.accept_token(stop[0]) and matcher.is_terminated()
        # Rejection must not advance the state. Compare before/after masks.
        bad = xg.GrammarMatcher(compiled)
        bad.fill_next_token_bitmask(mask)
        before = mask.clone()
        tid_bad = next(i for i in range(len(tok))
                       if not ((int(mask[0, i // 32]) & 0xffffffff) >> (i % 32)) & 1)
        rejected = not bad.accept_token(tid_bad)
        bad.fill_next_token_bitmask(mask)
        assert rejected and torch.equal(before, mask)
        cases = []
        for key, value in [('valid', TARGET), ('age_type', {**TARGET, 'age': '31'}),
                           ('age_range', {**TARGET, 'age': 200}),
                           ('city_enum', {**TARGET, 'city': 'London'}),
                           ('name_pattern', {**TARGET, 'name': 'zhang'}),
                           ('extra_key', {**TARGET, 'extra': True})]:
            m = xg.GrammarMatcher(compiled)
            accepted = m.accept_string(json.dumps(value, separators=(',', ':')))
            import jsonschema
            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
            cases.append({'case': key, 'matcher_accept': accepted,
                          'schema_valid': not errors})
        emit('cpu_summary', schema=label, compile_ms=timings,
             rejection_preserves_state=True, rejected_token_id=tid_bad,
             rollback_eos=True, validation=cases)
        save(out / f'{label}-summary.json', {'compile_ms': timings, 'cases': cases})
        # CPU fill cost on an identical legal trajectory: 5 warm + 30 measured.
        samples = []
        for rep in range(35):
            m = xg.GrammarMatcher(compiled)
            for tid in ids:
                t = time.perf_counter_ns()
                m.fill_next_token_bitmask(mask)
                ns = time.perf_counter_ns() - t
                assert m.accept_token(tid)
                if rep >= 5:
                    samples.append(ns)
        emit('fill_timing', schema=label, steps=len(samples),
             median_us=statistics.median(samples) / 1000,
             mean_us=statistics.mean(samples) / 1000)
    save(out / 'mask_steps.json', all_rows)


def bench(out, backend, repeats, batch):
    import torch
    import jsonschema
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    if 'L56_GPU_MEMORY_UTILIZATION' not in os.environ:
        raise ValueError('Set L56_GPU_MEMORY_UTILIZATION for this run; no fixed reservation policy')
    util = float(os.environ['L56_GPU_MEMORY_UTILIZATION'])
    assert 0 < util <= 1
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    fixtures = [('Zhang', 'Shenzhen', 31), ('Alice', 'Beijing', 22),
                ('Robert', 'Shanghai', 40), ('Helen', 'Beijing', 65)]
    prompts, expected = [], []
    for i in range(batch):
        name, city, age = fixtures[i % len(fixtures)]
        obj = {'name': name, 'city': city, 'age': age}
        expected.append(obj)
        messages = [{'role': 'user', 'content':
            f'Extract name, city, age from: {name} lives in {city} and is {age} years old. '
            'Return only a JSON object with these three keys. No explanation.'}]
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
                       add_generation_prompt=True, enable_thinking=False))
    config = dict(model=MODEL, gpu_memory_utilization=util, max_model_len=2048,
                  max_num_seqs=32, enforce_eager=True, enable_prefix_caching=False,
                  disable_log_stats=True, seed=7)
    if backend != 'none':
        config['structured_outputs_config'] = {'backend': backend}
    save(out / 'config.json', {'versions': versions(), 'llm': config,
                             'prompts': prompts, 'expected': expected,
                             'batch': batch, 'repeats': repeats})
    free, total = torch.cuda.mem_get_info()
    emit('gpu_before', free_bytes=free, total_bytes=total, budget_fraction=util)
    llm = LLM(**config)
    records, summaries = [], []
    try:
        for label, schema in [('loose', LOOSE), ('tight', TIGHT)]:
            sp = SamplingParams(temperature=0, max_tokens=128, seed=7)
            if backend != 'none':
                sp.structured_outputs = StructuredOutputsParams(json=schema)
            llm.generate(prompts[:1], sp, use_tqdm=False)
            validator = jsonschema.Draft202012Validator(schema)
            for rep in range(repeats):
                t = time.perf_counter()
                outputs = llm.generate(prompts, sp, use_tqdm=False)
                dt = time.perf_counter() - t
                rows = []
                for i, o in enumerate(outputs):
                    item = o.outputs[0]
                    text = item.text
                    try:
                        obj = json.loads(text)
                        errors = [e.message for e in validator.iter_errors(obj)]
                        parse = True
                    except (ValueError, TypeError) as e:
                        obj, errors, parse = None, [str(e)], False
                    row = {'backend': backend, 'schema': label, 'rep': rep, 'i': i,
                           'text': text, 'token_ids': list(item.token_ids),
                           'finish_reason': item.finish_reason,
                           'stop_reason': item.stop_reason, 'json_valid': parse,
                           'schema_valid': parse and not errors,
                           'semantic_match': obj == expected[i], 'errors': errors}
                    rows.append(row)
                records.extend(rows)
                total_tokens = sum(len(r['token_ids']) for r in rows)
                summary = {'backend': backend, 'schema': label, 'rep': rep,
                    'seconds': dt, 'tokens': total_tokens, 'tokens_per_second': total_tokens/dt,
                    'requests': len(rows), 'json_valid': sum(r['json_valid'] for r in rows),
                    'schema_valid': sum(r['schema_valid'] for r in rows),
                    'semantic_match': sum(r['semantic_match'] for r in rows),
                    'normal_stop': sum(r['finish_reason']=='stop' for r in rows),
                    'finish_reasons': [r['finish_reason'] for r in rows]}
                summaries.append(summary)
                emit('bench', **summary)
        # Truncation is a distinct outcome, not a grammar guarantee violation.
        sp = SamplingParams(temperature=0, max_tokens=2)
        if backend != 'none':
            sp.structured_outputs = StructuredOutputsParams(json=TIGHT)
        o = llm.generate(prompts[:1], sp, use_tqdm=False)[0].outputs[0]
        emit('truncation', text=o.text, tokens=list(o.token_ids), finish_reason=o.finish_reason)
    finally:
        save(out / 'outputs.json', records)
        save(out / 'summary.json', summaries)
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cpu', 'bench'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=['none', 'xgrammar', 'guidance', 'outlines'], default='none')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'run.json', {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              'mode': args.mode, 'backend': args.backend})
    if args.mode == 'cpu':
        cpu(args.out)
    else:
        bench(args.out, args.backend, args.repeats, args.batch)
