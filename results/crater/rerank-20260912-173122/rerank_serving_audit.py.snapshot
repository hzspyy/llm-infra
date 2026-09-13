#!/usr/bin/env python3
"""5.12: Qwen3 reranker HF reference and vLLM HTTP contract audit.

Run in a fresh --out directory after checking GPU occupancy. Each stage is a
separate process; reference exits before the server starts. No quality benchmark.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

QUERY = 'Which mechanism lets multiple requests reuse an identical prompt prefix?'
DOCS = [
    'Prefix caching reuses computed key and value blocks for matching prompt prefixes.',
    'Temperature scales logits before sampling the next token.',
    'A paged cache stores key and value tensors in fixed-size blocks.',
    'Gradient accumulation combines gradients before updating model parameters.',
    'A prefix cache lookup depends on the preceding token sequence and cache identity.',
    'The restaurant serves soup at noon.',
]


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def request(base, endpoint, payload, dest):
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(base + endpoint, data=raw,
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    elapsed = (time.perf_counter() - start) * 1000
    with dest.with_suffix('.bin').open('xb') as f:
        f.write(body)
    result = {'endpoint': endpoint, 'payload': payload, 'status': status,
              'client_ms': elapsed, 'response': json.loads(body)}
    save(dest.with_suffix('.json'), result)
    return result


def reference(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    prompts = [tok.apply_chat_template([{'role': 'query', 'content': QUERY},
                                        {'role': 'document', 'content': d}], tokenize=False) for d in DOCS]
    tokens = [tok.encode(p, add_special_tokens=False) for p in prompts]
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    inputs = tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
    ids = [tok.convert_tokens_to_ids(t) for t in ['no', 'yes']]
    with torch.inference_mode():
        h = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
        # Two selected LM-head rows in fp32: independent reference for the
        # algebraic head conversion. Also preserve native bf16 full-head logits.
        rows = model.lm_head.weight[ids].float()
        selected = h.float() @ rows.T
        native = model.lm_head(h)[:, ids].float()
    torch.cuda.synchronize()
    save(args.out / 'reference.json', {
        'model': str(args.model), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'query': QUERY, 'documents': DOCS, 'prompts': prompts, 'token_ids': tokens,
        'token_lengths': list(map(len, tokens)), 'label_token_ids': ids,
        'input_shape': list(inputs['input_ids'].shape), 'hidden_shape': list(h.shape),
        'hidden_stride': list(h.stride()), 'head_rows_shape': list(rows.shape),
        'selected_fp32_logits': selected.tolist(), 'native_bf16_logits': native.tolist(),
        'probabilities': selected.softmax(-1)[:, 1].tolist(),
        'native_probabilities': native.softmax(-1)[:, 1].tolist(),
        'logit_differences': (selected[:, 1] - selected[:, 0]).tolist(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'quality_scope': 'six authored smoke documents; no retrieval benchmark',
        'acceptance': {'probability_abs_tol': 0.02, 'logit_abs_tol': 0.25,
                       'ranking_top_must_be': 0},
    })


def online(args):
    ref = json.loads((args.out / 'reference.json').read_text())
    common = {'model': 'reranker'}
    score = request(args.base, '/score', {**common, 'queries': QUERY, 'documents': DOCS}, args.out / 'score')
    rerank = request(args.base, '/rerank', {**common, 'query': QUERY, 'documents': DOCS, 'top_n': 3}, args.out / 'rerank')
    classify = request(args.base, '/classify', {**common, 'input': ref['prompts'], 'add_special_tokens': False}, args.out / 'classify')
    raw = request(args.base, '/classify', {**common, 'input': ref['prompts'], 'add_special_tokens': False, 'use_activation': False}, args.out / 'classify-raw')
    checks = {}
    for name, r in [('score',score),('rerank',rerank),('classify',classify),('raw',raw)]:
        checks[name + '_200'] = r['status'] == 200
    if all(checks.values()):
        scores = [d['score'] for d in score['response']['data']]
        probs = [d['probs'][0] for d in classify['response']['data']]
        logits = [d['probs'][0] for d in raw['response']['data']]
        prob_error = max(abs(a-b) for a,b in zip(scores,ref['probabilities']))
        logit_error = max(abs(a-b) for a,b in zip(logits,ref['logit_differences']))
        checks.update(probability_close=prob_error <= 0.02, logit_close=logit_error <= 0.25,
                      score_classify_close=max(abs(a-b) for a,b in zip(scores,probs)) <= 1e-5,
                      activation_close=max(abs(1/(1+math.exp(-a))-b) for a,b in zip(logits,probs)) <= 1e-5,
                      top_document=rerank['response']['results'][0]['index'] == 0,
                      token_count_matches=score['response']['usage']['prompt_tokens'] == sum(ref['token_lengths']),
                      rerank_order=[d['index'] for d in rerank['response']['results']] == sorted(range(len(scores)),key=lambda i:-scores[i])[:3])
    else:
        prob_error = logit_error = None
    invalid = request(args.base,'/score',{**common,'queries':['one','two'],'documents':['a','b','c']},args.out/'invalid-lengths')
    checks['unequal_pair_lengths_rejected'] = invalid['status'] == 400
    # Warm-ups excluded; sequential HTTP requests, each with N documents.
    scan=[]
    for n in [1,4,16,64]:
        docs=[DOCS[i % len(DOCS)] for i in range(n)]
        payload={**common,'query':QUERY,'documents':docs}
        for rep in range(5):
            r=request(args.base,'/rerank',payload,args.out/f'scan-n{n}-r{rep}')
            scan.append({'documents':n,'repeat':rep,'warmup':rep<2,'status':r['status'],
                         'client_ms':r['client_ms'], 'usage':r['response'].get('usage')})
    checks['scan_all_200']=all(r['status']==200 for r in scan)
    save(args.out/'audit.json', {'checks':checks,'max_probability_error':prob_error,
        'max_logit_error':logit_error,'scan':scan,
        'scan_median_ms':{str(n):statistics.median(r['client_ms'] for r in scan if r['documents']==n and not r['warmup']) for n in [1,4,16,64]}})
    if not all(checks.values()):
        raise SystemExit('contract check failed; raw responses preserved')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['reference','online'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:8124')
    args=p.parse_args()
    {'reference':reference,'online':online}[args.stage](args)
