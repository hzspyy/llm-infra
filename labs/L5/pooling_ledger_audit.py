#!/usr/bin/env python3
"""重算容量账本；--check-meta 用 transformers meta 参数逐项核对，不加载权重/GPU。"""
import argparse
import hashlib
import json
from pathlib import Path
from pooling_batch_limit import arch, ledger, max_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check-meta', action='store_true')
    args = p.parse_args()
    report = {'kind': 'formula, not GPU measurement', 'budget_bytes': 30 * 2**30, 'models': {}}
    for kind, revision in [('bert', '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'),
                           ('qwen2', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306')]:
        raw = (args.configs / f'{kind}.json').read_bytes()
        cfg = json.loads(raw)
        a = arch(cfg)
        r = {'config_sha256': hashlib.sha256(raw).hexdigest(), 'revision': revision,
             'architecture': a, 'ledger': {}}
        if args.check_meta:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
            config = AutoConfig.for_model(kind, **{k: v for k, v in cfg.items() if k != 'model_type'})
            cls = AutoModel if kind == 'bert' else AutoModelForCausalLM
            with torch.device('meta'):
                model = cls.from_config(config)
                model.tie_weights()
            count = sum(p.numel() for p in model.parameters())
            r['meta_parameter_count'] = count
            r['transformers'] = transformers.__version__
            assert count == a['parameter_count'], (kind, count, a)
            del model
        mode = 'encode' if kind == 'bert' else 'decode'
        for length in ([512] if kind == 'bert' else [1024, 4096]):
            b = max_batch(a, mode, length, report['budget_bytes'])
            r['ledger'][str(length)] = {'max_batch_formula': b, 'at_max': ledger(a, mode, b, length)}
            assert ledger(a, mode, b, length)['total'] <= report['budget_bytes']
            assert ledger(a, mode, b + 1, length)['total'] > report['budget_bytes']
            # 验证零容量、恰好容纳一条和 dtype 改变。
            assert max_batch(a, mode, length, 0) == 0
            assert max_batch(a, mode, length, ledger(a, mode, 1, length)['total']) == 1
            assert ledger(a, mode, 1, length, 4)['total'] == 2 * ledger(a, mode, 1, length, 2)['total']
        report['models'][kind] = r
    with args.out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
