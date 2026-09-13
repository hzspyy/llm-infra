#!/usr/bin/env python3
"""M1: freeze predictions before probing linear; retain observations and inputs.

python labs/M/dispatch_evidence.py --output results/local/M1/<new-run>
CPU FP64 correctness/dispatch evidence only; no performance claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from version_manifest import capture_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def describe(x):
    return {'shape': list(x.shape), 'stride': list(x.stride()),
            'dtype': str(x.dtype), 'values': x.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cases = []
    for seed in range(3):
        for layout in ['matrix', 'cube', 'strided_cube', 'empty', 'zero', 'large']:
            for bias in [False, True]:
                # Static prediction: compare addmm fast path with matmul fallback.
                expected = 'aten.addmm.default' if bias and layout != 'strided_cube' else 'aten.mm.default'
                cases.append({'id': f'{seed}-{layout}-{bias}', 'seed': seed,
                              'layout': layout, 'bias': bias, 'expected_gemm': expected,
                              'naive_prediction': 'aten.addmm.default'})
    write(args.output / 'cases.json', cases)  # Stored BEFORE execution.
    manifest = capture_environment()
    manifest.update(task_ids=['M1-A', 'M1-B'], device='cpu', dtype='float64',
                    atol=1e-10, rtol=1e-12, timing='none; correctness and operator observation only',
                    warmup=0, repetitions=1, seeds=[0, 1, 2],
                    cases_sha256=digest(args.output / 'cases.json'),
                    script_sha256=digest(Path(__file__)),
                    model='none; synthetic linear', backend='PyTorch CPU; BLAS kernel identity not measured')
    write(args.output / 'manifest.json', manifest)
    rows = []
    for case in cases:
        torch.manual_seed(case['seed'])
        layout = case['layout']
        x = torch.randn(2, 3, dtype=torch.float64)
        if layout in ('cube', 'strided_cube'):
            x = torch.randn(2, 3, 3, dtype=torch.float64)
            if layout == 'strided_cube':
                x = x.transpose(0, 1)
        elif layout == 'empty':
            x = x[:0]
        elif layout == 'zero':
            x.zero_()
        elif layout == 'large':
            x *= 1e4
        w = torch.randn(4, 3, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64) if case['bias'] else None
        # Scalar reference avoids invoking the same linear/matmul implementation.
        reference = torch.empty((*x.shape[:-1], 4), dtype=torch.float64)
        for j, row in enumerate(x.reshape(-1, 3).tolist()):
            for o in range(4):
                reference.reshape(-1, 4)[j, o] = sum(row[k] * w[o, k].item() for k in range(3)) + (b[o].item() if b is not None else 0)
        module = torch.nn.Linear(3, 4, bias=b is not None, dtype=torch.float64)
        with torch.no_grad():
            module.weight.copy_(w)
            if b is not None:
                module.bias.copy_(b)
        hooks = []
        handle = module.register_forward_hook(lambda m, inp, out: hooks.append(type(m).__name__))
        with torch.no_grad(), Trace() as trace:
            actual = module(x)
        handle.remove()
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            plain = module(x)
        record = dict(case, inputs={'x': describe(x), 'w': describe(w), 'b': describe(b) if b is not None else None},
                      module_hooks=hooks, dispatch=trace.ops,
                      profiler=[event.key for event in prof.key_averages()],
                      output=actual.tolist(), reference=reference.tolist(),
                      max_abs_error=float((actual-reference).abs().max()) if actual.numel() else 0,
                      numerics_ok=torch.allclose(actual, reference, atol=1e-10, rtol=1e-12),
                      observer_output_equal=torch.equal(actual, plain),
                      prediction_correct=case['expected_gemm'] in trace.ops,
                      naive_prediction_correct=case['naive_prediction'] in trace.ops)
        rows.append(record)
    write(args.output / 'observations.json', rows)
    write(args.output / 'registration.json', {'schema': str(torch.ops.aten.linear.default._schema),
          'table': torch._C._dispatch_dump_table('aten::linear'),
          'binding_is_C_nn_linear': F.linear is torch._C._nn.linear})
    summary = {'cases': len(rows), 'numeric_pass': sum(r['numerics_ok'] for r in rows),
               'prediction_pass': sum(r['prediction_correct'] for r in rows),
               'naive_prediction_fail': sum(not r['naive_prediction_correct'] for r in rows),
               'observer_output_equal': all(r['observer_output_equal'] for r in rows),
               'max_abs_error': max(r['max_abs_error'] for r in rows)}
    write(args.output / 'summary.json', summary)
    ok = all(r['numerics_ok'] and r['prediction_correct'] and r['observer_output_equal'] for r in rows)
    (args.output / 'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
