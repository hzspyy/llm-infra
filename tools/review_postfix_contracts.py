"""Bounded CPU counterexamples for the 2026-09-14 post-fix chapter review.

Run with the project Python and external-disk TMPDIR. No historical results
are modified. Output must be a new file supplied by the caller.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import torch
from torch.utils.checkpoint import checkpoint


def module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    result = {'torch': torch.__version__, 'git': torch.version.git_version,
              'device': 'CPU', 'seed': 0, 'quality_or_gpu_measured': False}

    text = Path('src/L7/7.0-autograd-anatomy.md').read_text()
    blocks = re.findall(r'```python\n(.*?)```', text, re.S)
    graph_code = next(b for b in blocks if 'def print_grad_graph' in b)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(graph_code, '<chapter graph snippet>', 'exec'), {})
    result['literal_graph_snippet'] = {'code': graph_code, 'stdout': capture.getvalue()}

    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)
    h = x @ w.t()
    try:
        h.grad_fn.saved_tensors
    except AttributeError as error:
        result['builtin_saved_tensors'] = {'error': str(error), 'actual_fields':
            {name: list(getattr(h.grad_fn, name).shape) for name in dir(h.grad_fn)
             if name.startswith('_saved') and isinstance(getattr(h.grad_fn, name), torch.Tensor)}}
    g = torch.ones_like(h)
    try:
        g @ x.t()
    except RuntimeError as error:
        result['matmul_backward_formula'] = {'chapter_error': str(error),
            'correct_shape': list((g.t() @ x).shape), 'weight_shape': list(w.shape)}

    x = torch.tensor([1., 2.], requires_grad=True)
    y = x * 2
    try:
        x.add_(1)
    except RuntimeError as error:
        result['inplace_failure'] = {'stage': 'x.add_(1), before backward', 'error': str(error)}
    with torch.no_grad():
        x.add_(1)
    y.sum().backward()
    result['inplace_failure']['no_grad_mutation_backward_gradient'] = x.grad.tolist()
    x = torch.tensor([1., 2.], requires_grad=True)
    y = x.square()
    with torch.no_grad():
        x.add_(1)
    try:
        y.sum().backward()
    except RuntimeError as error:
        result['actual_saved_value_mutation'] = str(error)

    checkpoint_rows = []
    for enabled in (False, True):
        calls = [0]
        saved = []
        def heavy_function(value):
            calls[0] += 1
            return value * 2
        def pack(value):
            saved.append({'shape': list(value.shape), 'numel': value.numel()})
            return value.detach()
        x = torch.randn(2, 3, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            y = x
            for _ in range(3):
                y = checkpoint(heavy_function, y, use_reentrant=False) if enabled else heavy_function(y)
            calls_before = calls[0]
            y.sum().backward()
        checkpoint_rows.append({'checkpoint': enabled, 'forward_calls': calls_before,
                                'extra_backward_calls': calls[0]-calls_before, 'saved': saved,
                                'input_grad': x.grad.tolist()})
    result['checkpoint_multiply_by_constant'] = checkpoint_rows

    x = torch.randn(2, requires_grad=True)
    y = x + 1
    y.sum().backward()
    y.sum().backward()
    result['no_saved_tensor_repeated_backward'] = {'gradient': x.grad.tolist(), 'node_still_exists': str(y.grad_fn)}

    # Distinguish TensorImpl version counters from storage identity.
    a = torch.ones(2)
    b = torch.empty(0).set_(a.untyped_storage(), 0, a.shape, a.stride())
    before = [a._version, b._version]
    b.add_(1)
    result['same_storage_different_version'] = {'same_ptr': a.data_ptr() == b.data_ptr(),
            'before': before, 'after': [a._version, b._version], 'a': a.tolist(), 'b': b.tolist()}
    x = torch.tensor([1., 2.], requires_grad=True)
    a = x.sin()
    y = a.square()
    with torch.no_grad():
        a.add_(1)
    try:
        y.sum().backward()
    except RuntimeError as error:
        result['saved_value_producer_vs_consumer'] = {'producer': 'SinBackward0',
                'saving_consumer': 'PowBackward0', 'error': str(error)}
    result['aten_linear_exists'] = {'schema': str(torch.ops.aten.linear.default._schema),
            'dispatch': torch._C._dispatch_dump_table('aten::linear')}

    # The event analyzer must not turn per-rank activity extents into a shared step boundary.
    analyzer = module('labs/L7/training_trace_analysis.py')
    payload = {'source_kind': 'synthetic_fixture', 'source': 'shared-clock step [0,100] ms',
        'events': [{'rank': 0, 'category': 'compute', 'start_ms': 0., 'end_ms': 10.},
                   {'rank': 1, 'category': 'compute', 'start_ms': 90., 'end_ms': 100.}],
        'model_flops_per_step_estimate': 1e9, 'peak_tflops_per_device': 1., 'num_devices': 2}
    result['trace_step_boundary'] = {'input': payload, 'actual': analyzer.analyze_events(payload),
                                    'explicit_step_ms': 100., 'explicit_step_mfu_pct': .5}
    # Confirm teaching scalar fields do not validate an impossible launch mesh.
    inspector = module('labs/L7/recipe_inspector.py')
    result['recipe_zero_batch'] = inspector.validate({'kind': 'normalized_contract',
        'micro_batch_size': 0, 'accumulation': 1, 'dp': 2, 'global_batch_size': 0})

    paths = ['tools/review_postfix_contracts.py', 'labs/L7/training_trace_analysis.py',
             'labs/L7/recipe_inspector.py', 'src/L7/7.0-autograd-anatomy.md']
    result['sources'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('aten_linear_exists','literal_graph_snippet')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
