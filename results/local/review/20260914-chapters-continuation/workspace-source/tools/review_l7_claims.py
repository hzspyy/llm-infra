"""Small CPU counterexamples for the 2026-09-14 chapter review.

Writes only to a caller-selected NEW directory; never runs chapter main() functions.
Run with the project's Python and TMPDIR on /Volumes/data.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'labs/L7' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'torch_version': torch.__version__, 'torch_git': torch.version.git_version,
              'device': 'CPU', 'seed': 0, 'purpose': 'counterexamples, not GPU validation'}
    torch.manual_seed(0)
    dtypes = []
    for dtype in (torch.float32, torch.bfloat16):
        model = torch.nn.Linear(2, 2, bias=False).to(dtype)
        opt = torch.optim.AdamW(model.parameters())
        x = torch.ones(1, 2, dtype=dtype)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y = model(x)
            loss = y.float().square().mean()
        loss.backward()
        opt.step()
        p = model.weight
        dtypes.append({'parameter': str(p.dtype), 'output': str(y.dtype),
                       'gradient': str(p.grad.dtype),
                       'm': str(opt.state[p]['exp_avg'].dtype),
                       'v': str(opt.state[p]['exp_avg_sq'].dtype)})
    result['actual_dtype_ledger'] = dtypes
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=1.0)
    scaler.scale(p.square()).backward()
    before = p.item()
    ret = scaler.step(opt)
    scaler.update()
    successful = {'return': ret, 'before': before, 'after': p.item()}
    opt.zero_grad()
    scaler.scale(p.square()).backward()
    p.grad.fill_(float('inf'))
    ret_skip = scaler.step(opt)
    scaler.update()
    result['scaler_return_and_scale'] = {'successful': successful, 'skipped_return': ret_skip,
                                        'scale_after_overflow': scaler.get_scale()}
    result['rounding'] = {str(dtype): {str(d): float(torch.tensor(1., dtype=dtype) -
                                                  torch.tensor(d, dtype=dtype))
                                     for d in (0.0039, 0.00048, 1e-8)}
                          for dtype in (torch.float32, torch.float16, torch.bfloat16)}
    ratio = torch.tensor(10., dtype=torch.float64, requires_grad=True)
    advantage = -1.
    loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
    loss.backward()
    result['ppo_negative_advantage'] = {'ratio': ratio.item(), 'loss': loss.item(),
                                        'gradient_wrt_ratio': ratio.grad.item()}
    p = torch.nn.Parameter(torch.tensor(1.))
    try:
        torch.optim.AdamW([{'params': [p]}, {'params': [p]}])
    except ValueError as error:
        result['duplicate_optimizer_groups'] = str(error)
    theta = torch.tensor(0., requires_grad=True)
    big_loss = theta * .001 + 3.5
    small_loss = theta * 10 + .015
    result['loss_value_not_gradient'] = {
        'losses': [big_loss.item(), small_loss.item()],
        'gradients': [torch.autograd.grad(big_loss, theta)[0].item(),
                      torch.autograd.grad(small_loss, theta)[0].item()]}
    data = load('training_data_contract')
    collator = data.MiniCollator() if hasattr(data, 'MiniCollator') else None
    if collator is None:
        collator = next(v for v in vars(data).values()
                        if inspect.isclass(v) and hasattr(v, 'collate_packing'))()
    packed = collator.collate_packing([
        {'id': 'a', 'input_ids': [1, 2], 'labels': [1, 2]},
        {'id': 'b', 'input_ids': [3, 4], 'labels': [3, 4]}])
    result['packing_shift_boundary'] = {
        'input_ids': packed['input_ids'], 'labels': packed['labels'],
        'cu_seqlens': packed['cu_seqlens'],
        'cross_document_shift_target': packed['labels'][2],
        'required_ignore_index_for_standard_causal_shift': -100}
    mini = load('mini_autograd')
    x = mini.Tensor(np.array([2.]), requires_grad=True)
    y = mini.mul(x, x)
    z = mini.mul(y, y)
    z.backward()
    first = float(x.grad[0])
    z.backward()
    result['mini_backward_reuse'] = {'first': first, 'second_accumulated': float(x.grad[0]),
                                      'expected_accumulated_if_retain_supported': 64.0}
    mm = load('multimodal_train_contract')
    model = mm.UnifiedMultimodalModel()
    images = torch.randn(1, 4, 32)
    tokens = torch.randint(0, 64, (1, 8))
    result['multimodal_input_dependence'] = {
        'text_logits_change_when_image_changes': float((
            model.forward_vlm(images, tokens)[:, 4:] -
            model.forward_vlm(images + 10, tokens)[:, 4:]).abs().max().detach()),
        'flow_changes_when_latent_changes': float((
            model.forward_flow(images, tokens) -
            model.forward_flow(images + 10, tokens)).abs().max().detach())}
    # A size-one process group suffices to verify the API return contract.
    import torch.distributed as dist
    rendezvous = Path(os.environ['TMPDIR']) / ('review-rendezvous-' + str(os.getpid()))
    dist.init_process_group('gloo', init_method=rendezvous.as_uri(), rank=0, world_size=1)
    try:
        tensor = torch.tensor(2.)
        value = dist.all_reduce(tensor)
        result['all_reduce_return'] = {'return': value, 'tensor_after': tensor.item()}
    finally:
        dist.destroy_process_group()
    sources = {}
    for obj in (torch.amp.GradScaler.step, torch.optim.Optimizer.add_param_group,
                torch.amp.GradScaler.update):
        obj = inspect.unwrap(obj)
        path = Path(inspect.getsourcefile(obj))
        sources[obj.__qualname__] = {'path': str(path), 'line': inspect.getsourcelines(obj)[1],
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result['installed_source'] = sources
    (args.output / 'counterexamples.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
