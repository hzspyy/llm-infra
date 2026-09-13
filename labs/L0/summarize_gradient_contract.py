#!/usr/bin/env python3
"""0.0b-B: derived statistics recomputed from a saved gradient_contract run.

python labs/L0/summarize_gradient_contract.py \
    --run results/local/0.0b/<run> --output results/local/0.0b/<new-run>
"""
import argparse,hashlib,json
from pathlib import Path
import torch

MODES=['batch_token_mean','accumulate_tokens','mean_of_sample_means','unshifted_labels']


def flat(d,skip_buffers):
    """Concatenate every floating-point tensor of a state dict in a fixed order."""
    return torch.cat([v.flatten() for k,v in sorted(d.items())
                      if isinstance(v,torch.Tensor) and v.is_floating_point() and k not in skip_buffers])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.run/'accumulation.json').read_text())
    lr=json.loads((a.run/'manifest.json').read_text())['optimizer']['lr']
    seeds=[]
    for row in source:
        ref=torch.load(a.run/f"state-{row['seed']}-batch_token_mean.pt",weights_only=False)
        buffers={k for k in ref['before'] if k not in ref['gradients']}
        trained=flat(ref['after'],buffers)-flat(ref['before'],buffers)
        per_token=torch.tensor(row['modes']['batch_token_mean']['per_token_losses'],dtype=torch.float64)
        valid=torch.tensor(row['labels'])!=-100
        modes={}
        for mode in MODES:
            run=torch.load(a.run/f'state-{row["seed"]}-{mode}.pt',weights_only=False)
            deltas=torch.cat([(g-ref['gradients'][n]).flatten().abs() for n,g in run['gradients'].items()])
            scale=max(g.abs().max().item() for g in ref['gradients'].values())
            flips=sum(int(((g.sign()*ref['gradients'][n].sign())<0).sum()) for n,g in run['gradients'].items())
            step=flat(run['after'],buffers)-flat(ref['after'],buffers)
            modes[mode]=dict(loss=row['modes'][mode]['loss'],
                max_gradient_delta=deltas.max().item(),
                max_gradient_delta_relative=deltas.max().item()/scale,
                sign_flips=flips,parameter_count=int(deltas.numel()),
                sign_flip_fraction=flips/deltas.numel(),
                max_parameter_delta=step.abs().max().item(),
                median_parameter_delta=step.abs().median().item(),
                parameter_fraction_above_lr=float((step.abs()>lr).float().mean()))
        # AdamW first step: active elements move about one lr, zero-gradient ones
        # only get decoupled weight decay, w <- (1 - lr*wd)w.
        wd=json.loads((a.run/'manifest.json').read_text())['optimizer']['weight_decay']
        grad=torch.cat([g.flatten() for _,g in sorted(ref['gradients'].items())])
        before=flat(ref['before'],buffers);zero=grad==0
        decay=dict(weight_decay=wd,zero_gradient_elements=int(zero.sum()),total_elements=int(grad.numel()),
            zero_gradient_per_parameter={n:int((g==0).sum()) for n,g in sorted(ref['gradients'].items()) if (g==0).any()},
            max_error_vs_decay_only=(trained[zero].abs()-before[zero].abs()*lr*wd).abs().max().item(),
            active_step_over_lr=[(trained[~zero].abs().min()/lr).item(),(trained[~zero].abs().max()/lr).item()])
        seeds.append(dict(seed=row['seed'],effective_tokens=row['effective_tokens'],weight_decay_only=decay,
            valid_per_row=valid.sum(1).tolist(),
            per_token_loss_sum=per_token[valid].sum().item(),
            loss_from_per_token=(per_token[valid].sum()/row['effective_tokens']).item(),
            loss_reported=row['modes']['batch_token_mean']['loss'],
            first_step_abs_min=trained.abs().min().item(),
            first_step_abs_max=trained.abs().max().item(),
            first_step_relative_to_lr=[(trained.abs().min()/lr).item(),(trained.abs().max()/lr).item()],
            modes=modes))
    summary=dict(learning_rate=lr,run=str(a.run),seeds=[s['seed'] for s in seeds],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accumulation_sha256=hashlib.sha256((a.run/'accumulation.json').read_bytes()).hexdigest(),
        loss_reconstruction_max_error=max(abs(s['loss_from_per_token']-s['loss_reported']) for s in seeds),
        equivalent_modes=[m for m in MODES if all(s['modes'][m]['max_gradient_delta']<1e-12 for s in seeds)],
        divergent_modes=[m for m in MODES if any(s['modes'][m]['max_gradient_delta']>1e-6 for s in seeds)],
        first_step_relative_to_lr=[min(s['first_step_relative_to_lr'][0] for s in seeds),
                                   max(s['first_step_relative_to_lr'][1] for s in seeds)],
        active_step_over_lr=[min(s['weight_decay_only']['active_step_over_lr'][0] for s in seeds),
                             max(s['weight_decay_only']['active_step_over_lr'][1] for s in seeds)],
        max_error_vs_decay_only=max(s['weight_decay_only']['max_error_vs_decay_only'] for s in seeds))
    (a.output/'derived.json').write_text(json.dumps(dict(summary=summary,seeds=seeds),indent=2)+'\n')
    ok=(summary['loss_reconstruction_max_error']<1e-12
        and summary['equivalent_modes']==['batch_token_mean','accumulate_tokens']
        and summary['divergent_modes']==['mean_of_sample_means','unshifted_labels']
        and summary['max_error_vs_decay_only']<1e-15)
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(summary,indent=2));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
