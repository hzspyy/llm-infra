#!/usr/bin/env python3
"""0.0b A/B/C: manual shared gradients, token-normalized AdamW, failure evidence.

python labs/L0/gradient_contract.py --output results/local/0.0b/<new-run>
"""
import argparse,copy,hashlib,json,platform
from pathlib import Path
import torch
import torch.nn.functional as F
from tiny_lm import TinyLM


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def error(x,y):return (x-y).abs().max().item()


def manual(seed):
    gen=torch.Generator().manual_seed(seed)
    x=torch.randn(2,3,generator=gen,dtype=torch.float64,requires_grad=True)
    w=torch.randn(3,3,generator=gen,dtype=torch.float64,requires_grad=True)
    target=torch.tensor([0,2]);h=x@w.T;z=h@w+x
    loss=F.cross_entropy(z,target);loss.backward()
    with torch.no_grad():
        dz=z.softmax(-1);dz[range(2),target]-=1;dz/=2
        dh=dz@w.T;dw=dh.T@x+h.T@dz;dx=dh@w+dz
        eps=1e-6;fd=torch.zeros_like(w)
        for i in range(3):
            for j in range(3):
                plus=w.detach().clone();minus=plus.clone();plus[i,j]+=eps;minus[i,j]-=eps
                fd[i,j]=(F.cross_entropy((x@plus.T)@plus+x,target)-F.cross_entropy((x@minus.T)@minus+x,target))/(2*eps)
    return dict(seed=seed,x=x.detach().tolist(),w=w.detach().tolist(),target=target.tolist(),loss=loss.item(),
        manual_dw=dw.tolist(),autograd_dw=w.grad.tolist(),finite_difference_dw=fd.tolist(),
        manual_dx=dx.tolist(),autograd_dx=x.grad.tolist(),
        max_dw_error=error(dw,w.grad),max_dx_error=error(dx,x.grad),max_fd_error=error(fd,w.grad),epsilon=eps)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    save(a.output/'cases.json',dict(seeds=[0,1,2],effective_lengths=[3,7],raw_lengths=[4,8],
         modes=['batch_token_mean','accumulate_tokens','mean_of_sample_means','unshifted_labels']))
    save(a.output/'manifest.json',dict(torch=torch.__version__,torch_git=torch.version.git_version,python=platform.python_version(),
        dtype='float64',device='cpu',task_ids=['0.0b-A','0.0b-B','0.0b-C'],model='TinyLM(V=11,d=8,H=2,L=1,block=8)',
        optimizer=dict(name='AdamW',lr=0.001,betas=[0.9,0.999],eps=1e-8,weight_decay=0.01,foreach=False),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tiny_lm_sha256=hashlib.sha256((Path(__file__).parent/'tiny_lm.py').read_bytes()).hexdigest(),
        timing='none',tolerances=dict(manual=1e-12,finite_difference=1e-7,gradient_and_parameter=1e-10)))
    gradients=[manual(s) for s in range(3)];save(a.output/'manual.json',gradients)
    rows=[]
    for seed in range(3):
        torch.manual_seed(seed);base=TinyLM(11,d_model=8,n_head=2,n_layer=1,block_size=8).double().eval()
        sequences=[torch.randint(0,11,(n+1,)) for n in [3,7]]
        inputs=torch.zeros(2,7,dtype=torch.long);labels=torch.full((2,7),-100,dtype=torch.long)
        for i,tokens in enumerate(sequences):inputs[i,:len(tokens)-1]=tokens[:-1];labels[i,:len(tokens)-1]=tokens[1:]
        before=copy.deepcopy(base.state_dict());runs={}
        for mode in ['batch_token_mean','accumulate_tokens','mean_of_sample_means','unshifted_labels']:
            m=copy.deepcopy(base);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,foreach=False)
            losses=[]
            if mode in ['batch_token_mean','unshifted_labels']:
                used_labels=labels.clone()
                if mode=='unshifted_labels':used_labels[labels!=-100]=inputs[labels!=-100]
                per=F.cross_entropy(m(inputs).reshape(-1,11),used_labels.reshape(-1),reduction='none',ignore_index=-100).reshape(2,7)
                loss=per.sum()/10;loss.backward();losses=per.detach().tolist()
            else:
                total=0.
                for tokens in sequences:
                    n=len(tokens)-1;per=F.cross_entropy(m(tokens[:-1][None]).reshape(-1,11),tokens[1:],reduction='none')
                    loss=per.sum()/10 if mode=='accumulate_tokens' else per.mean()/2
                    total+=loss.item();loss.backward();losses.append(per.detach().tolist())
                loss=torch.tensor(total)
            grads={n:p.grad.clone() for n,p in m.named_parameters()};opt.step()
            after=copy.deepcopy(m.state_dict());state={n:{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in opt.state[p].items()} for n,p in m.named_parameters()}
            torch.save(dict(before=before,gradients=grads,after=after,optimizer_state=state),a.output/f'state-{seed}-{mode}.pt')
            runs[mode]=dict(loss=float(loss),per_token_losses=losses,gradients=grads,after=after,state=state)
        reference=runs['batch_token_mean'];deltas={}
        for name,r in runs.items():
            deltas[name]=dict(loss=r['loss'],per_token_losses=r['per_token_losses'],
                max_gradient_delta=max(error(g,reference['gradients'][n]) for n,g in r['gradients'].items()),
                max_parameter_delta=max(error(v,reference['after'][n]) for n,v in r['after'].items()),
                max_exp_avg_delta=max(error(v['exp_avg'],reference['state'][n]['exp_avg']) for n,v in r['state'].items()),
                max_exp_avg_sq_delta=max(error(v['exp_avg_sq'],reference['state'][n]['exp_avg_sq']) for n,v in r['state'].items()))
        rows.append(dict(seed=seed,sequences=[s.tolist() for s in sequences],inputs=inputs.tolist(),labels=labels.tolist(),effective_tokens=10,modes=deltas))
    save(a.output/'accumulation.json',rows)
    failures=[]
    for kind in ['inplace_saved','repeat_backward']:
        x=torch.tensor([2.],dtype=torch.float64,requires_grad=True);y=x.square().sum()
        try:
            if kind=='inplace_saved':
                with torch.no_grad():x.add_(1)
            else:y.backward()
            y.backward()
        except RuntimeError as exc:failures.append(dict(case=kind,error=str(exc)))
    m=copy.deepcopy(base);handle=m.tok_emb.register_forward_hook(lambda mod,inp,out:out.detach())
    F.cross_entropy(m(inputs).reshape(-1,11),labels.reshape(-1),ignore_index=-100).backward();handle.remove()
    failures.append(dict(case='detach_embedding',embedding_grad_none=m.tok_emb.weight.grad is None,lm_head_grad_present=m.lm_head.weight.grad is not None))
    save(a.output/'failures.json',failures)
    ok=all(r['max_dw_error']<1e-12 and r['max_dx_error']<1e-12 and r['max_fd_error']<1e-7 for r in gradients)
    ok=ok and all(r['modes']['accumulate_tokens']['max_gradient_delta']<1e-10 and r['modes']['accumulate_tokens']['max_parameter_delta']<1e-10 for r in rows)
    ok=ok and all(r['modes']['mean_of_sample_means']['max_gradient_delta']>1e-6 for r in rows) and len(failures)==3
    save(a.output/'summary.json',dict(passed=ok,manual_cases=len(gradients),accumulation_seeds=len(rows),failures=failures,
        max_finite_difference_error=max(r['max_fd_error'] for r in gradients)))
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n');print(json.dumps({'passed':ok,'max_fd_error':max(r['max_fd_error'] for r in gradients)}));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
