#!/usr/bin/env python3
"""M1 SDPA branch evidence, tiny CUDA inputs and CPU FP64 oracle; no timing claims."""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils._python_dispatch import TorchDispatchMode

class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.ops=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.ops.append(str(func)); return func(*args,**(kwargs or {}))

def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[dict(seed=s,backend=b,dtype=d,prediction=p) for s in range(3) for b,d,p in
           [('math','float32','math'),('default','bfloat16','flash'),('flash','bfloat16','flash'),('flash','float32','reject')]]
    save(a.output/'cases.json',cases)
    save(a.output/'manifest.json',dict(task='M1-A/B SDPA',torch=torch.__version__,torch_git=torch.version.git_version,
        cuda=torch.version.cuda,python=sys.version,device=torch.cuda.get_device_name(),shape=[1,2,7,8],
        dropout=0.0,causal=True,warmup=1,repeats=1,timing='none; profiler trace for branch identity only',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256((a.output/'cases.json').read_bytes()).hexdigest(),
        input_generation='CPU torch.Generator with case seed, then dtype conversion; saved converted values',
        tolerances={'float32':{'atol':1e-5,'rtol':1e-5},'bfloat16':{'atol':0.03,'rtol':0.02}},
        weights='none',working_set='tiny; no bandwidth/cache claim'))
    rows=[]
    for i,c in enumerate(cases):
        dtype=getattr(torch,c['dtype']); gen=torch.Generator().manual_seed(c['seed'])
        cpu=[torch.randn(1,2,7,8,generator=gen).to(dtype) for _ in range(3)]
        q,k,v=[x.double() for x in cpu]
        scores=q@k.transpose(-1,-2)/(8**0.5)
        scores.masked_fill_(~torch.ones(7,7,dtype=torch.bool).tril(),float('-inf'))
        ref=scores.softmax(-1)@v
        gpu=[x.cuda() for x in cpu]
        row=dict(c,inputs=[x.float().tolist() for x in cpu],stride=[list(x.stride()) for x in gpu])
        backends={'math':[SDPBackend.MATH],'flash':[SDPBackend.FLASH_ATTENTION],
                  'default':[SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH,SDPBackend.CUDNN_ATTENTION]}[c['backend']]
        # Default leaves backend priorities as installed, not forcing our list order.
        from contextlib import nullcontext
        def context(): return nullcontext() if c['backend']=='default' else sdpa_kernel(backends)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                with context(),torch.no_grad():
                    F.scaled_dot_product_attention(*gpu,is_causal=True); torch.cuda.synchronize()
                    with Trace() as trace: out=F.scaled_dot_product_attention(*gpu,is_causal=True)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        plain=F.scaled_dot_product_attention(*gpu,is_causal=True); torch.cuda.synchronize()
                prof.export_chrome_trace(str(a.output/f'trace-{i}.json'))
                actual=out.double().cpu()
                tol=(1e-5,1e-5) if dtype==torch.float32 else (0.03,0.02)
                row.update(dispatch=trace.ops,kernels=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA],
                    output=actual.tolist(),reference=ref.tolist(),max_abs_error=(actual-ref).abs().max().item(),
                    numeric_ok=torch.allclose(actual,ref,atol=tol[0],rtol=tol[1]),observer_equal=torch.equal(out,plain),
                    actual='flash' if any('flash_attention' in x for x in trace.ops) else 'other')
                if c['backend']=='math': row['actual']='math' if any('bmm' in x for x in trace.ops) else row['actual']
            except Exception as e: row.update(actual='reject',error_type=type(e).__name__,error=str(e))
            row['warnings']=[str(w.message) for w in caught]
        row['prediction_correct']=row['prediction']==row['actual']; rows.append(row)
        save(a.output/'observations.json',rows)
    # Prediction failures remain evidence, not numerical failures.
    ok=all(r.get('numeric_ok',r['prediction']=='reject') and r.get('observer_equal',True) for r in rows)
    save(a.output/'summary.json',dict(cases=len(rows),predictions_correct=sum(r['prediction_correct'] for r in rows),
        outputs_checked=sum('numeric_ok' in r for r in rows),numeric_pass=sum(r.get('numeric_ok',False) for r in rows),
        rejections=sum(r['actual']=='reject' for r in rows)))
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print((a.output/'summary.json').read_text()); raise SystemExit(0 if ok else 1)
if __name__=='__main__': main()
