#!/usr/bin/env python3
"""L5.9 actual installed sampling functions and engine logprobs benchmark."""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')


def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)


def emit(kind,**kw):print(json.dumps({'kind':kind,**kw}),flush=True)


def version():
    data={}
    for name in ['vllm','sglang','torch','flashinfer-python','transformers']:
        try:data[name]=md.version(name)
        except md.PackageNotFoundError:pass
    return data


def filters(out,engine):
    import torch
    cases=[('topk_then_topp',[.5,.3,.15,.05],2,.6,0.),
           ('minp_then_topp',[.4,.3,.2,.1],4,.55,.6),
           ('ties',[1.,1.,1.,.1],2,1.,0.)]
    rows=[]
    for label,weights,k,p,m in cases:
        probs=torch.tensor([weights],dtype=torch.float32)
        probs/=probs.sum(-1,keepdim=True)
        logits=probs.log()
        if engine=='vllm':
            from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
            from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
            if m:
                stub=SimpleNamespace(min_p_count=1,min_p=torch.tensor([[m]]))
                logits=MinPLogitsProcessor.apply(stub,logits)
            result=apply_top_k_top_p_pytorch(logits,torch.tensor([k]),torch.tensor([p]))
            post=result.softmax(-1)
        else:
            from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
            orig=torch.multinomial
            captured=[]
            def record(weights,*a,**kw):
                captured.append(weights.clone())
                return orig(weights,*a,**kw)
            torch.multinomial=record
            try:
                top_k_top_p_min_p_sampling_from_probs_torch(probs.clone(),torch.tensor([k]),
                    torch.tensor([p]),torch.tensor([m]),bool(m),None,torch.tensor([0]))
            finally:torch.multinomial=orig
            _,idx=probs.sort(descending=True)
            post=torch.zeros_like(probs).scatter(1,idx,captured[0])
            post/=post.sum(-1,keepdim=True)
        row={'engine':engine,'case':label,'input_probs':probs.tolist()[0],
             'k':k,'p':p,'min_p':m,'final_probs':post.tolist()[0],
             'allowed_ids':torch.nonzero(post[0]>0).flatten().tolist()}
        rows.append(row);emit('filter',**row)
    save(out/'filters.json',rows)


def gpu(out):
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler,apply_top_k_top_p,apply_top_k_top_p_pytorch,random_sample
    from vllm.v1.sample.sampler import Sampler
    from vllm.model_executor.layers.utils import apply_penalties
    from mini_sampler import process
    device='cuda'
    torch.manual_seed(7)
    raw=torch.tensor([[-1.,2.,1.,.5]],device=device)
    penalized=apply_penalties(raw.clone(),torch.tensor([[0,1]],device=device),
          torch.tensor([[3,3,1]],device=device),torch.tensor([.5],device=device),
          torch.tensor([.25],device=device),torch.tensor([1.5],device=device))
    expected,_=process(raw[0].tolist(),prompt=[0,1],output=[3,3,1],repetition=1.5,frequency=.25,presence=.5)
    assert torch.allclose(penalized,torch.tensor([expected['penalties']],device=device),atol=1e-6)
    emit('penalties',input=raw.tolist(),output=penalized.tolist(),reference=expected['penalties'])
    sampler=TopKTopPSampler('raw_logprobs')
    processed=TopKTopPSampler('processed_logprobs')
    emit('dispatch',raw_forward=sampler.forward.__name__,processed_forward=processed.forward.__name__)
    # Distribution-level check, not equality of random draws.
    probs=torch.tensor([[.5,.3,.15,.05]],device=device).repeat(30000,1)
    sample=random_sample(probs.clone(),{})
    counts=torch.bincount(sample,minlength=4).cpu().tolist()
    expected_counts=[15000,9000,4500,1500]
    for obs,p,exp in zip(counts,[.5,.3,.15,.05],expected_counts):
        assert abs(obs-exp) < 6*(30000*p*(1-p))**.5
    emit('categorical',n=30000,counts=counts,expected=expected_counts,within_6sigma=True)
    # Compare filtering supports on non-tied random logits across dispatch boundary.
    V=151936;records=[]
    for B in [1,7,8,64,256]:
        x=torch.randn((B,V),device=device,dtype=torch.float32)
        k=torch.full((B,),50,device=device,dtype=torch.int32)
        p=torch.full((B,),.9,device=device)
        a=apply_top_k_top_p(x.clone(),k,p)
        b=apply_top_k_top_p_pytorch(x.clone(),k,p)
        disagreements=int((torch.isfinite(a)!=torch.isfinite(b)).sum())
        emit('support_check',batch=B,disagreements=disagreements)
        if disagreements:
            mismatch=[]
            locations=torch.nonzero(torch.isfinite(a)!=torch.isfinite(b)).cpu().tolist()
            for row,col in locations:
                values,indices=x[row].topk(55)
                top_probs=values[:50].softmax(-1)
                rank=(indices==col).nonzero().flatten().tolist()
                mismatch.append({'row':row,'token_id':col,'rank_in_top55_zero_based':rank,
                    'dispatch_kept':bool(torch.isfinite(a[row,col])),
                    'sort_kept':bool(torch.isfinite(b[row,col])),
                    'top55_logits':values.cpu().tolist(),'top55_ids':indices.cpu().tolist(),
                    'top50_cumulative':top_probs.cumsum(-1).cpu().tolist()})
            save(out/f'mismatch-B{B}.json',mismatch)
            emit('support_mismatch_preserved',batch=B,count=disagreements)
        ids=x.argmax(-1).long()
        def with_logprobs(n):
            z=x.clone();s=z.argmax(-1)
            return Sampler.gather_logprobs(Sampler.compute_logprobs(z),n,s)
        arms={'clone':lambda:x.clone(),'greedy':lambda:x.clone().argmax(-1),
          'filter_dispatch':lambda:apply_top_k_top_p(x.clone(),k,p),
          'native_sampling':lambda:sampler.forward_native(x.clone(),{},k,p),
          'selected_sampling':lambda:sampler(x.clone(),{},k,p),
          'processed_sampling':lambda:processed(x.clone(),{},k,p),
          'greedy_logprobs0':lambda:with_logprobs(0),
          'greedy_logprobs5':lambda:with_logprobs(5),
          'greedy_logprobs20':lambda:with_logprobs(20)}
        for fn in arms.values():
            for _ in range(3):fn()
        torch.cuda.synchronize()
        samples={name:[] for name in arms}
        names=list(arms)
        for rep in range(5):
            for name in names[rep:]+names[:rep]:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):arms[name]()
                end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/10)
        for name,vals in samples.items():
            row={'batch':B,'vocab':V,'dtype':'float32','shape':[B,V],'stride':[V,1],
                 'input_bytes':x.numel()*x.element_size(),'input_fits_L2_96MiB':x.numel()*4<=96*1024**2,
                 'arm':name,'filter_support_disagreements':disagreements,'microseconds':vals,'median_us':statistics.median(vals)}
            records.append(row);emit('timing',**row)
    save(out/'gpu.json',records)


def engine(out):
    import torch
    from vllm import LLM,SamplingParams
    util=float(os.environ['L59_GPU_MEMORY_UTILIZATION'])
    cfg=dict(model='Qwen/Qwen3-1.7B',gpu_memory_utilization=util,enforce_eager=True,
             max_model_len=1024,max_num_seqs=32,enable_prefix_caching=False,disable_log_stats=True)
    llm=LLM(**cfg)
    prompts=['Write a short paragraph about GPU memory and scheduling.']*8
    arms=[None,0,1,5,20];rows=[];details=[]
    try:
        for n in arms:
            llm.generate(prompts,SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,logprobs=n),use_tqdm=False)
        for rep in range(3):
            for n in arms[rep:]+arms[:rep]:
                t=time.perf_counter()
                outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,logprobs=n),use_tqdm=False)
                ms=(time.perf_counter()-t)*1000
                tok=[list(o.outputs[0].token_ids) for o in outputs]
                returned=[]
                for o in outputs:
                    lp=o.outputs[0].logprobs
                    returned.append(None if lp is None else [
                      {str(k):{'logprob':v.logprob,'rank':v.rank,'decoded_token':v.decoded_token} for k,v in item.items()}
                      for item in lp])
                payload=json.dumps(returned,ensure_ascii=False).encode()
                count=sum(len(step) for per in returned if per is not None for step in per)
                row={'rep':rep,'logprobs':n,'batch':8,'tokens':sum(map(len,tok)),'wall_ms':ms,
                     'candidate_records':count,'serialized_logprobs_bytes':len(payload)}
                rows.append(row);details.append({'rep':rep,'logprobs':n,'tokens':tok,'values':returned});emit('engine',**row)
        assert all(d['tokens']==details[0]['tokens'] for d in details)
        emit('same_tokens_all_logprobs_arms',value=True)
    finally:
        save(out/'engine-summary.json',rows);save(out/'engine-outputs.json',details)
        save(out/'engine-config.json',cfg)
        llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['vllm','sglang','gpu','engine']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'versions.json',version())
    if a.mode in ['vllm','sglang']:filters(a.out,a.mode)
    elif a.mode=='gpu':gpu(a.out)
    else:engine(a.out)
