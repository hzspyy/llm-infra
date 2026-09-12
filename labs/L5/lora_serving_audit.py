#!/usr/bin/env python3
"""L5.10: synthetic adapters through the real vLLM multi-LoRA path.

Adapters are randomly generated system fixtures, not trained models.
No base weights are copied or merged on disk. Run under the serving environment.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING','0')
MODEL='Qwen/Qwen3-1.7B'
TRACE=True
EVENTS=[]
SEEN=set()


def emit(kind,**fields):
    row={'kind':kind,**fields}
    print(json.dumps(row),flush=True)
    EVENTS.append(row)


def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2)


def make_adapters(root):
    import torch
    from transformers import AutoConfig
    from safetensors.torch import save_file
    cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    hidden=cfg.hidden_size
    qout=cfg.num_attention_heads*getattr(cfg,'head_dim',hidden//cfg.num_attention_heads)
    manifest=[]
    for ident,rank,zero in [(1,8,False),(2,8,True),(3,16,False),(4,8,False),(5,32,False)]:
        folder=root/f'adapter-{ident}';folder.mkdir(parents=True)
        config={'base_model_name_or_path':MODEL,'peft_type':'LORA','task_type':'CAUSAL_LM',
                'inference_mode':True,'r':rank,'lora_alpha':rank,'target_modules':['q_proj'],
                'lora_dropout':0.,'bias':'none','use_dora':False}
        save(folder/'adapter_config.json',config)
        generator=torch.Generator().manual_seed(100+ident)
        weights={}
        for layer in range(cfg.num_hidden_layers):
            prefix=f'base_model.model.model.layers.{layer}.self_attn.q_proj'
            weights[prefix+'.lora_A.weight']=(torch.randn(rank,hidden,generator=generator)*.03).bfloat16()
            weights[prefix+'.lora_B.weight']=torch.zeros(qout,rank,dtype=torch.bfloat16) if zero else (torch.randn(qout,rank,generator=generator)*.03).bfloat16()
        path=folder/'adapter_model.safetensors';save_file(weights,str(path),metadata={'format':'pt'})
        raw=path.read_bytes();header_len=int.from_bytes(raw[:8],'little')
        save(folder/'safetensors_header.json',json.loads(raw[8:8+header_len]))
        row={'id':ident,'rank':rank,'zero_B':zero,'seed':100+ident,'config':config,
             'tensor_count':len(weights),'tensor_bytes':sum(t.numel()*t.element_size() for t in weights.values()),
             'file_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
             'sample_shapes':{k:list(v.shape) for k,v in list(weights.items())[:2]}}
        manifest.append(row);emit('adapter',**row)
    save(root/'manifest.json',{'model_revision':cfg._commit_hash,'torch':md.version('torch'),
                             'hidden':hidden,'qout':qout,'layers':cfg.num_hidden_layers,'adapters':manifest})


def instrument():
    from vllm.lora.model_manager import LoRAModelManager,LRUCacheLoRAModelManager
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    old=LRUCacheLoRAModelManager.activate_adapter
    def activate(self,lora_id,*args,**kw):
        before=list(self.lora_index_to_id)
        result=old(self,lora_id,*args,**kw)
        if TRACE and before!=self.lora_index_to_id:
            emit('gpu_slots',requested_id=lora_id,before=before,after=list(self.lora_index_to_id),
                 cpu_ids=sorted(self.list_adapters()))
        return result
    LRUCacheLoRAModelManager.activate_adapter=activate
    old_mapping=LoRAModelManager.set_adapter_mapping
    def mapping(self,value):
        result=old_mapping(self,value)
        key=(tuple(value.index_mapping),tuple(self.lora_index_to_id))
        if TRACE and key not in SEEN and len(SEEN)<16:
            SEEN.add(key)
            emit('mapping',token_adapter_ids=[int(v) for v in value.index_mapping],
                 prompt_adapter_ids=[int(v) for v in value.prompt_mapping],gpu_slots=list(self.lora_index_to_id))
        return result
    LoRAModelManager.set_adapter_mapping=mapping
    old_linear=PunicaWrapperGPU.add_lora_linear
    def linear(self,y,x,aa,bb,scale,output_slices,**kwargs):
        key=('linear',tuple(x.shape),tuple(output_slices))
        if TRACE and key not in SEEN and len(SEEN)<24:
            SEEN.add(key)
            emit('kernel_shapes',x=list(x.shape),x_stride=list(x.stride()),y=list(y.shape),
                 A=[list(t.shape) for t in aa],B=[list(t.shape) for t in bb],
                 scale=scale,output_slices=list(output_slices))
        return old_linear(self,y,x,aa,bb,scale,output_slices,**kwargs)
    PunicaWrapperGPU.add_lora_linear=linear


def run(root, adapter_root=None, trace_only=False):
    global TRACE
    import torch
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    instrument()
    cfg=dict(model=MODEL,enable_lora=True,max_loras=2,max_cpu_loras=3,max_lora_rank=16,
      gpu_memory_utilization=float(os.environ['L510_GPU_MEMORY_UTILIZATION']),
      max_model_len=1024,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=False,disable_log_stats=True)
    save(root/'engine_config.json',cfg)
    TRACE=False
    llm=LLM(**cfg)
    TRACE=True
    SEEN.clear()
    adapter_root=adapter_root or root/'adapters'
    reqs={i:LoRARequest(f'synthetic-{i}',i,str(adapter_root/f'adapter-{i}')) for i in range(1,6)}
    prompt='Write a short paragraph about GPU memory and scheduling.'
    prompts=[prompt]*8
    sp=SamplingParams(temperature=0,max_tokens=24,ignore_eos=True,logprobs=0)
    records=[]
    def generate(label,requests):
        if TRACE:SEEN.clear()
        t=time.perf_counter()
        out=llm.generate(prompts,sp,lora_request=requests,use_tqdm=False)
        duration=(time.perf_counter()-t)*1000
        tokens=[list(x.outputs[0].token_ids) for x in out]
        sample_lp=[next(iter(x.outputs[0].logprobs[0].values())).logprob for x in out]
        row={'label':label,'wall_ms':duration,'tokens':tokens,'first_logprobs':sample_lp,
             'finish_reasons':[x.outputs[0].finish_reason for x in out]}
        records.append(row)
        emit('request_batch',label=label,wall_ms=duration,token_count=sum(map(len,tokens)),
             first_tokens=[t[:4] for t in tokens])
        return tokens
    try:
        base=generate('base',None)
        a=generate('adapter1',reqs[1])
        z=generate('zero2',reqs[2])
        mixed=generate('mixed1_2',[reqs[1] if i%2==0 else reqs[2] for i in range(8)])
        emit('correctness',zero_matches_base=z==base,
             mixed_matches_separate=all(mixed[i]==(a[i] if i%2==0 else z[i]) for i in range(8)),
             nonzero_changes_tokens=a!=base)
        if trace_only:return
        # Touch three GPU IDs with only two physical slots, then four CPU IDs.
        generate('activate3',reqs[3]);generate('reactivate1',reqs[1])
        generate('load4_cpu_eviction',reqs[4]);generate('reload2',reqs[2])
        # End-to-end timings without trace serialization; warm each arm first.
        TRACE=False
        arms=[('base',None),('single1',reqs[1]),('mixed1_2',[reqs[1],reqs[2]]*4)]
        for label,r in arms:generate('warm_'+label,r)
        for rep in range(3):
            for label,r in arms[rep:]+arms[:rep]:generate(f'time_{rep}_{label}',r)
        # Validate a rank mismatch in the real loader, then check whether valid requests can still run.
        try:
            llm.generate(prompts[:1],sp,lora_request=reqs[5],use_tqdm=False)
        except Exception as e:
            emit('rank_error',type=type(e).__name__,message=str(e))
        else:
            emit('rank_error',unexpected_accept=True)
        generate('after_rank_error',reqs[1])
    finally:
        save(root/'requests.json',records);save(root/'events.json',EVENTS)
        llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--adapters',type=Path);p.add_argument('--trace-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.adapters is None:make_adapters(a.out/'adapters')
    run(a.out,a.adapters,a.trace_only)
