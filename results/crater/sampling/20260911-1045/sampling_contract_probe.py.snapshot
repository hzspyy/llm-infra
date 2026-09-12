#!/usr/bin/env python3
"""Small GPU probes for probability semantics, independent of model weights."""
import json
import sys
import torch
from types import SimpleNamespace

torch.manual_seed(11)
if sys.argv[1]=='sglang':
    from sglang.srt.layers.sampler import top_k_top_p_sampling_from_probs,top_k_renorm_prob,top_p_renorm_prob,min_p_sampling_from_probs
    for name,weights,k,p,m in [('joint',[.5,.3,.15,.05],2,.6,0.),
                              ('minp_last',[.4,.3,.2,.1],4,.55,.6),
                              ('ties',[1.,1.,1.,.1],2,1.,0.)]:
        probs=torch.tensor([weights],device='cuda').repeat(8192,1)
        probs/=probs.sum(-1,keepdim=True)
        ks=torch.full((8192,),k,device='cuda',dtype=torch.int32)
        ps=torch.full((8192,),p,device='cuda')
        if m:
            probs=top_k_renorm_prob(probs,ks)
            probs=top_p_renorm_prob(probs,ps)
            sample=min_p_sampling_from_probs(probs,torch.full((8192,),m,device='cuda'))
        else:sample=top_k_top_p_sampling_from_probs(probs,ks,ps,filter_apply_order='joint')
        print(json.dumps({'case':name,'n':8192,'counts':torch.bincount(sample.long(),minlength=4).cpu().tolist()}))
else:
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.logits_processor.state import LogitsProcessors
    logits=torch.tensor([[.5,.3,.15,.05]],device='cuda').log()
    meta=SamplingMetadata(temperature=torch.ones(1,device='cuda'),all_greedy=False,all_random=True,
        top_p=None,top_k=torch.ones(1,device='cuda',dtype=torch.int32),generators={},
        max_num_logprobs=1,no_penalties=True,prompt_token_ids=None,
        frequency_penalties=torch.zeros(1,device='cuda'),presence_penalties=torch.zeros(1,device='cuda'),
        repetition_penalties=torch.ones(1,device='cuda'),output_token_ids=[[]],
        allowed_token_ids_mask=None,bad_words_token_ids={},logitsprocs=LogitsProcessors())
    for mode in ['raw_logprobs','processed_logprobs']:
        sampler=Sampler(mode)
        result=sampler(logits.clone(),meta)
        print(json.dumps({'mode':mode,'sampled':result.sampled_token_ids.cpu().tolist(),
            'logprobs':result.logprobs_tensors.logprobs.cpu().tolist(),
            'sampler_path':sampler.topk_topp_sampler.forward.__name__}))
