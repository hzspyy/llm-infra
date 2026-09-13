#!/usr/bin/env python3
"""0.0 A/B: FP64 full-prefix/cache parity, parameter accounting, hand attention."""
import argparse,hashlib,json,math,platform
from pathlib import Path
import torch
from tiny_lm_cache import CachedTinyLM


def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def describe(t):return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    configs=[dict(seed=seed,tied=tied,length=length,variant=variant) for seed in range(3) for tied in [False,True]
             for length in [1,2,7,16] for variant in ['base','no_pos','no_scale']]
    save(a.output/'cases.json',configs)
    save(a.output/'manifest.json',dict(task_ids=['0.0-A','0.0-B'],torch=torch.__version__,torch_git=torch.version.git_version,
        python=platform.python_version(),dtype='float64',device='cpu',batch=2,d_model=16,n_head=4,n_layer=2,vocab=17,block_size=16,
        atol=1e-10,rtol=1e-10,timing='none; numerical/state evidence only',seed=[0,1,2],
        source_sha256={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['tiny_lm.py','tiny_lm_cache.py','verify_tiny_cache.py']},
        cases_sha256=hashlib.sha256((a.output/'cases.json').read_bytes()).hexdigest()))
    rows=[];parameters=[]
    for seed in range(3):
        for tied in [False,True]:
            torch.manual_seed(seed)
            model=CachedTinyLM(17,d_model=16,n_head=4,n_layer=2,block_size=16,tie_weights=tied).double().eval()
            torch.save(model.state_dict(),a.output/f'weights-{seed}-{tied}.pt')
            unique=sum(p.numel() for p in model.parameters())
            expected=(1 if tied else 2)*17*16+16*16+2*(12*16*16+4*16)+2*16
            parameters.append(dict(seed=seed,tied=tied,expected=expected,actual=unique,
                shared_object=model.tok_emb.weight is model.lm_head.weight,
                tensors=[dict(name=n,**describe(p),numel=p.numel()) for n,p in model.named_parameters()]))
            assert unique==expected
            tokens=torch.randint(0,17,(2,16))
            for variant in ['base','no_pos','no_scale']:
                model.use_pos=variant!='no_pos'
                for block in model.blocks:block.attn.scale=variant!='no_scale'
                for length in [1,2,7,16]:
                    ids=tokens[:,:length];cache=None;broken=None;steps=[]
                    with torch.no_grad():
                        for pos in range(length):
                            full=model(ids[:,:pos+1])[:,-1:,:]
                            old=[(k.clone(),v.clone()) for k,v in cache] if cache is not None else None
                            output,new_cache=model.forward_cached(ids[:,pos:pos+1],cache)
                            wrong,broken=model.forward_cached(ids[:,pos:pos+1],broken,reset_position=True)
                            assert old is None or all(torch.equal(k,ok) and torch.equal(v,ov) for (k,v),(ok,ov) in zip(cache,old))
                            steps.append(dict(position=pos,max_abs_error=(output-full).abs().max().item(),
                                argmax_equal=torch.equal(output.argmax(-1),full.argmax(-1)),
                                numeric_equal=torch.allclose(output,full,atol=1e-10,rtol=1e-10),
                                wrong_position_max_abs_error=(wrong-full).abs().max().item(),
                                cache=[dict(k=describe(k),v=describe(v)) for k,v in new_cache]))
                            cache=new_cache
                        # A multi-token chunk must also mask future tokens within the chunk.
                        chunked,_=model.forward_cached(ids)
                        complete=model(ids)
                    rows.append(dict(seed=seed,tied=tied,variant=variant,length=length,input=ids.tolist(),steps=steps,
                                     full_logits=complete.tolist(),chunk_error=(chunked-complete).abs().max().item()))
    save(a.output/'parameters.json',parameters);save(a.output/'parity.json',rows)
    # Capture a complete B=2,S=5 tensor flow and independently calculate one attention row.
    torch.manual_seed(0);model=CachedTinyLM(17,d_model=16,n_head=4,n_layer=2,block_size=16).double().eval()
    ids=torch.tensor([[1,2,3,4,5],[5,4,3,2,1]]);flow=[];handles=[]
    for name,module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(lambda m,inp,out,name=name:flow.append(dict(name=name,input=[describe(t) for t in inp if isinstance(t,torch.Tensor)],output=describe(out)))))
    trace={}
    with torch.no_grad():model(ids,trace)
    for h in handles:h.remove()
    query=trace['q'][0,0,4].tolist();keys=trace['k'][0,0].tolist()
    scores=[sum(x*y for x,y in zip(query,key))/2 for key in keys]
    exps=[math.exp(x-max(scores)) for x in scores];probs=[x/sum(exps) for x in exps]
    actual=trace['att'][0,0,4].tolist()
    save(a.output/'tensor-flow.json',dict(input=ids.tolist(),modules=flow,attention={k:describe(trace[k]) for k in ['qkv','q','k','v','att','attn_out']},
        hand_row=dict(layer=1,batch=0,head=0,query_position=4,q=query,k=keys,scaled_scores=scores,probabilities=probs,actual=actual,max_error=max(abs(x-y) for x,y in zip(probs,actual)))))
    errors=[]
    with torch.no_grad(): _,cache=model.forward_cached(ids)
    for name,t,c in [('empty',ids[:,:0],None),('overflow',torch.ones(2,12,dtype=torch.long),cache),('layers',ids[:,:1],cache[:1])]:
        try:model.forward_cached(t,c)
        except ValueError as e:errors.append(dict(case=name,error=str(e)))
    save(a.output/'errors.json',errors)
    passed=all(s['numeric_equal'] and s['argmax_equal'] for r in rows for s in r['steps']) and all(r['chunk_error']<1e-10 for r in rows) and len(errors)==3
    summary=dict(cases=len(rows),steps=sum(len(r['steps']) for r in rows),passed=passed,max_abs_error=max(s['max_abs_error'] for r in rows for s in r['steps']),
                 position_counterexamples=sum(any(s['wrong_position_max_abs_error']>1e-8 for s in r['steps']) for r in rows),
                 untied_parameters=parameters[0]['actual'],tied_parameters=parameters[1]['actual'])
    save(a.output/'summary.json',summary);(a.output/'exit.txt').write_text('0\n' if passed else '1\n');print(json.dumps(summary));raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
