#!/usr/bin/env python3
"""Targeted offset/mask and ablation controls; consumes a saved parity experiment."""
import argparse,json,math,hashlib
from pathlib import Path
import torch
from tiny_lm_cache import CachedTinyLM

def main():
    p=argparse.ArgumentParser();p.add_argument('--parity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(0)
    model=CachedTinyLM(17,d_model=16,n_head=4,n_layer=2,block_size=16).double().eval()
    ids=torch.tensor([[1,2,3,4,5],[5,4,3,2,1]]);trace={}
    with torch.no_grad():
        full=model(ids,trace);_,old=model.forward_cached(ids[:,:2]);chunk,cache=model.forward_cached(ids[:,2:],old)
    q=trace['q'][0,0,2].tolist();ks=trace['k'][0,0].tolist()
    raw=[sum(x*y for x,y in zip(q,k))/2 for k in ks]
    exps=[math.exp(x-max(raw[:3])) if i<=2 else 0 for i,x in enumerate(raw)]
    probs=[x/sum(exps) for x in exps];actual=trace['att'][0,0,2].tolist()
    source=json.loads(a.parity.read_text());index={(r['seed'],r['tied'],r['length'],r['variant']):r for r in source};ablations=[]
    for key,r in index.items():
        if key[-1]=='base':continue
        base=index[(*key[:3],'base')];x=torch.tensor(r['full_logits'],dtype=torch.float64);y=torch.tensor(base['full_logits'],dtype=torch.float64)
        ablations.append(dict(seed=key[0],tied=key[1],length=key[2],variant=key[3],max_abs_delta=(x-y).abs().max().item(),argmax_changes=int((x.argmax(-1)!=y.argmax(-1)).sum())))
    result=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parity_sha256=hashlib.sha256(a.parity.read_bytes()).hexdigest(),
        q=q,keys=ks,scaled_scores_before_mask=raw,masked_positions=[3,4],manual_probs=probs,actual_probs=actual,
        manual_error=max(abs(x-y) for x,y in zip(probs,actual)),offset_chunk_error=(chunk-full[:,2:]).abs().max().item(),ablations=ablations)
    (a.output/'controls.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['manual_error']<1e-14 and result['offset_chunk_error']<1e-10
    print(result['manual_error'],result['offset_chunk_error'])
if __name__=='__main__':main()
