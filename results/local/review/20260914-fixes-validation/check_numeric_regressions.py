from pathlib import Path
import sys,json,hashlib
from unittest.mock import patch
import torch
root=Path.cwd();sys.path[:0]=[str(root/'labs/L4'),str(root/'labs/L2')]
import quantize_reference as q
import alloc_trace
import alloc_graph_combo

def remaining_hessian_reference(weight,hessian,group):
    work=weight.clone();result=torch.zeros_like(work)
    scales=weight.reshape(weight.shape[0],-1,group).abs().amax(-1).clamp_min(1e-12)/7
    for index in range(weight.shape[1]):
        scale=scales[:,index//group]
        chosen=(work[:,index]/scale).round().clamp(-7,7)*scale
        error=work[:,index]-chosen
        result[:,index]=chosen
        if index+1<weight.shape[1]:
            # Directly solve the unconstrained remaining quadratic after fixing this column.
            response=torch.linalg.solve(hessian[index+1:,index+1:],hessian[index+1:,index])
            work[:,index+1:]+=error[:,None]*response[None,:]
    return result

rows=[]
for seed in (0,1,2):
    torch.manual_seed(seed)
    weight=torch.randn(3,8,dtype=torch.float64)
    x=torch.randn(24,8,dtype=torch.float64)
    h=x.T@x
    for group in (2,4,8):
        actual,_,_=q.gptq_quant(weight,h,group_size=group,damp=0)
        expected=remaining_hessian_reference(weight,h,group)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-12)
        _,rtn,_=q.rtn_quant(weight,group_size=group)
        _,clipped,_=q.clip_quant(weight,x,group_size=group,ratios=[1.])
        assert torch.equal(rtn,clipped)
        rows.append({'seed':seed,'group':group,'gradient_free_quantized_weight_max_error':(actual-expected).abs().max().item(),'unclipped_equal_rtn':True})
w=torch.tensor([[1.,.31,.346]],dtype=torch.float64)
h=torch.tensor([[4.,1.,1.],[1.,2.,1.],[1.,1.,2.]],dtype=torch.float64)
a=q.gptq_quant(w,h,group_size=3,damp=0)[0];e=remaining_hessian_reference(w,h,3)
torch.testing.assert_close(a,torch.tensor([[1.,2/7,3/7]],dtype=torch.float64),rtol=0,atol=1e-12)
torch.testing.assert_close(a,e,rtol=0,atol=1e-12)
small=torch.tensor([[1.,0.,6/7,4/7]],dtype=torch.float64)
group4=(q.rtn_quant(small,group_size=4)[1]-small).norm().item()
group2=(q.rtn_quant(small,group_size=2)[1]-small).norm().item()
assert group2>group4
snapshot=[{'total_size':16*2**20,'blocks':[{'state':'active_allocated','size':4*2**20},{'state':'inactive','size':8*2**20},{'state':'active_awaiting_free','size':4*2**20}]}]
with patch.object(torch.cuda,'memory_snapshot',return_value=snapshot),patch.object(torch.cuda,'memory_stats',return_value={'inactive_split_bytes.all.current':8*2**20}):
    parsed=alloc_trace.snapshot_summary();combo=alloc_graph_combo.snap()
    assert parsed==combo
    assert parsed['pending_mb']==4 and parsed['active_mb']==4
    assert parsed['inactive_split_mb']==8 and parsed['inactive_in_mixed_segments_mb']==8 and parsed['free_blocks']==1
result={'scope':'CPU numerical references and synthetic allocator schema; not GPU timing',
        'torch':torch.__version__,'quantizer_independent_reference':rows,
        'three_column_case':{'actual':a.tolist(),'remaining_hessian_reference':e.tolist()},
        'smaller_group_counterexample':{'group4_error':group4,'group2_error':group2},
        'cross_group_objective':{'full':4,'diagonal_only':2,'input':[[1,1]],'delta_weight':[[1,1]]},
        'allocator_synthetic_schema':parsed,
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'labs/L4/quantize_reference.py',root/'labs/L2/alloc_trace.py',root/'labs/L2/alloc_graph_combo.py']}}
assert (torch.tensor([[1.,1.]])@torch.tensor([[1.],[1.]])).square().item()==4
out=Path('/Volumes/data/llm-infra-review-fixes-ostfce6p/numeric-regressions.json')
with out.open('x') as f:json.dump(result,f,indent=2)
print('9 independent GPTQ cases, clipping control and allocator schema passed')
