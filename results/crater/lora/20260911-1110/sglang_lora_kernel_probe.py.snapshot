#!/usr/bin/env python3
"""Actual SGLang Triton segmented low-rank GEMMs against a dense reference."""
import json
import torch
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd

torch.manual_seed(19)
x=(torch.randn(6,64,device='cuda')*.1).half()
a=(torch.randn(3,16,64,device='cuda')*.1).half()
b=(torch.randn(3,48,16,device='cuda')*.1).half()
a[0].zero_();b[0].zero_()
info=LoRABatchInfo(use_cuda_graph=False,bs=3,num_segments=3,
    seg_indptr=torch.tensor([0,2,3,6],device='cuda',dtype=torch.int32),
    weight_indices=torch.tensor([1,0,2],device='cuda',dtype=torch.int64),
    lora_ranks=torch.tensor([0,8,16],device='cuda',dtype=torch.int32),
    scalings=torch.tensor([0.,1.,1.],device='cuda'),max_len=3,
    seg_lens=torch.tensor([2,1,3],device='cuda',dtype=torch.int32),permutation=None,
    expected_tokens=6,has_active_lora=True)
low=sgemm_lora_a_fwd(x,a,info,stack_num=1)
y=sgemm_lora_b_fwd(low,b,info,None)
reference=torch.zeros((6,48),device='cuda')
for start,end,id_,rank in [(0,2,1,8),(2,3,0,0),(3,6,2,16)]:
    if rank:reference[start:end]=(x[start:end].float()@a[id_,:rank].float().T)@b[id_,:,:rank].float().T
error=float((y.float()-reference).abs().max())
assert torch.allclose(y.float(),reference,atol=.003,rtol=.01)
assert (y[2]==0).all()
print(json.dumps({'x_shape':list(x.shape),'x_stride':list(x.stride()),'A_shape':list(a.shape),
 'B_shape':list(b.shape),'low_shape':list(low.shape),'output_shape':list(y.shape),
 'seg_indptr':info.seg_indptr.cpu().tolist(),'weight_indices':info.weight_indices.cpu().tolist(),
 'ranks':info.lora_ranks.cpu().tolist(),'max_abs_error':error,'base_row_zero':True,
 'output_prefix':y[:,:4].cpu().tolist()}))
