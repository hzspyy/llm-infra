#!/usr/bin/env python3
"""CPU SFT mask and one-layer LoRA update/save/load/merge contracts.

Synthetic inputs only. This is not a TRL/PEFT training or QLoRA kernel run.
"""
import hashlib
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


class SimpleLoRALinear(nn.Module):
    def __init__(self, in_features=16, out_features=32, r=4, lora_alpha=8.):
        super().__init__()
        self.r, self.alpha = r, lora_alpha
        self.scaling = lora_alpha / r
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float64) * .02,
                                   requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(r, in_features, dtype=torch.float64) * .02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float64))

    def forward(self, x):
        return F.linear(x, self.weight) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def merged(self):
        # Return a separate plain Linear, so the adapter is not applied twice.
        layer = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False).double()
        with torch.no_grad():
            layer.weight.copy_(self.weight + self.scaling * self.lora_B @ self.lora_A)
        return layer


def verify_sft_masks(seed):
    torch.manual_seed(seed)
    # EOS and padding share ID 2. Valid lengths, rather than token equality, distinguish them.
    ids = torch.tensor([[1, 3, 4, 2, 2], [1, 5, 6, 7, 2]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    response = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
    labels = ids.masked_fill(~(attention & response), -100)
    logits = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    shifted = labels[:, 1:]
    valid = shifted != -100
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 8), shifted.reshape(-1), reduction='sum') / valid.sum()
    logp = logits[:, :-1].log_softmax(-1)
    manual = -logp.gather(-1, shifted.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].sum() / valid.sum()
    official_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    manual_grad = torch.autograd.grad(manual, logits)[0]
    torch.testing.assert_close(loss, manual, rtol=0, atol=1e-12)
    torch.testing.assert_close(official_grad, manual_grad, rtol=0, atol=1e-12)
    assert official_grad[:, :-1][~valid].abs().max() == 0
    assert int((labels == 2).sum()) == 2  # two genuine EOS targets
    bad = labels.masked_fill(ids == 2, -100)
    assert int((bad[:, 1:] != -100).sum()) == 3
    # Truncating before the first answer yields an empty supervised set; reject before reducing.
    empty = labels[:, :2]
    assert not (empty[:, 1:] != -100).any()
    return {'seed':seed,'input_ids':ids.tolist(),'attention_mask':attention.tolist(),
            'response_mask':response.tolist(),'labels':labels.tolist(),'valid_targets':int(valid.sum()),
            'loss':loss.item(),'gradient_max_error':(official_grad-manual_grad).abs().max().item(),
            'wrong_eos_mask_valid_targets':int((bad[:, 1:]!=-100).sum()),
            'empty_answer_action':'reject before loss/optimizer',
            'scope':'independent logits CE reference; no TRL or causal-attention model executed'}


def verify_lora(seed, rank):
    torch.manual_seed(seed)
    model = SimpleLoRALinear(r=rank)
    x = torch.randn(3,16,dtype=torch.float64)
    target = torch.randn(3,32,dtype=torch.float64)
    initial_base = model.weight.detach().clone()
    initial_diff = (model(x)-F.linear(x,model.weight)).abs().max().item()
    assert initial_diff == 0
    optimizer = torch.optim.AdamW([model.lora_A,model.lora_B], lr=.01, weight_decay=0.)
    F.mse_loss(model(x),target).backward()
    grad_a,grad_b = model.lora_A.grad.norm().item(), model.lora_B.grad.norm().item()
    assert grad_a == 0 and grad_b > 0 and model.weight.grad is None
    optimizer.step()
    torch.testing.assert_close(model.weight,initial_base,rtol=0,atol=0)
    expected = model(x).detach()
    base_hash = hashlib.sha256(initial_base.numpy().tobytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='adapter-contract-',dir=os.environ['TMPDIR']) as directory:
        checkpoint = Path(directory)/'adapter.pt'
        torch.save({'r':rank,'alpha':model.alpha,'base_sha256':base_hash,
                    'A':model.lora_A.detach(),'B':model.lora_B.detach()},checkpoint)
        payload = torch.load(checkpoint,weights_only=True)
        restored = SimpleLoRALinear(r=payload['r'],lora_alpha=payload['alpha'])
        with torch.no_grad():
            restored.weight.copy_(initial_base)
            restored.lora_A.copy_(payload['A'])
            restored.lora_B.copy_(payload['B'])
        assert hashlib.sha256(restored.weight.numpy().tobytes()).hexdigest()==payload['base_sha256']
        reloaded_output = restored(x).detach()
        merged_output = restored.merged()(x).detach()
        torch.testing.assert_close(reloaded_output,expected,rtol=0,atol=1e-12)
        torch.testing.assert_close(merged_output,expected,rtol=0,atol=1e-12)
        checkpoint_bytes = checkpoint.stat().st_size
    return {'seed':seed,'rank':rank,'alpha':model.alpha,'initial_output_error':initial_diff,
            'base_max_change':(model.weight-initial_base).abs().max().item(),
            'first_gradient_norms':{'A':grad_a,'B':grad_b},
            'adapter_parameter_count':model.lora_A.numel()+model.lora_B.numel(),
            'adapter_optimizer_state_dtypes':sorted({str(t.dtype) for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t)}),
            'reload_max_error':(reloaded_output-expected).abs().max().item(),
            'merge_max_error':(merged_output-expected).abs().max().item(),
            'checkpoint_bytes':checkpoint_bytes,'base_sha256':base_hash,
            'expected_output':expected.tolist(),'updates':1}


def adapter_count_example():
    # Dimensions from the pinned SmolLM3 YAML. Counts are arithmetic, not a loaded-model census.
    shapes={'q_proj':[2048,2048],'k_proj':[512,2048],'v_proj':[512,2048],
            'o_proj':[2048,2048],'gate_proj':[11008,2048],
            'up_proj':[11008,2048],'down_proj':[2048,11008]}
    return {'source_kind':'formula_prediction','layers':36,'linear_shapes_out_in':shapes,
            'counts':{str(r):36*sum(r*(m+n) for m,n in shapes.values()) for r in (8,16,32)},
            'assumptions':'adapters on all seven listed matrices in every layer; no embedding adapter, bias or modules_to_save',
            'memory_rule':'state bytes use actual adapter parameter/gradient/m/v dtypes; activation and base storage are separate'}


def main():
    from _evidence import new_output,write_result
    torch.set_num_threads(1)
    out=new_output('Synthetic SFT mask and one-layer LoRA round-trip')
    result={'sft_masks':[verify_sft_masks(seed) for seed in (0,1,2)],
            'lora_round_trip':[verify_lora(seed,rank) for seed in (0,1,2) for rank in (2,4)],
            'adapter_count_formula':adapter_count_example()}
    write_result(out,'sft_lora_contract_report.json',result,
                 {'device':'CPU','dtype':'float64','seeds':[0,1,2],'lora_ranks':[2,4],
                  'updates_per_lora_case':1,'source_data':'synthetic tensors',
                  'reference':'PyTorch CE and one plain Linear after merge',
                  'TRL_PEFT_QLoRA_execution':False,'quality_measured':False},[__file__])


if __name__=='__main__':
    main()
