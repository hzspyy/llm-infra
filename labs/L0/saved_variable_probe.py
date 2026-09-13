#!/usr/bin/env python3
"""0.0b-C: version counter, saved tensors, retain_graph and the two ways to stop a gradient.

python labs/L0/saved_variable_probe.py --output results/local/0.0b/<new-run>
"""
import argparse,copy,hashlib,json,platform
from pathlib import Path
import torch
import torch.nn.functional as F
from tiny_lm import TinyLM


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def version_cases():
    """Two in-place edits with the same version bump; only one of them breaks backward."""
    rows=[]
    for name,needs_output in [('exp_saves_output',True),('mul_by_scalar_saves_nothing',False)]:
        x=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
        y=x.exp() if needs_output else x.mul(2.)
        z=y.sum();version_at_save=y._version
        with torch.no_grad():y.add_(1.)
        row=dict(case=name,forward_op='exp' if needs_output else 'mul(scalar)',
            version_at_save=version_at_save,version_at_backward=y._version)
        try:
            z.backward();row.update(raised=False,grad=x.grad.tolist())
        except RuntimeError as exc:row.update(raised=True,error=str(exc))
        rows.append(row)
    # same graph, no in-place edit: backward succeeds and reproduces d(exp)/dx = exp(x)
    x=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True);y=x.exp();y.sum().backward()
    rows.append(dict(case='exp_no_inplace',forward_op='exp',version_at_save=0,version_at_backward=y._version,
        raised=False,grad=x.grad.tolist(),reference=x.detach().exp().tolist(),
        max_error=(x.grad-x.detach().exp()).abs().max().item()))
    return rows


def retain_graph_cases():
    """Saved values are released by the first backward; retain_graph keeps them alive."""
    rows=[]
    x=torch.tensor([2.],dtype=torch.float64,requires_grad=True);y=x.square().sum();y.backward()
    row=dict(case='second_backward_without_retain',first_grad=x.grad.tolist())
    try:
        y.backward();row.update(raised=False)
    except RuntimeError as exc:row.update(raised=True,error=str(exc))
    rows.append(row)
    x=torch.tensor([2.],dtype=torch.float64,requires_grad=True);y=x.square().sum()
    y.backward(retain_graph=True);first=x.grad.clone();y.backward()
    rows.append(dict(case='second_backward_with_retain',raised=False,first_grad=first.tolist(),
        second_grad=x.grad.tolist(),analytic_single=(2*x.detach()).tolist(),
        accumulated_is_double=bool(torch.allclose(x.grad,2*first))))
    return rows


def saved_tensor_inventory(model,inputs,labels,vocab):
    """Record every tensor the backward graph holds between forward and backward.

    A saved weight view shares storage with its parameter, so data_ptr separates
    the parameters an operator saved from the intermediate values it produced.
    """
    owners={p.data_ptr():n for n,p in model.named_parameters()}
    packed=[]
    def pack(t):
        packed.append(dict(shape=list(t.shape),dtype=str(t.dtype).replace('torch.',''),
            bytes=t.numel()*t.element_size(),parameter=owners.get(t.data_ptr())));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        loss=F.cross_entropy(model(inputs).reshape(-1,vocab),labels.reshape(-1),ignore_index=-100)
    activations=[r for r in packed if r['parameter'] is None]
    unique={}
    for row in activations:unique.setdefault((tuple(row['shape']),row['dtype']),[0,row['bytes']])[0]+=1
    return dict(loss=loss.item(),saved_tensor_count=len(packed),
        saved_bytes=sum(r['bytes'] for r in packed),
        saved_parameter_views=len(packed)-len(activations),
        saved_parameter_bytes=sum(r['bytes'] for r in packed if r['parameter'] is not None),
        saved_activation_count=len(activations),
        saved_activation_bytes=sum(r['bytes'] for r in activations),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        input_tokens=int(inputs.numel()),
        activations_by_shape=[dict(shape=list(k[0]),dtype=k[1],count=v[0],bytes_each=v[1]) for k,v in
                  sorted(unique.items(),key=lambda kv:-kv[1][0]*kv[1][1])],
        saved_parameters=sorted({r['parameter'] for r in packed if r['parameter'] is not None}))


def stop_gradient_cases(base,inputs,labels,vocab):
    """detach cuts the path; requires_grad_(False) only skips that parameter."""
    rows=[]
    for mode in ['reference','detach_block_input','freeze_fc1']:
        m=copy.deepcopy(base);handle=None
        if mode=='detach_block_input':handle=m.blocks[0].register_forward_pre_hook(lambda mod,inp:(inp[0].detach(),))
        if mode=='freeze_fc1':m.blocks[0].mlp.fc1.weight.requires_grad_(False)
        F.cross_entropy(m(inputs).reshape(-1,vocab),labels.reshape(-1),ignore_index=-100).backward()
        if handle is not None:handle.remove()
        rows.append(dict(case=mode,**{n:(None if p.grad is None else round(p.grad.norm().item(),9))
            for n,p in m.named_parameters() if n in
            ['tok_emb.weight','blocks.0.attn.qkv.weight','blocks.0.mlp.fc1.weight','blocks.0.mlp.fc2.weight','lm_head.weight']}))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    save(a.output/'manifest.json',dict(torch=torch.__version__,torch_git=torch.version.git_version,
        python=platform.python_version(),dtype='float64',device='cpu',task_ids=['0.0b-C'],
        model='TinyLM(V=11,d=8,H=2,L=1,block=8)',timing='none',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tiny_lm_sha256=hashlib.sha256((Path(__file__).parent/'tiny_lm.py').read_bytes()).hexdigest(),
        tolerances=dict(analytic_gradient=1e-12)))
    torch.manual_seed(0);base=TinyLM(11,d_model=8,n_head=2,n_layer=1,block_size=8).double().eval()
    sequences=[torch.randint(0,11,(n+1,)) for n in [3,7]]
    inputs=torch.zeros(2,7,dtype=torch.long);labels=torch.full((2,7),-100,dtype=torch.long)
    for i,tokens in enumerate(sequences):
        inputs[i,:len(tokens)-1]=tokens[:-1];labels[i,:len(tokens)-1]=tokens[1:]
    versions=version_cases();retain=retain_graph_cases()
    inventory=saved_tensor_inventory(copy.deepcopy(base),inputs,labels,11)
    stops=stop_gradient_cases(base,inputs,labels,11)
    save(a.output/'version_counter.json',versions)
    save(a.output/'retain_graph.json',retain)
    save(a.output/'saved_tensors.json',inventory)
    save(a.output/'stop_gradient.json',stops)
    reference=next(r for r in stops if r['case']=='reference')
    detached=next(r for r in stops if r['case']=='detach_block_input')
    frozen=next(r for r in stops if r['case']=='freeze_fc1')
    ok=(versions[0]['raised'] and not versions[1]['raised'] and not versions[2]['raised']
        and versions[0]['version_at_save']==0 and versions[0]['version_at_backward']==1
        and versions[1]['version_at_backward']==1 and versions[2]['max_error']<1e-12
        and retain[0]['raised'] and retain[1]['accumulated_is_double']
        and detached['tok_emb.weight'] is None and detached['lm_head.weight']==reference['lm_head.weight']
        and frozen['blocks.0.mlp.fc1.weight'] is None
        and frozen['tok_emb.weight']==reference['tok_emb.weight'])
    save(a.output/'summary.json',dict(passed=ok,version_cases=len(versions),retain_cases=len(retain),
        saved_tensor_count=inventory['saved_tensor_count'],saved_bytes=inventory['saved_bytes'],
        saved_activation_count=inventory['saved_activation_count'],
        saved_activation_bytes=inventory['saved_activation_bytes'],
        saved_parameter_views=inventory['saved_parameter_views'],
        parameter_bytes=inventory['parameter_bytes'],stop_gradient_cases=len(stops)))
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(dict(passed=ok,saved_tensors=inventory['saved_tensor_count'],
        saved_activation_bytes=inventory['saved_activation_bytes'],
        parameter_bytes=inventory['parameter_bytes'])));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
