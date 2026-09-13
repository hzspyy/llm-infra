#!/usr/bin/env python3
"""Read pinned Qwen3 config/source and small safetensors slices; no inference."""
import argparse,hashlib,inspect,json,shutil
from pathlib import Path
import transformers
from transformers.models.qwen3 import modeling_qwen3
from safetensors import safe_open

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=a.model/'config.json';shutil.copyfile(config,a.output/'config.json')
    src=Path(inspect.getsourcefile(modeling_qwen3));shutil.copyfile(src,a.output/'modeling_qwen3.py')
    targets={'model.embed_tokens.weight','model.layers.0.input_layernorm.weight','model.layers.0.self_attn.q_proj.weight','model.layers.0.self_attn.k_proj.weight','model.layers.0.self_attn.v_proj.weight','model.layers.0.mlp.gate_proj.weight','model.layers.0.mlp.up_proj.weight','model.layers.0.mlp.down_proj.weight','lm_head.weight'}
    records=[]
    for file in sorted(a.model.glob('*.safetensors')):
        with safe_open(file,framework='pt',device='cpu') as f:
            for key in sorted(targets & set(f.keys())):
                view=f.get_slice(key);shape=view.get_shape();x=view[:8] if len(shape)==1 else view[:1,:8]
                records.append(dict(file=file.name,key=key,shape=shape,slice='[:8]' if len(shape)==1 else '[:1,:8]',dtype=str(x.dtype),values=x.float().tolist()))
    (a.output/'tensors.json').write_text(json.dumps(records,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(dict(model='Qwen/Qwen3-1.7B',revision=a.model.name,
        transformers=transformers.__version__,source_path=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='read-only small CPU weight slices and installed source; no forward or quality evaluation',missing_keys=sorted(targets-{r['key'] for r in records})),indent=2)+'\n')
    print(len(records))
if __name__=='__main__':main()
