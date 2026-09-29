#!/usr/bin/env python3
"""0.1-C: locate cancellation and back-end swap points in an installed engine.

Resolves each target symbol in the installed package and records its real
file:line plus the first source line, so the tables in the chapter point at code
that exists in the version actually used rather than at remembered paths.

/scratch/learn/envs/serve/bin/python labs/L0/engine_extension_points.py \
    --engine vllm --output results/crater/0.1/<run>/vllm
/scratch/learn/envs/sgl/bin/python labs/L0/engine_extension_points.py \
    --engine sglang --output results/crater/0.1/<run>/sglang
"""
import argparse,hashlib,importlib,inspect,json,platform
from pathlib import Path

TARGETS={
 'vllm':[
  ('cancel','client aborts the stream','vllm.v1.engine.async_llm:AsyncLLM.abort'),
  ('cancel','front end tells the engine core','vllm.v1.engine.core_client:MPClient.__init__'),
  ('cancel','engine core drops the request','vllm.v1.engine.core:EngineCore.abort_requests'),
  ('cancel','scheduler frees the blocks','vllm.v1.core.sched.scheduler:Scheduler.finish_requests'),
  ('cancel','HTTP disconnect is detected','vllm.entrypoints.serve.utils.api_utils:with_cancellation'),
  ('attention','backend is chosen here','vllm.v1.attention.selector:get_attn_backend'),
  ('attention','the cached resolver behind it','vllm.v1.attention.selector:_cached_get_attn_backend'),
  ('attention','platform maps name to class','vllm.platforms.cuda:CudaPlatformBase.get_attn_backend_cls'),
  ('sampler','sampling entry point','vllm.v1.sample.sampler:Sampler.forward'),
  ('sampler','logits are adjusted here','vllm.v1.sample.sampler:Sampler.apply_penalties'),
  ('sampler','custom logits processors plug in','vllm.v1.sample.logits_processor:build_logitsprocs'),
 ],
 'sglang':[
  ('cancel','client aborts the stream','sglang.srt.managers.tokenizer_manager:TokenizerManager.abort_request'),
  ('cancel','scheduler drops the request','sglang.srt.managers.scheduler:Scheduler.abort_request'),
  ('attention','backend is chosen here','sglang.srt.model_executor.model_runner:ModelRunner.init_attention_backends'),
  ('attention','name maps to class','sglang.srt.model_executor.model_runner:ModelRunner._get_attention_backend'),
  ('sampler','sampling entry point','sglang.srt.layers.sampler:Sampler.forward'),
 ],
}


def resolve(spec):
    module,_,qual=spec.partition(':')
    try:
        obj=importlib.import_module(module)
    except Exception as exc:
        return dict(spec=spec,found=False,error=f'{type(exc).__name__}: {exc}'[:200])
    for part in qual.split('.'):
        obj=getattr(obj,part,None)
        if obj is None:
            return dict(spec=spec,found=False,error=f'missing attribute {part}')
    target=inspect.unwrap(obj)
    try:
        file=inspect.getsourcefile(target);line=target.__code__.co_firstlineno
        source=inspect.getsource(target).splitlines()
    except (OSError,TypeError,AttributeError) as exc:
        return dict(spec=spec,found=False,error=f'no source: {exc}'[:200])
    package=spec.split('.')[0]
    short=file.split('site-packages/')[-1]
    return dict(spec=spec,found=True,file=short,line=line,
        signature=str(inspect.signature(target))[:160] if callable(target) else None,
        first_source_line=source[0].strip()[:160],source_lines=len(source),
        sha256=hashlib.sha256(Path(file).read_bytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',choices=sorted(TARGETS),required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    package=importlib.import_module(a.engine)
    rows=[dict(group=group,role=role,**resolve(spec)) for group,role,spec in TARGETS[a.engine]]
    found=[r for r in rows if r['found']]
    result=dict(engine=a.engine,version=getattr(package,'__version__','unknown'),
        python=platform.python_version(),host=platform.node(),task_ids=['0.1-C'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        targets=len(rows),resolved=len(found),
        by_group={g:sum(1 for r in found if r['group']==g) for g in sorted({r['group'] for r in rows})},
        symbols=rows)
    (a.output/'extension_points.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    ok=len(found)==len(rows)
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(dict(engine=a.engine,version=result['version'],resolved=len(found),targets=len(rows),
        missing=[r['spec'] for r in rows if not r['found']]),ensure_ascii=False))
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
