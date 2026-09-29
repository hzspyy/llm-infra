#!/usr/bin/env python3
"""0.1-B: link one QKV projection and one attention call down to kernels.

Profiles a single decode step of a real vLLM forward, then follows each chosen
operator through the trace: python-level op -> ATen op -> CUDA runtime launch ->
device kernel. Host submit and device completion are reported separately.

python labs/L0/op_to_kernel_trace.py --output results/crater/0.1/<run>
"""
import argparse,hashlib,json,os,platform
from pathlib import Path

PROMPT="Explain what a KV cache is in one sentence."
WARMUP_TOKENS=4
TRACE_TOKENS=2
# The three operators this chapter follows through one attention layer, matched
# against the exact operator name the profiler records.
TARGETS={'qkv_projection':['aten::linear'],
         'kv_cache_write':['_C_cache_ops::reshape_and_cache_flash'],
         'attention':['vllm::unified_attention_with_output']}


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def classify(name):
    for label,names in TARGETS.items():
        if name in names:return label
    return None


def analyse(trace_path,out):
    """Walk the chrome trace and join cpu_op -> cuda_runtime -> kernel by correlation."""
    raw=json.loads(Path(trace_path).read_text())
    events=raw['traceEvents']
    cpu_ops=[e for e in events if e.get('cat')in('cpu_op','user_annotation')and'dur'in e]
    runtime=[e for e in events if e.get('cat')in('cuda_runtime','cuda_driver')and'dur'in e]
    kernels=[e for e in events if e.get('cat')in('kernel','gpu_op')and'dur'in e]
    by_corr={}
    for e in runtime:
        c=e.get('args',{}).get('correlation')
        if c is not None:by_corr.setdefault(c,{})['runtime']=e
    for e in kernels:
        c=e.get('args',{}).get('correlation')
        if c is not None:by_corr.setdefault(c,{})['kernel']=e

    def launches_inside(op):
        """Runtime launches issued while this cpu op was on the stack."""
        lo,hi=op['ts'],op['ts']+op['dur']
        return [e for e in runtime if e['tid']==op['tid'] and lo<=e['ts']<=hi]

    def follow(op):
        pairs=[]
        for r in launches_inside(op):
            c=r.get('args',{}).get('correlation')
            k=by_corr.get(c,{}).get('kernel')
            if k is None:continue
            pairs.append(dict(runtime_name=r['name'],runtime_us=r['dur'],runtime_ts=r['ts'],
                kernel_name=k['name'],kernel_us=k['dur'],kernel_ts=k['ts'],correlation=c,
                grid=k.get('args',{}).get('grid'),block=k.get('args',{}).get('block'),
                registers_per_thread=k.get('args',{}).get('registers per thread'),
                shared_memory=k.get('args',{}).get('shared memory'),
                stream=k.get('args',{}).get('stream'),
                submit_to_start_us=k['ts']-r['ts'],
                host_returns_before_device_ends_us=(k['ts']+k['dur'])-(r['ts']+r['dur'])))
        if not pairs:return None
        return dict(cpu_op=op['name'],cpu_op_us=op['dur'],cpu_op_ts=op['ts'],
            input_shapes=op.get('args',{}).get('Input Dims'),input_types=op.get('args',{}).get('Input type'),
            launches=len(pairs),kernels=pairs)

    # The first match falls in the prefill step, the last one in the final decode
    # step; keeping both makes the shape difference between the phases visible.
    ordered=sorted(cpu_ops,key=lambda e:e['ts'])
    picked={}
    for label in TARGETS:
        matches=[op for op in ordered if classify(op['name'])==label]
        first=next((f for f in (follow(op) for op in matches) if f),None)
        last=next((f for f in (follow(op) for op in reversed(matches)) if f),None)
        if first is None:continue
        picked[label]=dict(label=label,occurrences=len(matches),prefill=first,decode=last)

    kernel_names={}
    for e in kernels:
        kernel_names[e['name']]=kernel_names.get(e['name'],0)+1
    summary=dict(trace=str(Path(trace_path).name),cpu_ops=len(cpu_ops),runtime_events=len(runtime),
        kernel_events=len(kernels),distinct_kernels=len(kernel_names),
        device_us_total=round(sum(e['dur'] for e in kernels),3),
        top_kernels=sorted(({'name':k,'count':v} for k,v in kernel_names.items()),
                           key=lambda d:-d['count'])[:10],
        followed=picked)
    save(out/'op_to_kernel.json',summary)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',default='Qwen/Qwen3-1.7B')
    ap.add_argument('--gpu-fraction',type=float,default=0.25)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ.setdefault('VLLM_LOGGING_LEVEL','WARNING')
    import torch
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.v1.worker.gpu_worker import Worker

    save(a.output/'manifest.json',dict(model=a.model,prompt=PROMPT,warmup_tokens=WARMUP_TOKENS,
        trace_tokens=TRACE_TOKENS,gpu_memory_utilization=a.gpu_fraction,task_ids=['0.1-B'],
        vllm=vllm.__version__,torch=torch.__version__,python=platform.python_version(),host=platform.node(),
        targets=TARGETS,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    state=dict(profiler=None,steps=0,decode_step=None)
    original=Worker.execute_model
    def execute_model(self,scheduler_output,*args,**kw):
        counts=dict(scheduler_output.num_scheduled_tokens)
        if state['profiler'] is None or not counts:
            return original(self,scheduler_output,*args,**kw)
        state['steps']+=1
        decode=all(v==1 for v in counts.values())
        if decode and state['decode_step'] is None:
            state['decode_step']=dict(step=state['steps'],num_scheduled_tokens=counts)
        return original(self,scheduler_output,*args,**kw)
    Worker.execute_model=execute_model

    llm=LLM(model=a.model,max_model_len=2048,gpu_memory_utilization=a.gpu_fraction,
        enforce_eager=True,enable_prefix_caching=False,disable_log_stats=True)
    llm.generate([PROMPT],SamplingParams(max_tokens=WARMUP_TOKENS,temperature=0.0,seed=0))
    torch.cuda.synchronize()

    trace_dir=a.output/'torch-profile';trace_dir.mkdir()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,with_stack=False) as prof:
        state['profiler']=prof
        outputs=llm.generate([PROMPT],SamplingParams(max_tokens=TRACE_TOKENS,temperature=0.0,seed=0))
        torch.cuda.synchronize()
    Worker.execute_model=original
    trace_path=trace_dir/'decode_step.json'
    prof.export_chrome_trace(str(trace_path))

    summary=analyse(trace_path,a.output)
    summary['generated_text']=outputs[0].outputs[0].text
    summary['profiled_steps']=state['steps']
    summary['first_decode_step']=state['decode_step']
    save(a.output/'op_to_kernel.json',summary)
    followed=summary['followed']
    ok=(set(followed)==set(TARGETS)
        and all(f['prefill']['launches']>=1 for f in followed.values())
        and all(k['submit_to_start_us']>=0 for f in followed.values()
                for phase in ('prefill','decode') if f[phase] for k in f[phase]['kernels'])
        and summary['kernel_events']>0)
    save(a.output/'summary.json',dict(passed=ok,followed=sorted(followed),
        kernel_events=summary['kernel_events'],distinct_kernels=summary['distinct_kernels'],
        profiled_steps=summary['profiled_steps']))
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(dict(passed=ok,followed=sorted(followed),kernels=summary['kernel_events'],
        distinct=summary['distinct_kernels'])));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
