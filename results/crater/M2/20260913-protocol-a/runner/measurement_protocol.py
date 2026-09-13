#!/usr/bin/env python3
"""M2 CUDA reduction protocol. New output directory required; no model weights.

python labs/M/measurement_protocol.py --output results/<machine>/M2/<run>
Measure stream intervals and complete calls separately. Cache capacity != hit rate.
"""
import time
IMPORT_START = time.perf_counter()
import torch
TORCH_IMPORT_SECONDS = time.perf_counter() - IMPORT_START
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def telemetry():
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw',
                        '--format=csv,noheader'], text=True, capture_output=True)
    return {'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}


def native(x):
    return x.sum()


def split(x):
    # Same sum, with two reductions and an add; not claimed to be an optimization.
    return x[:x.numel()//2].sum() + x[x.numel()//2:].sum()


def clocks(fn, x, count, mode):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Allocate event handles outside timing as well as Python event wrappers.
    start.record(); end.record(); end.synchronize()
    torch.cuda.synchronize()
    begin = time.perf_counter_ns()
    start.record()
    for _ in range(count):
        output = fn(x)
        if mode == 'sync_each': torch.cuda.synchronize()
    end.record()
    submitted = time.perf_counter_ns()
    end.synchronize()
    complete = time.perf_counter_ns()
    return dict(count=count,mode=mode,host_submit_us=(submitted-begin)/1e3,
                host_complete_us=(complete-begin)/1e3,event_us=start.elapsed_time(end)*1e3,
                value=float(output.item()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compile-probe',action='store_true')
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    init=time.perf_counter(); torch.cuda.init(); torch.cuda.synchronize()
    init_seconds=time.perf_counter()-init
    prop=torch.cuda.get_device_properties(0); l2=int(prop.L2_cache_size)
    rounds,reps,warmup=5,20,10
    cases=[]
    for ratio in [0.5,1,2,4]:
        size=int(l2*ratio); slots=max(2,math.ceil(4*l2/size))
        for access in ['resident','rotating']:
            cases.append(dict(ratio=ratio,bytes_per_input=size,slots=slots,access=access,
                              pool_bytes=size*slots,rounds=rounds,reps=reps,warmup=warmup))
    save(a.output/'cases.json',cases)
    save(a.output/'manifest.json',dict(torch=torch.__version__,torch_git=torch.version.git_version,
        python=sys.version,cuda=torch.version.cuda,device=str(prop),l2_bytes=l2,
        l2_evidence='torch.cuda.get_device_properties(0).L2_cache_size; device-reported capacity, not hit-rate measurement',
        task_ids=['M2-A','M2-B'],dtype='float32',stride=[1],input_recipe='ones(numel=bytes_per_input/4)',
        max_input_pool_bytes=max(c['pool_bytes'] for c in cases),seed=[0,1,2],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256((a.output/'cases.json').read_bytes()).hexdigest(),
        boundaries='event pairs per call, no intra-batch sync; host complete includes event record and final wait; no host/device transfers inside',
        repetitions='5 alternating rounds x 20 device-event samples; timing repetitions are correlated, not independent tasks',
        background_gpu=telemetry(),cache_counters='UNVERIFIED',model=None))
    # Store the exact installed API source alongside the experiment.
    from torch.cuda import streams
    import shutil
    source=Path(streams.__file__)
    shutil.copyfile(source,a.output/'torch_cuda_streams.py')
    save(a.output/'source.json',dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch_git=torch.version.git_version))
    startup={'torch_import_seconds':TORCH_IMPORT_SECONDS,'cuda_init_seconds':init_seconds,'allocations':[]}
    for i in range(3):
        torch.cuda.synchronize(); t=time.perf_counter()
        x=torch.empty(l2//4,device='cuda'); torch.cuda.synchronize()
        startup['allocations'].append(dict(index=i,seconds=time.perf_counter()-t,
                                           allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))
        del x; gc.collect()
    # Independent small correctness cases including signed values and a non-contiguous view.
    numeric=[]
    for seed in range(3):
        g=torch.Generator().manual_seed(seed)
        for layout in ['contiguous','strided','zero']:
            cpu=torch.randn(1024,generator=g)
            if layout=='strided': cpu=cpu[::2]
            if layout=='zero': cpu.zero_()
            x=torch.empty_strided(cpu.shape,cpu.stride(),device="cuda",dtype=cpu.dtype); x.copy_(cpu); ref=cpu.double().sum().item()
            for name,fn in [('native',native),('split',split)]:
                value=fn(x).item(); numeric.append(dict(seed=seed,layout=layout,method=name,input=cpu.tolist(),stride=list(x.stride()),reference=ref,
                    actual=value,passed=abs(value-ref)<=1e-4+1e-5*abs(ref)))
    save(a.output/'numeric.json',numeric)
    if not all(r['passed'] for r in numeric): raise RuntimeError('numerical gate failed; inputs saved')
    x=torch.ones(l2//8,device='cuda')
    if a.compile_probe:
        fn=torch.compile(native,fullgraph=True)
        startup['compiled_calls']=[]
        for i in range(3):
            torch.cuda.synchronize(); t=time.perf_counter(); y=fn(x); torch.cuda.synchronize()
            startup['compiled_calls'].append(dict(index=i,seconds=time.perf_counter()-t,value=y.item()))
        startup['compile_scope']='First call includes capture/codegen/cache lookup/execution; not pure compiler time. Cache location controlled by runner.'
    save(a.output/'startup.json',startup)
    for _ in range(warmup): native(x)
    torch.cuda.synchronize()
    boundary=[]
    for r in range(rounds):
        modes=[(1,'sync_end'),(20,'sync_end'),(20,'sync_each')]
        if r%2: modes.reverse()
        for count,mode in modes:
            boundary.append(dict(round=r,**clocks(native,x,count,mode)))
    save(a.output/'boundaries.json',boundary)
    del x
    # Profiler is separate from the normal timing loop.
    x=torch.ones(l2//8,device='cuda'); native(x); split(x); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        native(x); split(x); torch.cuda.synchronize()
    prof.export_chrome_trace(str(a.output/'profiler.json'))
    save(a.output/'kernel_names.json',[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA])
    del x
    rows=[]
    for c in cases:
        pool=[torch.ones(c['bytes_per_input']//4,device='cuda') for _ in range(c['slots'])]
        for fn in [native,split]:
            for i in range(warmup): fn(pool[i%c['slots']] if c['access']=='rotating' else pool[0])
        torch.cuda.synchronize()
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
        for start,end in events: start.record();end.record()
        torch.cuda.synchronize()
        for r in range(rounds):
            methods=[('native',native),('split',split)]
            if r%2: methods.reverse()
            state=telemetry()
            for name,fn in methods:
                torch.cuda.synchronize(); begin=time.perf_counter_ns()
                for i,(start,end) in enumerate(events):
                    x=pool[i%c['slots']] if c['access']=='rotating' else pool[0]
                    start.record(); y=fn(x);end.record()
                submitted=time.perf_counter_ns(); events[-1][1].synchronize();complete=time.perf_counter_ns()
                expected=float(c['bytes_per_input']//4)
                record=dict(case=c,round=r,method=name,telemetry=state,
                    event_us=[s.elapsed_time(e)*1e3 for s,e in events],
                    host_submit_us=(submitted-begin)/1e3,host_complete_us=(complete-begin)/1e3,
                    value=y.item(),expected=expected,correct=y.item()==expected)
                rows.append(record); save(a.output/'samples.json',rows)
                if not record['correct']: raise RuntimeError('large-input sum mismatch; record saved')
        del pool,x,y; gc.collect(); torch.cuda.synchronize()
    summary=[]
    for c in cases:
        for name in ['native','split']:
            group=[r for r in rows if r['case']==c and r['method']==name]
            samples=[v for r in group for v in r['event_us']]
            summary.append(dict(ratio=c['ratio'],access=c['access'],method=name,
                                event_median_us=statistics.median(samples),samples=len(samples),
                                round_medians_us=[statistics.median(r['event_us']) for r in group],
                                complete_per_call_us=[r['host_complete_us']/reps for r in group]))
    save(a.output/'summary.json',summary); save(a.output/'gpu-after.json',telemetry())
    (a.output/'exit.txt').write_text('0\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
