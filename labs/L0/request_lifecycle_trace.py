#!/usr/bin/env python3
"""0.1-A: follow one request id across processes, threads and engine steps.

Two phases, run separately because they answer different questions.

  inproc  in-process AsyncLLM with the four boundaries instrumented; gives the
          per-step record (one prefill step, then one decode step per token).
  server  a real `vllm serve` process tree; gives the process/thread boundaries
          and the same request id seen from HTTP and from the server log.

python labs/L0/request_lifecycle_trace.py inproc --output results/crater/0.1/<run>
python labs/L0/request_lifecycle_trace.py server --output results/crater/0.1/<run> \
    --base-url http://127.0.0.1:8010 --server-pid <pid> --server-log <path>
"""
import argparse,hashlib,json,os,platform,threading,time
from pathlib import Path

PROMPT="Explain what a KV cache is in one sentence."
MAX_TOKENS=3


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return time.time()


def where():
    """Identity of whoever is executing this hook right now."""
    t=threading.current_thread()
    return dict(pid=os.getpid(),tid=threading.get_ident(),thread=t.name)


# ---------------------------------------------------------------- inproc phase

def run_inproc(out):
    """Synchronous LLM with multiprocessing off: every layer lands in one process.

    AsyncLLM cannot be traced this way - core_client.py:89 make_client refuses
    asyncio_mode without multiprocess_mode, so the server path is always split.
    """
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ.setdefault('VLLM_LOGGING_LEVEL','WARNING')
    import inspect
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.core_client import EngineCoreClient,InprocClient
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.engine.output_processor import OutputProcessor

    events=[];step=[0];client_kind=[None];tracing=[False];warmup=[0];empty=[0]
    def record(**kw):
        if tracing[0]:events.append(dict(t=now(),**where(),**kw))

    def symbol(fn):
        f=inspect.unwrap(fn)
        return dict(file=inspect.getsourcefile(f).split('site-packages/')[-1],line=f.__code__.co_firstlineno)

    symbols=dict(add_request=symbol(LLMEngine.add_request),step=symbol(LLMEngine.step),
        make_client=symbol(EngineCoreClient.make_client),schedule=symbol(Scheduler.schedule),
        execute_model=symbol(Worker.execute_model),process_outputs=symbol(OutputProcessor.process_outputs))

    original=dict(add_request=LLMEngine.add_request,schedule=Scheduler.schedule,
        execute_model=Worker.execute_model,process_outputs=OutputProcessor.process_outputs,
        make_client=EngineCoreClient.make_client)

    def make_client(*a,**kw):
        client=original['make_client'](*a,**kw)
        client_kind[0]=type(client).__name__
        return client

    def add_request(self,request_id,*a,**kw):
        record(hook='LLMEngine.add_request',request_ids=[request_id])
        return original['add_request'](self,request_id,*a,**kw)

    def queues(sched):
        """Depth of the two queues the scheduler picks from."""
        return dict(waiting=len(getattr(sched,'waiting',()) or ()),
                    running=len(getattr(sched,'running',()) or ()))

    def schedule(self,*a,**kw):
        before=queues(self)
        output=original['schedule'](self,*a,**kw)
        counts=dict(output.num_scheduled_tokens)
        if counts:
            step[0]+=1
            record(hook='Scheduler.schedule',engine_step=step[0],request_ids=sorted(counts),
                num_scheduled_tokens=counts,total_num_scheduled_tokens=output.total_num_scheduled_tokens,
                new_reqs=[r.req_id for r in output.scheduled_new_reqs],
                phase='prefill' if output.scheduled_new_reqs else 'decode',
                queue_before=before,queue_after=queues(self),
                finished_req_ids=sorted(getattr(self,'finished_req_ids',()) or ()))
        elif tracing[0]:
            # An empty schedule still costs a loop iteration; count them separately.
            empty[0]+=1
        return output

    def execute_model(self,scheduler_output,*a,**kw):
        counts=dict(scheduler_output.num_scheduled_tokens)
        start=now();result=original['execute_model'](self,scheduler_output,*a,**kw)
        if counts:
            if tracing[0]:
                events.append(dict(t=start,**where(),hook='Worker.execute_model',engine_step=step[0],
                    request_ids=sorted(counts),num_scheduled_tokens=counts,returned_at=now()))
            else:warmup[0]+=1
        return result

    def process_outputs(self,engine_core_outputs,*a,**kw):
        ids=[o.request_id for o in engine_core_outputs]
        if ids:record(hook='OutputProcessor.process_outputs',engine_step=step[0],request_ids=sorted(set(ids)),
            new_token_ids=[list(o.new_token_ids) for o in engine_core_outputs],
            finished=[bool(o.finished) for o in engine_core_outputs])
        return original['process_outputs'](self,engine_core_outputs,*a,**kw)

    LLMEngine.add_request=add_request;Scheduler.schedule=schedule
    Worker.execute_model=execute_model;OutputProcessor.process_outputs=process_outputs
    EngineCoreClient.make_client=staticmethod(make_client)

    llm=LLM(model=MODEL,max_model_len=2048,gpu_memory_utilization=GPU_FRACTION,
        enforce_eager=True,enable_prefix_caching=False,disable_log_stats=True)
    tokenizer=llm.get_tokenizer();prompt_len=len(tokenizer.encode(PROMPT))
    tracing[0]=True
    record(hook='client.submit',request_ids=[REQUEST_ID],prompt_tokens=prompt_len)
    outputs=llm.generate([PROMPT],SamplingParams(max_tokens=MAX_TOKENS,temperature=0.0,seed=0))
    record(hook='client.done',request_ids=[REQUEST_ID])
    tracing[0]=False
    text=outputs[0].outputs[0].text;token_ids=list(outputs[0].outputs[0].token_ids)

    for k,v in original.items():
        target={'add_request':LLMEngine,'schedule':Scheduler,'execute_model':Worker,
                'process_outputs':OutputProcessor,'make_client':EngineCoreClient}[k]
        setattr(target,k,staticmethod(v) if k=='make_client' else v)

    steps=[e for e in events if e['hook']=='Scheduler.schedule']
    prefill=[s for s in steps if s['phase']=='prefill'];decode=[s for s in steps if s['phase']=='decode']
    frontend_ids=sorted({i for e in events if e['hook']=='LLMEngine.add_request' for i in e['request_ids']})
    engine_ids=sorted({i for e in events if e['hook'] in ('Scheduler.schedule','Worker.execute_model',
                       'OutputProcessor.process_outputs') for i in e.get('request_ids',[])})
    result=dict(vllm=vllm.__version__,python=platform.python_version(),model=MODEL,
        prompt=PROMPT,prompt_tokens=prompt_len,max_tokens=MAX_TOKENS,output_text=text,output_token_ids=token_ids,
        multiprocessing=os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'],engine_core_client=client_kind[0],
        enforce_eager=True,symbols=symbols,frontend_request_ids=frontend_ids,engine_request_ids=engine_ids,
        warmup_execute_model_calls=warmup[0],empty_schedule_calls=empty[0],
        queue_depth_per_step=[dict(step=e['engine_step'],phase=e['phase'],
            before=e['queue_before'],after=e['queue_after']) for e in steps],
        submit_to_first_schedule_s=(steps[0]['t']-next(e['t'] for e in events
            if e['hook']=='LLMEngine.add_request')) if steps else None,
        engine_steps=len(steps),prefill_steps=len(prefill),decode_steps=len(decode),
        prefill_scheduled_tokens=prefill[0]['total_num_scheduled_tokens'] if prefill else None,
        decode_scheduled_tokens=[d['total_num_scheduled_tokens'] for d in decode],
        distinct_pids=sorted({e['pid'] for e in events}),
        threads_by_hook={h:sorted({e['thread'] for e in events if e['hook']==h})
                         for h in sorted({e['hook'] for e in events})},
        events=events)
    save(out/'inproc_trace.json',result)
    ok=(result['prefill_steps']==1 and result['decode_steps']==MAX_TOKENS-1
        and result['prefill_scheduled_tokens']==prompt_len
        and result['decode_scheduled_tokens']==[1]*(MAX_TOKENS-1)
        and client_kind[0]=='InprocClient' and len(result['distinct_pids'])==1
        and len(frontend_ids)==1 and len(engine_ids)==1
        and engine_ids[0].startswith(frontend_ids[0]+'-')
        and sum(1 for e in events if e['hook']=='Worker.execute_model')==MAX_TOKENS)
    return ok,dict(engine_steps=result['engine_steps'],prefill_tokens=result['prefill_scheduled_tokens'],
        decode_tokens=result['decode_scheduled_tokens'],pids=result['distinct_pids'],
        client=client_kind[0],frontend_request_ids=frontend_ids,engine_request_ids=engine_ids,
        warmup_execute_model_calls=warmup[0],empty_schedule_calls=empty[0],
        execute_model_calls=sum(1 for e in events if e['hook']=='Worker.execute_model'))


# ---------------------------------------------------------------- server phase

def process_tree(pid):
    """Read the real process/thread layout of a running vllm serve from /proc."""
    def read(p,*rest):
        try:return Path('/proc',str(p),*rest).read_text()
        except OSError:return ''
    def describe(p):
        tasks=sorted(Path('/proc',str(p),'task').iterdir(),key=lambda x:int(x.name)) if Path('/proc',str(p),'task').is_dir() else []
        names={}
        for t in tasks:
            try:names[t.name]=(t/'comm').read_text().strip()
            except OSError:pass
        return dict(pid=int(p),cmdline=read(p,'cmdline').replace('\x00',' ').strip(),
            comm=read(p,'comm').strip(),threads=len(tasks),thread_names=sorted(set(names.values())))
    children=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        stat=read(entry.name,'status')
        for line in stat.splitlines():
            if line.startswith('PPid:') and line.split()[1]==str(pid):children.append(describe(entry.name))
    return dict(parent=describe(pid),children=sorted(children,key=lambda c:c['pid']))


def run_server(out,base_url,server_pid,server_log):
    import urllib.request
    tree=process_tree(server_pid) if server_pid else None
    body=json.dumps(dict(model=MODEL,prompt=PROMPT,max_tokens=MAX_TOKENS,temperature=0.0,seed=0,stream=False)).encode()
    req=urllib.request.Request(base_url.rstrip('/')+'/v1/completions',data=body,
        headers={'Content-Type':'application/json','X-Request-Id':REQUEST_ID})
    sent=now()
    with urllib.request.urlopen(req,timeout=120) as resp:
        raw=resp.read();headers=dict(resp.headers)
    received=now();payload=json.loads(raw)
    served_id=payload.get('id')
    log_lines=[]
    if server_log and Path(server_log).is_file():
        text=Path(server_log).read_text(errors='replace')
        for line in text.splitlines():
            if REQUEST_ID in line or (served_id and served_id in line):log_lines.append(line)
    result=dict(engine=ENGINE,model=MODEL,request_id_sent=REQUEST_ID,response_id=served_id,base_url=base_url,
        client_sent=sent,client_received=received,client_elapsed_s=received-sent,
        response_headers=headers,prompt_tokens=payload.get('usage',{}).get('prompt_tokens'),
        completion_tokens=payload.get('usage',{}).get('completion_tokens'),
        output_text=payload['choices'][0]['text'],finish_reason=payload['choices'][0]['finish_reason'],
        process_tree=tree,server_log_lines_matching_id=log_lines,server_log=str(server_log) if server_log else None)
    save(out/'server_trace.json',result)
    markers={'vllm':['EngineCore'],'sglang':['scheduler','detokenizer','Scheduler','Detokenizer']}[ENGINE]
    ok=(served_id is not None and result['completion_tokens']==MAX_TOKENS
        and tree is not None and any(any(m in c['comm'] or m in c['cmdline'] for m in markers)
                                     for c in tree['children']))
    return ok,dict(response_id=served_id,child_pids=[c['pid'] for c in tree['children']] if tree else None,
        log_lines=len(log_lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['inproc','server'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',default='Qwen/Qwen3-1.7B')
    ap.add_argument('--request-id',default='l01-trace-0001')
    ap.add_argument('--gpu-fraction',type=float,default=0.25)
    ap.add_argument('--base-url',default='http://127.0.0.1:8010')
    ap.add_argument('--server-pid',type=int)
    ap.add_argument('--server-log')
    ap.add_argument('--engine',choices=['vllm','sglang'],default='vllm')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    global MODEL,REQUEST_ID,GPU_FRACTION,ENGINE
    MODEL=a.model;REQUEST_ID=a.request_id;GPU_FRACTION=a.gpu_fraction;ENGINE=a.engine
    save(a.output/'manifest.json',dict(phase=a.phase,engine=ENGINE,model=MODEL,request_id=REQUEST_ID,prompt=PROMPT,
        max_tokens=MAX_TOKENS,gpu_memory_utilization=GPU_FRACTION,task_ids=['0.1-A'],timing='wall clock only',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),host=platform.node()))
    ok,summary=(run_inproc(a.output) if a.phase=='inproc'
                else run_server(a.output,a.base_url,a.server_pid,a.server_log))
    save(a.output/'summary.json',dict(phase=a.phase,passed=ok,**summary))
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(dict(phase=a.phase,passed=ok,**summary)));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
