#!/usr/bin/env python3
"""Small real-process watchdog and commit/replay protocol, without GPU training.

Workers exchange progress/ACK messages. Faults include process exit, a silent
worker, an invalid batch, delayed mock reward/teacher calls, and interrupted
checkpoint publication. Physical processing may repeat; committed IDs do not.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time


def worker(connection, rank):
    connection.send({'kind':'ready','rank':rank,'pid':os.getpid()})
    try:
        while True:
            command = connection.recv()
            if command['op'] == 'stop':
                return
            connection.send({'kind':'progress','rank':rank,'pid':os.getpid(),
                             'step':command['step'],'at':time.monotonic()})
            fault = command.get('fault')
            if fault == 'crash':
                os._exit(17)
            if fault in ('slow_rank','reward_timeout','teacher_timeout'):
                time.sleep(3.)
            values = command['sample_ids']
            if fault == 'bad_batch':
                values = None
            if not isinstance(values,list):
                connection.send({'kind':'error','rank':rank,'error':'invalid sample_ids batch'})
                return
            # An intentionally simple deterministic state update, not an NN step.
            next_state = command['state'] + sum(values)
            connection.send({'kind':'ack','rank':rank,'pid':os.getpid(),'step':command['step'],
                             'sample_ids':values,'state':next_state})
    finally:
        connection.close()


class ProcessCoordinator:
    def __init__(self, root, fault, world_size=2, timeout=.5):
        self.root = Path(root)
        self.fault = fault
        self.world_size = world_size
        self.timeout = timeout
        self.context = mp.get_context('spawn')
        self.processes = []
        self.connections = []
        self.all_processes = []
        self.events = []
        self.processed = []
        self.t0 = time.monotonic()

    def log(self, kind, **fields):
        self.events.append({'at_ms':(time.monotonic()-self.t0)*1000,'kind':kind,**fields})

    def start(self):
        for rank in range(self.world_size):
            parent,child = self.context.Pipe()
            process = self.context.Process(target=worker,args=(child,rank))
            process.start()
            child.close()
            self.processes.append(process)
            self.all_processes.append(process)
            self.connections.append(parent)
            if not parent.poll(10):
                raise TimeoutError('worker did not become ready')
            self.log('spawn',**{k:v for k,v in parent.recv().items() if k!='kind'})

    def stop(self):
        for conn, process in zip(self.connections,self.processes):
            if process.is_alive():
                try:conn.send({'op':'stop'})
                except (BrokenPipeError,EOFError,OSError):pass
        for conn,process in zip(self.connections,self.processes):
            process.join(.2)
            terminated = False
            if process.is_alive():
                process.terminate()
                terminated = True
                process.join(1.)
            if process.is_alive():
                process.kill()
                process.join(1.)
            self.log('reaped',pid=process.pid,exit_code=process.exitcode,terminated=terminated)
            assert not process.is_alive()
            conn.close()
        self.connections=[]
        self.processes=[]

    def publish(self,state,interrupt=False):
        payload=json.dumps(state,sort_keys=True).encode()
        pending=self.root/f"step-{state['step']}.pending"
        pending.write_bytes(payload)
        if interrupt:
            self.log('publication_interrupted',step=state['step'],pending_bytes=len(payload))
            raise RuntimeError('checkpoint not published')
        committed=self.root/f"step-{state['step']}.json"
        pending.replace(committed)
        latest=self.root/'latest.next'
        latest.write_text(committed.name)
        latest.replace(self.root/'latest')
        self.log('committed',step=state['step'],sample_ids=state['committed_ids'])

    def load(self):
        filename=(self.root/'latest').read_text()
        assert filename.endswith('.json')
        return json.loads((self.root/filename).read_text())

    def execute(self,state,attempt):
        step=state['step']+1
        for rank,conn in enumerate(self.connections):
            ids=list(range((step-1)*4+rank*2,(step-1)*4+rank*2+2))
            fault=self.fault if attempt==0 and step==2 and rank==1 else None
            conn.send({'op':'step','step':step,'state':state['rank_states'][rank],
                       'sample_ids':ids,'fault':fault})
        pending=set(range(self.world_size));acks={}
        last_progress={rank:time.monotonic() for rank in pending}
        while pending:
            now=time.monotonic()
            for rank in list(pending):
                conn=self.connections[rank]
                if conn.poll(.005):
                    try:message=conn.recv()
                    except EOFError:raise RuntimeError(f'rank {rank} exited before ACK')
                    last_progress[rank]=time.monotonic()
                    if message['kind']=='ack':
                        acks[rank]=message
                        self.processed.extend(message['sample_ids'])
                        pending.remove(rank)
                        self.log('processed',attempt=attempt,**{k:v for k,v in message.items() if k!='kind'})
                    elif message['kind']=='error':
                        raise RuntimeError(message['error'])
                    else:
                        self.log('heartbeat',attempt=attempt,**{k:v for k,v in message.items() if k!='kind'})
                elif not self.processes[rank].is_alive():
                    raise RuntimeError(f'rank {rank} exited before ACK')
                elif now-last_progress[rank]>self.timeout:
                    raise TimeoutError(f'rank {rank} progress deadline expired')
        return {'step':step,'rank_states':[acks[r]['state'] for r in range(self.world_size)],
                'committed_ids':state['committed_ids']+[i for r in range(self.world_size) for i in acks[r]['sample_ids']]}

    def run(self):
        state={'step':0,'rank_states':[0]*self.world_size,'committed_ids':[]}
        self.publish(state)
        attempt=0
        try:
            self.start()
            while state['step']<2:
                try:
                    candidate=self.execute(state,attempt)
                    interrupted=self.fault=='checkpoint_uncommitted' and attempt==0 and candidate['step']==2
                    self.publish(candidate,interrupted)
                    state=candidate
                except (RuntimeError,TimeoutError,EOFError,BrokenPipeError) as error:
                    self.log('failure',attempt=attempt,error_type=type(error).__name__,message=str(error))
                    self.stop()
                    state=self.load()
                    self.log('restore',step=state['step'],rank_states=state['rank_states'])
                    attempt+=1
                    if attempt>1:raise
                    self.start()
        finally:
            self.stop()
        assert state['committed_ids']==list(range(8))
        assert state['rank_states']==[10,18]
        assert all(not p.is_alive() for p in self.all_processes)
        return {'fault':self.fault,'world_size':self.world_size,'worker_pids':[p.pid for p in self.all_processes],
                'events':self.events,'physical_processing_ids':self.processed,
                'physical_repeats':len(self.processed)-len(set(self.processed)),
                'final_committed_state':state,'restarts':attempt,'no_live_children':True,
                'scope':'real OS process protocol; deterministic scalar updates, no GPU/model-quality measurement'}


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Real-process watchdog and committed-data replay contracts')
    parser.add_argument('--outdir',required=True,type=Path)
    parser.add_argument('--scratch',type=Path,default=Path('/Volumes/data/artifacts/llm-infra/scratch'),
                        help='临时工作区；默认写外置盘，不落仓库与内置盘')
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    args.scratch.mkdir(parents=True,exist_ok=True)
    faults=('crash','slow_rank','bad_batch','checkpoint_uncommitted','reward_timeout','teacher_timeout')
    rows=[]
    for fault in faults:
        with tempfile.TemporaryDirectory(prefix='job-contract-',dir=args.scratch) as root:
            rows.append(ProcessCoordinator(root,fault).run())
    (args.outdir/'training_job_simulation_recovery.json').write_text(
        json.dumps({'cases':rows},ensure_ascii=False,indent=2)+'\n')
    cases={'device':'CPU','world_size':2,'steps':2,'timeout_seconds':.5,
           'faults':[r['fault'] for r in rows],
           'checkpoint_protocol':'custom atomic JSON commit, not DCP',
           'service_timeouts':'mock provider delays inside a child process',
           'scratch_dir':str(args.scratch)}
    (args.outdir/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    manifest={'argv':sys.argv,'sources':[str(__file__)],
              'claim_scope':'real OS process protocol with deterministic scalar state; '
                            'no GPU training, model-quality or DCP measurement'}
    (args.outdir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for row in rows:
        print(f"{row['fault']:22s} restarts={row['restarts']} "
              f"physical_ids={len(row['physical_processing_ids'])} repeats={row['physical_repeats']} "
              f"committed={row['final_committed_state']['committed_ids']} "
              f"no_live_children={row['no_live_children']}")


if __name__=='__main__':
    main()
