#!/usr/bin/env python3
"""Paired cluster bootstrap for request/task/window or benchmark-round records.

Input JSON list: {method, cluster, value, status}; value in declared common unit.
Completed samples use status=ok. Failures are counted, never silently dropped.
This estimates latency conditional on successful completion; goodput/SLO needs a separate metric.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def quantile(values, q):
    values=sorted(values)
    if not values: raise ValueError('empty sample')
    index=(len(values)-1)*q; lower=int(index);upper=min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(index-lower)


def analyze(records, baseline, candidate, q=0.5, draws=2000, seed=0):
    if baseline==candidate: raise ValueError('methods must differ')
    if not 0<=q<=1 or draws<1: raise ValueError('invalid q/draws')
    groups={name:{} for name in (baseline,candidate)}; counts={name:{} for name in groups}
    for row in records:
        name=row['method']
        if name not in groups: continue
        cluster=str(row['cluster']);status=row['status']
        counts[name][status]=counts[name].get(status,0)+1
        groups[name].setdefault(cluster,[])
        if status=='ok':
            value=row['value']
            if not isinstance(value,(float,int)) or not math.isfinite(value): raise ValueError('invalid successful value')
            groups[name][cluster].append(float(value))
    if set(groups[baseline])!=set(groups[candidate]): raise ValueError('paired cluster identities differ')
    keys=sorted(groups[baseline]); n=len(keys)
    if n<2: raise ValueError('need at least two independent clusters')
    if any(not groups[m][k] for m in groups for k in keys):
        raise ValueError('empty-success cluster: conditional latency undefined; use failure/SLO metric')
    def metric(name, picked): return quantile([v for k in picked for v in groups[name][k]],q)
    rng=random.Random(seed); delta=[]
    for _ in range(draws):
        picked=rng.choices(keys,k=n)
        delta.append(metric(candidate,picked)-metric(baseline,picked))
    a,b=metric(baseline,keys),metric(candidate,keys)
    return dict(quantile=q,clusters=n,counts=counts,baseline=a,candidate=b,difference=b-a,
                percentile_interval_95=[quantile(delta,0.025),quantile(delta,0.975)],draws=draws,seed=seed,
                estimator='paired whole-cluster resampling; sample-weighted quantile over successful records',
                limitations=['cluster independence is a study-design assumption, not established by this function',
                             'small cluster counts limit interval reliability',
                             'failures counted separately; conditional latency alone cannot rank service quality'],
                p99_stability_claim=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--q',type=float,default=0.5);ap.add_argument('--draws',type=int,default=2000);ap.add_argument('--unit',required=True)
    a=ap.parse_args(); result=analyze(json.loads(a.input.read_text()),a.baseline,a.candidate,a.q,a.draws)
    result.update(unit=a.unit,input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__':main()
