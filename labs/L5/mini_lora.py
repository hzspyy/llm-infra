#!/usr/bin/env python3
"""Two adapters in one tiny batch; standard-library reference implementation."""
import json


def transpose(a):return [list(row) for row in zip(*a)]

def mm(a,b):
    assert len(a[0])==len(b)
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]

def add(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]


def mixed_linear(x,w,adapters,ids):
    y=mm(x,transpose(w))
    for i,adapter_id in enumerate(ids):
        if adapter_id is None:continue
        a,b,scale=adapters[adapter_id]
        low=mm([x[i]],transpose(a))
        delta=mm(low,transpose(b))[0]
        y[i]=[base+scale*d for base,d in zip(y[i],delta)]
    return y


if __name__=='__main__':
    x=[[1.,2.,3.],[3.,1.,2.],[2.,3.,1.]]
    w=[[1.,0.,1.],[0.,1.,1.]]
    adapters={1:([[1.,0.,0.]],[[1.],[2.]],1.),
              2:([[0.,1.,0.],[0.,0.,1.]],[[1.,0.],[0.,1.]],.5)}
    ids=[1,None,2]
    y=mixed_linear(x,w,adapters,ids)
    reference=[]
    for row,id_ in zip(x,ids):
        if id_ is None:merged=w
        else:
            a,b,scale=adapters[id_]
            delta=[[scale*z for z in rr] for rr in mm(b,a)]
            merged=add(w,delta)
        reference+=mm([row],transpose(merged))
    assert y==reference
    assert y[1]==mm([x[1]],transpose(w))[0]
    print(json.dumps({'x':x,'W':w,'adapters':adapters,'token_adapter_ids':ids,
                      'output':y,'per_row_merged_reference':reference,'equal':y==reference}))
    # Rows are grouped for efficient low-rank kernels, then scattered back.
    groups={id_:[i for i,v in enumerate(ids) if v==id_] for id_ in set(ids)}
    print('adapter_groups =',groups)
