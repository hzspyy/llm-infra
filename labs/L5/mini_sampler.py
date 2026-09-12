#!/usr/bin/env python3
"""Transparent finite-vocabulary sampler; vLLM-style filter order.

Tie policy: top-k keeps every token equal to the kth logit.
Top-p removes the ascending low-probability tail, preserving a maximum.
"""
from collections import Counter
import json
import math
import random


def softmax(logits):
    m = max(logits)
    if not math.isfinite(m):
        raise ValueError('no finite candidate')
    weights = [math.exp(x-m) for x in logits]
    total = math.fsum(weights)
    return [x/total for x in weights]


def process(logits, prompt=(), output=(), repetition=1., frequency=0.,
            presence=0., temperature=1., min_p=0., top_k=None, top_p=1.):
    if repetition <= 0 or temperature < 0 or not 0 <= min_p <= 1 or not 0 < top_p <= 1:
        raise ValueError('invalid sampling parameter')
    if any(math.isnan(x) or x == math.inf for x in logits):
        raise ValueError('invalid logits')
    z = list(logits)
    trace = {'raw': z.copy(), 'raw_probs': softmax(z)}
    seen, counts = set(prompt) | set(output), Counter(output)
    for i, x in enumerate(z):
        if i in seen:
            z[i] = x / repetition if x > 0 else x * repetition
        z[i] -= frequency * counts[i] + presence * int(counts[i] > 0)
    trace['penalties'] = z.copy()
    if temperature == 0:
        winner = max(range(len(z)), key=z.__getitem__)
        trace['greedy_id'] = winner
        return trace, [float(i == winner) for i in range(len(z))]
    z = [x / temperature for x in z]
    trace['temperature'] = z.copy()
    if min_p > 0:
        threshold = max(z) + math.log(min_p)
        z = [x if x >= threshold else -math.inf for x in z]
    trace['min_p'] = z.copy()
    if top_k is not None:
        if not 1 <= top_k <= len(z):
            raise ValueError('invalid top_k')
        threshold = sorted(z, reverse=True)[top_k-1]
        z = [x if x >= threshold else -math.inf for x in z]
    trace['top_k'] = z.copy()
    probs = softmax(z)
    order = sorted(range(len(z)), key=z.__getitem__)
    cumulative = 0.
    for i in order[:-1]:
        cumulative += probs[i]
        if cumulative <= 1-top_p:
            z[i] = -math.inf
    trace['top_p'] = z.copy()
    trace['processed_probs'] = softmax(z)
    return trace, trace['processed_probs']


def categorical(probs, rng):
    u, cumulative = rng.random(), 0.
    for i, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return i
    return max(i for i,p in enumerate(probs) if p > 0)


if __name__ == '__main__':
    trace, probs = process([-1.,2.,1.,.5], prompt=[0,1], output=[3,3,1],
                           repetition=1.5, frequency=.25, presence=.5,
                           temperature=.5, min_p=.3, top_k=2, top_p=.6)
    print(json.dumps({'trace':trace,'selected':categorical(probs,random.Random(7))}))
    assert probs == [0.,0.,1.,0.]
    tie,_ = process([1.,1.,1.,0.],top_k=2)
    assert sum(math.isfinite(x) for x in tie['top_k']) == 3
    greedy,p = process([1.,2.],temperature=0,top_k=1,min_p=1)
    assert greedy['greedy_id'] == 1
    try:softmax([-math.inf,-math.inf])
    except ValueError:pass
    else:raise AssertionError('empty distribution was accepted')
    print('checks: sign-aware repetition, output counts, ties, greedy and empty support passed')
