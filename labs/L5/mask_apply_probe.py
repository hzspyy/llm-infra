#!/usr/bin/env python3
"""Verify XGrammar's GPU mask application on captured model-vocabulary masks."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import xgrammar as xg

path = Path(sys.argv[1])
words = np.frombuffer(path.read_bytes(), dtype='<i4').copy()
mask = torch.from_numpy(words).reshape(1, -1).cuda()
V = 151936
logits = torch.arange(V, dtype=torch.float32, device='cuda').reshape(1, V)
original = logits.clone()
xg.apply_token_bitmask_inplace(logits, mask)
torch.cuda.synchronize()
i = torch.arange(V, device='cuda')
allowed = ((mask[0, i // 32] >> (i % 32)) & 1).bool()
assert torch.isneginf(logits[0, ~allowed]).all()
assert torch.equal(logits[0, allowed], original[0, allowed])
selected = int(logits.argmax())
assert bool(allowed[selected])
print(json.dumps({'backend': 'XGrammar GPU apply_token_bitmask_inplace',
 'logits_shape': list(logits.shape), 'logits_stride': list(logits.stride()),
 'mask_shape': list(mask.shape), 'mask_stride': list(mask.stride()),
 'allowed_count': int(allowed.sum()), 'masked_to_negative_inf': int((~allowed).sum()),
 'legal_logits_unchanged': True, 'selected_token_id': selected}))
