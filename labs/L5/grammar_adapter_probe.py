#!/usr/bin/env python3
"""Probe installed adapter termination contracts without loading model weights."""
import json
import sys
import torch
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from structured_output_audit import MODEL, TARGET, LOOSE, versions

mode = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
ids = tok.encode(json.dumps(TARGET, separators=(',', ':')), add_special_tokens=False)
print(json.dumps({'versions': versions(), 'tokenizer_eos': tok.eos_token_id,
                  'generation_eos': gen.eos_token_id}))
if mode == 'outlines':
    import outlines_core as oc
    from outlines_core import json_schema
    from vllm.v1.structured_output.utils import get_outlines_vocabulary
    from vllm.v1.structured_output.backend_outlines import OutlinesGrammar
    vocab = get_outlines_vocabulary(tok)
    index = oc.Index(json_schema.build_regex_from_schema(json.dumps(LOOSE)), vocab.inner)
    for eos in gen.eos_token_id:
        g = OutlinesGrammar(cfg.vocab_size, oc.Guide(index))
        assert g.accept_tokens('probe', ids)
        mask = torch.empty((1, (cfg.vocab_size+31)//32), dtype=torch.int32)
        g.fill_bitmask(mask, 0)
        allowed = bool((int(mask[0, eos//32]) >> (eos%32)) & 1)
        # Call in the same order as the scheduler: accept -> is_terminated.
        pre = g.is_terminated()
        accepted = g.accept_tokens('probe', [eos])
        post = g.is_terminated()
        print(json.dumps({'eos': eos, 'allowed_by_mask': allowed,
                          'adapter_accept': accepted, 'pre_terminated': pre,
                          'post_terminated': post}))
else:
    import xgrammar as xg
    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
    info = xg.TokenizerInfo.from_huggingface(tok, vocab_size=cfg.vocab_size,
                                           stop_token_ids=gen.eos_token_id)
    compiled = xg.GrammarCompiler(info).compile_json_schema(json.dumps(LOOSE))
    g = XGrammarGrammar(xg.GrammarMatcher(compiled, max_rollback_tokens=4),
                        cfg.vocab_size, compiled, gen.eos_token_id)
    mask = g.allocate_vocab_mask(cfg.vocab_size, 1, 'cpu')
    for tid in ids:
        g.fill_vocab_mask(mask, 0)
        assert (int(mask[0,tid//32]) >> (tid%32)) & 1
        g.accept_token(tid)
    g.accept_token(gen.eos_token_id[0])
    assert g.is_terminated()
    g.rollback(1)
    assert not g.is_terminated()
    fresh = g.copy()
    fresh.fill_vocab_mask(mask, 0)
    assert not ((int(mask[0,gen.eos_token_id[0]//32]) >> (gen.eos_token_id[0]%32)) & 1)
    print(json.dumps({'adapter': 'SGLang XGrammarGrammar', 'trajectory_tokens': len(ids),
                      'eos_terminated': True, 'rollback_restored': True,
                      'copy_is_fresh_request': True, 'mask_shape': list(mask.shape),
                      'mask_stride': list(mask.stride())}))
