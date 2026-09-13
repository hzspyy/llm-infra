#!/usr/bin/env python3
"""Explicit incremental attention using the existing TinyLM parameter objects.

Cache belongs to the caller; this inference-only extension never mutates old cache.
"""
import math
import torch
from tiny_lm import TinyLM


class CachedTinyLM(TinyLM):
    def __init__(self,*args,tie_weights=False,**kwargs):
        super().__init__(*args,**kwargs)
        if tie_weights:
            self.lm_head.weight=self.tok_emb.weight

    @torch.no_grad()
    def forward_cached(self,idx,cache=None,*,reset_position=False):
        if idx.ndim!=2 or idx.shape[1]==0:
            raise ValueError('idx must be [batch, nonempty tokens]')
        batch,count=idx.shape
        if cache is None:
            cache=[None]*len(self.blocks);offset=0
        else:
            if len(cache)!=len(self.blocks): raise ValueError('one KV pair per layer required')
            lengths=[]
            for block,item in zip(self.blocks,cache):
                if item is None or len(item)!=2: raise ValueError('incomplete KV pair')
                k,v=item
                if k.shape!=v.shape or k.ndim!=4 or k.shape[:2]!=(batch,block.attn.n_head) or k.shape[-1]!=block.attn.d_head:
                    raise ValueError('cache shape mismatch')
                if k.device!=idx.device or v.device!=idx.device or k.dtype!=self.tok_emb.weight.dtype or v.dtype!=k.dtype:
                    raise ValueError('cache device/dtype mismatch')
                lengths.append(k.shape[2])
            if len(set(lengths))!=1: raise ValueError('layer cache lengths differ')
            offset=lengths[0]
        if offset+count>self.block_size: raise ValueError('cache exceeds block_size; window rebasing is not implemented')
        x=self.tok_emb(idx)
        if self.use_pos:
            positions=torch.arange(count,device=idx.device)+(0 if reset_position else offset)
            x=x+self.pos_emb(positions)
        next_cache=[]
        for block,old in zip(self.blocks,cache):
            attn=block.attn; normalized=block.ln1(x)
            q,k,v=attn.qkv(normalized).chunk(3,dim=-1)
            def heads(t): return t.reshape(batch,count,attn.n_head,attn.d_head).transpose(1,2)
            q,k,v=map(heads,(q,k,v))
            if old is not None:
                k=torch.cat((old[0],k),dim=2);v=torch.cat((old[1],v),dim=2)
            scores=q@k.transpose(-1,-2)
            if attn.scale: scores=scores/math.sqrt(attn.d_head)
            key_pos=torch.arange(offset+count,device=idx.device)
            query_pos=offset+torch.arange(count,device=idx.device)
            scores=scores.masked_fill(key_pos[None,:]>query_pos[:,None],float('-inf'))
            mixed=(scores.softmax(-1)@v).transpose(1,2).contiguous().reshape(batch,count,-1)
            x=x+attn.proj(mixed)
            x=x+block.mlp(block.ln2(x))
            next_cache.append((k,v))
        return self.lm_head(self.ln_f(x)),next_cache
