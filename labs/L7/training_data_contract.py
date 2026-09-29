#!/usr/bin/env python3
"""文本清洗与 batch 组装：MinHash 近重复、污染排查、padding/packing 与 loss 边界。

覆盖内容：
1. 自制 30 条短文上的规则过滤、精确/近似去重与自制评测串的污染排查
2. padding 与 packing 两种 collator：cu_seqlens、position reset、跨文档 loss 与 attention 隔离
3. 逐文档 shift 与整段 shift 的梯度对拍，以及装不下的样本如何交还调度器

样本身份与拆分单元见 sample_schema_survey.py，混合权重见 data_mixture_sampling.py，
存储布局与供数队列见 data_supply_paths.py。

Usage:
    python labs/L7/training_data_contract.py --outdir "$RUN_DIR/data"
"""

import os
from pathlib import Path
import sys
import json
import math
import random
import hashlib
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

# 固定随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "local", "7.8"
)


# ==============================================================================
# 1. 文本清洗与去重（MinHash 与测试集污染排查）
# ==============================================================================

class MinHashDedup:
    """
    轻量 MinHash 文本近重复检测器。
    用于识别高 Jaccard 相似度的重复文档，并排查测试集泄漏。
    """
    def __init__(self, num_perm: int = 64, ngram: int = 3, seed: int = 42):
        self.num_perm = num_perm
        self.ngram = ngram
        self.seed = seed
        # 生成随机哈希函数参数: (a * hash + b) % prime
        self.prime = 4294967311  # 2^32 + 15
        rng = random.Random(seed)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

    def _get_shingles(self, text: str) -> set:
        tokens = text.lower().split()
        if len(tokens) < self.ngram:
            return {text.lower()}
        return {" ".join(tokens[i:i+self.ngram]) for i in range(len(tokens) - self.ngram + 1)}

    def compute_signature(self, text: str) -> List[int]:
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm
        
        # 将 shingle 映射为 32 位无符号整数
        shingle_hashes = [int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) for s in shingles]
        
        sig = []
        for i in range(self.num_perm):
            min_val = self.prime
            a_i = self.a[i]
            b_i = self.b[i]
            for h in shingle_hashes:
                val = (a_i * h + b_i) % self.prime
                if val < min_val:
                    min_val = val
            sig.append(min_val)
        return sig

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        assert len(sig1) == len(sig2) == self.num_perm
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / self.num_perm


def run_dedup_and_contamination_test() -> Dict[str, Any]:
    """
    构建 30 条文本样本，测试：
    1. 精确重复检测
    2. MinHash 近重复检测（相似度阈值 >= 0.75）
    3. 乱码/长度异常过滤
    4. 训练集 vs 自制测试文本的 MinHash / 40 字符窗口排查
    """
    dedup = MinHashDedup(num_perm=64, ngram=3, seed=42)
    
    # 构造 30 个自制样本
    base_docs = [
        "Distributed deep learning frameworks require careful coordination of all-reduce communication.",
        "FlashAttention tiles the softmax computation to minimize high-bandwidth memory transfers.",
        "Quantization reduces memory footprint by mapping floating point weights to lower precision integers.",
        "Speculative decoding utilizes a small draft model to generate candidates verified by the target model.",
        "KV cache management with paging eliminates memory fragmentation in high-throughput inference engines.",
        "Mixture of experts models route tokens dynamically to sparse feedforward networks.",
        "Reinforcement learning from human feedback aligns language models with human preferences via reward modeling.",
        "Stateful data loaders preserve exact worker cursors across spot preemption and checkpoint restart.",
        "AdamW decouples weight decay from gradient updates preventing coupled scale distortion.",
        "Direct preference optimization formulates policy training directly on preference pairs without explicit rewards."
    ]
    
    dataset = []
    # 0-9: 正常基础样本
    for i, doc in enumerate(base_docs):
        dataset.append({"id": f"train_{i:02d}", "text": doc, "type": "original"})
    
    # 10-12: 精确重复 (Duplicate)
    dataset.append({"id": "train_10", "text": base_docs[0], "type": "exact_dup_of_00"})
    dataset.append({"id": "train_11", "text": base_docs[3], "type": "exact_dup_of_03"})
    dataset.append({"id": "train_12", "text": base_docs[8], "type": "exact_dup_of_08"})
    
    # 13-16: 近重复 (Near Duplicate，微改动标点与虚词)
    dataset.append({"id": "train_13", "text": base_docs[1] + " This is crucial for GPU efficiency.", "type": "near_dup_of_01"})
    dataset.append({"id": "train_14", "text": "In modern setups, " + base_docs[4], "type": "near_dup_of_04"})
    dataset.append({"id": "train_15", "text": base_docs[5].replace("dynamically", "efficiently"), "type": "near_dup_of_05"})
    dataset.append({"id": "train_16", "text": base_docs[6].replace("human preferences", "user preferences"), "type": "near_dup_of_06"})
    
    # 17-20: 乱码与低质样本 (Garbage / Short)
    dataset.append({"id": "train_17", "text": "asdf qwer zxcv 12345 67890 !@#$%^", "type": "garbage"})
    dataset.append({"id": "train_18", "text": "OK.", "type": "too_short"})
    dataset.append({"id": "train_19", "text": "null undefined NaN [object Object]", "type": "garbage"})
    dataset.append({"id": "train_20", "text": "aaaa " * 40, "type": "repetitive_chars"})
    
    # 21-25: 更多正常独立样本
    additional_docs = [
        "Rotary position embedding encodes relative distances through complex coordinate rotations.",
        "Low rank adaptation injects trainable decomposition matrices into frozen linear layers.",
        "Grouped query attention shares key-value heads across multiple query heads to save cache memory.",
        "Pipeline parallelism partitions model layers across sequential workers with bubble minimization.",
        "Tensor parallelism splits weight matrices across multiple GPUs using row and column parallel primitives."
    ]
    for i, doc in enumerate(additional_docs, start=21):
        dataset.append({"id": f"train_{i:02d}", "text": doc, "type": "additional"})
    
    # 26-29: 训练-测试集跨 split 污染样本
    eval_benchmark = [
        "GSM8K benchmark question: John has 5 apples, buys 3 more, then gives 2 away. How many remain?",
        "MMLU computer science question: What is the worst-case time complexity of quicksort with median pivot?"
    ]
    dataset.append({"id": "train_26", "text": eval_benchmark[0], "type": "contaminated_gsm8k"})
    dataset.append({"id": "train_27", "text": eval_benchmark[0] + " Answer: 6 apples remain.", "type": "contaminated_gsm8k_solution"})
    dataset.append({"id": "train_28", "text": eval_benchmark[1], "type": "contaminated_mmlu"})
    dataset.append({"id": "train_29", "text": "Unrelated normal sentence discussing compiler optimization techniques.", "type": "normal"})
    
    # 运行过滤与去重
    results = {
        "total_samples": len(dataset),
        "exact_duplicates": [],
        "near_duplicates": [],
        "filtered_out": [],
        "test_set_contaminations": [],
        "kept_samples": []
    }
    
    # 计算签名与精确哈希
    signatures = {}
    exact_hashes = {}
    for item in dataset:
        text = item["text"]
        item_id = item["id"]
        
        # 1. 质量过滤规则
        words = text.split()
        if len(words) < 3:
            results["filtered_out"].append({"id": item_id, "reason": "length_below_threshold", "len": len(words)})
            continue
        # 重复字检查
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.25:
            results["filtered_out"].append({"id": item_id, "reason": "high_repetition_rate"})
            continue
        # 乱码启发式检查
        if "asdf" in text or "[object Object]" in text:
            results["filtered_out"].append({"id": item_id, "reason": "gibberish_pattern"})
            continue
        
        # 2. 精确哈希去重
        md5_h = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        if md5_h in exact_hashes:
            results["exact_duplicates"].append({"id": item_id, "dup_of": exact_hashes[md5_h]})
            continue
        exact_hashes[md5_h] = item_id
        
        # 3. MinHash 近重复去重
        sig = dedup.compute_signature(text)
        is_near_dup = False
        for prev_id, prev_sig in signatures.items():
            sim = dedup.jaccard_similarity(sig, prev_sig)
            if sim >= 0.75:
                results["near_duplicates"].append({"id": item_id, "dup_of": prev_id, "estimated_jaccard": sim})
                is_near_dup = True
                break
        if is_near_dup:
            continue
        
        # 4. 测试集污染排查
        is_contaminated = False
        for eval_idx, eval_text in enumerate(eval_benchmark):
            # 40 字符窗口匹配或高 MinHash 重叠；不是 13-token gram
            eval_sig = dedup.compute_signature(eval_text)
            sim_eval = dedup.jaccard_similarity(sig, eval_sig)
            if sim_eval >= 0.65 or any(eval_text[i:i+40] in text for i in range(0, max(1, len(eval_text)-40), 20)):
                results["test_set_contaminations"].append({
                    "id": item_id,
                    "eval_benchmark_idx": eval_idx,
                    "overlap_sim": sim_eval,
                    "snippet": text[:60]
                })
                is_contaminated = True
                break
        
        if not is_contaminated:
            signatures[item_id] = sig
            results["kept_samples"].append(item_id)
            
    expected_near = {item['id'] for item in dataset if item['type'].startswith('near_dup_of_')}
    detected_near = {item['id'] for item in results['near_duplicates']}
    pair_scores = []
    for item in dataset:
        if item['id'] not in expected_near:
            continue
        original = base_docs[int(item['type'].split('_')[-1])]
        a, b = dedup._get_shingles(original), dedup._get_shingles(item['text'])
        pair_scores.append({'id': item['id'], 'exact_shingle_jaccard': len(a & b)/len(a | b),
                            'estimated_jaccard': dedup.jaccard_similarity(dedup.compute_signature(original), dedup.compute_signature(item['text']))})
    results['fixture_samples'] = dataset
    results['filter_error_analysis'] = {'label_basis': 'synthetic edit provenance, not an assertion that Jaccard must exceed .75',
        'expected_near_ids': sorted(expected_near), 'missed_near_ids': sorted(expected_near-detected_near),
        'false_positive_near_ids': sorted(detected_near-expected_near), 'pair_scores': pair_scores,
        'public_benchmark_contamination_verified': False}
    assert sum(len(results[k]) for k in ['exact_duplicates','near_duplicates','filtered_out','test_set_contaminations','kept_samples']) == len(dataset)
    return results


# ==============================================================================
# 2. 多模态 Collator：Padding vs Packing 对比
# ==============================================================================

class DataCollatorContract:
    """
    展示 Padding 与 Packing（拼接打包）的核心数据流与损失归一化差异。
    """
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def collate_padding(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        标准 Padding 模式：对齐至当前批次最大长度。
        """
        seqs = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        batch_size = len(seqs)
        max_len = max(len(s) for s in seqs)
        
        padded_inputs = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, (seq, lab) in enumerate(zip(seqs, labels)):
            l = len(seq)
            padded_inputs[i, :l] = torch.tensor(seq, dtype=torch.long)
            padded_labels[i, :l] = torch.tensor(lab, dtype=torch.long)
            attention_mask[i, :l] = 1
            position_ids[i, :l] = torch.arange(l, dtype=torch.long)
            
        total_tokens = batch_size * max_len
        valid_tokens = sum(len(s) for s in seqs)
        padding_tokens = total_tokens - valid_tokens
        padding_ratio = padding_tokens / max(1, total_tokens)
        
        # 有效监督 token 数（排除 label=-100）
        supervised_tokens = (padded_labels[:, 1:] != -100).sum().item()
        
        return {
            "mode": "padding",
            "batch_size": batch_size,
            "max_len": max_len,
            "total_slots": total_tokens,
            "valid_tokens": valid_tokens,
            "padding_tokens": padding_tokens,
            "padding_ratio": float(padding_ratio),
            "supervised_tokens": supervised_tokens,
            "input_ids_shape": list(padded_inputs.shape),
            "sample_input_ids": padded_inputs.tolist(),
            "sample_labels": padded_labels.tolist(),
            "attention_mask": attention_mask.tolist(),
            "position_ids": position_ids.tolist()
        }

    def collate_packing(self, batch: List[Dict[str, Any]], max_packed_tokens: int = 64) -> Dict[str, Any]:
        """
        Packing 打包模式：将多个变长样本拼入单条连续向量中。
        产出：
        - input_ids: [1, packed_len]
        - cu_seqlens: Cumulative sequence lengths, 用于 FlashAttention varlen
        - position_ids: 每个样本内部从 0 重新编号
        - document boundary 隔离
        """
        all_inputs = []
        all_labels = []
        all_positions = []
        cu_seqlens = [0]
        sample_boundaries = []
        
        curr_offset = 0
        deferred = []
        rejected = []
        for item in batch:
            seq = item["input_ids"]
            lab = list(item["labels"])
            l = len(seq)
            if l != len(lab):
                raise ValueError(f"input/label length mismatch: {item['id']}")
            if l == 0 or l > max_packed_tokens:
                rejected.append({"id": item["id"], "reason": "empty" if l == 0 else "exceeds_capacity"})
                continue
            if curr_offset + l > max_packed_tokens:
                deferred.append(item["id"])
                continue
            # 全段 causal shift 会用上一文档末尾 logits 预测本条首 token。
            # 对每条首 label 设 ignore_index，使 loss 与独立逐文档 shift 一致。
            lab[0] = -100
            all_inputs.extend(seq)
            all_labels.extend(lab)
            all_positions.extend(list(range(l)))
            curr_offset += l
            cu_seqlens.append(curr_offset)
            sample_boundaries.append({"id": item["id"], "start": curr_offset - l, "end": curr_offset})
            
        packed_len = len(all_inputs)
        supervised_tokens = sum(1 for x in all_labels if x != -100)
        
        # 构造跨文档隔离的 2D block-diagonal attention mask (如果不用 varlen kernel 时)
        block_mask = np.zeros((packed_len, packed_len), dtype=int)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i+1]
            # 因果下三角 mask
            for r in range(start, end):
                for c in range(start, r + 1):
                    block_mask[r, c] = 1
                    
        return {
            "mode": "packing",
            "packed_length": packed_len,
            "max_packed_tokens": max_packed_tokens,
            "num_packed_samples": len(cu_seqlens) - 1,
            "cu_seqlens": cu_seqlens,
            "supervised_tokens": supervised_tokens,
            "padding_tokens": 0,
            "padding_ratio": 0.0,
            "input_ids": all_inputs,
            "position_ids": all_positions,
            "labels": all_labels,
            "sample_boundaries": sample_boundaries,
            "deferred_ids": deferred,
            "rejected": rejected,
            "loss_contract": "labels[..., 1:] with every document's first label ignored",
            "block_diagonal_mask_sample": block_mask.tolist()
        }


# ==============================================================================
# 主执行入口
# ==============================================================================

def verify_packing_loss_boundary():
    cases=[]
    collator=DataCollatorContract()
    for seed in (0,1,2):
        torch.manual_seed(seed)
        samples=[{'id':'a','input_ids':[1,2],'labels':[1,2]},
                 {'id':'b','input_ids':[3,4],'labels':[3,4]}]
        packed=collator.collate_packing(samples,max_packed_tokens=4)
        logits=torch.randn(4,8,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor(packed['labels'])
        packed_loss=torch.nn.functional.cross_entropy(logits[:-1],labels[1:],ignore_index=-100)
        independent=(torch.nn.functional.cross_entropy(logits[0:1],torch.tensor([2]))+
                     torch.nn.functional.cross_entropy(logits[2:3],torch.tensor([4])))/2
        g1=torch.autograd.grad(packed_loss,logits,retain_graph=True)[0]
        g2=torch.autograd.grad(independent,logits)[0]
        torch.testing.assert_close(g1,g2,rtol=0,atol=1e-12)
        assert packed['labels'][2]==-100
        tail=collator.collate_packing(samples+[{'id':'long','input_ids':[1]*5,'labels':[1]*5},
                                               {'id':'later','input_ids':[1],'labels':[1]}],4)
        assert tail['deferred_ids']==['later'] and tail['rejected']==[{'id':'long','reason':'exceeds_capacity'}]
        cases.append({'seed':seed,'labels':packed['labels'],'loss':packed_loss.item(),
                      'independent_loss':independent.item(),'gradient_max_error':float((g1-g2).abs().max()),
                      'deferred':tail['deferred_ids'],'rejected':tail['rejected']})
    return cases


def main():
    global OUTPUT_DIR
    from _evidence import new_output, write_result
    OUTPUT_DIR = str(new_output('Data schemas, packing boundaries and small data-flow examples'))
    random.seed(0)
    print("=== [7.8 文本清洗与 batch 组装] ===")

    # 1. 清洗、MinHash 去重与测试集污染排查
    dedup_results = run_dedup_and_contamination_test()
    dedup_path = os.path.join(OUTPUT_DIR, "dedup_filtration.json")
    with open(dedup_path, "w", encoding="utf-8") as f:
        json.dump(dedup_results, f, indent=2, ensure_ascii=False)
    print(f"[*] 1. 文本清洗与污染检测完成 (保留 {len(dedup_results['kept_samples'])}/{dedup_results['total_samples']} 样本) -> {dedup_path}")
    
    # 2. Collator 对拍（Padding vs Packing）
    toy_batch = [
        {"id": "doc_1", "input_ids": [101, 2054, 2003, 1037, 102], "labels": [-100, 2054, 2003, 1037, 102]},
        {"id": "doc_2", "input_ids": [101, 7592, 102], "labels": [-100, 7592, 102]},
        {"id": "doc_3", "input_ids": [101, 2129, 2024, 2017, 1029, 102], "labels": [-100, 2129, 2024, 2017, 1029, 102]},
        {"id": "doc_4", "input_ids": [101, 1037, 102], "labels": [-100, 1037, 102]}
    ]
    collator = DataCollatorContract(pad_token_id=0)
    pad_res = collator.collate_padding(toy_batch)
    pack_res = collator.collate_packing(toy_batch, max_packed_tokens=32)
    
    collator_path = os.path.join(OUTPUT_DIR, "collator_batch_contract.json")
    with open(collator_path, "w", encoding="utf-8") as f:
        json.dump({
            "padding_batch": pad_res,
            "packing_batch": pack_res,
            "comparison": {
                "padding_waste_ratio": pad_res["padding_ratio"],
                "packing_waste_ratio": pack_res["padding_ratio"],
                "cu_seqlens": pack_res["cu_seqlens"]
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"[*] 2. Collator Padding vs Packing 对比完成 (Padding 浪费率: {pad_res['padding_ratio']*100:.1f}%) -> {collator_path}")
    
    write_result(Path(OUTPUT_DIR), 'packing_loss_contract.json', {'cases':verify_packing_loss_boundary()},
                 {'device':'CPU','dtype':'float64','seeds':[0,1,2],'atol':1e-12,
                  'scope':'loss boundary only; no attention forward and no storage measurement'},[__file__])
    print("=== [7.8 完成] ===")


if __name__ == "__main__":
    main()
