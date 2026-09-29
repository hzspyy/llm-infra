#!/usr/bin/env python3
"""
6.3 任务 A/B/C：MoE 的 pack / dispatch / combine 与专家并行通信。

任务 A：只用整数与唯一 token ID 实现一遍 dispatch/combine 的计划构造与反排列，
        在单进程里验证「无丢失、无重复、无错位」，并覆盖零 token rank、空专家、
        倾斜分布三种边界。
任务 B：在 2/4 卡上用 NCCL 跑两条真实路线并分阶段计时：
        all_to_all 路线与 all_gather+reduce_scatter 路线；记录通信字节、padding、
        workspace 峰值，并与单机专家参考对拍。
任务 C：把 DeepEP V1/V2、UCCL-EP、NVSHMEM 的触发条件与当前平台的实际能力
        汇成支持矩阵（源码事实 + 本机探测，不运行不支持的路径）。

用法：
    python moe_dispatch.py A --tokens 512 --experts 8 --topk 2 --ranks 4 --skew 0.0 --out <dir>
    python moe_dispatch.py B --world-size 4 --tokens 256 --topk 8 --skew 0.2 --route alltoall --out <dir>
    python moe_dispatch.py capability --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# 本机主模型：OLMoE-1B-7B-0924
OLMOE = {"num_experts": 64, "top_k": 8, "hidden": 2048, "inter": 1024, "layers": 16,
         "vocab": 50304}


def env_pins():
    info = {"host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda}
    try:
        info["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        info["nccl"] = None
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
    return info


def expert_owner(expert_id, num_experts, ep_size):
    """专家到 rank 的静态连续切分：专家 e 归 rank e*ep_size//E。"""
    return expert_id * ep_size // num_experts


# ==========================================================================
# 计划构造（任务 A 的核心，任务 B 复用）
# ==========================================================================

def make_routing(tokens, num_experts, top_k, skew, seed=0, ep_size=1):
    """造一份确定性的路由：每个 token 选 top_k 个专家。

    skew ∈ [0,1) 把一部分 token 强制路由到同一个小专家集合，用来制造倾斜。
    返回 (expert_ids (T,K) int64, weights (T,K) float32)
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, num_experts, (tokens, top_k), generator=g)
    w = torch.rand((tokens, top_k), generator=g) + 0.1
    if skew > 0:
        hot = max(1, int(num_experts * 0.1))
        n_hot = int(tokens * skew)
        hot_ids = torch.randint(0, hot, (n_hot, top_k), generator=g)
        ids[:n_hot] = hot_ids
    return ids, w


def build_plan(expert_ids, ep_size, num_experts):
    """由路由结果算出 dispatch 计划：send/recv counts、offsets 与两组排列。

    排列用 (token_id, k_index) 作为唯一键，这是能证明「无错位」的最小信息量：
    接收端靠它把结果放回原来的位置。
    """
    T, K = expert_ids.shape
    owner = (expert_ids * ep_size // num_experts)          # (T,K) 目标 rank
    # 本 rank 需要发往各 rank 的 (token,k) 个数
    send_counts = [int((owner == r).sum()) for r in range(ep_size)]
    send_offsets = [0]
    for c in send_counts:
        send_offsets.append(send_offsets[-1] + c)
    # 发送顺序：按目标 rank 分段，段内按 (token,k) 原始顺序
    order = []
    for r in range(ep_size):
        idx = (owner == r).nonzero(as_tuple=False)
        order.extend([(int(t), int(k)) for t, k in idx])
    assert len(order) == sum(send_counts)
    # 每个 (token,k) 的发送位置
    src_pos = {}
    for pos, key in enumerate(order):
        src_pos[key] = pos
    # 接收侧：按 (专家) 分组，专家内部按到达顺序
    recv_experts = [e for e in range(num_experts)
                    if expert_owner(e, num_experts, ep_size) is not None]
    return {
        "owner": owner, "send_counts": send_counts, "send_offsets": send_offsets,
        "order": order, "src_pos": src_pos,
    }


def local_experts(rank, ep_size, num_experts):
    return [e for e in range(num_experts) if expert_owner(e, num_experts, ep_size) == rank]


def simulate_ep(expert_ids, weights, ep_size, num_experts, hidden):
    """在单进程里把 EP 的 dispatch→compute→combine 走一遍，返回每 rank 的观测量。

    专家计算用可复现的线性变换 y = x @ We^T，We 由专家 id 决定，便于与单机参考对拍。
    """
    T, K = expert_ids.shape
    owner = expert_ids * ep_size // num_experts
    g = torch.Generator().manual_seed(1234)
    W = torch.randn(num_experts, hidden, hidden, generator=g)
    x = torch.randn(T, hidden, generator=g)

    # 单机参考：所有专家都在本地，逐 (token,k) 直接算
    ref_per_k = torch.empty(T, K, hidden)
    for t in range(T):
        for k in range(K):
            e = int(expert_ids[t, k])
            ref_per_k[t, k] = x[t] @ W[e].t()
    ref = (ref_per_k * weights.unsqueeze(-1)).sum(dim=1)

    recv_log = []
    for rank in range(ep_size):
        mine = local_experts(rank, ep_size, num_experts)
        # 本 rank 要发出的 (token,k)
        send = [(t, k) for t in range(T) for k in range(K) if owner[t, k] == rank]
        # 本 rank 会收到哪些 (token,k)：专家属于本 rank 的那些
        recv = [(t, k) for t in range(T) for k in range(K)
                if int(expert_ids[t, k]) in mine]
        # 每个本地专家的分段（expert offsets）
        per_expert = {e: 0 for e in mine}
        for t, k in recv:
            per_expert[int(expert_ids[t, k])] += 1
        offsets = {}
        acc = 0
        for e in mine:
            offsets[e] = (acc, acc + per_expert[e])
            acc += per_expert[e]
        recv_log.append({
            "rank": rank,
            "local_experts": len(mine),
            "send_pairs": len(send),
            "recv_pairs": len(recv),
            "per_expert_counts": per_expert,
            "expert_offsets": {str(e): list(v) for e, v in offsets.items()},
            "empty_experts": [e for e in mine if per_expert[e] == 0],
        })

    # 数值检查：接收端的 (token,k) 集合必须与发送端完全一致
    total_send = sum(r["send_pairs"] for r in recv_log)
    total_recv = sum(r["recv_pairs"] for r in recv_log)
    return {"ref": ref, "recv_log": recv_log, "total_send": total_send,
            "total_recv": total_recv, "pairs_expected": T * K}


def run_A(args):
    os.makedirs(args.out, exist_ok=True)
    cases = [
        ("均匀", dict(tokens=args.tokens, skew=0.0)),
        ("倾斜20%", dict(tokens=args.tokens, skew=0.2)),
        ("倾斜50%", dict(tokens=args.tokens, skew=0.5)),
        ("空专家（专家数>>token）", dict(tokens=8, skew=0.0, experts=64)),
        ("全部倾斜到 rank0（零 token rank）", dict(tokens=args.tokens, skew=1.0)),
    ]
    results = []
    for name, kw in cases:
        E = kw.get("experts", args.experts)
        expert_ids, weights = make_routing(kw["tokens"], E, args.topk,
                                           kw["skew"], seed=7, ep_size=args.ranks)
        sim = simulate_ep(expert_ids, weights, args.ranks, E, hidden=32)
        # 唯一键集合检查：每个 (token,k) 必须落到唯一的目标 rank
        keys = [(t, k) for t in range(kw["tokens"]) for k in range(args.topk)]
        owner = expert_ids * args.ranks // E
        per_rank = [int((owner == r).sum()) for r in range(args.ranks)]
        # 用计划构造再验一次排列的完整性
        plan = build_plan(expert_ids, args.ranks, E)
        perm_ok = (len(plan["order"]) == len(keys)
                   and sorted(plan["order"]) == sorted(keys)
                   and len(set(plan["src_pos"].values())) == len(keys))
        zero_token_ranks = [r for r in range(args.ranks) if per_rank[r] == 0]
        empty_experts_total = sum(len(r["empty_experts"]) for r in sim["recv_log"])
        maxmin = (max(per_rank) / max(1, min(per_rank))) if min(per_rank) >= 0 else None
        rec = {
            "case": name, "tokens": kw["tokens"], "experts": E, "top_k": args.topk,
            "ep_size": args.ranks, "skew": kw["skew"],
            "pairs": kw["tokens"] * args.topk,
            "pairs_dispatched": sim["total_recv"],
            "pairs_sent": sim["total_send"],
            "no_loss_no_dup": (sim["total_recv"] == kw["tokens"] * args.topk
                               and perm_ok),
            "permutation_is_bijection": perm_ok,
            "per_rank_pairs": per_rank,
            "rank_load_max_over_min": maxmin,
            "zero_token_ranks": zero_token_ranks,
            "empty_expert_slots": empty_experts_total,
            "rank_detail": sim["recv_log"],
        }
        results.append(rec)
        print(f"  {name:<22} pairs={rec['pairs']:<6} 分发={rec['pairs_dispatched']:<6} "
              f"无丢失/重复/错位={rec['no_loss_no_dup']} "
              f"每 rank 负载={per_rank} 零 token rank={zero_token_ranks} "
              f"空专家槽={empty_experts_total}")

    payload = {"model": "OLMoE-1B-7B-0924", "config": OLMOE,
               "cases": results, "env": env_pins()}
    with open(os.path.join(args.out, "dispatch_plan.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[A] {len(results)} 个用例，全部无丢失/重复/错位="
          f"{all(r['no_loss_no_dup'] for r in results)}")
    return payload


# ==========================================================================
# 任务 B：真实通信
# ==========================================================================

def _bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / 1e3 for s, e in ev)


def run_B_worker(rank, world_size, out_dir, args):
    """真实通信：all_to_all 路线与 all_gather+all_reduce 路线。

    all_to_all 路线的接收端必须再排一次：发送方按 (目标 rank, 专家) 排序，收到的是
    「按发送方分段」的布局；要喂给 grouped GEMM，接收端还需要按本地专家做一次局部重排。
    这个局部重排不发通信，但它是 dispatch 实现里真实存在的一步。
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29970")
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=180))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    E, K, H = args.experts, args.topk, args.hidden
    T = args.tokens
    expert_ids, weights = make_routing(T, E, K, args.skew, seed=11, ep_size=world_size)
    expert_ids = expert_ids.to(dev)
    weights = weights.to(dev)
    x = torch.randn(T, H, device=dev)
    owner = expert_ids * world_size // E          # (T,K) 目标 rank
    my_experts = local_experts(rank, world_size, E)   # 本 rank 持有的专家 id
    is_local = torch.zeros(E, dtype=torch.bool, device=dev)
    is_local[torch.tensor(my_experts, device=dev, dtype=torch.int64)] = True

    result = {"rank": rank, "tokens": T, "top_k": K, "experts": E,
              "ep_size": world_size, "skew": args.skew, "hidden": H,
              "route": args.route, "local_experts": len(my_experts)}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()

    if args.route == "alltoall":
        # ---- 发送侧：按 (目标 rank, 专家) 排序，并记录每 (目的 rank, 专家) 的行数 ----
        rows, send_meta = [], []
        per_dest_expert = torch.zeros(world_size, E, dtype=torch.int64, device=dev)
        send_counts = [0] * world_size
        send_rows_by_dest = []
        for r in range(world_size):
            idx_r = (owner == r).nonzero(as_tuple=False)
            if idx_r.numel() == 0:
                send_rows_by_dest.append([])
                continue
            exps = expert_ids[idx_r[:, 0], idx_r[:, 1]]
            order = torch.argsort(exps, stable=True)
            idx_sorted = idx_r[order]
            send_rows_by_dest.append([(int(t), int(k), int(e)) for (t, k), e in
                                      zip(idx_sorted.tolist(), exps[order].tolist())])
            send_counts[r] = len(idx_sorted)
            for e in exps[order].tolist():
                per_dest_expert[r, int(e)] += 1
        flat = [row for sub in send_rows_by_dest for row in sub]
        total_send = len(flat)
        send_counts_t = torch.tensor(send_counts, device=dev, dtype=torch.int64)
        recv_counts_t = torch.empty(world_size, device=dev, dtype=torch.int64)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_total = int(recv_counts_t.sum())
        # 每 (发送方, 专家) 的行数：all_gather 一张 (E,) 的计数表
        local_counts = per_dest_expert.sum(dim=0)          # 本 rank 发往每个专家的行数
        all_counts = torch.empty(world_size, E, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(all_counts.view(-1), local_counts)

        buf = torch.empty(total_send, H + 3, device=dev)
        for i, (t, k, e) in enumerate(flat):
            buf[i, 0], buf[i, 1], buf[i, 2] = t, k, e
            buf[i, 3:] = x[t]
        out = torch.empty(recv_total, H + 3, device=dev)
        in_splits = [int(v) for v in send_counts_t.tolist()]
        out_splits = [int(v) for v in recv_counts_t.tolist()]

        def phase_counts():
            dist.all_to_all_single(recv_counts_t, send_counts_t)

        def phase_dispatch():
            dist.all_to_all_single(out, buf, out_splits, in_splits)

        def phase_combine():
            dist.all_to_all_single(buf, out, in_splits, out_splits)

        t_counts = _bench(phase_counts)
        t_disp = _bench(phase_dispatch)
        t_comb = _bench(phase_combine)
        phase_dispatch()
        torch.cuda.synchronize()

        # ---- 接收侧：局部重排成 expert-major，再算专家，再送回 ----
        recv_e = out[:, 2].long()
        local_expert_counts = all_counts[:, :].sum(dim=0)      # 每个专家总行数
        offsets, acc = {}, 0
        for e in my_experts:
            c = int(local_expert_counts[e])
            offsets[e] = (acc, acc + c)
            acc += c
        perm = torch.empty(recv_total, dtype=torch.int64, device=dev)
        cursor = {e: offsets[e][0] for e in my_experts}
        # 到达顺序是 (发送方, 专家)，按这个顺序逐个填进 expert-major 的空位
        pos = 0
        for s in range(world_size):
            for e in my_experts:
                c = int(all_counts[s, e])
                for _ in range(c):
                    perm[cursor[e]] = pos
                    cursor[e] += 1
                    pos += 1
        assert pos == recv_total
        packed = out[perm]                                      # expert-major 布局
        packed_e = packed[:, 2].long()
        # 分组校验：每个本地专家的段必须连续且全部属于该专家
        groups_ok = True
        for e in my_experts:
            a, b = offsets[e]
            if b > a and not bool((packed_e[a:b] == e).all()):
                groups_ok = False
        # 专家计算：按专家 id 定的确定性缩放，便于精确对拍
        scale = (1.0 + packed_e.to(torch.float32) * 0.001).unsqueeze(-1)
        packed_y = packed[:, 3:] * scale
        # 反排列 + combine
        back_rows = torch.empty_like(packed_y)
        back_rows[perm] = packed_y
        out[:, 3:] = back_rows
        phase_combine()
        torch.cuda.synchronize()
        # 回来的行在 buf 里（combine 的 output 是 buf），不是 out
        back_tok, back_k = buf[:, 0].long(), buf[:, 1].long()
        got = torch.zeros(T, K, H, device=dev)
        got[back_tok, back_k] = buf[:, 3:]
        expect = x.unsqueeze(1) * (1.0 + expert_ids.to(torch.float32) * 0.001).unsqueeze(-1)
        # 只比本 rank 负责的那些 (t,k)：其余位置本来就不该被本 rank 写
        mine_mask = (owner == rank)
        maxdiff = float((got[mine_mask] - expect[mine_mask]).abs().max().item())
        covered = int(mine_mask.sum().item())
        result.update({
            "phase_counts_s": t_counts, "phase_dispatch_s": t_disp,
            "phase_combine_s": t_comb,
            "send_counts": send_counts,
            "recv_counts": [int(v) for v in recv_counts_t.tolist()],
            "dispatch_bytes_per_rank": total_send * (H + 3) * 4,
            "combine_bytes_per_rank": recv_total * (H + 3) * 4,
            "payload_bytes_per_rank": total_send * H * 4,
            "padding_bytes_per_rank": total_send * 3 * 4,
            "counts_bytes_per_rank": world_size * 8 + E * 8,
            "expert_major_grouping_ok": groups_ok,
            "received_rows": recv_total,
            "roundtrip_max_abs_diff": maxdiff,
            "roundtrip_correct": maxdiff < 1e-5,
            "pairs_covered_by_this_rank": covered,
            "local_expert_offsets": {str(e): list(offsets[e]) for e in my_experts[:4]},
            "empty_local_experts": [e for e in my_experts if offsets[e][1] == offsets[e][0]],
        })
    else:
        # ---- all_gather + all_reduce 路线 ----
        gx = torch.empty(world_size * T, H, device=dev)
        gids = torch.empty(world_size * T, K, dtype=torch.int64, device=dev)
        gw = torch.empty(world_size * T, K, device=dev)

        def phase_gather():
            dist.all_gather_into_tensor(gx, x)

        def phase_meta():
            dist.all_gather_into_tensor(gids.view(-1), expert_ids.view(-1))
            dist.all_gather_into_tensor(gw.view(-1), weights.view(-1))

        t_meta = _bench(phase_meta)
        t_gather = _bench(phase_gather)
        phase_meta()
        phase_gather()
        torch.cuda.synchronize()
        # 每个 rank 只算自己那些专家，其余位置补零；再把所有贡献 all_reduce 起来
        keep = is_local[gids]                                    # (N*T, K)
        scale = (1.0 + gids.to(torch.float32) * 0.001).unsqueeze(-1)

        def local_contrib():
            """本 rank 只算自己那些专家，其余位置补零。"""
            y = gx.unsqueeze(1) * scale
            y = torch.where(keep.unsqueeze(-1), y, torch.zeros_like(y))
            return (y * gw.unsqueeze(-1)).sum(dim=1)

        # all_reduce 会把 buffer 就地改成和，重复调用会不断累加，
        # 所以每次都从零开始重算，不能复用同一个 buffer 做基准与正确性检查。
        def phase_allreduce():
            c = local_contrib()
            dist.all_reduce(c)
            return c

        t_ar = _bench(phase_allreduce)
        contrib = phase_allreduce()
        torch.cuda.synchronize()
        expect_all = (x.unsqueeze(1)
                      * (1.0 + expert_ids.to(torch.float32) * 0.001).unsqueeze(-1)
                      * weights.unsqueeze(-1)).sum(dim=1)
        got = contrib[rank * T:(rank + 1) * T]
        maxdiff = float((got - expect_all).abs().max().item())
        result.update({
            "phase_meta_gather_s": t_meta, "phase_gather_s": t_gather,
            "phase_allreduce_s": t_ar,
            "gather_bytes_per_rank": T * H * 4,
            "meta_bytes_per_rank": T * K * 8 + T * K * 4,
            "allreduce_bytes_per_rank": world_size * T * H * 4,
            "total_bytes_per_rank": T * H * 4 + T * K * 12 + world_size * T * H * 4,
            "roundtrip_max_abs_diff": maxdiff,
            "roundtrip_correct": maxdiff < 1e-5,
            "note": "all_gather 路线把每个 token 复制到所有 rank；all_reduce 的字节数随 rank 数线性增长",
        })

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        result["base_allocated_GiB"] = base_alloc / 2**30

    gathered = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{args.route}_t{T}_k{K}_s{int(args.skew * 100)}_n{world_size}_h{H}"
        with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"env": env_pins(), "ranks": gathered}, f,
                      ensure_ascii=False, indent=2)
        r = gathered[0]
        if args.route == "alltoall":
            print(f"[B:{tag}] 发送段={r['send_counts']} 接收段={r['recv_counts']} "
                  f"本地专家={r['local_experts']}")
            print(f"     counts={r['phase_counts_s'] * 1e6:.1f} us "
                  f"dispatch={r['phase_dispatch_s'] * 1e6:.1f} us "
                  f"combine={r['phase_combine_s'] * 1e6:.1f} us "
                  f"字节/rank={r['dispatch_bytes_per_rank'] / 1024:.0f} KiB "
                  f"expert-major连续={r['expert_major_grouping_ok']} "
                  f"往返正确={r['roundtrip_correct']} "
                  f"峰值={r.get('peak_allocated_GiB', 0):.3f} GiB")
        else:
            print(f"[B:{tag}] meta={r['phase_meta_gather_s'] * 1e6:.1f} us "
                  f"gather={r['phase_gather_s'] * 1e6:.1f} us "
                  f"allreduce={r['phase_allreduce_s'] * 1e6:.1f} us "
                  f"字节/rank={r['total_bytes_per_rank'] / 1024:.0f} KiB "
                  f"往返正确={r['roundtrip_correct']} "
                  f"峰值={r.get('peak_allocated_GiB', 0):.3f} GiB")
    dist.barrier()
    dist.destroy_process_group()
# ==========================================================================
# 任务 C：支持矩阵
# ==========================================================================

CAPABILITY = {
    "deep_ep_v1": {
        "backend": "NVSHMEM",
        "entry": "deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)",
        "source": [
            "DeepEP docs/legacy.md:7（V1 提供高吞吐与低延迟 all-to-all，即 MoE dispatch/combine）",
            "docs/legacy.md:125-147（Buffer.set_num_sms / Buffer(...) 的构造）",
            "docs/legacy.md:52,54,82（跨机需要 RDMA 网络；依赖 NVSHMEM；未设 NVSHMEM_DIR 时"
            "跨机与低延迟特性全部关闭）",
        ],
        "requires": ["NVSHMEM", "RDMA 网络（跨机）", "SM90 级 GPU（见 setup.py 架构门控）"],
    },
    "deep_ep_v2": {
        "backend": "NCCL Gin",
        "entry": "deep_ep.ElasticBuffer（高吞吐与低延迟统一接口）",
        "source": [
            "DeepEP README:9（V2 从 NVSHMEM 换到更轻量的 NCCL Gin 后端）",
            "README:14-18（复用已有 NCCL communicator；统一 ElasticBuffer；新的 GEMM 布局）",
            "README:65-72（要求 Hopper SM90 或支持 SM90 PTX ISA 的架构；CUDA 12.3+；"
            "NCCL 2.30.4+；跨机需要 RDMA 网络）",
            "README:29-30（V2 的 buffer 占用大于 V1；不再支持 0-SM RDMA 低延迟 EP）",
            "setup.py:130-148（默认 TORCH_CUDA_ARCH_LIST=9.0，并按 9.0 校验）",
        ],
        "requires": ["SM90+", "CUDA 12.3+", "NCCL 2.30.4+", "RDMA（跨机）"],
    },
    "uccl_ep": {
        "backend": "UCCL transport（面向异构 GPU/NIC）",
        "entry": "uccl.ep，接口与 DeepEP 相同",
        "source": [
            "UCCL ep/README.md:3（与 DeepEP 同接口，支持 Nvidia/AMD GPU 与 EFA/Broadcom/CX7 NIC）",
            "ep/README.md:23-26（默认用 ibv_reg_mr_iova2 注册 GPU 显存，需要 nvidia_peermem / "
            "efa_nv_peermem / ib_peer_mem 内核模块；否则需 USE_DMABUF=1 重编）",
            "ep/README.md:27-28（自动探测 /opt/amazon/efa 决定 EFA 路径）",
        ],
        "requires": ["RDMA NIC + 可用的 peer-memory 内核模块（或 DMABUF）"],
    },
    "nvshmem": {
        "backend": "NVSHMEM（PGAS）",
        "entry": "nvshmem_malloc / nvshmem_putmem 等；被 DeepEP V1 使用",
        "source": ["DeepEP docs/nvshmem.md（安装指引）", "docs/legacy.md:54-62"],
        "requires": ["与 CUDA 版本匹配的 NVSHMEM；跨机需要 IB/RoCE 设备"],
    },
}


def probe_platform():
    import glob
    import subprocess
    p = {
        "sm": None, "nccl": None, "cuda": torch.version.cuda,
        "dev_infiniband": os.path.exists("/dev/infiniband"),
        "ib_devices": sorted(os.path.basename(x)
                             for x in glob.glob("/sys/class/infiniband/*")),
        "nvlink_present": None, "efa": os.path.exists("/opt/amazon/efa"),
        "nvshmem_lib": bool(glob.glob("/usr/lib/**/libnvshmem*", recursive=False)),
    }
    try:
        p["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass
    if torch.cuda.is_available():
        p["sm"] = f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
        try:
            out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                 text=True, timeout=30).stdout
            # 只看矩阵行（以 GPU 开头），不能扫全表：即使没有 NVLink，
            # nvidia-smi 也会在图例里打印 "NV# = Connection traversing a bonded set of # NVLinks"
            matrix = [l for l in out.splitlines() if l.startswith("GPU")]
            cells = [c.strip() for l in matrix for c in l.split()[1:]]
            p["nvlink_present"] = any(c.startswith("NV") for c in cells)
            p["topo_links"] = sorted({c for c in cells if c in
                                      ("NV1", "NV2", "NV3", "NV4", "NV6", "NV8", "NV12",
                                       "PIX", "PXB", "PHB", "NODE", "SYS", "X")})
            p["topo_head"] = out.splitlines()[:8]
        except Exception:
            pass
    return p


def run_capability(args):
    os.makedirs(args.out, exist_ok=True)
    plat = probe_platform()
    verdict = {}
    sm_ok = plat["sm"] in ("sm_90", "sm_100", "sm_103", "sm_120")  # DeepEP 要求 SM90+
    nccl_ok = False
    try:
        v = torch.cuda.nccl.version()
        nccl_ok = (v[0], v[1], v[2] if len(v) > 2 else 0) >= (2, 30, 4)
    except Exception:
        pass
    verdict["deep_ep_v1"] = {"runnable": False,
                             "why": "需要 NVSHMEM + RDMA；本机无 /dev/infiniband、无 NVSHMEM"}
    verdict["deep_ep_v2"] = {
        "runnable": False,
        "why": (f"架构门控：本机 {plat['sm']}，DeepEP 要求 SM90+（{'不满足' if not sm_ok else '满足'}）；"
                f"NCCL 要求 ≥2.30.4，本机 {plat['nccl']}（{'不满足' if not nccl_ok else '满足'}）；"
                f"跨机还需要 RDMA")}
    verdict["uccl_ep"] = {"runnable": False,
                          "why": "需要 RDMA NIC 与 peer-memory 内核模块；本机无 /dev/infiniband"}
    verdict["nvshmem"] = {"runnable": False,
                          "why": "未安装 NVSHMEM；跨机需要 IB/RoCE 设备"}
    verdict["nccl_alltoall"] = {"runnable": True,
                                "why": "NCCL 的 alltoall/alltoallv 不需要 RDMA 与 NVSHMEM，"
                                       "单机 GPU 直连即可运行"}
    payload = {"platform": plat, "capability": CAPABILITY, "verdict": verdict,
               "env": env_pins()}
    with open(os.path.join(args.out, "capability_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[C] 平台：{plat['sm']} NCCL {plat['nccl']} CUDA {plat['cuda']} "
          f"/dev/infiniband={plat['dev_infiniband']} IB={plat['ib_devices']} "
          f"NVLink={plat['nvlink_present']} EFA={plat['efa']}")
    for k, v in verdict.items():
        print(f"    {k:<16} 可运行={v['runnable']}  {v['why']}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["A", "B", "capability"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--route", default="alltoall", choices=["alltoall", "allgather"])
    a = ap.parse_args()
    if a.task == "A":
        run_A(a)
    elif a.task == "capability":
        run_capability(a)
    else:
        import torch.multiprocessing as mp
        mp.spawn(run_B_worker, args=(a.world_size, a.out, a), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    sys.exit(main())
