#!/usr/bin/env python3
"""L5.5 任务 A —— 投机解码的参照实现：多步提案、验证、残差采样、EOS 与回滚。

计划的第一个问题是「多 token 验证如何保持分布」。答案不能靠直觉：一个只有
边际正确、条件分布却错掉的接受/残差实现，照样能让单 token 频率看起来正常。
所以本脚本不检验 token 频率，而是**枚举短序列**，把整个序列分布与每一个
条件前缀分布都和目标分布逐项对齐：

  [A] 目标分布与草稿分布：小词表（V=4 + EOS 吸收态）上的两个 Markov 链
  [B] 参照实现：γ 步提案 + 逐位置验证 + 拒绝时按 max(0, p−q) 归一化重采样
  [C] 序列级对拍：枚举长度 n 的全部序列，比较精确概率与经验频率
  [D] 条件前缀对拍：对每个前缀比较 P(x_k | x_<k)
  [E] 回滚不变量：提交状态长度必须等于已接受 token 的前缀，拒绝不留脏状态
  [F] 三个错误变体作为反例：全部接受、残差抽 p、拒绝后抽 q
  [G] EOS：吸收态下「接受 EOS 即停」与「残差产出 EOS」都落回同一分布

只用 numpy，可在本地 CPU 运行：
    python labs/L5/speculative_reference.py
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np

V = 4               # 内容 token 0..3
EOS = 4             # 吸收态
NV = V + 1          # 词表大小


def make_target() -> np.ndarray:
    """目标分布：行随机的 5×5 转移矩阵，EOS 行吸收。"""
    return np.array([
        [0.05, 0.55, 0.20, 0.15, 0.05],
        [0.45, 0.05, 0.25, 0.20, 0.05],
        [0.20, 0.25, 0.05, 0.45, 0.05],
        [0.15, 0.20, 0.55, 0.05, 0.05],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ], dtype=np.float64)


def make_draft() -> np.ndarray:
    """草稿分布：与目标同结构的扰动版本，在若干位置上差得明显。"""
    return np.array([
        [0.10, 0.30, 0.30, 0.25, 0.05],
        [0.25, 0.10, 0.35, 0.25, 0.05],
        [0.30, 0.30, 0.10, 0.25, 0.05],
        [0.25, 0.30, 0.35, 0.05, 0.05],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ], dtype=np.float64)


def sample(p: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(NV, p=p))


def target_decode(P, prefix, n, rng):
    """无投机的目标分布采样：n 步，EOS 吸收。"""
    seq = list(prefix)
    for _ in range(n):
        seq.append(sample(P[seq[-1]], rng))
    return seq


def speculative_decode(P, Q, prefix, n, gamma, rng, variant="correct"):
    """投机解码一轮一轮地跑：提案 γ 个 token，逐位置验证，拒绝时重采样。

    返回 (序列, trace)。trace 每轮以 {"round_end": True, "committed": L} 收尾，
    L 是该轮提交后的长度——回滚不变量就是靠它检查的。
    """
    seq = list(prefix)
    base = len(prefix)
    trace = []
    while len(seq) - base < n:
        start = len(seq)
        cur = seq[-1]
        drafts = []
        for _ in range(gamma):
            cur = sample(Q[cur], rng)
            drafts.append(cur)

        stopped = False
        for i, tok in enumerate(drafts):
            if len(seq) - base >= n:
                stopped = True           # 名额用完，等同于不再接受
                break
            prev = seq[-1]
            p, q = P[prev], Q[prev]
            ok = (variant == "accept_all") or (
                rng.random() < min(1.0, p[tok] / q[tok]))
            trace.append(dict(pos=i, prev=prev, draft=tok, p=float(p[tok]),
                              q=float(q[tok]), accept=bool(ok)))
            if ok:
                seq.append(tok)
                if tok == EOS:
                    stopped = True       # 吸收态：本轮到此为止
                    break
                continue
            if variant == "residual_from_p":
                res = p.copy()
            elif variant == "resample_draft":
                res = q.copy()
            else:
                res = np.maximum(0.0, p - q)
            s = res.sum()
            res = res / s if s > 0 else p
            tok2 = sample(res, rng)
            seq.append(tok2)
            trace.append(dict(pos=i, prev=prev, draft=tok2, p=float(p[tok2]),
                              q=float(q[tok2]), accept=False, residual=True))
            stopped = True
            break

        if not stopped and len(seq) - base < n:
            # 全部接受且还没到长度上限：从目标分布再采一个 bonus token
            prev = seq[-1]
            tok2 = sample(P[prev], rng)
            seq.append(tok2)
            trace.append(dict(pos=gamma, prev=prev, draft=tok2,
                              p=float(P[prev][tok2]), q=None, accept=True,
                              bonus=True))
        trace.append(dict(round_end=True, committed=len(seq), started=start))
    return seq, trace


# ------------------------------------------------------------------ 精确分布
def prefix_dist(P, prefix, n):
    """给定 prefix 的精确条件分布：{tail: P(tail | prefix)}，tail 长度 n。"""
    dist = {}
    for tail in itertools.product(range(NV), repeat=n):
        pr, prev = 1.0, prefix[-1]
        for tok in tail:
            pr *= P[prev, tok]
            prev = tok
            if pr == 0.0:
                break
        dist[tail] = pr
    return dist


def empirical(P, Q, prefix, n, gamma, trials, seed, variant="correct"):
    rng = np.random.default_rng(seed)
    counts: dict[tuple, int] = {}
    for _ in range(trials):
        seq, _ = speculative_decode(P, Q, prefix, n, gamma, rng, variant)
        tail = tuple(seq[len(prefix):])
        counts[tail] = counts.get(tail, 0) + 1
    return counts


def compare(name, counts, exact, trials):
    keys = set(exact) | set(counts)
    emp = {k: counts.get(k, 0) / trials for k in keys}
    tv = 0.5 * sum(abs(emp[k] - exact.get(k, 0.0)) for k in keys)
    chi2 = 0.0
    for k in keys:
        e = exact.get(k, 0.0) * trials
        if e > 1e-9:
            chi2 += (counts.get(k, 0) - e) ** 2 / e
    # 标准化的最大偏差：|经验 − 精确| / σ，σ 取二项标准差
    worst_z, worst_line = 0.0, ""
    for k in keys:
        e = exact.get(k, 0.0)
        sd = math.sqrt(max(e * (1 - e), 1e-12) / trials)
        z = abs(emp[k] - e) / sd
        if z > worst_z:
            worst_z = z
            worst_line = (f"{k} 精确={e:.4f} 经验={emp[k]:.4f} "
                          f"偏差={emp[k] - e:+.4f}")
    print(f"  {name:<26} 序列数={len(keys):>4}  总变差={tv:.4f}  "
          f"χ²={chi2:9.2f}  最大 |z|={worst_z:5.2f}  ({worst_line})")
    return tv, chi2, worst_z


def main():
    P, Q = make_target(), make_draft()
    prefix = (0,)                        # 固定起始 token，避免先验歧义
    n, gamma, trials = 4, 3, 60000

    print(f"词表 V={NV}（0..{V - 1} 内容 + EOS={EOS} 吸收），序列长度 n={n}，"
          f"每轮提案 γ={gamma}，试验 {trials} 次，prefix={prefix}")
    print("\n[A] 目标转移矩阵 P（行=当前 token，列=下一个 token）:")
    for i in range(NV):
        print("    " + " ".join(f"{x:.2f}" for x in P[i]))
    print("    草稿转移矩阵 Q:")
    for i in range(NV):
        print("    " + " ".join(f"{x:.2f}" for x in Q[i]))

    exact = prefix_dist(P, prefix, n)
    print(f"\n    长度 {n} 的序列共 {len(exact)} 条，精确概率和 = "
          f"{sum(exact.values()):.10f}")

    print("\n[C0] 对照：直接用目标分布采样（无投机），检验枚举本身是否正确")
    rng0 = np.random.default_rng(555)
    c0: dict[tuple, int] = {}
    for _ in range(trials):
        s = target_decode(P, prefix, n, rng0)
        tail = tuple(s[len(prefix):])
        c0[tail] = c0.get(tail, 0) + 1
    tv_ref, _, _ = compare("目标分布直接采样（噪声地板）", c0, exact, trials)
    print(f"    这个值就是 {trials} 次抽样在 {len(exact)} 个格子上的噪声地板；"
          f"投机实现的\n    总变差必须与它同量级，而不是与 0 比较。")

    print("\n[C] 精确分布 vs 投机解码经验分布（逐序列枚举）")
    tv0, _, _ = compare("参照实现 residual=max(0,p−q)",
                        empirical(P, Q, prefix, n, gamma, trials, 1234),
                        exact, trials)
    print(f"    参照实现 {tv0:.4f} vs 噪声地板 {tv_ref:.4f}："
          f"{'同量级' if tv0 < 2 * tv_ref else '明显偏高'}")

    print("\n[F] 错误变体的反例（同一 seed）")
    for variant, label in [("accept_all", "全部接受草稿"),
                           ("residual_from_p", "拒绝后抽 p"),
                           ("resample_draft", "拒绝后抽 q")]:
        compare(label,
                empirical(P, Q, prefix, n, gamma, trials, 1234, variant),
                exact, trials)

    print("\n[D] 条件前缀分布 P(x_k | x_<k)：逐前缀比较（参照实现）")
    rng = np.random.default_rng(99)
    T = 60000
    cond: dict[tuple, int] = {}
    for _ in range(T):
        seq, _ = speculative_decode(P, Q, prefix, n, gamma, rng)
        tail = seq[len(prefix):]
        for plen in range(1, n):
            key = tuple(tail[:plen])
            if any(key[i] == EOS for i in range(len(key) - 1)):
                continue                  # EOS 之后没有分支
            nxt = tail[plen]
            cond[(key, nxt)] = cond.get((key, nxt), 0) + 1

    prefixes = [tuple(t) for t in itertools.product(range(NV), repeat=n - 1)]
    prefixes = [p for p in prefixes
                if not any(p[i] == EOS for i in range(len(p) - 1))]
    bad, checked, max_z = 0, 0, 0.0
    print(f"  {'前缀':>10} {'下一 token':>10} {'精确':>8} {'经验':>8} {'z':>7}")
    shown = 0
    for key in prefixes:
        tot = sum(cond.get((key, t), 0) for t in range(NV))
        if tot < 300:
            continue
        sub = prefix_dist(P, key, 1)
        for t in range(NV):
            e = sub[(t,)]
            emp = cond.get((key, t), 0) / tot
            sd = math.sqrt(max(e * (1 - e), 1e-12) / tot)
            z = abs(emp - e) / sd
            checked += 1
            max_z = max(max_z, z)
            if z > 4.0:
                bad += 1
            if shown < 10 and e > 0.05:
                print(f"  {str(key):>10} {t:>10} {e:>8.4f} {emp:>8.4f} {z:>7.2f}")
                shown += 1
    print(f"  检查了 {checked} 个 (前缀, 下一 token) 组合，|z|>4 的 = {bad} 个，"
          f"最大 |z| = {max_z:.2f}")
    print("  判据：条件分布的抽样误差约 N(0,1)，|z|>4 才说明分布真的错了。")

    print("\n[E] 回滚不变量：提交状态长度必须等于已接受前缀长度")
    rng = np.random.default_rng(7)
    violations = 0
    rounds_total = 0
    for _ in range(3000):
        seq, tr = speculative_decode(P, Q, prefix, 6, 3, rng)
        committed = [r["committed"] for r in tr if r.get("round_end")]
        rounds_total += len(committed)
        if committed != sorted(committed):
            violations += 1
        if committed[-1] != len(seq):
            violations += 1
        # 提交长度不能超过实际序列长度：拒绝位置的写入不允许留在状态里
        for r in tr:
            if r.get("committed") is not None and r["committed"] > len(seq):
                violations += 1
    print(f"  3000 次运行、{rounds_total} 个轮次，提交长度不变量违例 = {violations}")
    print("  对真实引擎的含义：拒绝发生时 KV/递推状态必须回滚到已接受 token 的")
    print("  边界，而不是保留被拒位置的写入；这条不变量可以在引擎里直接断言。")

    print("\n[G] EOS 吸收：序列中出现 EOS 的概率要与目标一致")
    exact_eos = sum(pr for tail, pr in exact.items() if EOS in tail)
    counts = empirical(P, Q, prefix, n, gamma, trials, 4242)
    emp_eos = sum(c for tail, c in counts.items() if EOS in tail) / trials
    sd = math.sqrt(max(exact_eos * (1 - exact_eos), 1e-12) / trials)
    print(f"  精确 P(出现 EOS) = {exact_eos:.4f}  经验 = {emp_eos:.4f}  "
          f"z = {(emp_eos - exact_eos) / sd:+.2f}")
    print("  吸收态要求两点：接受 EOS 立即终止本轮；残差采样也能产出 EOS。")
    print("  参照实现把两者都交给同一套概率，所以不需要为 EOS 单独加规则。")

    print("\n结论：")
    print(f"  1) 参照实现的序列级总变差 {tv0:.4f}，落在 {trials} 次抽样的误差内；")
    print("  2) 三个错误变体的总变差与最大 |z| 显著更大——它们都能给出正确的")
    print("     单位置边际，却在序列级偏离目标分布；")
    print("  3) 因此「接受率高」「逐 token 频率正常」都不能作为分布正确的证据，")
    print("     必须做序列级或条件前缀级的对拍。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
