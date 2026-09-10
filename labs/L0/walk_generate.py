#!/usr/bin/env python3
"""L0.0 lab · 把一次生成的每一个张量打印出来。

问题：从「一串字符」到「下一个字符」，中间到底经过了哪些数？形状怎么变？

本脚本按顺序打印：
    [0] 词表与参数清单        —— 「模型」在内存里是什么
    [1] 输入 -> 张量          —— 字符怎么变成数
    [2] 注意力内部            —— Q/K/V、相似度矩阵、掩码、softmax
    [3] MLP 与残差            —— 一层里剩下的部分
    [4] logits -> 概率        —— 分数怎么变成概率
    [5] 采样与自回归          —— 一次生成三个字符
    [A] 消融：去掉 1/sqrt(d)  —— softmax 饱和
    [B] 消融：去掉位置编码    —— 模型对顺序不敏感

用法：python walk_generate.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tiny_lm import CORPUS, CharVocab, TinyLM, param_table, show

torch.manual_seed(0)
SEP = "-" * 78


def main() -> None:
    print(f"torch {torch.__version__}  device=cpu\n")

    # ---------------------------------------------------------------- [0]
    print("[0] 词表与参数清单")
    vocab = CharVocab(CORPUS)
    print(f"    语料 {len(CORPUS)} 个字符，去重后 {vocab.size} 个 -> 词表大小 {vocab.size}")
    print(f"    itos = {vocab.itos}")
    print(f"    '风' -> id {vocab.stoi['风']},  id 0 -> '{vocab.itos[0]}'")

    D, H, L, BS = 32, 4, 2, 16
    model = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=L, block_size=BS)
    rows, total = param_table(model)
    print(f"\n    模型：d_model={D}, n_head={H}(每头 {D//H} 维), n_layer={L}, block_size={BS}")
    print(f"    {'参数名':<28s} {'形状':<18s} {'元素数':>8s}")
    for n, shape, cnt in rows:
        print(f"    {n:<28s} {str(shape):<18s} {cnt:>8,d}")
    print(f"    {'合计':<28s} {'':<18s} {total:>8,d}  ({total*4/1024:.1f} KiB @ fp32)")
    print("    「模型」= 这张表里的所有数字 + 一段决定它们怎么相乘的代码。")
    print(SEP)

    # ---------------------------------------------------------------- [1]
    print("[1] 输入：字符 -> id -> 向量")
    text = "风吹过山谷"
    ids = vocab.encode(text)
    idx = torch.tensor([ids])                      # [1, T]，batch 维放前面
    print(f"    文本   {text!r}")
    print(f"    id     {ids}")
    print(f"    张量   {show('idx', idx)}")

    trace: dict = {}
    logits = model(idx, trace)
    T = idx.shape[1]

    print(f"\n    {show('tok_emb', trace['tok_emb'])}")
    print(f"    {show('x_in', trace['x_in'])}   ← tok_emb + pos_emb")
    print(f"\n    embedding 是一次查表：tok_emb[0, 0] 就是权重矩阵的第 {ids[0]} 行。")
    same = torch.allclose(trace["tok_emb"][0, 0], model.tok_emb.weight[ids[0]])
    print(f"    验证 tok_emb[0,0] == tok_emb.weight[{ids[0]}] : {same}")
    print(SEP)

    # ---------------------------------------------------------------- [2]
    print("[2] 注意力内部（打印的是第 2 层，trace 里被后写入的那层）")
    for k in ("qkv", "q", "k", "v"):
        print(f"    {show(k, trace[k])}")
    print(f"\n    qkv 一次线性层出 3C={3*D} 维，再切成 Q/K/V 三份，各 {D} 维。")
    print(f"    然后拆头：{D} 维拆成 {H} 个头 × {D//H} 维，head 维换到第 1 维，")
    print(f"    所以 q 的形状是 [B={1}, nh={H}, T={T}, hd={D//H}]。")

    raw = trace["att_logits"]
    att = trace["att"]
    print(f"\n    {show('att_logits', raw)}   ← q @ k^T，未缩放未掩码")
    print(f"    {show('att', att)}   ← 除以 sqrt({D//H})、掩码、softmax 之后")
    print(f"\n    第 0 个头的注意力矩阵（每行和为 1，上三角被掩成 0）：")
    a0 = att[0, 0]
    print(f"      {'':>6s}" + "".join(f"{vocab.itos[i]:>8s}" for i in ids))
    for r in range(T):
        cells = "".join(f"{a0[r, c].item():8.3f}" for c in range(T))
        print(f"      {vocab.itos[ids[r]]:>4s}  {cells}   行和={a0[r].sum().item():.4f}")
    print(f"\n    掩码保证第 r 行只有前 r+1 列非零：因果性就是这么实现的。")
    print(f"    {show('attn_out', trace['attn_out'])}   ← att @ v 再合头、再过 proj")
    print(SEP)

    # ---------------------------------------------------------------- [3]
    print("[3] MLP 与残差")
    for k in ("mlp_hidden", "mlp_act", "mlp_out"):
        print(f"    {show(k, trace[k])}")
    print(f"    fc1 把 {D} 维升到 {4*D} 维，gelu 之后 fc2 降回 {D} 维。")
    print(f"    每个子层的输出都加回输入（残差）：x = x + attn(ln(x))，x = x + mlp(ln(x))。")
    print(f"    {show('x_final', trace['x_final'])}   ← 两层之后再过最后一个 LayerNorm")
    print(SEP)

    # ---------------------------------------------------------------- [4]
    print("[4] logits -> 概率")
    print(f"    {show('logits', logits)}")
    print(f"    logits 的最后一维是 {vocab.size}，正好是词表大小：每个词表项一个分数。")
    last = logits[0, -1]                       # 只有最后一个位置用来预测下一个字符
    probs = F.softmax(last, dim=-1)
    print(f"\n    softmax(z)_i = exp(z_i) / Σ_j exp(z_j)")
    print(f"      分子把分数变成正数，分母做归一化，所以结果非负且和为 1。")
    print(f"      代入：最大 logit = {last.max().item():+.4f}，最小 = {last.min().item():+.4f}")
    print(f"      概率和 = {probs.sum().item():.6f}")
    top = torch.topk(probs, 5)
    print(f"\n    位置 {T-1}（'{text[-1]}'）之后最可能的 5 个字符（未训练，接近均匀）：")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"      '{vocab.itos[i]}'  logit={last[i].item():+.4f}  p={p:.4f}")
    print(f"    均匀分布应为 1/{vocab.size} = {1/vocab.size:.4f}，可见模型还没学到东西。")
    print(SEP)

    # ---------------------------------------------------------------- [5]
    print("[5] 采样与自回归")
    g = torch.Generator().manual_seed(1234)
    cur = idx.clone()
    print(f"    起点 {vocab.decode(cur[0].tolist())!r}")
    for step in range(3):
        lg = model(cur[:, -BS:])[:, -1, :]
        p = F.softmax(lg, dim=-1)
        nxt = torch.multinomial(p, 1, generator=g)
        cur = torch.cat([cur, nxt], dim=1)
        tid = nxt.item()
        print(f"    第 {step+1} 步：喂进去 {cur.shape[1]-1} 个 token，"
              f"只取最后一个位置的 logits，采到 id={tid} '{vocab.itos[tid]}' "
              f"(p={p[0, tid].item():.4f}) -> {vocab.decode(cur[0].tolist())!r}")
    print(f"\n    注意每一步都把整个前缀重算了一遍。这正是 KV cache 要解决的浪费（L5.2）。")
    print(SEP)

    # ---------------------------------------------------------------- [A]
    print("[A] 消融：去掉 1/sqrt(d_head)")
    print("    注意力分数是 d_head 个乘积之和。若各分量独立、方差为 1，")
    print(f"    这个和的方差就是 d_head={D//H}，标准差 sqrt(d_head)={math.sqrt(D//H):.3f}。")
    print("    softmax 对输入尺度敏感：分数越大，分布越尖。除以 sqrt(d_head) 把尺度拉回来。")
    print("    下面固定 64 个 key、扫 d_head，每格取 200 次随机试验的平均：\n")
    NKEY, TRIALS = 64, 200
    print(f"    {'d_head':>8s} {'缩放':>6s} {'分数标准差':>12s} {'平均最大概率':>14s} "
          f"{'平均熵':>10s} {'softmax 梯度尺度':>18s}")
    for dh in (8, 64, 256):
        for sc in (True, False):
            stds, maxes, ents, grads = [], [], [], []
            for _ in range(TRIALS):
                q = torch.randn(1, dh)
                k = torch.randn(NKEY, dh)
                a = (q @ k.T)
                if sc:
                    a = a / math.sqrt(dh)
                p = F.softmax(a, dim=-1)
                stds.append(a.std().item())
                maxes.append(p.max().item())
                ents.append(-(p * p.clamp_min(1e-30).log()).sum().item())
                # softmax 的雅可比对角元是 p(1-p)，它决定梯度还剩多少
                grads.append((p * (1 - p)).sum().item())
            m = lambda xs: sum(xs) / len(xs)          # noqa: E731
            print(f"    {dh:>8d} {'是' if sc else '否':>6s} {m(stds):>12.3f} "
                  f"{m(maxes):>14.4f} {m(ents):>10.4f} {m(grads):>18.6f}")
    print(f"\n    不缩放时 d_head 越大，分数的标准差越大（≈ sqrt(d_head)），")
    print(f"    最大概率趋近 1、熵趋近 0——分布塌成 one-hot。")
    print(f"    最后一列是 softmax 雅可比对角元之和 Σp(1-p)：p 接近 0 或 1 时它趋于 0，")
    print(f"    也就是**梯度消失**。这是缩放的真正理由，不只是「数值好看」。")
    print(f"    参照：{NKEY} 个 key 的均匀分布，熵 = ln({NKEY}) = {math.log(NKEY):.4f}，"
          f"Σp(1-p) = {1 - 1/NKEY:.4f}")
    print(SEP)

    # ---------------------------------------------------------------- [B]
    print("[B] 消融：去掉位置编码")
    print("    注意力算的是「查询与每个键的相似度」，它本身不知道谁在前谁在后。")
    print("    不加位置编码时，交换前面两个 token（保持最后一个不变），")
    print("    最后一个位置看到的键值**集合**完全没变，输出应当相同。")
    print("    要干净地验证这一点，必须满足两个条件：\n")
    print("      1. 同一组权重（不是两个随机初始化的模型），只切换 use_pos；")
    print("      2. 单层。多层时第 1 层在位置 0 和 1 上的输出本身就不同，")
    print("         第 2 层看到的集合已经变了，等式不再成立。\n")

    m1 = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=1, block_size=BS)
    a_ids = vocab.encode("风吹过")
    b_ids = vocab.encode("吹风过")              # 只交换前两个，末字符相同
    print(f"    A = {vocab.decode(a_ids)!r} ids={a_ids}    B = {vocab.decode(b_ids)!r} ids={b_ids}")
    print(f"    {'层数':>4s} {'位置编码':>8s} {'末位 logits 最大差':>20s}")
    for nl, m in ((1, m1), (2, model)):
        for up in (False, True):
            m.use_pos = up
            with torch.no_grad():
                d1 = m(torch.tensor([a_ids]))[0, -1]
                d2 = m(torch.tensor([b_ids]))[0, -1]
            print(f"    {nl:>4d} {'有' if up else '无':>8s} "
                  f"{(d1 - d2).abs().max().item():>20.3e}")
        m.use_pos = True
    print("\n    单层无位置编码那一行接近机器精度：两次前向在数学上是同一个和，")
    print("    只是浮点加法的顺序不同（对照 L2.4 关于累加顺序的讨论）。")
    print("    加上位置编码后差值涨了约 5 个数量级——模型这才分得清「风吹」和「吹风」。")
    print("    两层时即使没有位置编码差值也不为零：第 1 层在位置 0 与 1 上")
    print("    分别看到 {风} 和 {风,吹}，A、B 两种输入下这两个集合不同。")


if __name__ == "__main__":
    main()
