#!/usr/bin/env python3
"""L0.0b lab · 把一次参数更新的每一步打印出来。

问题：loss 是一个标量，它怎么变成每个参数的一个数？「更新一次参数」到底改了什么？

本脚本按顺序打印：
    [0] 训练样本            —— 输入与 labels 的对齐关系
    [1] 逐 token loss       —— 一个标量是怎么从 T 个数平均出来的
    [2] 手算梯度 vs autograd —— 先在一个两参数的函数上对拍，再回到模型
    [3] 反向保存了什么      —— 前向的哪些中间值必须活到反向
    [4] 一次参数更新        —— SGD 与 AdamW 各改了哪些字节
    [5] 训练几十步          —— loss 真的会降
    [A] 失败案例：标签没移位 —— loss 迅速趋零，模型无用
    [B] 失败案例：detach     —— 梯度在哪里断掉

用法：python walk_train.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiny_lm import CORPUS, CharVocab, TinyLM, show

torch.manual_seed(0)
SEP = "-" * 78


def main() -> None:
    print(f"torch {torch.__version__}  device=cpu\n")
    vocab = CharVocab(CORPUS)
    D, H, L, BS = 32, 4, 2, 16
    model = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=L, block_size=BS)

    # ---------------------------------------------------------------- [0]
    print("[0] 训练样本：输入与 labels")
    text = "风吹过山谷，云落在水面"
    ids = vocab.encode(text)
    x = torch.tensor([ids[:-1]])          # 输入：去掉最后一个
    y = torch.tensor([ids[1:]])           # 标签：去掉第一个 —— 整体左移一位
    T = x.shape[1]
    print(f"    原文  {text!r}  ({len(ids)} 个字符)")
    print(f"    {'位置':>4s} {'输入 x':>8s} {'标签 y':>8s}   含义")
    for t in range(T):
        print(f"    {t:>4d} {vocab.itos[x[0,t]]:>8s} {vocab.itos[y[0,t]]:>8s}"
              f"   看到 {vocab.decode(ids[:t+1])!r} 之后应当预测 {vocab.itos[y[0,t]]!r}")
    print(f"\n    labels 就是输入右移一位。一条长度 {T} 的样本同时提供了 {T} 个训练信号，")
    print("    因为因果掩码保证位置 t 看不到 t 之后的内容——这叫 teacher forcing：")
    print("    训练时每一步喂的都是真实的前缀，而不是模型自己上一步生成的字符。")
    print(SEP)

    # ---------------------------------------------------------------- [1]
    print("[1] 从 logits 到一个标量 loss")
    logits = model(x)                                   # [1, T, V]
    V = logits.shape[-1]
    print(f"    {show('logits', logits)}")
    flat_logits = logits.view(-1, V)                    # [T, V]
    flat_y = y.view(-1)                                 # [T]
    per_tok = F.cross_entropy(flat_logits, flat_y, reduction="none")
    loss = per_tok.mean()
    print(f"    {show('per_token_loss', per_tok, k=6)}")
    print(f"    loss = per_token_loss.mean() = {loss.item():.6f}")
    print(f"\n    交叉熵：位置 t 的 loss = -log p(正确字符)，p 来自 softmax(logits[t])。")
    t0 = 0
    p0 = F.softmax(logits[0, t0], dim=-1)
    print(f"    验证位置 {t0}：正确字符 {vocab.itos[y[0,t0]]!r}，"
          f"p={p0[y[0,t0]].item():.6f}，-log p={-p0[y[0,t0]].log().item():.6f}"
          f"（脚本算出 {per_tok[t0].item():.6f}）")
    import math
    print(f"    未训练时模型接近均匀分布，预期 loss ≈ ln(V) = ln({V}) = {math.log(V):.4f}，"
          f"实测 {loss.item():.4f}")
    print(f"\n    mean() 的分母是 {T}（有效 token 数）。真实训练里有 padding，")
    print("    那时必须用 mask 把 padding 位置排除，分母也只数有效 token（见 L7.0b）。")
    print(SEP)

    # ---------------------------------------------------------------- [2]
    print("[2] 手算梯度 vs autograd")
    print("    先在一个两参数的函数上对拍，把链式法则写出来：")
    print("      f(a, b) = (a * b + a)^2，  取 a=2, b=3")
    print("      设 u = a*b + a = a(b+1)，f = u^2")
    print("      df/du = 2u")
    print("      du/da = b + 1,   du/db = a")
    print("      df/da = 2u(b+1), df/db = 2u·a")
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    f = (a * b + a) ** 2
    f.backward()
    u = 2.0 * 3.0 + 2.0
    print(f"      代入 u = {u}:  手算 df/da = {2*u*(3+1)}, df/db = {2*u*2.0}")
    print(f"      autograd:      df/da = {a.grad.item()}, df/db = {b.grad.item()}")
    print(f"      一致：{a.grad.item() == 2*u*(3+1) and b.grad.item() == 2*u*2.0}")

    print("\n    再看模型里最靠近 loss 的那一层：lm_head。")
    print("    交叉熵 + softmax 的梯度有一个简洁形式：dL/dlogits = (p - onehot(y)) / N")
    print(f"    其中 p 是 softmax 输出，N 是参与平均的 token 数（这里 N={T}）。")
    model.zero_grad()
    logits2 = model(x)
    logits2.retain_grad()
    loss2 = F.cross_entropy(logits2.view(-1, V), flat_y)
    loss2.backward()
    p_all = F.softmax(logits2.detach().view(-1, V), dim=-1)
    onehot = F.one_hot(flat_y, V).float()
    manual = (p_all - onehot) / T
    got = logits2.grad.view(-1, V)
    print(f"    手算与 autograd 的最大绝对差 = {(manual - got).abs().max().item():.3e}")
    print(f"    位置 0 的梯度前 6 个：手算 "
          f"[{', '.join(f'{v:+.6f}' for v in manual[0, :6].tolist())}]")
    print(f"    {'':>26s}autograd "
          f"[{', '.join(f'{v:+.6f}' for v in got[0, :6].tolist())}]")
    print(f"\n    正确字符那一项的梯度是 (p-1)/N < 0，其余是 p/N > 0：")
    print(f"      降低正确字符的 logit 会让 loss 变大，所以梯度为负、更新时朝增大方向走。")
    print(SEP)

    # ---------------------------------------------------------------- [3]
    print("[3] 反向需要前向的哪些中间值")
    print("    反向不是「再跑一遍前向」，它需要前向算出来的一些张量。")
    print("    PyTorch 把它们挂在计算图的节点上：")
    node = loss2.grad_fn
    chain = []
    for _ in range(6):
        chain.append(type(node).__name__)
        if not node.next_functions or node.next_functions[0][0] is None:
            break
        node = node.next_functions[0][0]
    print(f"      loss.grad_fn 链（前 6 跳）：{' -> '.join(chain)}")
    print("\n    以 softmax+交叉熵为例：反向要用到前向算出的 p。")
    print("    所以 p 必须一直留在显存里，直到反向用完。这就是 activation memory。")
    print(f"    本例中 logits 本身就是 {logits2.numel()} 个 fp32 = "
          f"{logits2.numel()*4} 字节；真实模型里这一项是显存大头（L7.0/L7.1）。")
    print("    L2.6b 打印过完整的反向图：前向 11 个节点，反向 25 个。")
    print(SEP)

    # ---------------------------------------------------------------- [4]
    print("[4] 一次参数更新改了什么")
    w = model.lm_head.weight
    before = w.detach().clone()
    g = w.grad.detach().clone()
    print(f"    {show('lm_head.weight', w)}")
    print(f"    {show('  .grad', g)}")
    print(f"    梯度和参数**形状完全相同**：每个参数一个数。")
    print(f"    梯度范数 |g| = {g.norm().item():.6f}")

    print("\n    (a) 最朴素的 SGD：w <- w - lr * g")
    lr = 0.1
    manual_sgd = before - lr * g
    opt = torch.optim.SGD([w], lr=lr)
    opt.step()
    print(f"      手算与 optimizer 的最大差 = {(manual_sgd - w.detach()).abs().max().item():.3e}")
    print(f"      w[0,0]: {before[0,0].item():+.6f} -> {w[0,0].item():+.6f}  "
          f"(变化 {(w[0,0]-before[0,0]).item():+.6f} = -{lr} × {g[0,0].item():+.6f})")

    print("\n    (b) AdamW：多了两个和参数等大的状态张量")
    w.detach().copy_(before)                      # 还原
    w.grad = g.clone()
    opt2 = torch.optim.AdamW([w], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt2.step()
    st = opt2.state[w]
    print(f"      状态键：{sorted(k for k in st)}")
    for k in ("exp_avg", "exp_avg_sq"):
        print(f"      {show(k, st[k])}")
    print(f"      w[0,0]: {before[0,0].item():+.6f} -> {w[0,0].item():+.6f}")
    print(f"\n      第一步时 m/(sqrt(v)+eps) 经偏差修正后接近 sign(g)，")
    print(f"      所以每个参数的变化幅度都接近 lr={lr}，与梯度大小基本无关。")
    print(f"      代价：每个参数要多存 2 个 fp32 状态。参数 {w.numel()} 个 -> "
          f"权重 {w.numel()*4} B + 状态 {w.numel()*8} B = {w.numel()*12} B。")
    print(f"      **优化器状态是权重的 2 倍**，这是训练显存账本的关键一项（L7.0b/L7.1）。")
    print(SEP)

    # ---------------------------------------------------------------- [5]
    print("[5] 训练 200 步，看 loss 下降")
    torch.manual_seed(0)
    m = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=L, block_size=BS)
    xs = torch.tensor([ids[:-1]])
    ys = torch.tensor([ids[1:]])
    opt3 = torch.optim.AdamW(m.parameters(), lr=3e-3)
    print(f"    {'step':>6s} {'loss':>10s} {'困惑度 exp(loss)':>18s}")
    for step in range(201):
        lg = m(xs)
        ls = F.cross_entropy(lg.view(-1, V), ys.view(-1))
        opt3.zero_grad()
        ls.backward()
        opt3.step()
        if step % 40 == 0:
            print(f"    {step:>6d} {ls.item():>10.4f} {ls.exp().item():>18.3f}")
    print(f"\n    起点 ln(V)={math.log(V):.4f} 对应困惑度 {V}（等于随机猜）。")
    print("    这条样本被记住了——只有一条训练数据时这叫过拟合，但它证明梯度确实在起作用。")
    g2 = torch.Generator().manual_seed(7)
    out = m.generate(torch.tensor([vocab.encode("风吹")]), 8, temperature=0.5, generator=g2)
    print(f"    训练后从 '风吹' 续写：{vocab.decode(out[0].tolist())!r}")
    print(SEP)

    # ---------------------------------------------------------------- [A]
    print("[A] 失败案例：labels 没有移位")
    torch.manual_seed(0)
    m_bad = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=L, block_size=BS)
    xb = torch.tensor([ids[:-1]])
    yb = torch.tensor([ids[:-1]])            # 错误：用 x 预测 x
    opt4 = torch.optim.AdamW(m_bad.parameters(), lr=3e-3)
    print(f"    {'step':>6s} {'错误 loss':>12s} {'正确 loss':>12s}")
    torch.manual_seed(0)
    m_ok = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=L, block_size=BS)
    opt5 = torch.optim.AdamW(m_ok.parameters(), lr=3e-3)
    for step in range(81):
        lb = F.cross_entropy(m_bad(xb).view(-1, V), yb.view(-1))
        opt4.zero_grad(); lb.backward(); opt4.step()
        lo = F.cross_entropy(m_ok(xs).view(-1, V), ys.view(-1))
        opt5.zero_grad(); lo.backward(); opt5.step()
        if step % 20 == 0:
            print(f"    {step:>6d} {lb.item():>12.4f} {lo.item():>12.4f}")
    print("\n    两条 loss 曲线几乎一模一样，错误版本没有任何异常表现。")
    print("    因为它学的是恒等映射：位置 t 的输入就是位置 t 的答案，")
    print("    模型只要把 embedding 抄到输出即可——这同样能把 loss 压到很低。")
    print("    **光看 loss 曲线分辨不出这个错误。** 分辨它要看生成结果：")
    g3 = torch.Generator().manual_seed(7)
    ob = m_bad.generate(torch.tensor([vocab.encode("风吹")]), 8, temperature=0.5, generator=g3)
    g4 = torch.Generator().manual_seed(7)
    oo = m_ok.generate(torch.tensor([vocab.encode("风吹")]), 8, temperature=0.5, generator=g4)
    print(f"      正确模型：{vocab.decode(oo[0].tolist())!r}")
    print(f"      错误模型：{vocab.decode(ob[0].tolist())!r}   ← 在复读输入")
    print("\n    训练脚本至少要有一个不依赖 loss 的检查：定期生成一小段看，")
    print("    或者在留出集上算指标（本例只有一条数据，无法留出）。")
    print(SEP)

    # ---------------------------------------------------------------- [B]
    print("[B] 失败案例：detach 把梯度断在哪里")
    m2 = TinyLM(vocab.size, d_model=D, n_head=H, n_layer=1, block_size=BS)
    for tag, use_detach in (("正常", False), ("对 embedding 输出 detach", True)):
        m2.zero_grad()
        emb = m2.tok_emb(xs)
        if use_detach:
            emb = emb.detach()
        h = emb + m2.pos_emb(torch.arange(xs.shape[1]))
        for blk in m2.blocks:
            h = blk(h)
        lg2 = m2.lm_head(m2.ln_f(h))
        F.cross_entropy(lg2.view(-1, V), ys.view(-1)).backward()
        ge = m2.tok_emb.weight.grad
        gh = m2.lm_head.weight.grad
        print(f"    {tag:<24s} tok_emb.grad = "
              f"{'None' if ge is None else f'范数 {ge.norm().item():.6f}'}"
              f" | lm_head.grad 范数 = {gh.norm().item():.6f}")
    print("\n    detach 切断的是**这一条路径**上的梯度回传，它下游的参数照常拿到梯度。")
    print("    这正是冻结部分模块的做法（VLM 冻结 ViT、LoRA 冻结主干），见 L4.6 / L7.5。")


if __name__ == "__main__":
    main()
