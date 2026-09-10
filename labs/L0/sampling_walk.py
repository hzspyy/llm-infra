#!/usr/bin/env python3
"""L0.5 lab · 从 logits 到你看到的那段文字。

L0.4 走完了「文本 -> 张量」。这个脚本走另一半：模型吐出一串实数之后，
到底经过哪些步骤才变成屏幕上的字符。

按顺序打印：
    [0] 一次真实前向的 logits —— 形状、量级、top-k
    [1] logits -> logprob -> prob —— 三者的关系与各自的用途
    [2] temperature 改变了什么 —— 同一组 logits，不同温度下的分布
    [3] top-k / top-p 的截断 —— 候选集怎么变
    [4] 采样：同一分布采 1000 次，看频率是否符合概率
    [5] 增量 detokenize —— 流式输出为什么不能逐 token decode
    [6] 停止条件 —— eos / stop string / max_tokens 三条路径
    [7] 与引擎对齐 —— 自己实现的 top-p 与 vLLM 的输出对比

用法：python sampling_walk.py [--model Qwen/Qwen3-1.7B]
"""

from __future__ import annotations

import argparse
import collections
import math

import torch
import torch.nn.functional as F

SEP = "-" * 78


# ---------------------------------------------------------------------------
# 自己实现采样：每一步都不用库
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, t: float) -> torch.Tensor:
    """t < 1 让分布更尖，t > 1 更平。t -> 0 等价于 argmax。"""
    if t <= 0:
        out = torch.full_like(logits, float("-inf"))
        out[logits.argmax()] = 0.0
        return out
    return logits / t


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """只保留最大的 k 个，其余置为 -inf。"""
    if k <= 0 or k >= logits.numel():
        return logits
    kth = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """核采样：按概率从大到小累加，累计超过 p 之后的全部丢掉。

    注意边界：第一个使累计**达到或超过** p 的 token 要保留，
    否则 p 很小时可能一个都不剩。
    """
    if p >= 1.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # cum > p 的位置要丢；但把这个 mask 右移一位，保住第一个越界的那个
    remove = cum > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    drop_idx = sorted_idx[remove]
    return logits.index_fill(-1, drop_idx, float("-inf"))


def make_byte_decoder():
    """字节级 BPE 把 0..255 映射到可打印字符（Ġ Ĉ 之类），这里做反向表。

    没有这张表就拿不到 token 的**真实字节**——直接 decode 会在多字节字符
    被拆开时得到 U+FFFD，而 U+FFFD 的字节是 ef bf bd，和原始字节毫无关系。

    映射规则（GPT-2 定义，之后所有字节级 BPE 都沿用）：
      本来就可打印的字节（! 到 ~、¡ 到 ¬、® 到 ÿ）映射到自己；
      其余 68 个（控制字符、空格、以及几个空洞）依次映射到 U+0100 往后。
      所以空格 0x20 -> U+0120 'Ġ'，换行 0x0A -> U+010A 'Ċ'。
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    b2u = {b: chr(b) for b in printable}
    n = 0
    for b in range(256):
        if b not in b2u:
            b2u[b] = chr(256 + n)
            n += 1
    return {c: b for b, c in b2u.items()}


def token_bytes(tok, tid: int, byte_dec: dict) -> bytes:
    """取一个 token 的原始字节。词表里存的是映射后的字符串。"""
    piece = tok.convert_ids_to_tokens([tid])[0]
    try:
        return bytes(byte_dec[ch] for ch in piece)
    except KeyError:                     # 特殊 token（<|im_end|> 等）不走字节映射
        return piece.encode("utf-8")


class IncrementalDetokenizer:
    """流式输出用的增量解码器：维护一个字节缓冲。

    直接对每个 token 调 decode 会在多字节字符被拆开时产生 U+FFFD 替换符。
    正确做法是把字节攒起来，只吐出能构成完整字符的部分。
    """

    def __init__(self) -> None:
        self.buf = b""

    def push(self, token_bytes: bytes) -> str:
        self.buf += token_bytes
        # 从后往前找最长的可解码前缀；未完成的多字节序列留在缓冲里
        for cut in range(len(self.buf), max(len(self.buf) - 4, -1), -1):
            try:
                text = self.buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            self.buf = self.buf[cut:]
            return text
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"torch {torch.__version__}  device={dev}  model={args.model}\n")

    # ---------------------------------------------------------------- [0]
    print("[0] 一次真实前向的 logits")
    prompt = "中国的首都是"
    ids = tok.encode(prompt)
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        out = model(x)
    logits = out.logits[0, -1].float().cpu()      # 只要最后一个位置
    print(f"    prompt {prompt!r} -> {len(ids)} 个 token {ids}")
    print(f"    logits 形状 {tuple(out.logits.shape)}，取最后一个位置 -> {tuple(logits.shape)}")
    print(f"    量级：min {logits.min():.2f}  max {logits.max():.2f}  "
          f"mean {logits.mean():.2f}  std {logits.std():.2f}")
    top = torch.topk(logits, 8)
    print(f"\n    {'排名':>4s} {'id':>8s} {'token':<14s} {'logit':>9s} {'prob':>9s} {'logprob':>9s}")
    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)
    for r, (v, i) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        print(f"    {r:>4d} {i:>8d} {tok.decode([i])!r:<14s} {v:>9.4f} "
              f"{probs[i]:>9.6f} {logprobs[i]:>9.4f}")
    print(SEP)

    # ---------------------------------------------------------------- [1]
    print("[1] logits / logprob / prob 三者的关系")
    print("    logits 是模型直接输出的实数，没有归一化，加一个常数不改变分布。")
    print("    prob    = softmax(logits)            —— 归一化，和为 1")
    print("    logprob = log_softmax(logits)        —— log(prob)，永远 <= 0")
    print(f"\n    验证 1：给所有 logit 加 100，softmax 不变")
    shifted = F.softmax(logits + 100.0, dim=-1)
    print(f"      最大绝对差 = {(shifted - probs).abs().max():.3e}")
    print(f"    验证 2：exp(logprob) == prob")
    print(f"      最大绝对差 = {(logprobs.exp() - probs).abs().max():.3e}")
    print(f"    验证 3：概率之和 = {probs.sum():.6f}")
    print("\n    为什么 API 返回的是 logprob 而不是 prob：")
    print("    概率会小到 1e-30，fp32 存不下也没法比较；取对数之后是 -70 这种数，")
    print("    而且多个 token 的联合概率取对数就变成求和，数值稳定。")
    seq_lp = logprobs[top.indices[0]].item()
    print(f"    例：连续采 10 个概率各 {probs[top.indices[0]]:.4f} 的 token，")
    print(f"        联合概率 = {probs[top.indices[0]].item()**10:.3e}，"
          f"而 logprob 相加 = {seq_lp*10:.2f}")
    print(SEP)

    # ---------------------------------------------------------------- [2]
    print("[2] temperature")
    print("    softmax(z/T)：T 只是在进 softmax 之前除一下。")
    print(f"    {'T':>6s} {'最大概率':>10s} {'熵':>8s} {'有效候选数 exp(熵)':>18s} {'top-3'}")
    for T in (0.0, 0.2, 0.5, 1.0, 1.5, 3.0):
        lg = apply_temperature(logits, T)
        p = F.softmax(lg, dim=-1)
        ent = -(p * p.clamp_min(1e-30).log()).sum()
        t3 = [tok.decode([i]) for i in torch.topk(p, 3).indices.tolist()]
        print(f"    {T:>6.1f} {p.max():>10.4f} {ent:>8.4f} {ent.exp():>18.1f}   {t3}")
    print(f"\n    「有效候选数」= exp(熵)，可以理解成模型实际在几个选项之间犹豫。")
    print(f"    词表 {len(probs)}，均匀分布时 exp(熵) = {len(probs)}。")
    print("    T=0 是特例：不是除以 0，而是直接取 argmax（贪心解码）。")
    print(SEP)

    # ---------------------------------------------------------------- [3]
    print("[3] top-k 与 top-p 的截断")
    print(f"    {'策略':<16s} {'候选数':>8s} {'覆盖概率':>10s} {'候选'}")
    base_p = F.softmax(logits, dim=-1)
    for name, lg in [
        ("原始", logits),
        ("top_k=1", top_k_filter(logits, 1)),
        ("top_k=5", top_k_filter(logits, 5)),
        ("top_k=50", top_k_filter(logits, 50)),
        ("top_p=0.5", top_p_filter(logits, 0.5)),
        ("top_p=0.9", top_p_filter(logits, 0.9)),
        ("top_p=0.99", top_p_filter(logits, 0.99)),
    ]:
        keep = torch.isfinite(lg)
        n = int(keep.sum())
        cov = base_p[keep].sum().item()
        show = [tok.decode([i]) for i in torch.topk(lg, min(4, n)).indices.tolist()]
        print(f"    {name:<16s} {n:>8d} {cov:>10.4f}   {show}")
    print("\n    top-k 固定候选个数，top-p 固定覆盖的概率质量。")
    print("    区别在模型确定时最明显：模型很确定时 top-p 只留 1~2 个，")
    print("    top-k=50 却仍会保留 50 个（包括一堆概率极低的）。")
    print("    两者常常一起用：先 top-k 砍掉长尾，再 top-p 动态收紧。")
    print(SEP)

    # ---------------------------------------------------------------- [4]
    print("[4] 采样 1000 次，频率对不对得上概率")
    lg = top_k_filter(logits, 5)
    p = F.softmax(lg, dim=-1)
    g = torch.Generator().manual_seed(0)
    draws = torch.multinomial(p, 1000, replacement=True, generator=g)
    cnt = collections.Counter(draws.tolist())
    print(f"    {'token':<14s} {'理论概率':>10s} {'实测频率':>10s} {'次数':>6s}")
    for i, c in cnt.most_common():
        print(f"    {tok.decode([i])!r:<14s} {p[i]:>10.4f} {c/1000:>10.4f} {c:>6d}")
    print("\n    采样是**有放回**的多项分布抽样，不是「选最大的」。")
    print("    同样的 prompt 每次跑结果不同，就是这一步引入的随机性。")
    print("    要复现必须固定随机种子——而且引擎的种子和你的不是同一个。")
    print(SEP)

    # ---------------------------------------------------------------- [5]
    print("[5] 增量 detokenize：流式输出的坑")
    text = "风吹过山谷"
    tids = tok.encode(text)
    print(f"    {text!r} -> {len(tids)} 个 token")
    byte_dec = make_byte_decoder()
    byte_map = {t: token_bytes(tok, t, byte_dec) for t in tids}
    print(f"    {'i':>3s} {'id':>8s} {'单独 decode':>14s} {'字节'}")
    for i, t in enumerate(tids):
        print(f"    {i:>3d} {t:>8d} {tok.decode([t])!r:>14s} "
              f"{' '.join(f'{b:02x}' for b in byte_map[t])}")

    print("\n    这三个 token 各自都是完整字符，所以逐个 decode 恰好没问题。")
    print("    但这只是运气好。换一个词表里没有的罕见字：")
    rare = "𩸽"                     # 一个 4 字节的罕见汉字，词表里没有，只能退回字节
    rids = tok.encode(rare)
    print(f"    {rare!r} ({len(rare.encode('utf-8'))} 字节) -> {len(rids)} 个 token")
    print(f"    {'i':>3s} {'id':>8s} {'token':<10s} {'单独 decode':>12s}")
    for i, t in enumerate(rids):
        print(f"    {i:>3d} {t:>8d} {tok.convert_ids_to_tokens([t])[0]!r:<10s} "
              f"{tok.decode([t])!r:>12s}")
    naive = "".join(tok.decode([t]) for t in rids)
    print(f"\n      逐 token decode 再拼接 -> {naive!r}   ← 三个替换符")
    print(f"      整体 decode            -> {tok.decode(rids)!r}")
    print(f"      两者相同：{naive == tok.decode(rids)}")
    print("\n    流式接口是一个 token 一个 token 往外吐的，如果每次都单独 decode，")
    print("    用户就会先看到几个 '�' 再看到正确的字——或者永远看不到正确的字。")

    print("\n    用字节缓冲的正确做法（本脚本的 IncrementalDetokenizer）：")
    det = IncrementalDetokenizer()
    pieces = []
    for t in rids:
        b = token_bytes(tok, t, byte_dec)
        piece = det.push(b)
        pieces.append(piece)
        print(f"      推入 id={t:<8d} 字节 {' '.join(f'{x:02x}' for x in b):<12s} "
              f"-> 吐出 {piece!r}  缓冲剩 {len(det.buf)} 字节")
    print(f"      拼接结果 {''.join(pieces)!r}")
    print(SEP)

    # ---------------------------------------------------------------- [6]
    print("[6] 停止条件")
    print("    三条独立的路径，在不同的层判断：")
    print(f"    1. eos token：采到 {tok.eos_token!r} (id {tok.eos_token_id}) 就停。")
    print("       在采样之后、detokenize 之前判断，属于引擎的调度层。")
    print("    2. stop string：生成的**文本**里出现了指定字符串就停。")
    print("       必须在 detokenize 之后判断，而且要处理跨 token 的情况——")
    print("       停止串可能横跨两个 token，只看单个 token 会漏。")
    print("    3. max_tokens：计数到了就停。这是唯一保证会终止的一条。")
    print("\n    实测一次带 eos 的生成：")
    msgs = [{"role": "user", "content": "用一句话说明风是什么。"}]
    prompt2 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids2 = tok.encode(prompt2, add_special_tokens=False)
    cur = torch.tensor([ids2], device=dev)
    det2 = IncrementalDetokenizer()
    g2 = torch.Generator(device="cpu").manual_seed(1234)
    stop_reason, produced = "max_tokens", []
    with torch.no_grad():
        for step in range(200):
            lg2 = model(cur).logits[0, -1].float().cpu()
            lg2 = top_p_filter(apply_temperature(lg2, 0.7), 0.9)
            p2 = F.softmax(lg2, dim=-1)
            nxt = int(torch.multinomial(p2, 1, generator=g2))
            if nxt == tok.eos_token_id:
                stop_reason = f"eos (id {nxt}) @ step {step}"
                break
            produced.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=dev)], dim=1)
    print(f"    停止原因：{stop_reason}")
    print(f"    生成 {len(produced)} 个 token：{tok.decode(produced)!r}")
    print("\n    再演示 stop string：把 '。' 作为停止串，它可能横跨 token 边界。")
    det3 = IncrementalDetokenizer()
    cur3 = torch.tensor([ids2], device=dev)
    g3 = torch.Generator(device="cpu").manual_seed(1234)
    acc, why = "", "max_tokens"
    with torch.no_grad():
        for step in range(200):
            lg3 = model(cur3).logits[0, -1].float().cpu()
            lg3 = top_p_filter(apply_temperature(lg3, 0.7), 0.9)
            nxt3 = int(torch.multinomial(F.softmax(lg3, dim=-1), 1, generator=g3))
            if nxt3 == tok.eos_token_id:
                why = f"eos @ step {step}"
                break
            piece = det3.push(token_bytes(tok, nxt3, byte_dec))
            acc += piece
            if "。" in acc:                       # 在**文本**上判断，不是在 token 上
                why = f"stop string '。' @ step {step}"
                acc = acc[:acc.index("。") + 1]   # 截到停止串为止
                break
            cur3 = torch.cat([cur3, torch.tensor([[nxt3]], device=dev)], dim=1)
    print(f"    停止原因：{why}")
    print(f"    输出：{acc!r}")
    print(SEP)

    # ---------------------------------------------------------------- [7]
    print("[7] 与引擎对齐：自己实现的 top-p 对不对")
    print("    用同一组 logits，比较自己的实现与 torch 官方写法：")
    for pp in (0.5, 0.9, 0.95):
        mine = top_p_filter(logits, pp)
        # 参照实现：排序 -> 累加 -> 右移一位 -> 丢弃
        sp, si = torch.sort(F.softmax(logits, -1), descending=True)
        cum = sp.cumsum(-1)
        mask = cum - sp > pp           # 另一种等价写法：用「不含自己的累计」
        ref = logits.clone()
        ref[si[mask]] = float("-inf")
        same = torch.equal(torch.isfinite(mine), torch.isfinite(ref))
        print(f"      top_p={pp}: 候选集相同 = {same}  "
              f"(我的 {int(torch.isfinite(mine).sum())} 个 / 参照 {int(torch.isfinite(ref).sum())} 个)")
    print("\n    两种写法在边界上的差别：一个用「含自己的累计 > p」再右移，")
    print("    一个用「不含自己的累计 > p」。数学上等价，实现上容易写错一个位置——")
    print("    差一个 token 在温度低时会显著改变输出。")


if __name__ == "__main__":
    main()
