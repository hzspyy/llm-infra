#!/usr/bin/env python3
"""L0.4 lab · 手写 BPE：从 UTF-8 字节到词表，每一轮合并都打印出来。

不用 tokenizers、不用 transformers，只用标准库。目的是把「分词」这件事
彻底摊开：它不是什么语言学处理，就是一个在字节序列上反复做「把最常见的
相邻对合并成一个新符号」的贪心算法。

按顺序打印：
    [0] 一个字符串在内存里是什么   —— UTF-8 编码，逐字节看
    [1] 为什么从字节开始           —— 字节级 BPE 的初始词表就是 0..255
    [2] 训练：每一轮合并           —— 统计相邻对 -> 合并最高频 -> 重复
    [3] 学出来的 merges 表          —— 顺序很重要，它就是「算法」
    [4] 编码：用 merges 切一段新文本 —— 逐步展示
    [5] 解码与往返一致性
    [6] 词表大小的影响             —— 同一句话在不同词表下切成几个 token
    [A] 为什么一个中文字常是 2~3 个 token
    [B] 坑：解码时按 token 边界切 UTF-8 会乱码

用法：python bpe_from_scratch.py
"""

from __future__ import annotations

import collections

SEP = "-" * 78

# 一小段中英混合语料。真实 tokenizer 用几百 GB 文本训练，
# 这里只用两千多字，好处是每一轮合并都能看清楚。
#
# 注意语料要有**多样性**：如果只是同一句话重复几遍，BPE 会把整句合并成
# 一个 token，得出的结论会完全失真（这本身也是一个值得看到的现象，见 [D]）。
CORPUS = (
    "风吹过山谷，云落在水面。山不动，水在动，风来了又走。"
    "山上有树，树下有石，石边有溪。溪水流过石头，声音很轻。"
    "云在天上走，影子在地上走。天很高，地很远，人在中间。"
    "早上有雾，中午有太阳，晚上有月亮。一天就这样过去了。"
    "水从山上下来，流进河里，河流进海里。海很大，看不到边。"
    "风从海上来，带着水的味道。树叶动了，人抬起头看天。"
    "石头不说话，水也不说话，只有风一直在说。"
    "山高水长，天大地大。走过山，走过水，走过很多地方。"
    "the wind blows over the valley, the cloud falls on the water. "
    "the mountain stands still, the water moves, the wind comes and goes. "
    "there are trees on the mountain and stones under the trees. "
    "water flows over the stones and the sound is very light. "
    "clouds walk in the sky and shadows walk on the ground. "
    "the sky is high, the ground is far, and people are in between. "
    "in the morning there is fog, at noon there is sun, at night the moon. "
    "water comes down from the mountain into the river and into the sea. "
    "the sea is large and you cannot see its edge. "
    "the wind comes from the sea carrying the taste of water. "
)


# ---------------------------------------------------------------------------
# 预切分（pre-tokenization）
# ---------------------------------------------------------------------------

def pre_tokenize(text: str) -> list[str]:
    """把文本先切成若干「块」，BPE 只在块内部合并，不跨块。

    真实实现用一个正则（GPT-2 的那条最出名）：
        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    Python 标准库的 re 不支持 \p{L}，所以这里用字符类别手写一个等价物：
    按「字母 / 数字 / 空白 / 其他」分组，前导空格并入后面的块
    （这就是为什么真实词表里有大量以空格开头的 token，比如 ' the'）。
    """
    def kind(ch: str) -> str:
        if ch.isspace():
            return "s"
        if ch.isalpha():
            return "L"
        if ch.isdigit():
            return "N"
        return "P"

    chunks: list[str] = []
    cur, cur_kind = "", ""
    for ch in text:
        k = kind(ch)
        if k != cur_kind and cur:
            chunks.append(cur)
            cur = ""
        cur_kind = k
        cur += ch
    if cur:
        chunks.append(cur)

    # 把单独的空白块并到后一个块的前面： ['the',' ','wind'] -> ['the',' wind']
    merged: list[str] = []
    pending = ""
    for c in chunks:
        if c.isspace():
            pending += c
        else:
            merged.append(pending + c)
            pending = ""
    if pending:
        merged.append(pending)
    return merged


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def get_pair_counts(seqs: list[list[int]]) -> collections.Counter:
    """统计所有序列里相邻符号对的出现次数。"""
    cnt: collections.Counter = collections.Counter()
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            cnt[(a, b)] += 1
    return cnt


def merge_pair(seqs: list[list[int]], pair: tuple[int, int], new_id: int) -> list[list[int]]:
    """把每个序列里所有出现的 pair 替换成 new_id。从左到右、不重叠。"""
    out = []
    for seq in seqs:
        res, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                res.append(new_id)
                i += 2
            else:
                res.append(seq[i])
                i += 1
        out.append(res)
    return out


def train_bpe(text: str, n_merges: int, verbose_rounds: int = 12,
              use_pretok: bool = True
              ) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """返回 (merges 有序列表, id -> 字节串)。

    初始词表固定是 0..255 这 256 个字节。之后每合并一次就多一个 id。
    use_pretok=False 时不做预切分，用来演示预切分为什么必要（见 [C]）。
    """
    chunks = pre_tokenize(text) if use_pretok else [text]
    seqs = [list(c.encode("utf-8")) for c in chunks if c]

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []

    print(f"    初始：{len(vocab)} 个字节 token，语料切成 {len(seqs)} 段，"
          f"共 {sum(len(s) for s in seqs)} 个符号")
    print(f"    {'轮':>3s} {'合并的对':>22s} {'次数':>6s} {'新 id':>6s} "
          f"{'新符号':<14s} {'剩余符号数':>10s}")

    for r in range(n_merges):
        counts = get_pair_counts(seqs)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:                       # 只出现一次的对，合并没有意义
            print(f"    第 {r} 轮：最高频的对只出现 {freq} 次，停止")
            break
        new_id = 256 + len(merges)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        seqs = merge_pair(seqs, pair, new_id)
        merges.append(pair)

        if r < verbose_rounds:
            a, b = show_token(vocab[pair[0]]), show_token(vocab[pair[1]])
            print(f"    {r:>3d} {a + ' + ' + b:>22s} {freq:>6d} {new_id:>6d} "
                  f"{show_token(vocab[new_id]):<14s} {sum(len(s) for s in seqs):>10d}")
        elif r == verbose_rounds:
            print(f"    ... （后面 {n_merges - verbose_rounds} 轮省略）")

    return merges, vocab


def show_token(b: bytes) -> str:
    """把一个 token 的字节串显示成可读形式。

    完整的 UTF-8 序列就显示成字符；不完整的（半个汉字）显示成十六进制。
    这个区分本身就是本 lab 要讲的重点之一。
    """
    try:
        s = b.decode("utf-8")
        return repr(s)
    except UnicodeDecodeError:
        return "<" + " ".join(f"{x:02x}" for x in b) + ">"


# ---------------------------------------------------------------------------
# 编码 / 解码
# ---------------------------------------------------------------------------

def encode(text: str, merges: list[tuple[int, int]], trace: bool = False,
           use_pretok: bool = True) -> list[int]:
    """按 merges 的**顺序**反复合并。顺序不能变——它就是学到的那个算法。"""
    if use_pretok:
        out: list[int] = []
        for chunk in pre_tokenize(text):
            out.extend(_encode_chunk(chunk, merges, trace))
        return out
    return _encode_chunk(text, merges, trace)


def _encode_chunk(text: str, merges: list[tuple[int, int]], trace: bool = False) -> list[int]:
    ids = list(text.encode("utf-8"))
    rank = {pair: i for i, pair in enumerate(merges)}
    step = 0
    while len(ids) >= 2:
        # 找当前序列里 rank 最小（即最早学到）的那个可合并对
        pairs = set(zip(ids, ids[1:]))
        best = min(pairs, key=lambda p: rank.get(p, float("inf")))
        if best not in rank:
            break
        ids = merge_pair([ids], best, 256 + rank[best])[0]
        step += 1
        if trace and step <= 8:
            print(f"      第 {step} 步：合并 {best} -> {256 + rank[best]}，"
                  f"剩 {len(ids)} 个 token")
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------

def main() -> None:
    # ---------------------------------------------------------------- [0]
    print("[0] 一个字符串在内存里是什么")
    s = "风"
    print(f"    '{s}' 的 Unicode 码点：U+{ord(s):04X}")
    print(f"    UTF-8 编码：{list(s.encode('utf-8'))}  "
          f"= {' '.join(f'{b:02x}' for b in s.encode('utf-8'))}  （3 个字节）")
    for t in ("a", "é", "风", "🌊"):
        bs = t.encode("utf-8")
        print(f"    {t!r:<6s} U+{ord(t):05X}  {len(bs)} 字节  "
              f"{' '.join(f'{b:02x}' for b in bs)}")
    print("\n    UTF-8 是变长的：ASCII 1 字节，多数汉字 3 字节，emoji 4 字节。")
    print("    模型看不到「字符」，它看到的是 token id；而 token 是从字节合并出来的。")
    print(SEP)

    # ---------------------------------------------------------------- [1]
    print("[1] 为什么从字节开始")
    print("    如果从「字符」开始，词表就要覆盖所有 Unicode 字符（十几万个），")
    print("    而且永远会遇到没见过的字符（生僻字、新 emoji）-> 需要 <UNK>。")
    print("    从**字节**开始只要 256 个起始符号，任何输入都能表示，永远不会 OOV。")
    print("    代价：一个汉字起手就是 3 个 token，要靠合并把它压回 1 个。")
    print(SEP)

    # ---------------------------------------------------------------- [2]
    print("[2] 训练：每一轮合并")
    merges, vocab = train_bpe(CORPUS, n_merges=120, verbose_rounds=12)
    print(f"\n    共合并 {len(merges)} 次，词表 = 256 + {len(merges)} = {256+len(merges)}")
    print(SEP)

    # ---------------------------------------------------------------- [3]
    print("[3] merges 表：它是有序的，顺序就是算法")
    print(f"    {'序号':>4s} {'左':>12s} {'右':>12s} {'合并结果':>16s}")
    for i in list(range(6)) + list(range(len(merges) - 4, len(merges))):
        a, b = merges[i]
        print(f"    {i:>4d} {show_token(vocab[a]):>12s} {show_token(vocab[b]):>12s} "
              f"{show_token(vocab[256+i]):>16s}")
        if i == 5:
            print(f"    {'...':>4s}")
    print("\n    编码时必须**按这个顺序**尝试合并。换顺序会切出不同的结果，")
    print("    模型就认不出来了——所以 merges 表和词表一样，是模型文件的一部分。")
    print(SEP)

    # ---------------------------------------------------------------- [4]
    print("[4] 编码：把一段新文本切开")
    for probe in ("风吹过山谷", "风很大", "the wind blows", "量子力学"):
        ids = encode(probe, merges)
        toks = [show_token(vocab[i]) for i in ids]
        n_bytes = len(probe.encode("utf-8"))
        print(f"    {probe!r:<18s} {n_bytes:>2d} 字节 -> {len(ids):>2d} 个 token: "
              f"{' | '.join(toks)}")
    print("\n    三件值得注意的事：")
    print("    1. 语料里频繁出现的字（风、过、山、很、大）已经被压成 1 个 token；")
    print("       没出现过的字（吹、谷、量、子）还是 2~3 个字节 token。")
    print("    2. 'the wind blows' 开头的 'the' 被切成 't' | 'he'，")
    print("       而中间的 ' wind' 是一个 token —— 因为词表里学到的是 ' the'（带空格），")
    print("       句首的 'the' 没有前导空格，匹配不上。**同一个词在不同位置切法不同。**")
    print("    3. 空格归到后一个词的前面，所以真实词表里满是 ' the' ' and' 这种 token。")
    print("\n    逐步看 '风吹过' 的合并过程：")
    encode("风吹过", merges, trace=True)
    print(SEP)

    # ---------------------------------------------------------------- [5]
    print("[5] 解码与往返一致性")
    for probe in ("风吹过山谷，云落在水面。", "the wind blows over the valley"):
        ids = encode(probe, merges)
        back = decode(ids, vocab)
        print(f"    {probe!r}")
        print(f"      -> {len(ids)} 个 token -> {back!r}   往返一致：{back == probe}")
    print("\n    字节级 BPE 的往返总是无损的：token 就是字节串，拼起来就是原文。")
    print(SEP)

    # ---------------------------------------------------------------- [6]
    print("[6] 词表大小的影响")
    probe = "风吹过山谷，云落在水面。"
    print(f"    对同一句 {probe!r}（{len(probe.encode('utf-8'))} 字节）：")
    print(f"    {'合并次数':>8s} {'词表大小':>8s} {'token 数':>8s} {'压缩比':>8s}")
    for n in (0, 20, 60, 120, 300):
        m, v = train_bpe_quiet(CORPUS, n)
        ids = encode(probe, m)
        nb = len(probe.encode("utf-8"))
        print(f"    {n:>8d} {256+len(m):>8d} {len(ids):>8d} {nb/len(ids):>8.2f}")
    print("\n    合并越多，词表越大，同一句话的 token 越少。")
    print("    真实模型的取舍：词表大 -> 序列短（省算力）但 embedding 表大（占显存）。")
    print("    Qwen3 的词表约 15 万，Llama 3 约 12.8 万，GPT-2 只有 5 万。")
    print(SEP)

    # ---------------------------------------------------------------- [A]
    print("[A] 为什么一个中文字常常是 2~3 个 token")
    m, v = train_bpe_quiet(CORPUS, 120)
    print(f"    {'字':>4s} {'在语料中出现':>12s} {'token 数':>8s}  切法")
    for ch in "风水量":
        ids = encode(ch, m)
        print(f"    {ch:>4s} {CORPUS.count(ch):>12d} {len(ids):>8d}  "
              f"{' | '.join(show_token(v[i]) for i in ids)}")
    print("\n    '量' 在本语料里出现 0 次，却只切成 2 个 token 而不是 3 个。")
    print("    因为 '量' = e9 87 8f，而语料里有 '里' = e9 87 8c——")
    print("    前两个字节相同，<e9 87> 这个合并是从 '里' 学来的，'量' 白捡了。")
    print("    **字节级 BPE 的一个副作用：没见过的字也能部分复用别的字学到的合并。**")
    print("\n    一个汉字 3 个字节。只有当它在训练语料里足够频繁，")
    print("    这 3 个字节才会被合并成 1 个 token；否则它就是 2~3 个 token。")
    print("    所以中文在英文为主的 tokenizer 上「token 效率」低——")
    print("    同样的意思要更多 token，推理更贵、上下文窗口更快用完。")
    print(SEP)

    # ---------------------------------------------------------------- [B]
    print("[B] 坑：token 边界不是字符边界")
    text = "量子"
    ids = encode(text, m)
    print(f"    {text!r} -> {len(ids)} 个 token")
    print(f"    {'i':>3s} {'id':>6s} {'字节':>14s} {'单独 decode':>16s}")
    for i, tid in enumerate(ids):
        b = v[tid]
        try:
            单独 = repr(b.decode("utf-8"))
        except UnicodeDecodeError:
            单独 = "无法单独解码"
        print(f"    {i:>3d} {tid:>6d} {' '.join(f'{x:02x}' for x in b):>14s} {单独:>16s}")
    print("\n    流式输出时如果每收到一个 token 就单独 decode，会得到乱码或替换字符。")
    print("    正确做法是维护一个字节缓冲：拼上新 token 的字节，")
    print("    只把能构成完整字符的部分吐出去，剩下的留到下一个 token。")
    print("    这就是「增量 detokenize」，L0.5 会实现它。")
    print("\n    验证：逐个 token 单独 decode 再拼接")
    naive = "".join(v[t].decode("utf-8", errors="replace") for t in ids)
    print(f"      朴素做法 -> {naive!r}")
    print(f"      正确做法 -> {decode(ids, v)!r}")
    print(SEP)

    # ---------------------------------------------------------------- [C]
    print("[C] 预切分（pre-tokenization）为什么必要")
    print("    BPE 只统计「相邻对」，它不知道什么是词、什么是句子。")
    print("    不做预切分时，它会合并跨越空格甚至跨越标点的组合：\n")
    m_no, v_no = train_bpe_quiet(CORPUS, 120, use_pretok=False)
    m_yes, v_yes = train_bpe_quiet(CORPUS, 120, use_pretok=True)
    print(f"    {'':>6s} {'不预切分学到的 token':<40s} {'预切分后学到的 token'}")
    for i in (10, 30, 60, 90, 119):
        a = show_token(v_no[256 + i]) if 256 + i in v_no else "-"
        b = show_token(v_yes[256 + i]) if 256 + i in v_yes else "-"
        print(f"    第{i:>3d}轮 {a:<40s} {b}")
    cross = [show_token(v_no[256 + i]) for i in range(len(m_no))
             if b" " in v_no[256 + i][1:]]
    print(f"\n    不预切分时，学到的 token 里有 {len(cross)} 个**内部含空格**，例如：")
    print(f"      {' '.join(cross[:6])}")
    print(f"    也就是说 {len(cross)}/{len(m_no)} = {len(cross)/len(m_no)*100:.0f}% 的词表预算")
    print("    花在了「某个词 + 它后面那个空格」这种依赖上下文的组合上。")
    print("\n    在**训练语料里出现过**的句子上，不预切分反而更省 token：")
    seen = "the wind comes from the sea"          # 这句在语料里
    for tag, mm, uu in (("不预切分", m_no, False), ("预切分", m_yes, True)):
        idsx = encode(seen, mm, use_pretok=uu)
        print(f"      {tag}：{len(idsx):>2d} 个 token   （{seen!r}）")
    print("    因为它把整段上下文背下来了。换成没见过的句子：")
    unseen = "a bird sings near the old bridge"   # 这句不在语料里
    for tag, mm, uu in (("不预切分", m_no, False), ("预切分", m_yes, True)):
        idsx = encode(unseen, mm, use_pretok=uu)
        print(f"      {tag}：{len(idsx):>2d} 个 token   （{unseen!r}）")
    print("\n    **两者一样。** 在这个语料规模上，预切分并没有带来压缩优势——")
    print("    这是个诚实的负面结果，不要硬说它更省 token。")
    print("    预切分真正的理由是词表预算：上面那 25 个含空格的 token 各占一个词表槽，")
    print("    它们只在特定上下文里有用。真实词表有 15 万个槽、训练语料有几百 GB 时，")
    print("    这类浪费会累积成显著差距，而且会让同一个词在不同位置切法不一致。")
    print("    另一个理由是可控性：预切分保证 token 不跨越数字、标点与文字的边界，")
    print("    这对后面要讲的约束解码（L5.6）和数字处理很重要。")
    print(SEP)

    # ---------------------------------------------------------------- [D]
    print("[D] 一个容易得出错误结论的陷阱：语料没有多样性")
    tiny = "风吹过山谷，云落在水面。" * 20
    m_t, v_t = train_bpe_quiet(tiny, 60)
    ids_t = encode("风吹过山谷，云落在水面。", m_t)
    print(f"    如果语料只是同一句话重复 20 遍，训练 60 轮之后：")
    print(f"      这句话被切成 {len(ids_t)} 个 token："
          f"{' | '.join(show_token(v_t[i]) for i in ids_t)}")
    print("    整句变成一个 token，压缩比看起来惊人——但这毫无意义：")
    print("    换任何一句新话都会退回逐字节。")
    print("\n    这类实验最容易得出的错误结论是「BPE 压缩率很高」。")
    print("    评价一个 tokenizer 必须用**它没见过的文本**：")
    for tag, mm, vv in (("重复语料训练的", m_t, v_t), ("多样语料训练的", m_yes, v_yes)):
        held_out = "早上有雾，晚上有月亮。"
        idsh = encode(held_out, mm)
        nb = len(held_out.encode("utf-8"))
        print(f"      {tag}词表在留出句上：{len(idsh):>2d} 个 token，"
              f"压缩比 {nb/len(idsh):.2f}")


def train_bpe_quiet(text: str, n_merges: int, use_pretok: bool = True):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return train_bpe(text, n_merges, verbose_rounds=0, use_pretok=use_pretok)


if __name__ == "__main__":
    main()
