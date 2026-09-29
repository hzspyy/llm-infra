#!/usr/bin/env python3
"""L3.2-B（源码部分）—— 固定 revision 下的 FA3 / FA4 源码审计。

FA3 只能在 Hopper 上跑，FA4 只能在 9.x/10.x/11.x 上跑，crater 是 sm_120。
所以这一节不做性能实测，而是把**固定 commit**的源码里和机制有关的位置逐行标出来：
barrier 与 phase、warp 分工、TMA、wgmma、ping-pong、rescale 条件、
TMEM 分配、2-CTA 数据流，以及 FA4 的软件 exp。

产出：
    <L3_OUT>/source/<file>.excerpt.txt    带行号的摘录
    <L3_OUT>/source/manifest.json         每个文件的 URL / SHA256 / 匹配行数

用法：
    L3_OUT=<目录> python fa3_source_audit.py
"""

import hashlib
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Harness                                     # noqa: E402

REPO = "Dao-AILab/flash-attention"
COMMIT = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"        # tag v2.8.3
BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"

HOPPER = "hopper/"
CUTE = "flash_attn/cute/"
FILES = [
    HOPPER + "flash_fwd_kernel_sm90.h",
    HOPPER + "mainloop_fwd_sm90_tma_gmma_ws.hpp",
    HOPPER + "softmax.h",
    HOPPER + "named_barrier.hpp",
    HOPPER + "sm90_pipeline_no_cluster.hpp",
    HOPPER + "tile_scheduler.hpp",
    HOPPER + "flash.h",
    HOPPER + "block.h",
    CUTE + "flash_fwd_sm100.py",
    CUTE + "flash_fwd.py",
    CUTE + "softmax.py",
    CUTE + "blackwell_helpers.py",
    CUTE + "mma_sm100_desc.py",
    CUTE + "pipeline.py",
    CUTE + "named_barrier.py",
    CUTE + "fast_math.py",
    CUTE + "tile_scheduler.py",
]

PAPER = "https://arxiv.org/html/2603.05451"

# 主题 -> 正则。每条命中都带 file:line，正文只引用这个索引。
TOPICS = {
    "barrier/phase": r"NamedBarrier|named_barrier|barrier_arrive|barrier_wait|"
                     r"PipelineState|phase\b|arrive_and_wait|barrier::",
    "warp 分工": r"Producer|Consumer|warp_group|WarpGroup|warp_idx|is_producer|"
                 r"role\b|softmax_warp|load_warp",
    "TMA": r"tma_|TMA|cp_async_bulk|make_tma_copy|tma_load|tma_store|TmaDescriptor",
    "wgmma/mma": r"wgmma|WGMMA|GMMA|make_tiled_mma|tcgen05|umma|MMA_Atom",
    "ping-pong": r"pingpong|Pingpong|ping_pong|PingPong",
    "寄存器归属": r"tCrS|tPrP|tOrO|tPgP|accum|tensor_acc|Fragment|Registers",
    "rescale": r"rescale|scale_apply|row_scale|m_new|alpha\b|correction|"
               r"scale_factor|log2_scale",
    "TMEM": r"tmem|TMEM|TensorMemory|tmem_alloc|tcgen05_alloc|tmem_dealloc",
    "2-CTA": r"cta_group|CTA_2|2cta|2CTA|cluster|multicast|mcast",
    "软件 exp": r"exp2|fast_exp|fast_math|poly|fma\b|ex2\b|hexp|exp_approx|"
                r"exp2f|__expf",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "llm-infra-lab"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def excerpt(text, patterns, ctx=1, max_per_topic=12):
    lines = text.splitlines()
    hits = {}
    for topic, pat in patterns.items():
        rx = re.compile(pat)
        rows = []
        used = set()
        for i, ln in enumerate(lines):
            if rx.search(ln):
                lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                if any(j in used for j in range(lo, hi)):
                    continue
                used.update(range(lo, hi))
                rows.append((i + 1, lines[i].rstrip()))
                if len(rows) >= max_per_topic:
                    break
        hits[topic] = rows
    return hits


def main():
    h = Harness("3.2-B-source", "3.2", out=os.environ.get("L3_OUT"),
                backend="none（源码审计，不需 GPU）",
                notes=f"{REPO}@{COMMIT} (tag v2.8.3)")
    src_dir = h.path("source")
    src_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": REPO, "commit": COMMIT, "tag": "v2.8.3", "files": {},
                "paper": PAPER}
    topic_index = {t: [] for t in TOPICS}

    print("=" * 78)
    print(f"[源码审计] {REPO} @ {COMMIT}  (tag v2.8.3)")
    print("=" * 78)
    for f in FILES:
        url = BASE + f
        try:
            raw = fetch(url)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {f:<55} 下载失败: {str(exc)[:50]}")
            continue
        text = raw.decode("utf-8", errors="replace")
        sha = hashlib.sha256(raw).hexdigest()
        hits = excerpt(text, TOPICS)
        name = f.replace("/", "__") + ".excerpt.txt"
        with open(src_dir / name, "w", encoding="utf-8") as fh:
            fh.write(f"# {REPO}@{COMMIT}  {f}\n# {url}\n# sha256 {sha}\n\n")
            for topic, rows in hits.items():
                if not rows:
                    continue
                fh.write(f"--- {topic} ---\n")
                for lineno, line in rows:
                    fh.write(f"{lineno:>6}: {line}\n")
                fh.write("\n")
        manifest["files"][f] = {
            "url": url, "sha256": sha, "lines": len(text.splitlines()),
            "excerpt": name,
            "matched": {t: len(rows) for t, rows in hits.items() if rows},
        }
        for topic, rows in hits.items():
            for lineno, line in rows:
                topic_index[topic].append((f, lineno, line.strip()[:110]))
        print(f"  {f:<55} {len(text.splitlines()):>6} 行  "
              f"命中 {sum(len(v) for v in hits.values()):>3} 处")

    # 论文：只存与机制有关的段落
    try:
        paper = fetch(PAPER).decode("utf-8", errors="replace")
        paper = re.sub(r"<script.*?</script>", " ", paper, flags=re.S)
        paper = re.sub(r"<style.*?</style>", " ", paper, flags=re.S)
        paper_text = re.sub(r"<[^>]+>", " ", paper)
        paper_text = re.sub(r"\s+", " ", paper_text)
        keys = ["conditional", "rescal", "exp", "TMEM", "2-CTA", "two-CTA",
                "software", "softmax"]
        sents = re.split(r"(?<=[.!?]) ", paper_text)
        picked = []
        for s in sents:
            if any(k.lower() in s.lower() for k in keys) and 40 < len(s) < 400:
                picked.append(s.strip())
            if len(picked) >= 40:
                break
        with open(src_dir / "paper_fa4_sentences.txt", "w", encoding="utf-8") as fh:
            fh.write(f"# {PAPER}\n# 与机制相关的句子摘录（未改写）\n\n")
            for s in picked:
                fh.write(s + "\n\n")
        manifest["paper_sentences"] = len(picked)
        print(f"\n  论文 {PAPER}: 摘录 {len(picked)} 句")
        for s in picked[:12]:
            print(f"    · {s[:150]}")
    except Exception as exc:                                      # noqa: BLE001
        print(f"  论文抓取失败: {str(exc)[:80]}")

    print("\n--- 主题索引（正文引用 file:line 时用这里） ---")
    for topic, rows in topic_index.items():
        print(f"\n  [{topic}]  命中 {len(rows)} 处")
        for f, lineno, line in rows[:6]:
            print(f"    {f}:{lineno}  {line}")
        h.case(id=f"topic_{topic}", hits=len(rows),
               samples=[{"file": f, "line": n, "text": t} for f, n, t in rows[:8]])

    (src_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    h.finish({"verdict": f"固定 {COMMIT} 下按主题标出了 FA3/FA4 的机制位置；"
                         "性能不在本章断言范围内。",
              "commit": COMMIT, "topics": {t: len(r) for t, r in topic_index.items()}})


if __name__ == "__main__":
    main()
