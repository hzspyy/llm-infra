#!/usr/bin/env python3
"""查漏：把库里可枚举的选项空间和正文对一遍。

动机（2026-09-11）：5.5 写完之后被指出**完全没提 MTP**，
而 MTP 就明明白白写在 vLLM 的 `SpeculativeMethod` 里（26 个模型类型）。
这类漏是可以机械查出来的——与其等人指出来，不如每写完一章跑一遍。

用法：
    python tools/coverage_check.py                 # 全部
    python tools/coverage_check.py vllm.CUDA       # 只看某个前缀

**只查「机制/方法」层面的空间，不查实例枚举。**
  查：投机解码方法、图执行模式、量化格式族、KV dtype 族、编译后端
      —— 漏掉其中一项是概念缺口（MTP 就是）。
  不查：vLLM 支持的 40 个 MoE 架构名、每种哈希算法、每个 fp8 变体
      —— 那是实例清单，本书没有义务逐个点名，列出来只会淹没真正的缺口。

判读：
  「未提到」不等于「该写」。判断之后决定「补写」还是「明确说明不覆盖」。
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPACES = ROOT / "results" / "crater" / "option_spaces.json"
SRC = ROOT / "src"

# 一个选项名要怎么在正文里算「被提到」：把它规约成一个族名
FAMILY = [
    (re.compile(r".*_mtp$|^mtp$"), "MTP"),
    (re.compile(r"^eagle\d?$"), "EAGLE"),
    (re.compile(r"^ngram(_gpu)?$"), "ngram"),
    (re.compile(r"^medusa$"), "Medusa"),
    (re.compile(r"^draft_model$"), "draft model"),
    (re.compile(r"^suffix$"), "suffix decoding"),
    (re.compile(r"^mlp_speculator$"), "MLP speculator"),
    (re.compile(r"^(gptq|gptq_marlin|gptq_bitblas).*"), "GPTQ"),
    (re.compile(r"^(awq|awq_marlin|awq_bitblas).*"), "AWQ"),
    (re.compile(r".*fp8.*"), "FP8"),
    (re.compile(r".*fp4.*|.*nvfp4.*|.*mxfp4.*"), "FP4"),
    (re.compile(r"^(bitsandbytes|bnb).*"), "bitsandbytes"),
    (re.compile(r"^(compressed[-_]tensors|modelopt).*"), "compressed-tensors"),
]


def family(name):
    for pat, fam in FAMILY:
        if pat.match(name):
            return fam
    return name


def main():
    if not SPACES.exists():
        print(f"缺 {SPACES}")
        print("先在 crater 上跑 tools/extract_option_spaces.py 并把 JSON 拉回来。")
        return 1
    spaces = json.loads(SPACES.read_text(encoding="utf-8"))
    want_prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    # 只保留机制层面的空间；实例清单排除（见文件头）
    KEEP = ("vllm.speculative.SpeculativeMethod",
            "vllm.speculative.MTPModelTypes",
            "vllm.CUDAGraphMode", "vllm.QUANTIZATION_METHODS",
            "vllm.CacheDType", "torch.dynamo_backends")
    spaces = {k: v for k, v in spaces.items()
              if k in KEEP or k.startswith("_err")}

    docs = {}
    for md in sorted(SRC.rglob("*.md")):
        docs[md.relative_to(SRC).as_posix()] = md.read_text(encoding="utf-8").lower()
    alltext = "\n".join(docs.values())

    print(f"正文 {len(docs)} 篇，选项空间 "
          f"{len([k for k in spaces if not k.startswith('_err')])} 组\n")
    missing_total = []
    for key, info in spaces.items():
        if key.startswith("_err") or not key.startswith(want_prefix):
            continue
        fams = {}
        for v in info["values"]:
            fams.setdefault(family(v), []).append(v)
        hit, miss = [], []
        for fam, members in sorted(fams.items()):
            # 族名或任一成员出现即算覆盖
            probes = [fam.lower()] + [m.lower() for m in members]
            where = [d for d, t in docs.items() if any(p in t for p in probes)]
            (hit if where else miss).append((fam, len(members), where[:3]))
        cov = len(hit) / max(len(fams), 1)
        flag = "✅" if not miss else ("⚠" if cov >= 0.5 else "❌")
        print(f"{flag} {key}  {len(hit)}/{len(fams)} 族被提到  ({info['note']})")
        for fam, n, where in miss:
            print(f"     未提到: {fam:<24} ({n} 个成员)")
            missing_total.append((key, fam, n))
        if miss and hit:
            print(f"     已提到: {', '.join(f for f, _, _ in hit)}")
        print()

    if missing_total:
        print("=" * 70)
        print(f"共 {len(missing_total)} 个族在全部正文里都没出现：")
        for key, fam, n in missing_total:
            print(f"  {fam:<26} ({n} 个成员)  来自 {key}")
        print("\n注意：没出现**不等于**该写。有些族与本书范围无关，")
        print("      判断之后可以决定「补写」或「明确说明不覆盖」。")
    else:
        print("所有族都至少被某一篇提到。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
