#!/usr/bin/env python3
"""L5.13 —— SGLang 混合架构支持的版本核查（静态）。

5.13 的正文以 RecurrentGemma 为载体讲混合架构的定长状态与回滚。
要在 SGLang 上做同一件事，第一步是问：**这个版本支持哪些混合架构？**

这个脚本只读安装包，不改任何东西，把三类证据打出来：
  1. 型号注册与模型文件里有没有目标架构；
  2. 混合/线性注意力的运行时设施在不在（缓存池、radix、slot）；
  3. 对照架构（Qwen3-Next、NemotronH 等）在不在。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

TARGETS = ["RecurrentGemma", "recurrent_gemma"]
HYBRID_ARCHS = ["MambaForCausalLM", "Mamba2ForCausalLM", "Qwen3NextForCausalLM",
                "NemotronHForCausalLM", "ZambaForCausalLM", "JambaForCausalLM",
                "BailingMoeV2ForCausalLM", "GraniteMoeHybridForCausalLM"]
FACILITIES = ["hybrid_cache", "mamba_radix_cache", "mamba_slot_fused",
              "mamba_checkpoint_pool"]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--site-packages", default="/scratch/learn/envs/sgl/lib/python3.12/site-packages")
    ap.add_argument("--python", default="/scratch/learn/envs/sgl/bin/python",
                    help="用来读 sglang.__version__ 的解释器（venv 根下的那个）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sgl = f"{args.site_packages}/sglang"
    version = sh(f"{args.python} -c 'import sglang; print(sglang.__version__)'")
    models = sh(f"ls {sgl}/srt/models/").split()

    target_hits = {}
    for t in TARGETS:
        target_hits[t] = int(sh(f"grep -rl '{t}' {sgl}/ --include=*.py 2>/dev/null | wc -l"))
    arch_hits = {a: int(sh(f"grep -rl '{a}' {sgl}/srt/ --include=*.py 2>/dev/null | wc -l"))
                 for a in HYBRID_ARCHS}
    facilities = {}
    for f in FACILITIES:
        n = sh(f"ls {sgl}/srt/mem_cache/ | grep -c '{f}'")
        facilities[f] = int(n or 0)
    hybrid_cache_dir = sh(f"ls {sgl}/srt/mem_cache/hybrid_cache/ 2>/dev/null").split()
    mamba_refs = int(sh(f"grep -rl 'mamba_radix_cache\\|hybrid_cache\\|MambaRadixCache' "
                        f"{sgl}/srt/ --include=*.py 2>/dev/null | wc -l"))
    gemma_models = [m for m in models if "gemma" in m.lower()]

    report = dict(
        sglang_version=version, site_packages=args.site_packages,
        target_arch_hits=target_hits,
        hybrid_arch_hits=arch_hits,
        facility_dirs=facilities, hybrid_cache_dir=hybrid_cache_dir,
        files_referencing_mamba_or_hybrid_cache=mamba_refs,
        gemma_variants_present=gemma_models,
        conclusion=None)
    has_target = any(v > 0 for v in target_hits.values())
    has_machinery = bool(hybrid_cache_dir) and mamba_refs > 0
    report["conclusion"] = dict(
        recurrent_gemma_supported=has_target,
        hybrid_machinery_present=has_machinery,
        supported_hybrid_archs=[a for a, n in arch_hits.items() if n > 0],
        unsupported_hybrid_archs=[a for a, n in arch_hits.items() if n == 0],
    )
    (args.out / "sglang_hybrid_survey.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SGLang {version}")
    print(f"目标架构命中：{target_hits}")
    print(f"混合设施：hybrid_cache 目录 {hybrid_cache_dir}，"
          f"引用 mamba/hybrid 缓存的文件 {mamba_refs} 个")
    print(f"支持的混合架构：{report['conclusion']['supported_hybrid_archs']}")
    print(f"缺席的混合架构：{report['conclusion']['unsupported_hybrid_archs']}")
    print(f"包内 gemma 变体：{gemma_models}")


if __name__ == "__main__":
    main()
