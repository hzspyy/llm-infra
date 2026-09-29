#!/usr/bin/env python3
"""L9.1 三类 agent 任务的生成、工具实现与评分。

三类任务各 100 个，全部由固定 seed 确定性生成：

* ``compute``   —— 本地计算工具：模型必须调用 ``calculate`` 求一个大整数表达式的值；
* ``retrieval`` —— 多轮 NFCorpus 检索：模型调用 ``search_corpus`` 找论据，最后引用文档 id；
* ``codefix``   —— 固定小代码仓库的读取与修复：模型用文件工具读代码、写补丁，由仓库自带测试判定。

工具全部在本地可核对地实现：计算器是 AST 白名单求值，检索是纯 Python BM25（语料与
qrels 来自本地 NFCorpus 快照），代码仓库是 ``labs/L9/fixtures/codefix`` 的副本加单点
注入变异。任务清单、工具 schema、评分规则都在这里，harness 只负责跑模型循环。

用法（本地校验 100 个变异都能被测试捕获）::

    python labs/L9/agent_tasks.py --selftest

用法（打印任务清单）::

    python labs/L9/agent_tasks.py --dump-tasks out/tasks.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE_REPO = HERE / "fixtures" / "codefix"

# 语料与标注：crater 上的 HF 缓存快照（ENVIRONMENTS.md 记录 revision）。
NFCORPUS_CORPUS = (
    "/scratch/learn/models/hf/hub/datasets--BeIR--nfcorpus/snapshots/"
    "b5026a0e96e8a7ac4f95f482a596389289d46269/corpus/corpus-00000-of-00001.parquet"
)
NFCORPUS_QUERIES = (
    "/scratch/learn/models/hf/hub/datasets--BeIR--nfcorpus/snapshots/"
    "b5026a0e96e8a7ac4f95f482a596389289d46269/queries/queries-00000-of-00001.parquet"
)
NFCORPUS_QRELS = (
    "/scratch/learn/models/hf/hub/datasets--BeIR--nfcorpus-qrels/snapshots/"
    "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65/test.tsv"
)

TASK_CLASSES = ("compute", "retrieval", "codefix")

# --------------------------------------------------------------------------------------
# 计算工具：AST 白名单求值
# --------------------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> int:
    """求值一个只含整数、括号与 + - * // % ** 的表达式。"""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise ValueError("only integers allowed")
            return node.value
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
            return op(ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unary operator not allowed")
            return op(ev(node.operand))
        raise ValueError(f"node not allowed: {type(node).__name__}")

    return ev(ast.parse(expr, mode="eval"))


def _gen_expression(rng: random.Random, depth: int = 3) -> str:
    """生成一个深度不超过 depth 的整数表达式，保证整除与幂不会爆炸。"""
    if depth <= 0:
        return str(rng.randint(2, 9999))
    kind = rng.choice(["add", "sub", "mul", "floordiv", "mod", "pow", "leaf"])
    if kind == "leaf":
        return str(rng.randint(2, 9999))
    if kind == "add":
        return f"({_gen_expression(rng, depth - 1)} + {_gen_expression(rng, depth - 1)})"
    if kind == "sub":
        return f"({_gen_expression(rng, depth - 1)} - {rng.randint(1, 5000)})"
    if kind == "mul":
        return f"({_gen_expression(rng, depth - 1)} * {rng.randint(2, 500)})"
    if kind == "floordiv":
        return f"({_gen_expression(rng, depth - 1)} // {rng.randint(2, 500)})"
    if kind == "mod":
        return f"({_gen_expression(rng, depth - 1)} % {rng.randint(3, 500)})"
    return f"({rng.randint(2, 40)} ** {rng.randint(2, 4)})"


# --------------------------------------------------------------------------------------
# 代码修复任务：单点变异
# --------------------------------------------------------------------------------------

# (相对路径, 原文, 变异后文本)。每条的原文在文件内唯一，且至少让一个测试失败；
# 由 --selftest 逐个验证。
MUTATIONS: list[tuple[str, str, str]] = [
    ("textstat/stats.py", "acc += x\n        out.append(acc / (i + 1))", "acc -= x\n        out.append(acc / (i + 1))"),
    ("textstat/stats.py", "acc += x\n        out.append(acc / (i + 1))", "acc *= x\n        out.append(acc / (i + 1))"),
    ("textstat/stats.py", "out.append(acc / (i + 1))", "out.append(acc / i)"),
    ("textstat/stats.py", "out.append(acc / (i + 1))", "out.append(acc / (i + 2))"),
    ("textstat/stats.py", "out.append(acc / (i + 1))", "out.append(acc)"),
    ("textstat/stats.py", "acc = 0.0", "acc = 1.0"),
    ("textstat/stats.py", "for i, x in enumerate(xs):", "for i, x in enumerate(xs[1:]):"),
    ("textstat/stats.py", "mid = n // 2", "mid = n // 2 + 1"),
    ("textstat/stats.py", "if n % 2 == 1:", "if n % 2 == 0:"),
    ("textstat/stats.py", "return s[mid]", "return s[mid - 1]"),
    ("textstat/stats.py", "return (s[mid - 1] + s[mid]) / 2.0", "return (s[mid] + s[mid + 1]) / 2.0"),
    ("textstat/stats.py", "return (s[mid - 1] + s[mid]) / 2.0", "return (s[mid - 1] + s[mid]) / 4.0"),
    ("textstat/stats.py", "s = sorted(xs)\n    n = len(s)", "s = list(xs)\n    n = len(s)"),
    ("textstat/stats.py", "n = len(s)", "n = len(s) - 1"),
    ("textstat/stats.py", "if p <= 0:\n        return s[0]", "if p <= 0:\n        return s[-1]"),
    ("textstat/stats.py", "if p >= 100:\n        return s[-1]", "if p >= 100:\n        return s[0]"),
    ("textstat/stats.py", "if p >= 100:\n        return s[-1]", "if p >= 100:\n        return s[1]"),
    ("textstat/stats.py", "rank = int(round(p / 100.0 * (len(s) - 1)))", "rank = round(p / 100.0 * (len(s) - 1)) + 1"),
    ("textstat/stats.py", "rank = int(round(p / 100.0 * (len(s) - 1)))", "rank = int(round(p / 100.0 * (len(s) - 1))) - 1"),
    ("textstat/stats.py", "return s[rank]", "return s[0]"),
    ("textstat/stats.py", "if p >= 100:\n        return s[-1]", "if p >= 100:\n        return s[rank - 1]"),
    ("textstat/stats.py", "total = sum(d.values())", "total = max(d.values())"),
    ("textstat/stats.py", "total = sum(d.values())", "total = sum(d.keys())"),
    ("textstat/stats.py", "if total <= 0:", "if total < 0:"),
    ("textstat/stats.py", "return {k: v / total for k, v in d.items()}", "return {k: v * total for k, v in d.items()}"),
    ("textstat/stats.py", "lo = max(0, i - w + 1)", "lo = max(0, i - w)"),
    ("textstat/stats.py", "lo = max(0, i - w + 1)", "lo = max(0, i - 1)"),
    ("textstat/stats.py", "out.append(max(xs[lo:i + 1]))", "out.append(max(xs[lo:i]))"),
    ("textstat/stats.py", "out.append(max(xs[lo:i + 1]))", "out.append(min(xs[lo:i + 1]))"),
    ("textstat/stats.py", "out.append(max(xs[lo:i + 1]))", "out.append(max(xs[lo:i + 2]))"),
    ("textstat/stats.py", "if w <= 0:", "if w <= 0 or w == 1:"),
    ("textstat/stats.py", "for i in range(len(xs)):", "for i in range(len(xs) - 1):"),
    ("textstat/stats.py", "s = sorted(xs)\n    if not s:", "s = sorted(xs)\n    if s:"),
    ("textstat/stats.py", "raise ValueError(\"non-positive total\")", "pass"),
    ("textstat/seq.py", "return [xs[i:i + n] for i in range(0, len(xs), n)]", "return [xs[i:i + n] for i in range(0, len(xs), n - 1)]"),
    ("textstat/seq.py", "return [xs[i:i + n] for i in range(0, len(xs), n)]", "return [xs[i:i + n] for i in range(1, len(xs), n)]"),
    ("textstat/seq.py", "return [xs[i:i + n] for i in range(0, len(xs), n)]", "return [xs[i:i + n - 1] for i in range(0, len(xs), n)]"),
    ("textstat/seq.py", "if n <= 0:", "if n <= 0:\n        return []\n    if n < 0:"),
    ("textstat/seq.py", "acc += x\n        out.append(acc)", "acc -= x\n        out.append(acc)"),
    ("textstat/seq.py", "acc += x\n        out.append(acc)", "out.append(acc)"),
    ("textstat/seq.py", "out = []\n    acc = 0", "out = []\n    acc = 1"),
    ("textstat/seq.py", "for x in xs:\n        acc += x", "for x in xs:\n        acc += x * 2"),
    ("textstat/seq.py", "if x in seen:\n            continue", "if x not in seen:\n            continue"),
    ("textstat/seq.py", "seen.add(x)\n        out.append(x)", "seen.add(x)\n        out.insert(0, x)"),
    ("textstat/seq.py", "out.append(x)\n    return out", "out.append(x)\n    return out[::-1]"),
    ("textstat/seq.py", "for part in s.split(\",\"):", "for part in s.split(\";\"):"),
    ("textstat/seq.py", "out.extend(range(int(lo), int(hi) + 1))", "out.extend(range(int(lo), int(hi)))"),
    ("textstat/seq.py", "out.append(int(part))", "out.append(part)"),
    ("textstat/seq.py", "if \"-\" in part:", "if \"-\" not in part:"),
    ("textstat/stats.py", "return {k: v / total for k, v in d.items()}", "return {k: 1.0 / total for k, v in d.items()}"),
    ("textstat/seq.py", "counts = [0] * n", "counts = [0] * (n + 1)"),
    ("textstat/seq.py", "width = (hi - lo) / n", "width = (hi - lo) / (n - 1)"),
    ("textstat/seq.py", "if v < lo or v > hi:", "if v <= lo or v > hi:"),
    ("textstat/seq.py", "if v < lo or v > hi:", "if v < lo or v >= hi:"),
    ("textstat/seq.py", "idx = int((v - lo) / width)", "idx = int((v - lo) / width) + 1"),
    ("textstat/seq.py", "if idx >= n:\n            idx = n - 1", "if idx > n:\n            idx = n - 1"),
    ("textstat/seq.py", "counts[idx] += 1", "counts[idx] = 1"),
    ("textstat/seq.py", "if hi <= lo:", "if hi < lo:"),
    ("textstat/seq.py", "return counts", "return counts[:-1]"),
    ("textstat/stats.py", "if w <= 0:\n        raise ValueError(\"window must be positive\")", "if w <= 0:\n        w = 1"),
    ("textstat/text.py", "return re.findall(r\"[a-z0-9]+\", text.lower())", "return re.findall(r\"[a-z]+\", text.lower())"),
    ("textstat/text.py", "return re.findall(r\"[a-z0-9]+\", text.lower())", "return re.findall(r\"[a-z0-9]+\", text)"),
    ("textstat/text.py", "return re.findall(r\"[a-z0-9]+\", text.lower())", "return text.lower().split()"),
    ("textstat/text.py", "counts[w] = counts.get(w, 0) + 1", "counts[w] = counts.get(w, 0)"),
    ("textstat/text.py", "key=lambda kv: (-kv[1], kv[0])", "key=lambda kv: (-kv[1],)"),
    ("textstat/text.py", "key=lambda kv: (-kv[1], kv[0])", "key=lambda kv: (kv[1], kv[0])"),
    ("textstat/text.py", "return items[:k]", "return items[:k + 1]"),
    ("textstat/text.py", "return items[:k]", "return items[-k:]"),
    ("textstat/text.py", "if n < 5:", "if n < 0:"),
    ("textstat/text.py", "if len(s) <= n:", "if len(s) < n:"),
    ("textstat/text.py", "keep = n - 3", "keep = n"),
    ("textstat/text.py", "left = (keep + 1) // 2", "left = (keep + 2) // 2"),
    ("textstat/text.py", "right = keep - left", "right = keep"),
    ("textstat/text.py", "return s[:left] + \"...\" + (s[len(s) - right:] if right else \"\")", "return s[:left] + \"..\" + (s[len(s) - right:] if right else \"\")"),
    ("textstat/text.py", "return s[:left] + \"...\" + (s[len(s) - right:] if right else \"\")", "return s[:left] + \"...\" + s[:-right]"),
    ("textstat/text.py", "s = re.sub(r\"[^a-z0-9]+\", \"-\", s.lower())", "s = re.sub(r\"[^a-z0-9]+\", \"_\", s.lower())"),
    ("textstat/text.py", "return s.strip(\"-\")", "return s.strip(\"_\")"),
    ("textstat/text.py", "return s.strip(\"-\")", "return \"-\" + s"),
    ("textstat/text.py", "s = re.sub(r\"[^a-z0-9]+\", \"-\", s.lower())", "s = re.sub(r\"[^a-z0-9]+\", \"-\", s)"),
    ("textstat/__init__.py", "from .seq import bucket, chunk, cumsum, dedupe_stable, parse_ranges", "from .seq import chunk, cumsum, dedupe_stable, parse_ranges"),
    ("textstat/stats.py", "mid = n // 2", "mid = n // 2 - 1"),
    ("textstat/__init__.py", "from .stats import median, normalize_scores, percentile, rolling_max, running_mean", "from .stats import median, normalize_scores, rolling_max, running_mean"),
    ("textstat/__init__.py", "from .text import slugify, tokenize, truncate_middle, word_freq", "from .text import slugify, tokenize, word_freq"),
    ("textstat/stats.py", "out = []\n    for i, x in enumerate(xs):", "out = []\n    for i, x in enumerate(reversed(xs)):"),
    ("textstat/stats.py", "mid = n // 2", "mid = 0"),
    ("textstat/stats.py", "if n % 2 == 1:\n        return s[mid]", "if n % 2 == 1:\n        return s[-1]"),
    ("textstat/stats.py", "if p <= 0:\n        return s[0]", "if p <= 0:\n        return s[1]"),
    ("textstat/stats.py", "rank = int(round(p / 100.0 * (len(s) - 1)))\n    return s[rank]", "rank = int(p / 100.0 * (len(s) - 1))\n    return s[rank]"),
    ("textstat/stats.py", "return {k: v / total for k, v in d.items()}", "return {k: round(v / total, 2) for k, v in d.items()}"),
    ("textstat/stats.py", "lo = max(0, i - w + 1)", "lo = min(0, i - w + 1)"),
    ("textstat/stats.py", "out.append(max(xs[lo:i + 1]))", "out.append(sum(xs[lo:i + 1]))"),
    ("textstat/stats.py", "return (s[mid - 1] + s[mid]) / 2.0", "return (s[mid - 1] - s[mid]) / 2.0"),
    ("textstat/seq.py", "if n <= 0:\n        raise ValueError(\"n must be positive\")", "if n <= 0:\n        n = 1"),
    ("textstat/seq.py", "out = []\n    acc = 0", "out = []\n    acc = 0.5"),
    ("textstat/seq.py", "seen.add(x)\n        out.append(x)", "seen.discard(x)\n        out.append(x)"),
    ("textstat/seq.py", "out.extend(range(int(lo), int(hi) + 1))", "out.extend(range(int(lo) + 1, int(hi) + 1))"),
    ("textstat/seq.py", "for part in s.split(\",\"):", "for part in s.split(\",\")[:1]:"),
    ("textstat/seq.py", "if hi <= lo:\n        raise ValueError(\"hi must exceed lo\")", "if hi <= lo:\n        pass"),
    ("textstat/seq.py", "return counts", "return counts[::-1]"),
    ("textstat/seq.py", "width = (hi - lo) / n", "width = (hi - lo) / n * 2"),
    ("textstat/text.py", "return re.findall(r\"[a-z0-9]+\", text.lower())", "return re.findall(r\"[a-z0-9]+\", text.upper())"),
    ("textstat/text.py", "return items[:k]", "return items[1:k + 1]"),
    ("textstat/text.py", "keep = n - 3", "keep = n - 4"),
    ("textstat/text.py", "return s.strip(\"-\")", "return s.replace(\"-\", \"_\")"),
]


def repo_files() -> dict[str, str]:
    """返回 fixture 仓库的相对路径 → 文本内容。

    只收 UTF-8 文本：``__pycache__``、编辑器临时文件、以及 macOS/bsdtar 顺带产生的
    AppleDouble ``._*`` 资源叉（非 UTF-8）都要跳过，否则一个杂散文件会让整个 harness
    在 build_tasks 阶段直接崩掉。
    """
    out = {}
    skipped: list[str] = []
    for p in sorted(FIXTURE_REPO.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(FIXTURE_REPO))
        if p.name.startswith("._") or p.name.startswith("."):
            skipped.append(rel)
            continue
        try:
            out[rel] = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append(rel)
    if skipped:
        print(f"[repo_files] skipped non-text/hidden files: {skipped}", flush=True)
    return out


def write_mutant(dest: Path, mutation: tuple[str, str, str] | None) -> None:
    """把仓库写到 dest，并按需注入单点变异。"""
    files = repo_files()
    if mutation is not None:
        rel, old, new = mutation
        text = files[rel]
        if text.count(old) != 1:
            raise ValueError(f"mutation anchor not unique in {rel}: {old!r} x{text.count(old)}")
        files[rel] = text.replace(old, new)
    for rel, text in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def run_repo_tests(repo: Path, python: str | None = None) -> dict:
    """在 repo 根目录跑测试套件，返回退出码与输出尾部。"""
    py = python or sys.executable
    proc = subprocess.run(
        [py, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-14:]
    return {"returncode": proc.returncode, "tail": "\n".join(tail)}


# --------------------------------------------------------------------------------------
# 检索工具：BM25（纯 Python）
# --------------------------------------------------------------------------------------

class BM25Index:
    """对 NFCorpus 语料建 BM25 索引。"""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.ids = [d["_id"] for d in docs]
        self.titles = [d.get("title") or "" for d in docs]
        self.texts = [d["text"] for d in docs]
        self.k1 = k1
        self.b = b
        self.doc_tf: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        for text in self.texts:
            toks = re.findall(r"[a-z0-9]+", text.lower())
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(self.texts)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def search(self, query: str, k: int = 5) -> list[dict]:
        q_toks = re.findall(r"[a-z0-9]+", query.lower())
        scores = [0.0] * self.n
        for t in q_toks:
            n_t = self.df.get(t)
            if not n_t:
                continue
            idf = math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5))
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        order = sorted(range(self.n), key=lambda i: (-scores[i], self.ids[i]))
        out = []
        for i in order[: max(1, int(k))]:
            out.append(
                {
                    "doc_id": self.ids[i],
                    "score": round(scores[i], 6),
                    "title": self.titles[i],
                    "snippet": self.texts[i][:280],
                }
            )
        return out


_RETRIEVAL_CACHE: dict[str, object] = {}


def load_nfcorpus() -> dict:
    """加载本地 NFCorpus 快照（语料、查询、qrels）。"""
    if "data" in _RETRIEVAL_CACHE:
        return _RETRIEVAL_CACHE["data"]  # type: ignore[return-value]
    from datasets import load_dataset

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    corpus = load_dataset("parquet", data_files=NFCORPUS_CORPUS, split="train")
    queries = load_dataset("parquet", data_files=NFCORPUS_QUERIES, split="train")
    qrels: dict[str, dict[str, int]] = {}
    with open(NFCORPUS_QRELS, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    data = {
        "docs": [dict(d) for d in corpus],
        "queries": [dict(q) for q in queries],
        "qrels": qrels,
        "index": BM25Index([dict(d) for d in corpus]),
    }
    _RETRIEVAL_CACHE["data"] = data
    return data


# --------------------------------------------------------------------------------------
# 任务生成
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一个严谨的任务执行助手。你可以使用给定的工具。"
    "先调用工具获取事实，再给出结论；不要凭记忆猜数或猜文档编号。"
    "最后一条回复必须以 `FINAL: <答案>` 结尾，`<答案>` 只放最简形式的结果。"
)

COMPUTE_PROMPT = (
    "请用 calculate 工具计算下面表达式的精确整数值（必须调用工具，禁止心算）：\n{expr}\n"
    "拿到工具结果后，最后一行写 `FINAL: <整数>`。"
)

RETRIEVAL_PROMPT = (
    "请用 search_corpus 工具在 NFCorpus 医学文献库里检索以下问题，必要时可以换关键词多轮检索：\n"
    "“{query}”\n"
    "最后引用一个最相关的文档 id（形如 MED-1234），并写一行 `FINAL: <doc_id>`。"
)

CODEFIX_PROMPT = (
    "仓库里有三个模块和一套单元测试，其中一个函数被改动了一行（运算符、常数或边界），测试因此失败。"
    "请先用 list_files / read_file 读代码，用 run_tests 复现失败，再用 edit_file 把那一行改回去"
    "（edit_file 会替换文件里唯一匹配的 old_string；只改 textstat/ 下的源码，不要改 tests/）。"
    "修好后最后一行写 `FINAL: <被修复的函数名>`。"
)

CALC_TOOL = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算一个只含整数、括号与 + - * // % ** 的算术表达式的精确整数值",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "算术表达式，例如 (12+7)*3"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}]

SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "在 NFCorpus 医学文献语料上做 BM25 检索，返回最相关的文档 id、标题与摘要片段",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词"},
                "k": {"type": "integer", "description": "返回文档数，1-20，默认 5"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}]

CODEFIX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出仓库内的文件路径",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取仓库内一个文件的全部内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对仓库根目录的路径"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "覆盖写入 textstat/ 下的一个源码文件（不允许写 tests/）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "文件的新内容"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "把文件里唯一的 old_string 替换成 new_string（改一行用这个，不要整文件重写）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string", "description": "必须与文件内容逐字符匹配，且在文件中唯一"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "在仓库根目录运行单元测试并返回结果",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class ComputeEnv:
    task_class = "compute"
    tools = CALC_TOOL

    def __init__(self, sandbox: Path):
        self.sandbox = sandbox

    def call(self, name: str, args: dict) -> str:
        if name != "calculate":
            return f"ERROR: unknown tool {name}"
        try:
            return str(safe_eval(str(args.get("expression", ""))))
        except Exception as exc:  # noqa: BLE001 - 工具错误要原样回给模型
            return f"ERROR: {type(exc).__name__}: {exc}"

    def score(self, final_text: str) -> dict:
        m = re.findall(r"FINAL:\s*(-?\d+)", final_text or "")
        predicted = int(m[-1]) if m else None
        return {
            "score": 1.0 if predicted == self.truth else 0.0,
            "predicted": predicted,
            "expected": self.truth,
        }


class RetrievalEnv:
    task_class = "retrieval"
    tools = SEARCH_TOOL

    def __init__(self, sandbox: Path, task: dict):
        self.sandbox = sandbox
        self.gold = {k for k, v in task.get("gold", {}).items() if v > 0}
        self.primary = {k for k, v in task.get("gold", {}).items() if v >= 2}
        self.index: BM25Index = task["_index"]
        self.search_calls = 0

    def call(self, name: str, args: dict) -> str:
        if name != "search_corpus":
            return f"ERROR: unknown tool {name}"
        self.search_calls += 1
        k = int(args.get("k") or 5)
        k = max(1, min(20, k))
        hits = self.index.search(str(args.get("query", "")), k)
        return json.dumps(hits, ensure_ascii=False)

    def score(self, final_text: str) -> dict:
        cited = re.findall(r"MED-\d+", final_text or "")
        hit = next((c for c in cited if c in self.gold), None)
        primary = next((c for c in cited if c in self.primary), None)
        return {
            "score": 1.0 if hit else 0.0,
            "primary_hit": 1.0 if primary else 0.0,
            "cited": cited[:5],
            "gold_size": len(self.gold),
        }


class CodefixEnv:
    task_class = "codefix"
    tools = CODEFIX_TOOLS

    def __init__(self, sandbox: Path, task: dict):
        self.sandbox = sandbox
        self.function = task["function"]
        self.rejected_writes: list[str] = []
        self.writes = 0
        self.edits = 0
        self.reads = 0
        self.test_runs: list[int] = []

    def _resolve(self, rel: str) -> Path | None:
        p = (self.sandbox / rel).resolve()
        try:
            p.relative_to(self.sandbox.resolve())
        except ValueError:
            return None
        return p

    def call(self, name: str, args: dict) -> str:
        if name == "list_files":
            files = [
                str(p.relative_to(self.sandbox))
                for p in sorted(self.sandbox.rglob("*.py"))
                if "__pycache__" not in p.parts
            ]
            return json.dumps(files, ensure_ascii=False)
        if name == "read_file":
            path = self._resolve(str(args.get("path", "")))
            if path is None or not path.is_file():
                return f"ERROR: no such file {args.get('path')!r}"
            self.reads += 1
            return path.read_text(encoding="utf-8")
        if name == "write_file":
            rel = str(args.get("path", ""))
            if rel.startswith("tests/") or rel.startswith("tests\\"):
                self.rejected_writes.append(rel)
                return "ERROR: tests/ is read-only for this task"
            path = self._resolve(rel)
            if path is None or not rel.endswith(".py"):
                self.rejected_writes.append(rel)
                return f"ERROR: refused to write {rel!r}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("content", "")), encoding="utf-8")
            self.writes += 1
            return f"wrote {rel} ({len(str(args.get('content', '')))} bytes)"
        if name == "edit_file":
            rel = str(args.get("path", ""))
            if rel.startswith("tests/") or rel.startswith("tests\\"):
                self.rejected_writes.append(rel)
                return "ERROR: tests/ is read-only for this task"
            path = self._resolve(rel)
            if path is None or not path.is_file():
                return f"ERROR: no such file {rel!r}"
            text = path.read_text(encoding="utf-8")
            old = str(args.get("old_string", ""))
            new = str(args.get("new_string", ""))
            n = text.count(old)
            if n != 1:
                return f"ERROR: old_string matches {n} times in {rel}; it must match exactly once"
            path.write_text(text.replace(old, new), encoding="utf-8")
            self.edits += 1
            return f"edited {rel}: 1 replacement"
        if name == "run_tests":
            res = run_repo_tests(self.sandbox)
            self.test_runs.append(res["returncode"])
            return json.dumps({"returncode": res["returncode"], "tail": res["tail"]}, ensure_ascii=False)
        return f"ERROR: unknown tool {name}"

    def score(self, final_text: str) -> dict:
        res = run_repo_tests(self.sandbox)
        m = re.findall(r"FINAL:\s*([A-Za-z_][A-Za-z0-9_]*)", final_text or "")
        return {
            "score": 1.0 if res["returncode"] == 0 else 0.0,
            "named_function": m[-1] if m else None,
            "expected_function": self.function,
            "named_correct": int(bool(m) and m[-1] == self.function),
            "writes": self.writes,
            "edits": self.edits,
            "reads": self.reads,
            "rejected_writes": len(self.rejected_writes),
            "test_runs": self.test_runs,
            "final_returncode": res["returncode"],
        }


def build_tasks(task_class: str, n: int, seed: int = 0) -> list[dict]:
    """确定性生成 n 个任务；返回的 dict 可直接序列化（除检索任务的 ``_index``）。"""
    rng = random.Random(f"{task_class}-{seed}")
    tasks: list[dict] = []
    if task_class == "compute":
        seen = set()
        while len(tasks) < n:
            expr = _gen_expression(rng)
            if expr in seen:
                continue
            seen.add(expr)
            tasks.append(
                {
                    "task_id": f"compute-{len(tasks):03d}",
                    "task_class": "compute",
                    "prompt": COMPUTE_PROMPT.format(expr=expr),
                    "meta": {"expression": expr},
                    "ground_truth": safe_eval(expr),
                }
            )
    elif task_class == "retrieval":
        data = load_nfcorpus()
        qrels = data["qrels"]
        candidates = []
        for q in data["queries"]:
            rel = qrels.get(q["_id"], {})
            if not rel:
                continue
            key = hashlib.sha256(f"{seed}:{q['_id']}".encode()).hexdigest()
            candidates.append((key, q, rel))
        candidates.sort(key=lambda t: t[0])
        if len(candidates) < n:
            raise ValueError(f"only {len(candidates)} labelled queries available")
        for i, (_key, q, rel) in enumerate(candidates[:n]):
            tasks.append(
                {
                    "task_id": f"retrieval-{i:03d}",
                    "task_class": "retrieval",
                    "prompt": RETRIEVAL_PROMPT.format(query=q["text"]),
                    "meta": {"query_id": q["_id"], "query": q["text"]},
                    "gold": rel,
                }
            )
    elif task_class == "codefix":
        if len(MUTATIONS) < n:
            raise ValueError(f"only {len(MUTATIONS)} mutations defined")
        order = list(range(len(MUTATIONS)))
        random.Random(seed).shuffle(order)
        for i, idx in enumerate(order[:n]):
            rel, _old, _new = MUTATIONS[idx]
            fn = _mutated_function(rel, _old)
            tasks.append(
                {
                    "task_id": f"codefix-{i:03d}",
                    "task_class": "codefix",
                    "prompt": CODEFIX_PROMPT,
                    "meta": {
                        "mutation_index": idx,
                        "file": rel,
                        "function": fn,
                        "mutated_anchor": _old[:80],
                    },
                    "function": fn,
                }
            )
    else:
        raise ValueError(f"unknown task class {task_class}")
    return tasks


def _mutated_function(rel: str, anchor: str) -> str:
    """从变异锚点往上找最近的函数名，用于评分时核对 FINAL 里的函数名。"""
    text = repo_files()[rel]
    pos = text.index(anchor)
    head = text[:pos]
    names = re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)", head, flags=re.M)
    return names[-1] if names else "<module>"


def make_env(task: dict, sandbox: Path):
    """为任务构造工具环境与评分器。"""
    if task["task_class"] == "compute":
        env = ComputeEnv(sandbox)
        env.truth = task["ground_truth"]
        return env
    if task["task_class"] == "retrieval":
        task = dict(task)
        task["_index"] = load_nfcorpus()["index"]
        return RetrievalEnv(sandbox, task)
    if task["task_class"] == "codefix":
        mutation = MUTATIONS[task["meta"]["mutation_index"]]
        write_mutant(sandbox, mutation)
        return CodefixEnv(sandbox, task)
    raise ValueError(task["task_class"])


# --------------------------------------------------------------------------------------
# 自检：每个变异都必须被测试捕获，且基线必须通过
# --------------------------------------------------------------------------------------

def selftest(limit: int | None = None) -> dict:
    import contextlib

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "baseline"
        write_mutant(base, None)
        baseline = run_repo_tests(base)
        results = []
        n = len(MUTATIONS) if limit is None else min(limit, len(MUTATIONS))
        for i in range(n):
            d = Path(tmp) / f"m{i:03d}"
            write_mutant(d, MUTATIONS[i])
            res = run_repo_tests(d)
            results.append(
                {
                    "index": i,
                    "file": MUTATIONS[i][0],
                    "anchor": MUTATIONS[i][1][:60],
                    "caught": res["returncode"] != 0,
                }
            )
            shutil.rmtree(d, ignore_errors=True)
        caught = sum(1 for r in results if r["caught"])
        return {
            "baseline_returncode": baseline["returncode"],
            "mutations": len(results),
            "caught": caught,
            "uncaught": [r for r in results if not r["caught"]],
        }


def dump_tasks(path: str, n: int, seed: int) -> dict:
    out = {}
    for cls in TASK_CLASSES:
        tasks = build_tasks(cls, n, seed)
        out[cls] = [
            {k: v for k, v in t.items() if not k.startswith("_")} for t in tasks
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {cls: len(v) for cls, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="L9.1 任务/工具/评分")
    ap.add_argument("--selftest", action="store_true", help="校验变异都能被测试捕获")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump-tasks", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(args.limit), ensure_ascii=False, indent=1))
        return 0
    if args.dump_tasks:
        print(json.dumps(dump_tasks(args.dump_tasks, args.n, args.seed), ensure_ascii=False))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
