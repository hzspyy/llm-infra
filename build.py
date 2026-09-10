#!/usr/bin/env python3
"""Build the LLM/VLM AI-infra study site.

Markdown (src/) -> fully static HTML (site/). No JavaScript, no CDN, no network
at read time: math is rendered to MathML and code is highlighted by Pygments at
build time, so the output works offline from Finder, iCloud Drive and iPadOS.

Run:  /Volumes/data/venvs/llm-infra/bin/python build.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import markdown
from latex2mathml.converter import convert as latex_to_mathml
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SITE = ROOT / "site"
ASSETS_SRC = ROOT / "assets"

STATUS_LABEL = {"done": "有正文", "verified": "已核对", "draft": "有初稿",
                "evidence": "证据待补", "wip": "编写中", "planned": "待写"}


# --------------------------------------------------------------------------
# math / code protection
# --------------------------------------------------------------------------

def _protect(text: str, pattern: re.Pattern, store: list, tag: str) -> str:
    def repl(m):
        store.append(m.group(0))
        return f"zZ{tag}{len(store) - 1}Zz"
    return pattern.sub(repl, text)


FENCE_RE = re.compile(r"^```.*?^```", re.M | re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\$)")


ADMONITION_RE = re.compile(
    r"^::: *(note|tip|warning|danger|info|fold|fold\+)?[ \t]*(.*?)\n(.*?)^:::[ \t]*$",
    re.M | re.S)

KNOWN_KINDS = {"note", "tip", "warning", "danger", "info", "fold", "fold+"}


def convert_admonitions(md_text: str) -> str:
    """把 `::: warning 标题 … :::` 转成带 markdown="1" 的 div。

    不用 python-markdown 的 `!!! note` + 四空格缩进语法：
    缩进四格之后，块里的代码围栏 ``` 会被 fenced_code 当成缩进代码块而不再识别，
    整块内容都会散掉。改成 md_in_html 的 div，内容保持零缩进，围栏正常工作。

    `fold` / `fold+` 走 <details>：长源码、长原始输出默认收起，
    `fold+` 默认展开。details/summary 都在 python-markdown 的
    BLOCK_LEVEL_ELEMENTS 里，所以 md_in_html 的 markdown="1" 对它有效。
    """
    def repl(m):
        kind = m.group(1) or "note"
        title = (m.group(2) or "").strip()
        body = m.group(3).rstrip("\n")
        if kind.startswith("fold"):
            openness = " open" if kind == "fold+" else ""
            summary = html.escape(title or "展开")
            return (f'<details class="fold"{openness} markdown="1">\n'
                    f'<summary>{summary}</summary>\n\n'
                    + body + "\n\n</details>\n")
        head = (f'<p class="admonition-title">{html.escape(title)}</p>\n\n'
                if title else "")
        return (f'<div class="admonition {kind}" markdown="1">\n\n'
                + head + body + "\n\n</div>\n")

    # 无 kind 时第一个词可能就是标题的一部分，交给 KNOWN_KINDS 判定
    def repl_smart(m):
        if m.group(1) is None and m.group(2):
            first = m.group(2).split(maxsplit=1)
            if first and first[0] in KNOWN_KINDS:
                pass  # 正则已经处理
        return repl(m)

    return ADMONITION_RE.sub(repl_smart, md_text)


def render_math(md_text: str, src_store: list[str] | None = None
                ) -> tuple[str, list[str]]:
    """Replace $...$ / $$...$$ with placeholders, returning rendered MathML.

    `src_store` 若给出，则先展开 {{src:...}} 指令；它生成的是成品 HTML，
    必须像数学一样用占位符跳过 markdown 转换。
    """
    code_store: list[str] = []
    md_text = _protect(md_text, FENCE_RE, code_store, "CODE")
    if src_store is not None:
        md_text = expand_src_directives(md_text, src_store)
    md_text = convert_admonitions(md_text)      # 代码块已保护，不会误伤
    md_text = _protect(md_text, INLINE_CODE_RE, code_store, "CODE")

    math_store: list[str] = []

    def block(m):
        try:
            ml = latex_to_mathml(m.group(1).strip(), display="block")
        except Exception as exc:  # noqa: BLE001 - keep source visible on failure
            ml = f'<pre class="math-error">MATH ERROR: {html.escape(str(exc))}\n{html.escape(m.group(1))}</pre>'
        math_store.append(f'<div class="math-block">{ml}</div>')
        return f"\n\nzZMATH{len(math_store) - 1}Zz\n\n"

    def inline(m):
        try:
            ml = latex_to_mathml(m.group(1).strip(), display="inline")
        except Exception:  # noqa: BLE001
            ml = f"<code>{html.escape(m.group(1))}</code>"
        math_store.append(f'<span class="math-inline">{ml}</span>')
        return f"zZMATH{len(math_store) - 1}Zz"

    md_text = BLOCK_MATH_RE.sub(block, md_text)
    md_text = INLINE_MATH_RE.sub(inline, md_text)

    # restore code before markdown runs, so fences are processed normally.
    # 注意缩进：admonition 转换会给占位符加 4 空格前缀，还原时必须把
    # 代码块的每一行都同样缩进，否则围栏的首行缩进而内容不缩进，整块会散掉。
    for i, snippet in enumerate(code_store):
        token = f"zZCODE{i}Zz"
        while True:
            pos = md_text.find(token)
            if pos < 0:
                break
            line_start = md_text.rfind("\n", 0, pos) + 1
            indent = md_text[line_start:pos]
            if indent.strip():                     # 行内代码，原样替换
                md_text = md_text[:pos] + snippet + md_text[pos + len(token):]
                continue
            body = "\n".join((indent + ln) if ln.strip() else ln
                              for ln in snippet.splitlines())
            md_text = md_text[:line_start] + body + md_text[pos + len(token):]
    return md_text, math_store


# --------------------------------------------------------------------------
# front matter
# --------------------------------------------------------------------------

FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def split_front_matter(text: str) -> tuple[dict, str]:
    m = FM_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------

ASSET_VER = "0"          # main() 里按 CSS 内容哈希填入，避免浏览器缓存旧样式


def page_shell(*, title: str, depth: int, body: str, sidebar: str = "",
               head_extra: str = "", main_class: str = "page") -> str:
    up = "../" * depth
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{up}assets/style.css?v={ASSET_VER}">
<link rel="stylesheet" href="{up}assets/code.css?v={ASSET_VER}">
{head_extra}
</head>
<body>
{sidebar}
<main class="{main_class}">
{body}
</main>
</body>
</html>
"""


def build_toc_sidebar(toc_html: str, up: str, mod_id: str, mod_title: str) -> str:
    if not toc_html.strip():
        inner = ""
    else:
        inner = toc_html
    return f"""<aside class="sidebar">
  <a class="sidebar-home" href="{up}index.html">&#8592; 全景地图</a>
  <div class="sidebar-id">{html.escape(mod_id)}</div>
  <div class="sidebar-title">{html.escape(mod_title)}</div>
  <nav class="toc">{inner}</nav>
</aside>"""


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def make_md() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=[
            "extra",           # tables, fenced_code, footnotes, attr_list, def_list, md_in_html
            "codehilite",
            "toc",
            "admonition",
            "sane_lists",
            "smarty",
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "linenums": False,
                           "css_class": "highlight"},
            "toc": {"permalink": False, "toc_depth": "2-3"},
            "smarty": {"smart_quotes": False},
        },
    )


def flat_modules(outline: dict) -> list[dict]:
    out = []
    for layer in outline["layers"]:
        for mod in layer["modules"]:
            m = dict(mod)
            m["layer_id"] = layer["id"]
            m["layer_name"] = layer["name"]
            out.append(m)
    return out


def module_path(mod: dict) -> Path:
    return SRC / mod["layer_id"] / f"{mod['id']}-{mod['slug']}.md"


def module_url(mod: dict) -> str:
    return f"{mod['layer_id']}/{mod['id']}-{mod['slug']}.html"


# --------------------------------------------------------------------------
# 模块间交叉引用
# --------------------------------------------------------------------------

# 正文里写 `L3.2` / `M2` / `4.0` 时自动变成链接。
# 只匹配这几种形态，避免误伤 "L2 cache"、"FP4"、"sm_120" 之类。
XREF_RE = re.compile(r"(?<![\w.])(L\d\.\d[a-z]?|M[123])(?![\w.])")

_XREF_INDEX: dict[str, dict] = {}


def build_xref_index(mods: list[dict]) -> None:
    """把 `L5.2` 这种写法映射到模块。模块 id 本身是 `5.2`，层是 `L5`。"""
    _XREF_INDEX.clear()
    for m in mods:
        key = m["id"] if m["layer_id"] == "M" else f"{m['layer_id'][0]}{m['id']}"
        _XREF_INDEX[key] = m


def link_xrefs(html_text: str, *, depth: int) -> str:
    """在**已生成的 HTML** 上做替换，并且跳过 <code>/<pre>/<a> 内部。

    在 HTML 阶段做而不是 markdown 阶段，是为了能精确避开代码块——
    源码走读里满是 `v1/core/...` 这类路径，误伤会很难看。
    """
    up = "../" * depth
    # 把 <pre>...</pre>、<code>...</code>、<a ...>...</a> 整段挖出来保护
    holes: list[str] = []

    def stash(m):
        holes.append(m.group(0))
        return f"\x00H{len(holes) - 1}\x00"

    protected = re.sub(r"<pre.*?</pre>|<code.*?</code>|<a\b.*?</a>|<svg.*?</svg>",
                       stash, html_text, flags=re.S)

    def repl(m):
        key = m.group(1)
        mod = _XREF_INDEX.get(key)
        if not mod:
            return m.group(0)
        title = html.escape(mod["title"])
        if mod.get("_built"):
            return (f'<a class="xref" href="{up}{module_url(mod)}" '
                    f'title="{title}">{key}</a>')
        return f'<span class="xref pending" title="{title}（尚未写）">{key}</span>'

    linked = XREF_RE.sub(repl, protected)
    for i, hole in enumerate(holes):
        linked = linked.replace(f"\x00H{i}\x00", hole)
    return linked


def build_module(mod: dict, prev_mod, next_mod, md: markdown.Markdown) -> bool:
    src_path = module_path(mod)
    if not src_path.exists():
        return False

    raw = src_path.read_text(encoding="utf-8")
    meta, body_md = split_front_matter(raw)
    src_store: list[str] = []
    body_md, math_store = render_math(body_md, src_store)

    md.reset()
    body_html = md.convert(body_md)
    toc_html = getattr(md, "toc", "")

    for i, ml in enumerate(math_store):
        body_html = body_html.replace(f"zZMATH{i}Zz", ml)
        toc_html = toc_html.replace(f"zZMATH{i}Zz", "")

    for i, blk in enumerate(src_store):
        body_html = body_html.replace(f"<p>zZSRC{i}Zz</p>", blk)
        body_html = body_html.replace(f"zZSRC{i}Zz", blk)
        toc_html = toc_html.replace(f"zZSRC{i}Zz", "")

    body_html = link_xrefs(body_html, depth=1)

    meta_bits = []
    if meta.get("machine"):
        meta_bits.append(f'<span class="chip">实测机器 {html.escape(meta["machine"])}</span>')
    if meta.get("measured"):
        meta_bits.append(f'<span class="chip">数据日期 {html.escape(meta["measured"])}</span>')
    if meta.get("deps"):
        meta_bits.append(f'<span class="chip">前置 {html.escape(meta["deps"])}</span>')
    meta_html = f'<div class="chips">{"".join(meta_bits)}</div>' if meta_bits else ""

    nav = []
    if prev_mod and prev_mod.get("_built"):
        nav.append(f'<a class="nav-prev" href="../{module_url(prev_mod)}">&#8592; {html.escape(prev_mod["id"])} {html.escape(prev_mod["title"])}</a>')
    else:
        nav.append("<span></span>")
    if next_mod and next_mod.get("_built"):
        nav.append(f'<a class="nav-next" href="../{module_url(next_mod)}">{html.escape(next_mod["id"])} {html.escape(next_mod["title"])} &#8594;</a>')
    else:
        nav.append("<span></span>")
    nav_html = f'<nav class="pagenav">{"".join(nav)}</nav>'

    header = f"""<header class="page-head">
  <div class="crumb"><a href="../index.html">全景地图</a> <span>&#8250;</span> {html.escape(mod["layer_id"])} · {html.escape(mod["layer_name"])}</div>
  <div class="mod-id">{html.escape(mod["id"])}</div>
  <h1>{html.escape(mod["title"])}</h1>
  <p class="lede">{html.escape(mod.get("brief", ""))}</p>
  {meta_html}
</header>"""

    body = header + f'<article class="prose">{body_html}</article>' + nav_html
    sidebar = build_toc_sidebar(toc_html, "../", mod["id"], mod["title"])
    page = page_shell(title=f'{mod["id"]} {mod["title"]}', depth=1,
                      body=body, sidebar=sidebar)

    out = SITE / module_url(mod)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return True


HARDWARE_TABLE = """
<table class="hw">
<thead><tr><th>机器</th><th>GPU</th><th>架构</th><th>本站用它讲什么</th></tr></thead>
<tbody>
<tr><td><code>crater</code></td><td>1&times; RTX 5090 D 32GB</td><td>sm_120 Blackwell 消费级</td><td>FP4/FP8 tensor core；<b>无 tcgen05 / TMEM</b>，走 <code>mma.sync</code> 路线</td></tr>
<tr><td><code>crater2</code></td><td>1&times; RTX 5090 32GB</td><td>sm_120</td><td>第二台单卡：A/B 对照与跨节点实验</td></tr>
<tr><td><code>worldvln</code></td><td><b>5&times; L40S 48GB</b></td><td>sm_89 Ada</td><td><b>无 NVLink，纯 PCIe + 跨 NUMA</b>：通信代价看得见；192 核 / 188GB RAM / RoCE</td></tr>
<tr><td><code>spark</code></td><td>GB10 (DGX Spark)</td><td>Blackwell + Grace</td><td>统一内存 / NVLink-C2C / aarch64：弱算力大内存的对照组</td></tr>
<tr><td><code>jetson-64</code></td><td>Jetson</td><td>边缘</td><td>端侧推理与功耗约束</td></tr>
</tbody></table>
"""



# --------------------------------------------------------------------------
# 源码浏览：每个 lab 文件生成一个可逐行阅读、可深链的页面
# --------------------------------------------------------------------------

LABS = ROOT / "labs"
CODE_DIR = SITE / "code"

LEXER_BY_SUFFIX = {".py": "python", ".cu": "cuda", ".cuh": "cuda",
                   ".sh": "bash", ".c": "c", ".cpp": "cpp", ".h": "c",
                   ".json": "json", ".md": "markdown",
                   # 原始工件也要能建源码页，否则引用它们的章节全是断链
                   ".txt": "text", ".log": "text", ".tsv": "text",
                   ".jsonl": "json", ".ptx": "text", ".sass": "text"}

# 每种语言里「值得进大纲」的东西。第 1 组是符号名。
OUTLINE_PAT = {
    "python": re.compile(r"^(?:class|def|    def|async def)\s+(\w+)|^([A-Z_][A-Z0-9_]{2,})\s*="),
    "cuda":   re.compile(r"^\s*(?:__global__|__device__|template\s*<[^>]*>\s*__global__)"
                         r"[\w\s:*&<>,]*?\b(\w+)\s*\(|^\s*(?:struct|class)\s+(\w+)"
                         r"|^\s*(?:static\s+)?[\w:<>*&\s]+\b(main)\s*\("),
    "bash":   re.compile(r"^(\w+)\s*\(\)\s*\{|^#\s*-+\n?|^#\s*(\d+\..*)$"),
}


def source_rel_paths() -> list[Path]:
    """所有需要生成源码页的文件。

    = labs/ 下的全部源码 + 正文用 {{src:}}/{{srcfold:}} 引用到的任何文件。
    第二部分是必需的：节选块底下那条「在完整文件中打开」指向源码页，
    如果只给 labs/ 建页，引用 results/ 里原始工件的章节就会全是断链。
    """
    out: list[Path] = []
    seen: set[Path] = set()

    def add(rel: Path) -> None:
        if rel not in seen and (ROOT / rel).is_file():
            seen.add(rel)
            out.append(rel)

    if LABS.exists():
        for f in sorted(LABS.rglob("*")):
            if f.is_file() and f.suffix in LEXER_BY_SUFFIX and "__pycache__" not in f.parts:
                add(f.relative_to(ROOT))

    if SRC.exists():
        for md in sorted(SRC.rglob("*.md")):
            if "_pre-merge" in md.parts:            # 合并前的存档，不建页
                continue
            for m in SRC_DIRECTIVE.finditer(md.read_text(encoding="utf-8")):
                add(Path(m.group(2)))
    return out


def source_url(rel: Path, depth: int) -> str:
    return "../" * depth + "code/" + "/".join(rel.parts) + ".html"


def _outline_rows(text: str, lang: str) -> list[tuple[int, str, int]]:
    """(行号, 符号名, 缩进层级)。用于页内目录。"""
    pat = OUTLINE_PAT.get(lang)
    if pat is None:
        return []
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        m = pat.match(line)
        if not m:
            continue
        name = next((g for g in m.groups() if g), None)
        if not name:
            continue
        indent = 1 if line.startswith(("    def", "    class")) else 0
        rows.append((i, name, indent))
    return rows


def render_source_page(rel: Path) -> tuple[str, int]:
    """返回 (文件标题, 行数)，并写出 site/code/<rel>.html。"""
    abs_path = ROOT / rel
    text = abs_path.read_text(encoding="utf-8", errors="replace")
    lang = LEXER_BY_SUFFIX.get(abs_path.suffix, "text")
    n_lines = len(text.splitlines())
    slug = "-".join(rel.parts).replace(".", "-")

    try:
        lexer = get_lexer_by_name(lang)
    except Exception:                                        # noqa: BLE001
        lexer = get_lexer_by_name("text")
    fmt = HtmlFormatter(cssclass="highlight srcview", linenos="table",
                        lineanchors=f"{slug}-L", anchorlinenos=True,
                        linenostart=1)
    code_html = highlight(text, lexer, fmt)

    rows = _outline_rows(text, lang)
    if rows:
        items = "".join(
            f'<li class="lvl{ind}"><a href="#{slug}-L-{ln}">'
            f'<span class="ol-name">{html.escape(name)}</span>'
            f'<span class="ol-line">{ln}</span></a></li>'
            for ln, name, ind in rows)
        outline_html = f'<nav class="src-outline"><h2>文件结构</h2><ul>{items}</ul></nav>'
    else:
        outline_html = ""

    depth = 1 + len(rel.parts) - 1          # site/code/<parts...>.html
    up = "../" * depth
    doc = text.split('"""')
    blurb = ""
    if abs_path.suffix == ".py" and len(doc) > 1:
        blurb = doc[1].strip().splitlines()[0] if doc[1].strip() else ""
    elif abs_path.suffix in (".cu", ".cuh", ".sh"):
        for ln in text.splitlines():
            if ln.startswith(("//", "#")) and len(ln) > 3 and "!/" not in ln:
                blurb = ln.lstrip("/# ").strip()
                break

    crumb = " / ".join(html.escape(pp) for pp in rel.parts)
    body = f"""<div class="src-head">
  <a class="src-back" href="{up}code/index.html">&#8592; 全部源码</a>
  <h1><code>{crumb}</code></h1>
  {f'<p class="src-blurb">{html.escape(blurb)}</p>' if blurb else ''}
  <p class="src-meta">{n_lines} 行 &middot; {html.escape(lang)} &middot;
     点击左侧行号可得到该行的固定链接</p>
</div>
{outline_html}
<div class="src-body">{code_html}</div>
"""
    sidebar = f"""<aside class="sidebar">
  <a class="sidebar-home" href="{up}index.html">&#8592; 全景地图</a>
  <div class="sidebar-id">源码</div>
  <div class="sidebar-title">{html.escape(rel.name)}</div>
  <nav class="toc">{outline_html or ''}</nav>
</aside>"""
    out_path = CODE_DIR / Path(*rel.parts).with_suffix(rel.suffix + ".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        page_shell(title=f"{rel.name} · 源码", depth=depth, body=body,
                   sidebar=sidebar, main_class="page page-wide"),
        encoding="utf-8")
    return crumb, n_lines


def build_source_pages() -> list[tuple[Path, int]]:
    rels = source_rel_paths()
    built = []
    for rel in rels:
        _, n = render_source_page(rel)
        built.append((rel, n))

    groups: dict[str, list[tuple[Path, int]]] = {}
    for rel, n in built:
        groups.setdefault(rel.parts[1] if len(rel.parts) > 1 else "labs", []).append((rel, n))

    secs = []
    for g in sorted(groups):
        rows = "".join(
            f'<li><a href="{"/".join(rel.parts)}.html"><code>{html.escape(rel.name)}</code></a>'
            f'<span class="src-lines">{n} 行</span></li>'
            for rel, n in sorted(groups[g]))
        secs.append(f'<section class="src-group"><h2>{html.escape(g)}</h2>'
                    f'<ul class="src-list">{rows}</ul></section>')
    total = sum(n for _, n in built)
    body = f"""<header class="hero">
  <h1>全部源码</h1>
  <p class="lede">{len(built)} 个文件，{total:,} 行。每个文件都可以逐行阅读，
  行号是固定链接；正文里的节选都能跳到这里的对应行。</p>
</header>
{''.join(secs)}
<p class="src-foot"><a href="../index.html">&#8592; 回到全景地图</a></p>"""
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    (CODE_DIR / "index.html").write_text(
        page_shell(title="全部源码", depth=1, body=body,
                   main_class="page page-wide"), encoding="utf-8")
    return built


SRC_DIRECTIVE = re.compile(
    r"^\{\{(src|srcfold):([^:}]+)(?::(\d+)-(\d+))?\}\}[ \t]*$", re.M)

# 超过这么多行的整文件展开自动折叠，避免正文被源码淹没
AUTOFOLD_LINES = 40


def expand_src_directives(md_text: str, store: list[str]) -> str:
    """把 {{src:labs/L0/tiny_lm.py}} 或 {{src:...:56-64}} 展开成真实代码。

    节选直接从磁盘上的文件读取，所以正文永远不会和源码漂移；
    每段都带一条指向完整文件对应行的链接。

    `{{srcfold:...}}` 强制折叠；整文件展开若超过 AUTOFOLD_LINES 行也自动折叠。
    折叠壳是原生 <details>，零 JS，离线与打印都正常。
    """
    def repl(m: re.Match) -> str:
        fold = m.group(1) == "srcfold"
        rel = Path(m.group(2))
        abs_path = ROOT / rel
        if not abs_path.exists():
            store.append(f'<div class="srcref"><code>{html.escape(str(rel))}</code>'
                         f'<span>（文件不存在）</span></div>')
            return f"zZSRC{len(store) - 1}Zz"
        lang = LEXER_BY_SUFFIX.get(abs_path.suffix, "text")
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        slug = "-".join(rel.parts).replace(".", "-")
        url = source_url(rel, 1)

        if m.group(3):
            a, b = int(m.group(3)), int(m.group(4))
            snippet = "\n".join(lines[a - 1:b])
            anchor = f"{url}#{slug}-L-{a}"
            label = f"{rel} 第 {a}–{b} 行"
            n_shown = b - a + 1
            fmt = HtmlFormatter(cssclass="highlight", linenos="inline",
                                linenostart=a)
        else:
            snippet = "\n".join(lines)
            anchor = url
            label = f"{rel} 全文（{len(lines)} 行）"
            n_shown = len(lines)
            fold = fold or n_shown > AUTOFOLD_LINES
            fmt = HtmlFormatter(cssclass="highlight", linenos="inline", linenostart=1)

        try:
            lexer = get_lexer_by_name(lang)
        except Exception:                                    # noqa: BLE001
            lexer = get_lexer_by_name("text")
        code = highlight(snippet, lexer, fmt)
        bar = (f'<div class="srcref"><code>{html.escape(label)}</code>'
               f'<a href="{anchor}">在完整文件中打开 &rarr;</a></div>')
        block = f'<div class="srcblock">{code}{bar}</div>'
        if fold:
            block = (f'<details class="fold srcfold">'
                     f'<summary>{html.escape(label)}</summary>'
                     f'{block}</details>')
        store.append(block)
        return f"zZSRC{len(store) - 1}Zz"

    return SRC_DIRECTIVE.sub(repl, md_text)

def build_index(outline: dict, mods: list[dict]) -> None:
    written = sum(1 for m in mods if module_path(m).exists())
    total = len(mods)

    def md_bold(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s))

    sections = []
    for layer in outline["layers"]:
        rows = []
        for raw_mod in layer["modules"]:
            mod = {**raw_mod, "layer_id": layer["id"]}
            st = mod.get("status", "planned")
            exists = module_path(mod).exists()
            if exists:
                link = f'<a href="{module_url(mod)}">{html.escape(mod["title"])}</a>'
                badge = '<span class="badge done">有正文</span>'
            else:
                link = f'<span class="pending">{html.escape(mod["title"])}</span>'
                badge = f'<span class="badge {st}">{STATUS_LABEL.get(st, st)}</span>'
            rows.append(
                f'<li><span class="mid">{html.escape(mod["id"])}</span>'
                f'<div class="mbody"><div class="mtitle">{link} {badge}</div>'
                f'<div class="mbrief">{md_bold(mod.get("brief", ""))}</div></div></li>'
            )
        sections.append(f"""<section class="layer" id="{layer['id']}">
  <div class="layer-head">
    <span class="layer-id">{html.escape(layer['id'])}</span>
    <h2>{html.escape(layer['name'])}<span class="layer-tag">{html.escape(layer['tagline'])}</span></h2>
  </div>
  <p class="layer-why">{html.escape(layer['why'])}</p>
  <ol class="modlist">{''.join(rows)}</ol>
</section>""")

    def md_bold(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s))

    principles = "".join(f"<li>{md_bold(p)}</li>" for p in outline.get("principles", []))

    body = f"""<header class="hero">
  <h1>{html.escape(outline['title'])}</h1>
  <p class="lede">{html.escape(outline['subtitle'])}</p>
  <div class="progress"><span>{written} / {total} 模块有正文（不代表验收完成）</span>
  <div class="bar"><i style="width:{(written / total * 100):.1f}%"></i></div></div>
</header>

<section class="intro">
  <h2>写作与证据要求</h2>
  <ol class="principles">{principles}</ol>
</section>

<section class="intro">
  <h2>怎么读这套材料</h2>
  <p>顺序不是纯自底向上，也不是纯自顶向下。<b>L0 先建立一本资源账本</b>——一次请求的时间、字节、
  FLOP 分别去了哪；它是后面每一章的挂载点。然后 <b>L1–L5 自底向上打穿</b>，每一层都在解决下一层
  暴露出的约束。最后 <b>L6–L8 横向展开</b>，把单机的结论放大到多卡、训练与集群。</p>
  <p>每个模块固定十段：<b>本章回答的三个问题 &rarr; 心智模型 &rarr; 机制拆解 &rarr;
  工程实现·源码走读 &rarr; 工程实现·自己写一遍 &rarr; 动手 lab（含实测数字）&rarr;
  原始现场 &rarr; 前沿 &rarr; 陷阱 &rarr; 自测题</b>。</p>
  <p class="rule"><b>数字诚实性</b>：站内每个数字都标注机器、日期与产生它的命令，原始输出留在
  <code>results/</code>。没有实测的数字一律显式标注 <span class="unverified">未实测</span>，
  不会把别处 blog 的加速比当作已验证事实呈现。</p>
  <h3>快车道</h3>
  <p>时间紧时可以只走推理主干：<b>L0 &rarr; L5 &rarr; L6 &rarr; L8</b>，底层（L1–L3）与训练侧（L7）后补。
  每章开头会标出真正的前置依赖，跳读不会断层。</p>
</section>

<section class="intro">
  <h2>实验硬件底座</h2>
  <p>同一份代码在四种硬件上跑出不同数字，才能讲清楚「瓶颈来自哪个硬件参数」。</p>
  {HARDWARE_TABLE}
</section>

{''.join(sections)}

<footer class="site-foot">构建于 {date.today().isoformat()} · 纯静态、无 JavaScript、无外部资源，可离线阅读</footer>
"""
    page = page_shell(title=outline["title"], depth=0, body=body)
    (SITE / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    outline = json.loads((ROOT / "outline.json").read_text(encoding="utf-8"))
    mods = flat_modules(outline)

    # 先扫一遍谁有源文件，交叉引用才知道该链过去还是标成「尚未写」
    for m in mods:
        m["_built"] = module_path(m).exists()
    build_xref_index(mods)

    SITE.mkdir(exist_ok=True)
    (SITE / "assets").mkdir(exist_ok=True)
    shutil.copy2(ASSETS_SRC / "style.css", SITE / "assets" / "style.css")
    global ASSET_VER
    ASSET_VER = hashlib.sha1(
        (ASSETS_SRC / "style.css").read_bytes()).hexdigest()[:8]

    # Pygments stylesheet: light + dark via prefers-color-scheme
    light = HtmlFormatter(style="friendly", cssclass="highlight").get_style_defs(".highlight")
    dark = HtmlFormatter(style="monokai", cssclass="highlight").get_style_defs(".highlight")
    dark = "\n".join("  " + ln for ln in dark.splitlines())
    (SITE / "assets" / "code.css").write_text(
        f"{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n", encoding="utf-8")

    md = make_md()
    built = 0
    for i, mod in enumerate(mods):
        # 上下页跳到**最近的已写章节**，不指向还没写的模块
        prev_mod = next((m for m in reversed(mods[:i]) if m.get("_built")), None)
        next_mod = next((m for m in mods[i + 1:] if m.get("_built")), None)
        if build_module(mod, prev_mod, next_mod, md):
            built += 1

    srcs = build_source_pages()
    build_index(outline, mods)
    print(f"built {built} module pages + index + {len(srcs)} source pages -> {SITE}")

    # ---- 孤儿检查 ----
    # 写了正文但文件名和 outline 的 slug 对不上 => 这一章根本不会被构建，
    # 而且**不报错**（2026-09-10 就这样丢过一次 3.2）。这里显式查出来。
    expected = {module_path(m).resolve() for m in mods}
    on_disk = {f.resolve() for f in SRC.rglob("*.md")}
    orphans = sorted(on_disk - expected)
    if orphans:
        print("\n⚠ 有 Markdown 没有对应的 outline 模块（不会被构建）：")
        for f in orphans:
            print(f"    {f.relative_to(ROOT)}")
        print("  检查 outline.json 里该模块的 slug 与文件名是否一致。")
    missing = sorted(m["id"] for m in mods
                     if m.get("status") == "done" and not module_path(m).exists())
    if missing:
        print("\n⚠ outline 标了 done 但文件不存在：" + ", ".join(missing))

    # ---- slug 冲突检查 ----
    # 两个模块用同一个 slug => 它们映射到同一个文件，后写的静默覆盖前一个。
    # 和孤儿检查是同一类问题（2026-09-11 发现 7.5/7.6 都叫 rl-infra）。
    by_path: dict[Path, list[str]] = {}
    for m in mods:
        by_path.setdefault(module_path(m).resolve(), []).append(m["id"])
    dups = {p: ids for p, ids in by_path.items() if len(ids) > 1}
    if dups:
        print("\n⚠ 多个模块映射到同一个文件（会互相覆盖）：")
        for p, ids in sorted(dups.items()):
            print(f"    {p.relative_to(ROOT)}  <-  {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
