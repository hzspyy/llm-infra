#!/usr/bin/env python3
"""Browser checks for the L5.12 delivery; use a dedicated browser session."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pooling-review-20260912-execution-final"
OUT.mkdir(exist_ok=False)
NODE = "/Users/nyxri/.workbuddy-ai/binaries/node/versions/22.22.2-2/bin/node"
CLI = "/Users/nyxri/.workbuddy-ai/binaries/node/workspace/node_modules/agent-browser/bin/agent-browser.js"
BASE = "http://127.0.0.1:8742"
env = dict(os.environ, PATH=str(Path(NODE).parent) + ":" + os.environ["PATH"])
records = []
def call(*args):
    run = subprocess.run([NODE, CLI, "--session", "l512-execution", *args], env=env,
                         capture_output=True, text=True, timeout=40)
    records.append({"args": args, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
    if run.returncode:
        raise RuntimeError(run.stderr or run.stdout)
    return run.stdout


def decoded(value):
    obj = json.loads(value)
    return json.loads(obj) if isinstance(obj, str) else obj


def check_page():
    result = decoded(call("eval", CHECK))
    assert result["scrollWidth"] <= result["width"], result
    assert not result["overflow"] and not result["externalResources"], result
    assert result["mathErrors"] == result["svgParagraphs"] == 0, result
    if "5.12" in result["title"]:
        assert result["svg"] == 2, result



CHECK = """JSON.stringify({title:document.title, width:innerWidth, scrollWidth:document.documentElement.scrollWidth,
math:document.querySelectorAll('math').length, svg:document.querySelectorAll('article svg').length,
svgChildren:[...document.querySelectorAll('article svg')].map(s=>({shapes:s.querySelectorAll('rect,path,polyline,circle').length,text:s.querySelectorAll('text').length})),
svgParagraphs:document.querySelectorAll('svg p').length,
folds:document.querySelectorAll('details.fold').length, mathErrors:document.querySelectorAll('.math-error').length,
externalResources:performance.getEntriesByType('resource').filter(e=>!e.name.startsWith(location.origin)).map(e=>e.name),
resources:performance.getEntriesByType('resource').map(e=>e.name),
overflow:[...document.querySelectorAll('article > p,article > table,article > svg,article > .math-block')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).map(e=>({tag:e.tagName,text:e.textContent.slice(0,80),right:e.getBoundingClientRect().right})),
background:getComputedStyle(document.body).backgroundColor,color:getComputedStyle(document.body).color})"""
try:
    call("open", BASE + "/index.html")
    call("snapshot", "-i")
    check_page()
    call("open", BASE + "/L5/5.12-non-generative-serving.html")
    call("snapshot", "-i")
    for theme in ["light", "dark"]:
        call("set", "media", theme)
        for width,height in [(1280,900),(834,1112)]:
            call("set", "viewport", str(width), str(height))
            check_page()
            call("eval", "document.querySelector('article svg').scrollIntoView({block:'center'}); true")
            call("screenshot", str(OUT / f"mind-{theme}-{width}.png"))
            call("eval", "document.querySelectorAll('article svg')[1].scrollIntoView({block:'center'}); true")
            call("screenshot", str(OUT / f"curve-{theme}-{width}.png"))
    call("eval", "document.querySelectorAll('details.fold').forEach(e=>e.open=true); true")
    check_page()
    call("eval", "JSON.stringify({openFolds:document.querySelectorAll('details.fold[open]').length,highlight:document.querySelectorAll('.highlight').length})")
    call("eval", "location.href=document.querySelector('a[href*=\"pooling_runner.py.html\"]').href; true")
    anchor = decoded(call("eval", "JSON.stringify({hash:location.hash,anchorExists:!!document.getElementById(decodeURIComponent(location.hash.slice(1))),sourceFolds:document.querySelectorAll('details').length})"))
    assert anchor["anchorExists"] and anchor["sourceFolds"] == 0, anchor
    call("open", BASE + "/L5/5.12-non-generative-serving.html")
    links = decoded(call("eval", "Promise.all([...document.querySelectorAll('a[href*=\"/raw/\"]')].map(async a=>{let r=await fetch(a.href,{method:'HEAD'});return {url:a.href,status:r.status}})).then(x=>JSON.stringify(x))"))
    assert links and all(x["status"] == 200 for x in links), links

finally:
    try:
        call("close")
    finally:
        (OUT / "browser-checks.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(str(OUT / "browser-checks.json"))
