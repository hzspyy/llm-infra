import re, pathlib, sys
site = pathlib.Path("site")
pages = list(site.rglob("*.html"))
bad = []
for p in pages:
    t = p.read_text(encoding="utf-8")
    for m in re.finditer(r'(?:href|src)="([^"#][^"]*?)"', t):
        h = m.group(1)
        if h.startswith(("http", "mailto", "data:")): continue
        path = h.split("#")[0].split("?")[0]
        if not path: continue
        if not (p.parent / path).resolve().exists():
            bad.append((str(p.relative_to(site)), h))
print(f"{len(pages)} 个页面, 断链 {len(bad)}")
for b in bad[:20]: print("  ", b)
