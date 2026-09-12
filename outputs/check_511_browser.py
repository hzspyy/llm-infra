from pathlib import Path
import os, subprocess, json

ROOT = Path(__file__).resolve().parents[1]
BIN = '/Users/nyxri/.workbuddy-ai/binaries/node/workspace/node_modules/.bin/agent-browser'
OUT = ROOT / 'outputs' / 'api-layer-review-20260912'
if not OUT.exists():          # 文件系统代理对 EEXIST 会直接抛错，先判存在
    OUT.mkdir(parents=True)
env = dict(os.environ, PATH='/Users/nyxri/.workbuddy-ai/binaries/node/versions/22.22.2-2/bin:' + os.environ['PATH'])
rows = []


def run(*args):
    p = subprocess.run([BIN, '--session', 'llm-511', '--json', *args],
                       env=env, text=True, capture_output=True, timeout=90)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    d = json.loads(p.stdout)
    if not d.get('success'):
        raise RuntimeError(d)
    return d.get('data', {})


def inspect():
    return run('eval', '''(() => {
      const root=document.documentElement;
      const folds=[...document.querySelectorAll('article details')];
      const sample=folds.find(x=>x.querySelector(':scope > summary'));
      let toggles=null;
      if(sample){ const old=sample.open; sample.querySelector('summary').click(); toggles=sample.open!==old; sample.open=old; }
      const wide=[...document.querySelectorAll('article > *, main > *')].filter(e=>{
        const r=e.getBoundingClientRect(); return r.width && r.right>innerWidth+1;
      }).map(e=>({tag:e.tagName,cls:String(e.className).slice(0,40),text:e.textContent.slice(0,60)}));
      const tables=[...document.querySelectorAll('article table')].filter(t=>{
        const r=t.getBoundingClientRect(); return r.right>innerWidth+1;}).length;
      const pre=[...document.querySelectorAll('article pre')].filter(t=>{
        const r=t.getBoundingClientRect(); return r.right>innerWidth+1;}).length;
      return {title:document.title,viewport:[innerWidth,innerHeight],scrollWidth:root.scrollWidth,
        background:getComputedStyle(document.body).backgroundColor,color:getComputedStyle(document.body).color,
        math:document.querySelectorAll('math').length,mathErrors:document.querySelectorAll('.math-error').length,
        svg:document.querySelectorAll('article svg').length,
        svgStrayP:[...document.querySelectorAll('article svg p')].length,
        folds:folds.length,toggles,highlighted:document.querySelectorAll('.highlight').length,
        overflow:wide,overflowTables:tables,overflowPre:pre,
        externalResources:performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name),
        styles:[...document.styleSheets].map(s=>s.href)};
    })()''')['result']


def anchors(url):
    """源码页里被正文引用的行锚点必须真实存在。"""
    run('open', url)
    run('snapshot', '-i')
    return run('eval', '''(() => {
      const ids=new Set([...document.querySelectorAll('[id]')].map(e=>e.id));
      const want=['-L-19','-L-494','-L-772','-L-321','-L-65','-L-36','-L-75'];
      return {total:ids.size, anchors:want.map(w=>[w,[...ids].some(i=>i.endsWith(w))])};
    })()''')['result']


try:
    run('set', 'offline', 'on')
    pages = ['index.html', 'L5/5.11-api-protocol-layer.html']
    for vw, vh, tag in [('1280', '900', 'desktop'), ('834', '1112', 'ipad')]:
        run('set', 'viewport', vw, vh)
        for name in pages:
            run('open', (ROOT / 'site' / name).as_uri())
            run('snapshot', '-i')
            for theme in ['light', 'dark']:
                run('set', 'media', theme)
                r = inspect()
                r.update(path=name, theme=f'{theme}-{tag}')
                rows.append(r)
            if name.startswith('L5/'):
                run('set', 'media', 'light')
                run('screenshot', str(OUT / f'chapter-light-{tag}.png'))
                run('set', 'media', 'dark')
                run('screenshot', str(OUT / f'chapter-dark-{tag}.png'))
    # 源码页行锚点
    src = ROOT / 'site' / 'code' / 'results' / 'crater' / 'api' / '20260912-0325' / 'source' / 'vllm' / 'v1' / 'engine' / 'async_llm.py.html'
    anchor_report = anchors(src.as_uri())
finally:
    run('close')
    (OUT / 'browser-checks.json').write_text(json.dumps(
        {'rows': rows, 'anchors': anchor_report}, ensure_ascii=False, indent=2))

issues = [r for r in rows
          if r['scrollWidth'] > r['viewport'][0] + 1 or r['mathErrors']
          or r['externalResources'] or r['toggles'] is False or r['overflow']]
print(json.dumps({'checks': len(rows), 'pages': sorted(set(r['path'] for r in rows)),
                  'issues': issues, 'anchors': anchor_report}, ensure_ascii=False, indent=2))
