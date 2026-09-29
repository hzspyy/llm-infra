"""6.0b 页面验收：多视口/深浅色、横向溢出、折叠、SVG 文本碰撞与越界。"""
from pathlib import Path
import os, subprocess, json, sys

ROOT = Path(__file__).resolve().parents[1]
BIN = '/Users/nyxri/.workbuddy-ai/binaries/node/workspace/node_modules/.bin/agent-browser'
env = dict(os.environ,
           PATH='/Users/nyxri/.workbuddy-ai/binaries/node/versions/22.22.2-2/bin:' + os.environ['PATH'])
PAGE = (ROOT / 'site/L4/4.3-quantization.html').as_uri()


def run(*args):
    p = subprocess.run([BIN, '--session', 'l43', '--json', *args],
                       env=env, text=True, capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    d = json.loads(p.stdout)
    if not d.get('success'):
        raise RuntimeError(d)
    return d.get('data', {})


INSPECT = '''(() => {
  const root=document.documentElement;
  const folds=[...document.querySelectorAll('main details')];
  let toggles=null;
  if(folds.length){const s=folds[0]; if(s.querySelector(':scope > summary')){
    const old=s.open; s.querySelector('summary').click(); toggles=s.open!==old; s.open=old;}}
  const overflow=[...document.querySelectorAll('article > *, main > *')].filter(e=>{
    const r=e.getBoundingClientRect(); return r.width && r.right>innerWidth+1;})
    .map(e=>({tag:e.tagName,cls:String(e.className).slice(0,50),text:e.textContent.slice(0,60)}));
  const svg=[];
  document.querySelectorAll('svg').forEach((s,i)=>{
    const r=s.getBoundingClientRect();
    const texts=[...s.querySelectorAll('text')].map(t=>{const q=t.getBoundingClientRect();
      return {t:t.textContent.slice(0,40),l:q.left-r.left,r:q.right-r.left,tp:q.top-r.top,b:q.bottom-r.top};});
    const hits=[];
    for(let i2=0;i2<texts.length;i2++)for(let j=i2+1;j<texts.length;j++){
      const a=texts[i2],b=texts[j];
      const ox=Math.min(a.r,b.r)-Math.max(a.l,b.l), oy=Math.min(a.b,b.b)-Math.max(a.tp,b.tp);
      if(ox>1.5&&oy>1.5) hits.push({a:a.t,b:b.t});}
    const outside=texts.filter(t=>t.l<-1||t.r>r.width+1||t.tp<-1||t.b>r.height+1).map(t=>t.t);
    svg.push({index:i,w:Math.round(r.width),texts:texts.length,hits,outside});
  });
  return {title:document.title,viewport:[innerWidth,innerHeight],width:root.scrollWidth,
    background:getComputedStyle(document.body).backgroundColor,
    mathErrors:document.querySelectorAll('.math-error').length,
    folds:folds.length,toggles,highlighted:document.querySelectorAll('.highlight').length,
    pre:document.querySelectorAll('pre').length,svg,
    externalResources:performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name),
    overflow};
})()'''

rows = []
for theme in ['light', 'dark']:
    run('set', 'offline', 'on')
    run('set', 'media', theme)
    for label, w, h in [('desktop', 1440, 900), ('ipad', 834, 1112), ('phone', 390, 844)]:
        run('set', 'viewport', str(w), str(h))
        run('open', PAGE)
        r = run('eval', INSPECT)['result']
        r.update(theme=theme, device=label)
        rows.append(r)
        print(f"{theme}/{label}: width={r['width']} 溢出={len(r['overflow'])} "
              f"折叠={r['folds']}(切换={r['toggles']}) 高亮={r['highlighted']} "
              f"外链={len(r['externalResources'])} mathErr={r['mathErrors']} "
              f"SVG={[(s['w'], len(s['hits']), len(s['outside'])) for s in r['svg']]}")
        for o in r['overflow'][:3]:
            print('   OVERFLOW', o)
        for s in r['svg']:
            for hh in s['hits'][:3]:
                print('   HIT', s['index'], hh)
            for t in s['outside'][:3]:
                print('   OUT', s['index'], t)
    run('set', 'viewport', '1440', '900')
    run('open', PAGE)
    run('screenshot', str(ROOT / 'outputs' / f'4.3-review-{theme}.png'))
    run('set', 'media', 'light')
(ROOT / 'outputs' / '4.3-browser-review.json').write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved outputs/4.3-browser-review.json')
