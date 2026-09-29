"""检查 6.0 页面内联 SVG 的文本碰撞与越界。"""
from pathlib import Path
import os, subprocess, json

ROOT = Path(__file__).resolve().parents[1]
BIN = '/Users/nyxri/.workbuddy-ai/binaries/node/workspace/node_modules/.bin/agent-browser'
env = dict(os.environ,
           PATH='/Users/nyxri/.workbuddy-ai/binaries/node/versions/22.22.2-2/bin:' + os.environ['PATH'])


def run(*args):
    p = subprocess.run([BIN, '--session', 'l6svg', '--json', *args],
                       env=env, text=True, capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    d = json.loads(p.stdout)
    if not d.get('success'):
        raise RuntimeError(d)
    return d.get('data', {})


EXPR = '''(() => {
  const res=[];
  document.querySelectorAll('svg').forEach((svg,si)=>{
    const svgR=svg.getBoundingClientRect();
    const texts=[...svg.querySelectorAll('text')].map(t=>{
      const r=t.getBoundingClientRect();
      return {t:t.textContent.slice(0,42), l:r.left-svgR.left, r:r.right-svgR.left,
              tp:r.top-svgR.top, b:r.bottom-svgR.top};
    });
    const hits=[];
    for(let i=0;i<texts.length;i++){
      for(let j=i+1;j<texts.length;j++){
        const a=texts[i], b=texts[j];
        const ox=Math.min(a.r,b.r)-Math.max(a.l,b.l);
        const oy=Math.min(a.b,b.b)-Math.max(a.tp,b.tp);
        if(ox>1.5 && oy>1.5) hits.push({a:a.t,b:b.t,ox:Math.round(ox),oy:Math.round(oy)});
      }
    }
    const outside=texts.filter(t=>t.l<-1||t.r>svgR.width+1||t.tp<-1||t.b>svgR.height+1)
                       .map(t=>({t:t.t,l:Math.round(t.l),r:Math.round(t.r),b:Math.round(t.b)}));
    res.push({index:si, w:Math.round(svgR.width), h:Math.round(svgR.height),
              texts:texts.length, hits:hits, outside:outside});
  });
  return res;
})()'''

run('set', 'offline', 'on')
run('set', 'media', 'light')
rows = []
for label, w, h in [('desktop', 1440, 900), ('ipad', 834, 1112)]:
    run('set', 'viewport', str(w), str(h))
    run('open', (ROOT / 'site/L6/6.0-concurrency-semantics.html').as_uri())
    for r in run('eval', EXPR)['result']:
        r['device'] = label
        rows.append(r)
        print(f"{label} svg#{r['index']} {r['w']}x{r['h']} 文本 {r['texts']} 碰撞 {len(r['hits'])} 越界 {len(r['outside'])}")
        for x in r['hits'][:6]:
            print('   HIT', x)
        for x in r['outside'][:6]:
            print('   OUT', x)
(ROOT / 'outputs' / '6.0-svg-check.json').write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
