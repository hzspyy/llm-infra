from pathlib import Path
import os, subprocess, json, sys

ROOT = Path(__file__).resolve().parents[1]
BIN = '/Users/nyxri/.workbuddy-ai/binaries/node/workspace/node_modules/.bin/agent-browser'
env = dict(os.environ,
           PATH='/Users/nyxri/.workbuddy-ai/binaries/node/versions/22.22.2-2/bin:' + os.environ['PATH'])


def run(*args):
    p = subprocess.run([BIN, '--session', 'l6review', '--json', *args],
                       env=env, text=True, capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    d = json.loads(p.stdout)
    if not d.get('success'):
        raise RuntimeError(d)
    return d.get('data', {})


def inspect(theme):
    return run('eval', '''(() => {
      const root=document.documentElement;
      const folds=[...document.querySelectorAll('article details, main details')];
      let toggles=null;
      if(folds.length){
        const s=folds[0];
        if(s.querySelector(':scope > summary')){
          const old=s.open; s.querySelector('summary').click(); toggles=s.open!==old; s.open=old;
        }
      }
      const out=[...document.querySelectorAll('article > *, main > *')].filter(e=>{
        const r=e.getBoundingClientRect(); return r.width && r.right>innerWidth+1;
      }).map(e=>({tag:e.tagName,cls:String(e.className).slice(0,60),text:e.textContent.slice(0,70)}));
      const svg=[...document.querySelectorAll('svg')].map(s=>{
        const r=s.getBoundingClientRect(); return {w:Math.round(r.width),right:Math.round(r.right)};
      });
      return {title:document.title,viewport:[innerWidth,innerHeight],width:root.scrollWidth,
        background:getComputedStyle(document.body).backgroundColor,
        math:document.querySelectorAll('math').length,
        mathErrors:document.querySelectorAll('.math-error').length,
        folds:folds.length,toggles,highlighted:document.querySelectorAll('.highlight').length,
        pre:document.querySelectorAll('pre').length, svg:svg,
        externalResources:performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name),
        overflow:out};
    })()''')['result']


out = []
for theme in ['light', 'dark']:
    run('set', 'offline', 'on')
    run('set', 'media', theme)
    for label, w, h in [('desktop', 1440, 900), ('ipad', 834, 1112), ('phone', 390, 844)]:
        run('set', 'viewport', str(w), str(h))
        run('open', (ROOT / 'site/L6/6.0-concurrency-semantics.html').as_uri())
        info = inspect(theme)
        info['theme'] = theme
        info['device'] = label
        out.append(info)
        print(json.dumps(info, ensure_ascii=False))
    run('set', 'viewport', '1440', '900')
    run('open', (ROOT / 'site/L6/6.0-concurrency-semantics.html').as_uri())
    run('screenshot', str(ROOT / 'outputs' / f'6.0-review-{theme}.png'))
    if theme == 'dark':
        run('set', 'viewport', '834', '1112')
        run('open', (ROOT / 'site/L6/6.0-concurrency-semantics.html').as_uri())
        run('screenshot', str(ROOT / 'outputs' / '6.0-review-ipad-dark.png'))
    run('set', 'media', 'light')

(ROOT / 'outputs' / '6.0-browser-review.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved', ROOT / 'outputs/6.0-browser-review.json')
