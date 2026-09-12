from pathlib import Path
import os, subprocess, json
ROOT=Path(__file__).resolve().parents[1]
BIN='/Users/nyxri/.workbuddy-ai/binaries/node/workspace/node_modules/.bin/agent-browser'
env=dict(os.environ,PATH='/Users/nyxri/.workbuddy-ai/binaries/node/versions/22.22.2-2/bin:'+os.environ['PATH'])
rows=[]

def run(*args):
    p=subprocess.run([BIN,'--session','llm-review','--json',*args],env=env,text=True,capture_output=True,timeout=60)
    if p.returncode: raise RuntimeError(p.stderr or p.stdout)
    d=json.loads(p.stdout)
    if not d.get('success'): raise RuntimeError(d)
    return d.get('data',{})

def inspect():
    return run('eval','''(() => {
      const root=document.documentElement;
      const folds=[...document.querySelectorAll('article details')];
      const sample=folds.find(x=>x.querySelector(':scope > summary'));
      let toggles=null;
      if(sample){ const old=sample.open; sample.querySelector('summary').click(); toggles=sample.open!==old; sample.open=old; }
      const out=[...document.querySelectorAll('article > *, main > *')].filter(e=>{
        const r=e.getBoundingClientRect(); return r.width && r.right>innerWidth+1;
      }).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,80)}));
      return {title:document.title,viewport:[innerWidth,innerHeight],width:root.scrollWidth,
        background:getComputedStyle(document.body).backgroundColor,color:getComputedStyle(document.body).color,
        math:document.querySelectorAll('math').length,mathErrors:document.querySelectorAll('.math-error').length,
        folds:folds.length,toggles,highlighted:document.querySelectorAll('.highlight').length,
        mobileToc:!!document.querySelector('.mobile-toc'),overflow:out,
        externalResources:performance.getEntriesByType('resource').filter(r=>/^https?:/.test(r.name)).map(r=>r.name),
        styles:[...document.styleSheets].map(s=>s.href)};
    })()''')['result']

try:
    run('set','offline','on')
    run('set','viewport','834','1112')
    pages=[ROOT/'site/index.html']+sorted(p for p in (ROOT/'site').glob('L*/*.html') if not p.name.startswith('5.10-'))
    for p in pages:
        run('open',p.as_uri())
        run('snapshot','-i')
        for theme in ['light','dark']:
            run('set','media',theme)
            r=inspect();r.update(path=str(p.relative_to(ROOT/'site')),theme=theme);rows.append(r)
        if p.name.startswith(('5.3-','5.5-')):
            run('screenshot',str(ROOT/'outputs'/f'{p.stem}-ipad-dark.png'))
    run('set','viewport','1280','900')
    for name in ['index.html','L5/5.3-scheduling.html','L5/5.5-speculative-decoding.html','code/labs/L5/speculative.py.html']:
        run('open',(ROOT/'site'/name).as_uri());run('snapshot','-i');run('set','media','light')
        r=inspect();r.update(path=name,theme='light-desktop');rows.append(r)
        if name=='index.html':run('screenshot',str(ROOT/'outputs/home-review-light.png'))
    run('open',(ROOT/'site/L5/5.5-speculative-decoding.html').as_uri())
    run('set','viewport','834','1112');run('set','media','light')
    run('screenshot',str(ROOT/'outputs/speculative-review-light.png'))
finally:
    run('close')
    (ROOT/'outputs/browser-review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
issues=[r for r in rows if r['width']>r['viewport'][0]+1 or r['mathErrors'] or r['externalResources'] or r['toggles'] is False]
print(json.dumps({'checks':len(rows),'pages':len(set(r['path'] for r in rows)),'issues':issues},ensure_ascii=False,indent=2))
