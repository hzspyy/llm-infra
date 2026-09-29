import json,re,traceback
from pathlib import Path
from urllib.parse import urlsplit,unquote
from playwright.sync_api import sync_playwright
base=Path('/Volumes/data/llm-infra-review-fixes-ostfce6p')
out=base/'browser-qa';out.mkdir(exist_ok=False)
paths=sorted(Path('site/L7').glob('*.html'))+[Path(p) for p in ['site/L2/2.0c-runtime-memory.html','site/L4/4.3-quantization.html','site/L4/4.0-checkpoint-format.html','site/L2/2.0-tensor-and-framework.html','site/L1/1.2-tensor-core-lineage.html','site/L3/3.3-decode-attention.html']]
modes=[('wide',1440,1000),('ipad',820,1050),('phone',390,844),('sidebar-below',1087,950),('sidebar-above',1088,950)]
records=[];failures=[];images=[];source_records=[]
with sync_playwright() as p:
 ctx=p.chromium.launch_persistent_context(str(base/'chrome-qa-profile'),executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True,viewport={'width':1440,'height':1000},offline=True)
 page=ctx.pages[0]
 requests=[];errors=[]
 page.on('request',lambda r: requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
 page.on('pageerror',lambda e:errors.append(str(e)))
 for path in paths:
  for mode,w,h in modes:
   for theme in ('light','dark'):
    page.set_viewport_size({'width':w,'height':h});page.emulate_media(color_scheme=theme)
    page.goto(path.resolve().as_uri(),wait_until='load')
    info=page.evaluate('''() => {
      const a=document.querySelector('article');
      const svgs=[...a.querySelectorAll('svg')];
      return {width:innerWidth,documentWidth:document.documentElement.scrollWidth,bodyWidth:document.body.scrollWidth,
        svgCount:svgs.length,fencedSvg:[...a.querySelectorAll('pre')].filter(e=>e.textContent.includes('<svg')).length,
        mathCount:a.querySelectorAll('math').length,sourceBlocks:a.querySelectorAll('.srcblock').length,
        macroResidue:a.textContent.includes('{{src:'),background:getComputedStyle(document.body).backgroundColor,
        sidebarPosition:getComputedStyle(document.querySelector('.sidebar')).position,
        brokenFragments:[...document.querySelectorAll('a[href^="#"]')].map(a=>a.getAttribute('href').slice(1)).filter(x=>x&&!document.getElementById(decodeURIComponent(x))),
        svgTextOverflow:svgs.flatMap(s=>{const b=s.getBoundingClientRect();return [...s.querySelectorAll('text')].filter(t=>{const r=t.getBoundingClientRect();return r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2}).map(t=>t.textContent)})};
    }''')
    rec={'page':str(path),'mode':mode,'theme':theme,**info}
    if info['documentWidth']>w+1 or info['bodyWidth']>w+1:failures.append({'kind':'page overflow',**rec})
    if '/L7/' in str(path) and (info['svgCount']==0 or info['fencedSvg']):failures.append({'kind':'L7 SVG',**rec})
    if info['macroResidue'] or info['brokenFragments'] or info['svgTextOverflow']:failures.append({'kind':'content rendering',**rec})
    # Exercise native folds and compact navigation without relying on JS handlers.
    for selector in ('article details.srcfold','article details.fold:not(.srcfold)', '.mobile-toc'):
     details=page.locator(selector)
     if details.count() and details.first.is_visible():
      target=details.first
      if not target.get_attribute('open'):
       target.locator('summary').first.click()
       assert target.evaluate('(e)=>e.open')
       if page.evaluate('document.documentElement.scrollWidth')>w+1:failures.append({'kind':'open fold overflow','page':str(path),'mode':mode,'theme':theme,'selector':selector})
       target.locator('summary').first.click()
    if mode in ('wide','ipad','phone'):
     svg=page.locator('article svg').first
     if svg.count():svg.scroll_into_view_if_needed()
     filename=f'{path.stem}-{mode}-{theme}.png'
     page.screenshot(path=str(out/filename));images.append(filename)
     if mode=='wide' and svg.count():
      svg.screenshot(path=str(out/f'{path.stem}-figure-{theme}.png'))
    records.append(rec)
  # One complete-source navigation per affected page.
  page.set_viewport_size({'width':1440,'height':1000});page.emulate_media(color_scheme='light')
  page.goto(path.resolve().as_uri())
  links=page.locator('article .srcref a')
  if links.count():
   href=links.first.evaluate('(a)=>a.href')
   page.goto(href)
   fragment=unquote(urlsplit(href).fragment)
   found=page.evaluate('(id)=>!id||!!document.getElementById(id)',fragment)
   item={'chapter':str(path),'url':href,'fragment_exists':found,'source_highlight_spans':page.locator('.highlight span').count(),'page_overflow':page.evaluate('document.documentElement.scrollWidth>innerWidth+1')}
   if not found or item['page_overflow']:failures.append({'kind':'source page',**item})
   source_records.append(item)
 # Offline landing page and the source index.
 for name in ['site/index.html','site/code/index.html']:
  page.goto(Path(name).resolve().as_uri());assert page.title()
  if page.evaluate('document.documentElement.scrollWidth>innerWidth+1'):failures.append({'kind':'index overflow','page':name})
 ctx.close()
result={'pages':[str(p) for p in paths],'modes':modes,'records':records,'source_pages':source_records,
        'external_requests':requests,'page_errors':errors,'failures':failures,'screenshots':images,
        'scope':'file:// offline Chrome, 19 affected chapters, source links, folds, light/dark and responsive layout'}
(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pages':len(paths),'layout_checks':len(records),'source_pages':len(source_records),'external_requests':len(requests),'page_errors':errors,'failures':failures},ensure_ascii=False))
