from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,re,ast,hashlib
root=Path.cwd();site=root/'site';base=Path('/Volumes/data/llm-infra-review-fixes-ostfce6p')
chapters=sorted(Path('src/L7').glob('*.md'))+[Path(x) for x in ['src/L2/2.0c-runtime-memory.md','src/L4/4.3-quantization.md','src/L4/4.0-checkpoint-format.md','src/L2/2.0-tensor-and-framework.md','src/L1/1.2-tensor-core-lineage.md','src/L3/3.3-decode-attention.md']]
class Html(HTMLParser):
 def __init__(self,text):
  super().__init__();self.ids=set();self.refs=[];self.svg=0;self.math_errors=0;self.feed(text)
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.add(a['id'])
  for k in ['href','src']:
   if k in a:self.refs.append(a[k])
  self.svg+=tag=='svg'
  self.math_errors+='math-error' in a.get('class','').split()
cache={}
def parse(p):
 if p not in cache:cache[p]=Html(p.read_text())
 return cache[p]
errors=[];refs=[];commands=[]
for path in chapters:
 text=path.read_text()
 for m in re.finditer(r'\{\{(?:src|srcfold):([^:}]+)(?::(\d+)-(\d+))?\}\}',text):
  source=Path(m[1]);record={'chapter':str(path),'source':str(source),'start':m[2],'end':m[3]}
  if not source.is_file():errors.append({'missing_source':record})
  else:
   record['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
   if m[3] and int(m[3])>len(source.read_text().splitlines()):errors.append({'invalid_source_range':record})
  refs.append(record)
 for match in re.finditer(r'(?<![\w/])labs/[\w./-]+\.py',text):
  commands.append({'chapter':str(path),'lab':match[0]})
  if not Path(match[0]).is_file():errors.append({'missing_lab':commands[-1]})
 page=(site/path.relative_to('src')).with_suffix('.html');doc=parse(page)
 if path.parent.name=='L7' and doc.svg<1:errors.append({'missing_svg':str(page)})
 if doc.math_errors:errors.append({'math_errors':str(page),'count':doc.math_errors})
 for href in doc.refs:
  url=urlsplit(href)
  if url.scheme or href.startswith('//'):continue
  target=(page.parent/unquote(url.path)).resolve() if url.path else page.resolve()
  if not target.exists():errors.append({'missing_site_target':href,'page':str(page)});continue
  if not target.is_relative_to(site):errors.append({'outside_site_dependency':href,'page':str(page)})
  if target.suffix=='.html' and url.fragment and unquote(url.fragment) not in parse(target).ids:errors.append({'missing_anchor':href,'page':str(page)})
for path in list(Path('labs/L7').glob('*.py'))+[Path('labs/L2/alloc_trace.py'),Path('labs/L2/alloc_graph_combo.py'),Path('labs/L4/quantize_reference.py')]:
 try:ast.parse(path.read_text(),filename=str(path))
 except SyntaxError as error:errors.append({'syntax':str(path),'message':str(error)})
# Current core dependency DAG, plan mapping and slug uniqueness.
outline=json.loads(Path('outline.json').read_text());modules=[m for layer in outline['layers'] for m in layer['modules']];ids={m['id'] for m in modules}
assert len(ids)==len(modules)
graph={m['id']:m.get('spec',{}).get('core_dependencies',[]) for m in modules}
visited=set();active=set()
def visit(key):
 if key in active:raise AssertionError('core dependency cycle: '+key)
 if key in visited:return
 active.add(key)
 for dep in graph[key]:
  if dep not in graph:raise AssertionError('unknown core dependency: '+dep)
  visit(dep)
 active.remove(key);visited.add(key)
for key in graph:visit(key)
for m in modules:
 entry=m.get('spec',{}).get('execution_plan')
 if entry:
  file,_,fragment=entry.partition('#')
  if not Path(file).is_file() or (fragment and fragment not in Path(file).read_text()):errors.append({'plan_mapping':m['id'],'entry':entry})
result={'chapters':[str(p) for p in chapters],'source_macros':refs,'lab_references':commands,'errors':errors,'core_dag_nodes':len(visited),'module_count':len(modules),'body_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in chapters}}
(base/'source-reference-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'chapters':len(chapters),'source_macros':len(refs),'lab_references':len(commands),'core_dag_nodes':len(visited),'errors':errors},ensure_ascii=False))
assert not errors
