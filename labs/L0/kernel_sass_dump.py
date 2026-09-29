#!/usr/bin/env python3
"""0.1-B: disassemble the kernels that op_to_kernel_trace.py followed.

cuobjdump cannot disassemble a kernel straight out of a host shared library, so
the chain is: extract every embedded cubin, keep the ones built for the target
architecture, find the cubin whose symbol table defines the kernel, then run
nvdisasm on it and cut out that function's block.

python labs/L0/kernel_sass_dump.py --trace <run>/op_to_kernel.json \
    --output <new-run> --cuobjdump /path/to/cuobjdump --nvdisasm /path/to/nvdisasm \
    --workdir /scratch/learn/work/<scratch-dir>
"""
import argparse,collections,hashlib,json,re,shutil,subprocess
from pathlib import Path

COMMON={'void','const','unsigned','cutlass','float','double','type','enable_if','integral_constant'}


def needles(kernel):
    """Mangled-symbol substrings that identify a kernel from its demangled name.

    The outermost template name alone is ambiguous - libtorch and vLLM both ship
    a flash_fwd_splitkv_kernel - so the namespace is rebuilt as an Itanium
    mangled prefix and paired with the longest token from the template args.
    """
    head=kernel.split('<')[0].replace('void ','').strip()
    parts=[p for p in head.split('::') if p]
    out=[]
    if len(parts)>=2:out.append('_ZN'+''.join(f'{len(p)}{p}' for p in parts))
    elif parts:out.append(parts[-1])
    args=kernel[kernel.find('<')+1:] if '<' in kernel else ''
    tokens=[t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]{7,}',args) if t not in COMMON]
    if tokens:out.append(max(tokens,key=len))
    return out


def run(cmd,cwd=None,timeout=1800):
    return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,
                          errors='replace',cwd=cwd)


def find_library(cuobjdump,libraries,keys):
    for lib in libraries:
        r=run([cuobjdump,'--dump-elf-symbols',lib])
        for line in r.stdout.splitlines():
            if 'STT_FUNC' in line and all(k in line for k in keys):
                return lib,line.split()[-1]
    return None,None


def arch_inventory(cuobjdump,lib):
    def histogram(flag):
        return dict(sorted(collections.Counter(
            re.findall(r'sm_\d+a?',run([cuobjdump,flag,lib]).stdout)).items()))
    return dict(sass=histogram('--list-elf'),ptx=histogram('--list-ptx'))


def extract_cubins(cuobjdump,lib,workdir):
    """cuobjdump -xelf writes into the working directory, so give it a fresh one."""
    workdir.mkdir(parents=True,exist_ok=True)
    run([cuobjdump,'-xelf','all',str(lib)],cwd=str(workdir))
    return sorted(workdir.glob('*.cubin'))


def cut_function(text,symbol):
    """Keep only the section belonging to this kernel, up to the next section."""
    keep=[];on=False
    for line in text.splitlines():
        if line.startswith('//----') or line.lstrip().startswith('.section'):
            on=symbol in line
        if on:keep.append(line)
    return '\n'.join(keep)


def histogram_of(text):
    opcodes=collections.Counter()
    for line in text.splitlines():
        m=re.match(r'\s+/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',line)
        if m:opcodes[m.group(1).split('.')[0]]+=1
    return opcodes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cuobjdump',required=True)
    ap.add_argument('--nvdisasm',required=True)
    ap.add_argument('--workdir',type=Path,required=True,help='scratch space for extracted cubins')
    ap.add_argument('--search-root',default='/scratch/learn/envs/serve/lib/python3.12/site-packages')
    ap.add_argument('--arch',default='sm_120')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(a.search_root)
    # cuBLAS/cuDNN live under nvidia/ and hold the GEMM kernels torch dispatches to.
    libraries=[str(p) for p in sorted(root.rglob('*.so*'),key=lambda p:-p.stat().st_size)
               if p.is_file() and any(k in str(p) for k in ('vllm','torch','flash','nvidia'))]
    followed=json.loads(a.trace.read_text())['followed']
    results={}
    for label,entry in followed.items():
        kernel=entry['prefill']['kernels'][0]['kernel_name']
        keys=needles(kernel)
        lib,symbol=find_library(a.cuobjdump,libraries,keys)
        if lib is None:
            results[label]=dict(kernel=kernel,searched=keys,
                error='symbol not found in searched libraries');continue
        inventory=arch_inventory(a.cuobjdump,lib)
        arch=a.arch if a.arch in inventory['sass'] else max(inventory['sass'],default=a.arch)
        scratch=a.workdir/label
        if scratch.exists():shutil.rmtree(scratch)
        cubins=[c for c in extract_cubins(a.cuobjdump,lib,scratch) if arch in c.name]
        base=dict(kernel=kernel,searched=keys,library=lib,symbol=symbol,requested_arch=a.arch,
            dumped_arch=arch,fatbin_archs=inventory,cubins_for_arch=len(cubins))
        # A cubin can name a symbol without defining it, so keep scanning and
        # take the one that actually carries instructions.
        best=None
        for cubin in cubins:
            text=run([a.nvdisasm,'-c',str(cubin)],timeout=600).stdout
            if symbol not in text:continue
            body=cut_function(text,symbol)
            count=sum(histogram_of(body).values())
            if best is None or count>best[2]:best=(cubin,body,count)
            if count:break
        if best is None or best[2]==0:
            results[label]=dict(base,error=f'{symbol} has no instructions in any {arch} cubin',
                cubins_naming_symbol=(0 if best is None else 1))
            shutil.rmtree(scratch,ignore_errors=True);continue
        cubin,body,_=best
        out=a.output/f'{label}.sass';out.write_text(body+'\n')
        opcodes=histogram_of(body)
        results[label]=dict(base,cubin=cubin.name,sass_file=out.name,
            sass_lines=len(body.splitlines()),instruction_count=sum(opcodes.values()),
            top_opcodes=[dict(opcode=o,count=c) for o,c in opcodes.most_common(12)])
        shutil.rmtree(scratch,ignore_errors=True)
    summary=dict(trace=str(a.trace),target_arch=a.arch,libraries_searched=len(libraries),
        cuobjdump_version=run([a.cuobjdump,'--version']).stdout.strip().splitlines(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernels=results)
    (a.output/'sass_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    ok=all('error' not in v for v in results.values()) and len(results)==len(followed)
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps({k:(v.get('error') or f"{v['dumped_arch']} {v['instruction_count']} insts")
                      for k,v in results.items()},ensure_ascii=False))
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
