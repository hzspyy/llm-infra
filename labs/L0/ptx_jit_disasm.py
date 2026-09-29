#!/usr/bin/env python3
"""0.1-B: reproduce the machine code the driver JIT-compiles for a PTX-only kernel.

When a fat binary carries no cubin for the running architecture, the driver
compiles the shipped PTX at load time. That step can be reproduced offline with
the same toolchain: pull the PTX module that defines the kernel, run ptxas for
the target architecture, then disassemble the result.

python labs/L0/ptx_jit_disasm.py --library <path.so> --symbol <mangled> \\
    --arch sm_120 --output <new-run> \\
    --cuobjdump ... --ptxas ... --nvdisasm ... --workdir /scratch/...
"""
import argparse,collections,hashlib,json,re,shutil,subprocess
from pathlib import Path


def run(cmd,cwd=None,timeout=1800):
    return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,
                          errors='replace',cwd=cwd)


def histogram_of(text):
    opcodes=collections.Counter()
    for line in text.splitlines():
        m=re.match(r'\s+/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)',line)
        if m:opcodes[m.group(1).split('.')[0]]+=1
    return opcodes


def cut_function(text,symbol):
    keep=[];on=False
    for line in text.splitlines():
        if line.startswith('//----') or line.lstrip().startswith('.section'):
            on=symbol in line
        if on:keep.append(line)
    return '\n'.join(keep)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',required=True)
    ap.add_argument('--symbol',required=True)
    ap.add_argument('--arch',default='sm_120')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cuobjdump',required=True)
    ap.add_argument('--ptxas',required=True)
    ap.add_argument('--nvdisasm',required=True)
    ap.add_argument('--workdir',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    work=a.workdir;shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True)

    run([a.cuobjdump,'-xptx','all',a.library],cwd=str(work))
    modules=sorted(work.glob('*.ptx'))
    hit=next((m for m in modules if a.symbol in m.read_text(errors='replace')),None)
    if hit is None:
        result=dict(library=a.library,symbol=a.symbol,ptx_modules=len(modules),
                    error='symbol not found in any extracted PTX module')
        (a.output/'ptx_jit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (a.output/'exit.txt').write_text('1\n');print(json.dumps(result));raise SystemExit(1)

    ptx_text=hit.read_text(errors='replace')
    ptx_target=re.search(r'\.target\s+(\S+)',ptx_text)
    # The whole module is several MB; keep a readable excerpt plus the kernel's
    # own entry declaration, and record the hash so it can be re-derived.
    lines=ptx_text.splitlines()
    entry=next((i for i,l in enumerate(lines) if a.symbol in l and '.entry' in l),None)
    excerpt=lines[:120]
    if entry is not None:
        excerpt+=['','// ---- kernel entry ----']+lines[entry:entry+80]
    (a.output/'module-excerpt.ptx').write_text('\n'.join(excerpt)+'\n')
    cubin=work/'jit.cubin'
    compile_cmd=[a.ptxas,f'-arch={a.arch}','-o',str(cubin),str(hit)]
    c=run(compile_cmd,timeout=1800)
    result=dict(library=a.library,symbol=a.symbol,arch=a.arch,
        ptx_modules=len(modules),ptx_module=hit.name,
        ptx_bytes=hit.stat().st_size,ptx_lines=len(lines),
        ptx_sha256=hashlib.sha256(hit.read_bytes()).hexdigest(),
        ptx_target=ptx_target.group(1) if ptx_target else None,
        ptxas_command=' '.join(compile_cmd),ptxas_returncode=c.returncode,
        ptxas_stderr=c.stderr[:600])
    if c.returncode!=0 or not cubin.is_file():
        result['error']='ptxas failed'
        (a.output/'ptx_jit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (a.output/'exit.txt').write_text('1\n');print(json.dumps(result));raise SystemExit(1)

    text=run([a.nvdisasm,'-c',str(cubin)],timeout=900).stdout
    body=cut_function(text,a.symbol)
    (a.output/'jit.sass').write_text(body+'\n')
    opcodes=histogram_of(body)
    result.update(cubin_bytes=cubin.stat().st_size,
        cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
        sass_lines=len(body.splitlines()),instruction_count=sum(opcodes.values()),
        top_opcodes=[dict(opcode=o,count=c2) for o,c2 in opcodes.most_common(12)])
    (a.output/'ptx_jit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    shutil.rmtree(work,ignore_errors=True)
    ok=result['instruction_count']>0
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps(dict(arch=a.arch,ptx_target=result['ptx_target'],
        instructions=result['instruction_count'])))
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
