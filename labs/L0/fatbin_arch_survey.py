#!/usr/bin/env python3
"""0.1-B: which GPU architectures a shipped .so actually carries.

A fat binary can hold native machine code (SASS, one cubin per architecture),
PTX (which the driver JIT-compiles at load time), or both. Running the same
survey over several builds shows which of them can start without a JIT on a
given device.

python labs/L0/fatbin_arch_survey.py --cuobjdump /path/to/cuobjdump \\
    --output results/crater/0.1/<run> --label vendored=/path/to/_vllm_fa2_C.abi3.so
"""
import argparse,collections,hashlib,json,re,subprocess
from pathlib import Path


def histogram(cuobjdump,flag,lib,timeout=900):
    r=subprocess.run([cuobjdump,flag,lib],capture_output=True,text=True,timeout=timeout,errors='replace')
    return dict(sorted(collections.Counter(re.findall(r'sm_\d+a?',r.stdout)).items()))


def survey(cuobjdump,label,lib,target):
    path=Path(lib)
    if not path.is_file():return dict(label=label,library=lib,error='file not found')
    sass=histogram(cuobjdump,'--list-elf',lib)
    ptx=histogram(cuobjdump,'--list-ptx',lib)
    return dict(label=label,library=lib,bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sass_archs=sass,ptx_archs=ptx,
        native_for_target=target in sass,ptx_for_target=target in ptx,
        # Without a native cubin the driver must build one from PTX at load time.
        needs_jit_for_target=(target not in sass) and bool(ptx))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cuobjdump',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--target',default='sm_120',help='architecture of the device in use')
    ap.add_argument('--label',action='append',required=True,metavar='NAME=PATH')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for item in a.label:
        label,_,lib=item.partition('=')
        rows.append(survey(a.cuobjdump,label,lib,a.target))
    version=subprocess.run([a.cuobjdump,'--version'],capture_output=True,text=True).stdout.strip().splitlines()
    result=dict(target=a.target,cuobjdump_version=version,task_ids=['0.1-B'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),libraries=rows)
    (a.output/'fatbin_archs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    ok=all('error' not in r for r in rows)
    (a.output/'exit.txt').write_text('0\n' if ok else '1\n')
    for r in rows:
        if 'error' in r:print(r['label'],r['error']);continue
        print(f"{r['label']:<22} SASS={r['sass_archs']} PTX={r['ptx_archs']} "
              f"native_{a.target}={r['native_for_target']} needs_jit={r['needs_jit_for_target']}")
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
