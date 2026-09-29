"""Small lab artifact helpers: explicit new output directory, source pin and cases."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys


def new_output(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--outdir', required=True, type=Path,
                        help='new directory for small evidence files')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    return args.outdir


def write_result(outdir, filename, result, cases, sources):
    import torch
    with (outdir / filename).open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with (outdir / 'cases.json').open('x') as stream:
        json.dump(cases, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'argv': sys.argv, 'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'torch_git': torch.version.git_version,
        'result': filename, 'source_sha256': {
            str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [__file__, *sources]},
        'claim_scope': 'Only the numerical or protocol checks listed in cases.json; no training quality claim',
    }
    with (outdir / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
