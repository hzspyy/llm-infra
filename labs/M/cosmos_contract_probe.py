#!/usr/bin/env python3
"""Run the pinned Cosmos pipeline's original check_inputs in isolation.

No model imports, weights, inference, GPU timing or quality validation.
python labs/M/cosmos_contract_probe.py --source SOURCE.py --output NEW_DIR
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Cosmos3OmniPipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'check_inputs')
    # Compile original AST method, untouched, with deferred annotations. No module imports/initialization.
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    isolated = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    namespace = {}
    exec(compile(isolated, str(args.source), 'exec'), namespace)
    check = namespace['check_inputs']
    cases = [
        ('i2v-valid', {'image': 'sentinel'}, {}, 'accept'),
        ('bad-prompt', {'prompt': 42}, {}, 'ValueError'),
        ('sound-without-tokenizer', {'enable_sound': True}, {}, 'ValueError'),
        ('unsupported-callback', {'callback_on_step_end_tensor_inputs': ['sound_latents']}, {}, 'ValueError'),
        ('unaligned-width', {'width': 831}, {}, 'ValueError'),
        ('zero-frames', {'num_frames': 0}, {}, 'ValueError'),
        ('image-and-video', {'image': 'sentinel', 'video': 'sentinel'}, {}, 'ValueError'),
        ('video-one-frame', {'video': 'sentinel', 'num_frames': 1}, {}, 'ValueError'),
        ('video-empty-indexes', {'video': 'sentinel', 'condition_frame_indexes_vision': ()}, {}, 'ValueError'),
        ('video-index-outside', {'video': 'sentinel', 'condition_frame_indexes_vision': (31,)}, {}, 'ValueError'),
        ('video-last-valid-index', {'video': 'sentinel', 'condition_frame_indexes_vision': (30,)}, {}, 'accept'),
        ('sound-model-disabled', {'enable_sound': True}, {'sound_tokenizer': 'sentinel'}, 'ValueError'),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'cases.json', cases)
    save(args.output/'manifest.json', {'task':'M1-C', 'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'symbol':'Cosmos3OmniPipeline.check_inputs', 'line':method.lineno, 'end_line':method.end_lineno,
         'python':sys.version, 'scope':'original AST method with synthetic configuration; no pipeline instantiation',
         'checkpoint':'nvidia/Cosmos3-Edge@a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba',
         'diffusers_commit':'c419dac0152186060246c93a095bc1bfaea342b3',
         'timing':'none', 'weights_loaded':False})
    observations = []
    for name, overrides, attributes, expected in cases:
        self = SimpleNamespace(sound_tokenizer=None, transformer=SimpleNamespace(config=SimpleNamespace(sound_gen=False)),
             vae=SimpleNamespace(config=SimpleNamespace(scale_factor_spatial=16,scale_factor_temporal=4)),
             _callback_tensor_inputs=['latents'])
        for key, value in attributes.items(): setattr(self,key,value)
        inputs = dict(prompt='A fixed scene.',negative_prompt=None,image=None,height=480,width=832,
                      num_frames=121,guidance_scale=6.0,enable_sound=False,
                      callback_on_step_end_tensor_inputs=['latents'])
        inputs.update(overrides)
        try:
            check(self,**inputs)
            actual, message = 'accept', None
        except Exception as error:
            actual, message = type(error).__name__, str(error)
        observations.append(dict(id=name,inputs=inputs,config_overrides=attributes,
                                 expected=expected,actual=actual,message=message,passed=actual==expected))
    save(args.output/'observations.json', observations)
    ok=all(x['passed'] for x in observations)
    save(args.output/'summary.json', {'cases':len(cases),'passed':sum(x['passed'] for x in observations),'full_pipeline':'UNVERIFIED'})
    (args.output/'exit.txt').write_text('0\n' if ok else '1\n')
    print(json.dumps({'cases':len(cases),'passed':sum(x['passed'] for x in observations)}))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
