"""Small synthetic CPU compatibility/timing probe, never an official language fit."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flm.selection_language import load_catalog, prepare_case
from flm.language_learning_train import Settings, optimizer_for, update


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier probe output')
    identity_path = ROOT/'reports/selection-language/study-identity.json'
    identity = json.loads(identity_path.read_text())
    for name, expected in identity['sources'].items():
        if digest(ROOT/'flm'/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    torch.set_num_threads(4)
    corpus = SimpleNamespace(binding=identity['corpus_binding'], lexicon=SimpleNamespace(
        vocabulary=4096, sha256=identity['corpus_binding']['tokenizer_sha256']))
    catalog = load_catalog(ROOT)
    checks, timings, arrays = [], [], {}
    labels = ['KCg-d-R-t5/candidate/measured/s42', 'KCg-d-R-t5/candidate/null101/s42']
    for row in identity['conditions']:
        c = row['condition']
        model, binding = prepare_case(catalog, corpus, c['graph'], c['seed'])
        differences = [key for key in binding if binding[key] != row['base_binding'].get(key)]
        checks.append(dict(condition=c, differing_binding_fields=differences))
        if c['label'] not in labels:
            continue
        settings = Settings(**row['settings'])
        optimizer = optimizer_for(model, settings)
        x = ((torch.arange(16*96).reshape(16, 96)*37+11) % 4094)+2
        y = ((x+13-2) % 4094)+2
        records = []
        for step in range(1, 11):
            started = time.perf_counter()
            metrics = update(model, optimizer, x, y, settings, 'bptt', step)
            records.append(dict(step=step, seconds=time.perf_counter()-started, metrics=metrics))
        for name, p in model.named_parameters():
            arrays[c['label']+'::'+name] = p.detach().cpu().numpy().copy()
        timings.append(dict(condition=c, settings=asdict(settings), updates=records,
                            mean_seconds_after_first_two=float(np.mean([r['seconds'] for r in records[2:]]))))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'synthetic-final-parameters.npz', **arrays)
    report = dict(format='flm-mac-cpu-probe-v1', identity_sha256=digest(identity_path),
                  probe_source_sha256=digest(Path(__file__)), platform=platform.platform(),
                  machine=platform.machine(), python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                  threads=torch.get_num_threads(), torch_build=torch.__config__.show(),
                  conditions_checked=len(checks), bindings=checks, timings=timings,
                  parameter_file_sha256=digest(args.output/'synthetic-final-parameters.npz'),
                  scope='All registered initialization bindings; two disposable ten-update synthetic-token CPU probes. '
                        'No corpus, validation or test payload opened. No official checkpoint or language-quality result.')
    (args.output/'probe.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:report[k] for k in ('platform','machine','torch','numpy','conditions_checked')}))
    print('Binding mismatches:', sum(bool(r['differing_binding_fields']) for r in checks))
    print('Measured seconds/update:', [r['mean_seconds_after_first_two'] for r in timings])


if __name__ == '__main__':
    main()
