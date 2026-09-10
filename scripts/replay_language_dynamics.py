"""Replay the pulse diagnostic from exported recurrent parameters only.

Run this script inside the extracted dynamics archive. It does not need the
training repository, original full checkpoints, raw text, or network access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from flm.inference import bundle_path
from flm.language_dynamics import CORE_GROUPS, SWAPS, core_hash, pulse_directions
from flm.language_dynamics_study import (FOLDER, GRAPH, LEXICON, SOURCES, INPUTS,
    conditions, measure, read, summary_record, verified_case)
from flm.language_train import construct
from flm.provenance import sha256
from flm.tokenizer import Lexicon


def verify_bundle(root):
    manifest = read(root / 'manifest.json')
    if manifest.get('format') != 'flm-language-dynamics-v1':
        raise ValueError('Unsupported dynamics archive')
    required = set(SOURCES) | set(INPUTS) | {
        'scripts/replay_language_dynamics.py', 'reports/language-dynamics/identity.json',
        'reports/language-dynamics/pulses.npz', 'reports/language-dynamics/replay.json',
        'reports/language-dynamics/summary.json'}
    required |= {f'{FOLDER.as_posix()}/case-{c["label"]}.{suffix}'
                 for c in conditions() for suffix in ('json', 'npz')}
    if not required.issubset(manifest['files']):
        raise ValueError('Dynamics archive is missing required files')
    for name, record in manifest['files'].items():
        path = bundle_path(root, name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Archive file changed: ' + name)
    identity = read(root / FOLDER / 'identity.json')
    identity_hash = sha256(root / FOLDER / 'identity.json')
    if (identity_hash != manifest['study_identity_sha256'] or
            identity['conditions'] != conditions() or set(identity['sources']) != set(SOURCES)):
        raise ValueError('Dynamics declaration changed')
    for group in ('sources', 'inputs'):
        for name, expected in identity[group].items():
            if sha256(bundle_path(root, name)) != expected:
                raise ValueError('Frozen file changed: ' + name)
    if identity['torch'] != str(torch.__version__) or identity['numpy'] != str(np.__version__):
        raise ValueError('Exact replay requires the recorded PyTorch and NumPy versions')
    if sha256(root / FOLDER / 'pulses.npz') != identity['pulse_sha256']:
        raise ValueError('Frozen pulse directions changed')
    cases = [verified_case(root, c, identity_hash) for c in conditions()]
    summary = read(root / FOLDER / 'summary.json')
    if summary != summary_record(identity, identity_hash, cases):
        raise ValueError('Summary differs from case records')
    replay = read(root / FOLDER / 'replay.json')
    hashes = {f'case-{c["label"]}.{s}': sha256(root / FOLDER / f'case-{c["label"]}.{s}')
              for c in conditions() for s in ('json', 'npz')}
    if (replay['study_identity_sha256'] != identity_hash or replay['conditions'] != 7 or
            replay['exact_metric_and_array_replay'] is not True or
            replay['summary_sha256'] != sha256(root / FOLDER / 'summary.json') or
            replay['case_artifact_sha256'] != hashes):
        raise ValueError('Replay record no longer binds this archive')
    return identity, cases


def restore_cores(root, identity, cases):
    vocabulary = Lexicon(root / LEXICON).vocabulary
    models = {}
    for case in cases:
        condition = case['condition']
        model = construct('flm', root / GRAPH, vocabulary, condition['seed'] or 42).eval()
        with np.load(root / FOLDER / case['arrays'], allow_pickle=False) as arrays, torch.no_grad():
            for name in CORE_GROUPS:
                source = arrays['parameter_' + name]
                target = model.get_parameter(name)
                if source.dtype != np.float32 or source.shape != tuple(target.shape) or not np.isfinite(source).all():
                    raise ValueError('Invalid recurrent parameter: ' + name)
                target.copy_(torch.from_numpy(source.copy()))
        if core_hash(model) != case['metrics']['core_sha256']:
            raise ValueError('Restored recurrent core differs: ' + condition['label'])
        models[condition['label']] = model
    initial = models['initial_shared']
    if core_hash(initial) != identity['shared_initial_core_sha256']:
        raise ValueError('Shared initial core changed')
    # Check declared swaps against the included trained and initial tensors.
    for seed in (42, 43):
        trained = models[f'trained-s{seed}']
        if core_hash(trained) != identity['trained_core_sha256'][str(seed)]:
            raise ValueError('Trained recurrent core changed')
        for mode in ('edges_gain_only', 'time_constants_only'):
            for name in CORE_GROUPS:
                expected = (trained if name in SWAPS[mode] else initial).get_parameter(name)
                if not torch.equal(models[f'{mode}-s{seed}'].get_parameter(name), expected):
                    raise ValueError('Acute swap differs from its declared source group')
    return models


def replay(root):
    torch.set_num_threads(1)
    identity, cases = verify_bundle(root)
    models = restore_cores(root, identity, cases)
    with np.load(root / FOLDER / 'pulses.npz', allow_pickle=False) as archive:
        directions = archive['directions'].copy()
    np.testing.assert_array_equal(directions, pulse_directions(1024))
    with np.load(root / GRAPH, allow_pickle=False) as graph:
        body_ids = graph['body_ids'].copy()
    for case in cases:
        label = case['condition']['label']
        metrics, arrays = measure(models[label], directions)
        arrays['body_ids'] = body_ids
        if metrics != case['metrics']:
            raise ValueError('Recomputed metrics differ: ' + label)
        with np.load(root / FOLDER / case['arrays'], allow_pickle=False) as expected:
            if (set(expected.files) != set(arrays) or
                    any(expected[name].dtype != value.dtype or not np.array_equal(expected[name], value)
                        for name, value in arrays.items())):
                raise ValueError('Recomputed arrays differ: ' + label)
        print(json.dumps(dict(condition=label, exact_metrics_and_arrays=True)), flush=True)
    return dict(conditions=7, exact_metrics_and_arrays=True, threads=1,
                study_identity_sha256=sha256(root / FOLDER / 'identity.json'),
                imported_measure_source=str(Path(sys.modules[measure.__module__].__file__).resolve()),
                original_full_checkpoints_needed=False, corpus_tokens_read=False,
                torch=str(torch.__version__), numpy=str(np.__version__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional fresh replay result JSON')
    args = parser.parse_args()
    result = replay(ROOT)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))
