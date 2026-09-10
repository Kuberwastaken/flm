"""Freeze, measure and replay the declared language-core dynamics diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .inference import RUNTIME_FILES, state_hash
from .language_dynamics import (CORE_GROUPS, SWAPS, SNAPSHOT_TIMES, core_hash, intervene,
    parameter_summary, pulse_directions, pulse_response, spectrum)
from .language_train import construct, restore
from .provenance import sha256, write_json
from .tokenizer import Lexicon


FOLDER = Path('reports/language-dynamics')
GRAPH = Path('data/graphs/central-1024/graph.npz')
LEXICON = Path('data/tokenizers/wikitext2-4096/tokenizer.json')
PROTOCOL = 'docs/LANGUAGE-DYNAMICS-PROTOCOL.md'
SOURCES = (*RUNTIME_FILES, 'flm/language_dynamics.py', 'flm/language_dynamics_study.py')
INPUTS = (PROTOCOL, GRAPH.as_posix(), LEXICON.as_posix(),
          GRAPH.with_name('graph-card.json').as_posix(),
          'reports/language-core/identity.json', 'reports/language-topology/selection.json',
          'reports/language-dynamics/software-preflight.json')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def conditions():
    return [dict(label='initial_shared', seed=None, mode='initial')] + [
        dict(label=f'{mode}-s{seed}', seed=seed, mode=mode) for seed in (42, 43) for mode in SWAPS]


def source_models(root, references):
    """Restore exact completed references without any corpus-cache access."""
    lexicon = Lexicon(root / LEXICON)
    initial = {seed: construct('flm', root / GRAPH, lexicon.vocabulary, seed) for seed in (42, 43)}
    if core_hash(initial[42]) != core_hash(initial[43]):
        raise ValueError('The declared shared initial dynamics differ across seeds')
    trained = {}
    for seed in (42, 43):
        reference = references[str(seed)]
        model, saved = restore(root / reference['checkpoint'], root / GRAPH, lexicon)
        if (saved['_file_sha256'] != reference['checkpoint_sha256'] or
                saved['step'] != reference['checkpoint_step'] or saved['run']['seed'] != seed or
                asdict(model.config) != asdict(initial[seed].config)):
            raise ValueError('A completed dynamics source checkpoint changed')
        expected = initial[seed].state_dict(); values = model.state_dict()
        if values.keys() != expected.keys():
            raise ValueError('Source model tensor inventory changed')
        for name, value in values.items():
            if value.dtype != expected[name].dtype or value.shape != expected[name].shape or not torch.isfinite(value).all():
                raise ValueError('Invalid source model tensor: ' + name)
        for name, value in initial[seed].named_buffers():
            if not torch.equal(values[name], value):
                raise ValueError('Source graph/pooling buffer changed: ' + name)
        trained[seed] = model.eval()
    return initial, trained


def verify_identity(root, identity):
    if (identity['conditions'] != conditions() or identity['torch'] != str(torch.__version__) or
            identity['numpy'] != str(np.__version__) or
            identity['schedule'] != dict(amplitudes=[.001, 1.], directions=8, seed=91011, silence=256,
                                         snapshots=list(SNAPSHOT_TIMES), threads=1, nonlinear_dtype='float32', linear_dtype='float64')):
        raise ValueError('Dynamics diagnostic declaration or environment changed')
    if set(identity['sources']) != set(SOURCES) or set(identity['inputs']) != set(INPUTS):
        raise ValueError('Dynamics source/input inventory changed')
    for group in ('sources', 'inputs'):
        for name, expected in identity[group].items():
            if sha256(root / name) != expected:
                raise ValueError('Frozen dynamics ' + group + ' changed: ' + name)
    if sha256(root / FOLDER / 'pulses.npz') != identity['pulse_sha256']:
        raise ValueError('Frozen dynamics pulses changed')
    upstream = read(root / 'reports/language-core/identity.json')
    for seed, reference in identity['references'].items():
        if (reference['checkpoint_sha256'] != upstream['reference_checkpoints']['full-s' + seed] or
                sha256(root / reference['checkpoint']) != reference['checkpoint_sha256']):
            raise ValueError('Frozen dynamics reference changed')


def prepare(root):
    path = root / FOLDER / 'identity.json'
    if path.exists():
        result = read(path); verify_identity(root, result); return result
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all', '--', *SOURCES, *INPUTS], cwd=root, text=True)
    if changed.strip():
        raise ValueError('Commit diagnostic sources, protocol and fixture evidence before freezing')
    upstream = read(root / 'reports/language-core/identity.json')
    selection = read(root / 'reports/language-topology/selection.json')
    references = {str(row['seed']): row for row in selection['runs'] if row['reference']}
    if set(references) != {'42', '43'} or any(row['label'] != 'measured-s' + seed for seed, row in references.items()):
        raise ValueError('The two original measured references are required')
    for seed, reference in references.items():
        if reference['checkpoint_sha256'] != upstream['reference_checkpoints']['full-s' + seed]:
            raise ValueError('Upstream selected reference differs')
    for name in (GRAPH.as_posix(), LEXICON.as_posix()):
        if sha256(root / name) != upstream['inputs'][name]:
            raise ValueError('Original measured graph/tokenizer changed')
    torch.set_num_threads(1)
    initial, trained = source_models(root, references)
    if initial[42].config.neurons != 1024 or initial[42].config.vocabulary != 4096:
        raise ValueError('Use the declared 1024-neuron, 4096-token language model')
    folder = root / FOLDER; folder.mkdir(parents=True, exist_ok=True)
    pulse_path = folder / 'pulses.npz'
    np.savez_compressed(pulse_path, directions=pulse_directions(1024))
    identity = dict(study='Language-trained recurrent dynamics diagnostic v1',
        declared_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        sources={name: sha256(root / name) for name in SOURCES}, inputs={name: sha256(root / name) for name in INPUTS},
        torch=str(torch.__version__), numpy=str(np.__version__), references=references, conditions=conditions(),
        schedule=dict(amplitudes=[.001, 1.], directions=8, seed=91011, silence=256,
                      snapshots=list(SNAPSHOT_TIMES), threads=1, nonlinear_dtype='float32', linear_dtype='float64'),
        pulse_sha256=sha256(pulse_path), shared_initial_core_sha256=core_hash(initial[42]),
        initial_dynamics_match_across_seeds=True,
        trained_core_sha256={str(seed): core_hash(model) for seed, model in trained.items()},
        source_full_state_sha256={str(seed): state_hash(model) for seed, model in trained.items()},
        corpus_tokens_read=False, scope='Direct-drive core diagnostic; no language scoring, fitting, or behavioral transfer')
    write_json(path, identity)
    verify_identity(root, identity)
    return identity


def measure(model, directions):
    before = state_hash(model); rng = torch.get_rng_state().clone()
    constants = model.constants()
    spectral, eigenvalues = spectrum(constants)
    arrays = dict(fast_eigenvalues=eigenvalues)
    arrays.update({'parameter_' + name: model.get_parameter(name).detach().cpu().numpy().copy() for name in CORE_GROUPS})
    pulses = []
    for index, amplitude in enumerate((.001, 1.)):
        response = pulse_response(model, directions, amplitude)
        for name in ('curves', 'normalized_linear_deviation', 'snapshot_times', 'snapshots'):
            arrays[f'amp_{index}_{name}'] = response[name]
        pulses.append(dict(amplitude=amplitude, directions=response['summary']))
    if before != state_hash(model) or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError('Dynamics measurement changed source tensors or RNG')
    return dict(core_sha256=core_hash(model), parameters=parameter_summary(model), spectrum=spectral, pulses=pulses), arrays


def verified_case(root, condition, identity_hash):
    path = root / FOLDER / ('case-' + condition['label'] + '.json')
    record = read(path)
    if record['condition'] != condition or record['study_identity_sha256'] != identity_hash:
        raise ValueError('Dynamics case identity changed')
    expected = 'case-' + condition['label'] + '.npz'
    if record['arrays'] != expected or sha256(root / FOLDER / expected) != record['arrays_sha256']:
        raise ValueError('Dynamics case array payload changed')
    return record


def summary_record(identity, identity_hash, records):
    return dict(study=identity['study'], study_identity_sha256=identity_hash, cases=records,
                scope=identity['scope'], initial_is_shared=True, new_language_test_scores=False)


def case_hashes(root):
    return {f'case-{condition["label"]}.{suffix}': sha256(root / FOLDER / f'case-{condition["label"]}.{suffix}')
            for condition in conditions() for suffix in ('json', 'npz')}


def verified_results(root):
    identity = read(root / FOLDER / 'identity.json'); verify_identity(root, identity)
    identity_hash = sha256(root / FOLDER / 'identity.json')
    records = [verified_case(root, condition, identity_hash) for condition in conditions()]
    summary = read(root / FOLDER / 'summary.json')
    if summary != summary_record(identity, identity_hash, records):
        raise ValueError('Dynamics summary differs from its condition records')
    replay = read(root / FOLDER / 'replay.json')
    if (replay['study_identity_sha256'] != identity_hash or replay['conditions'] != 7 or
            replay['exact_metric_and_array_replay'] is not True or
            replay['summary_sha256'] != sha256(root / FOLDER / 'summary.json') or
            replay['case_artifact_sha256'] != case_hashes(root)):
        raise ValueError('Dynamics replay does not bind the current result artifacts')
    return summary


def run(root, replay=False):
    identity = read(root / FOLDER / 'identity.json'); verify_identity(root, identity)
    identity_hash = sha256(root / FOLDER / 'identity.json')
    torch.set_num_threads(1)
    initial, trained = source_models(root, identity['references'])
    if (core_hash(initial[42]) != identity['shared_initial_core_sha256'] or
            any(core_hash(model) != identity['trained_core_sha256'][str(seed)] or
                state_hash(model) != identity['source_full_state_sha256'][str(seed)] for seed, model in trained.items())):
        raise ValueError('Source dynamics tensors changed after declaration')
    with np.load(root / FOLDER / 'pulses.npz', allow_pickle=False) as source:
        directions = source['directions'].copy()
    np.testing.assert_array_equal(directions, pulse_directions(1024))
    with np.load(root / GRAPH, allow_pickle=False) as source:
        body_ids = source['body_ids'].copy()
    records = []
    for condition in conditions():
        verify_identity(root, identity)
        marker = root / FOLDER / ('case-' + condition['label'] + '.json')
        if marker.exists() and not replay:
            records.append(verified_case(root, condition, identity_hash)); continue
        previous = verified_case(root, condition, identity_hash) if replay else None
        model = initial[42] if condition['seed'] is None else intervene(initial[condition['seed']], trained[condition['seed']], condition['mode'])
        metrics, arrays = measure(model, directions)
        arrays['body_ids'] = body_ids
        if previous:
            if previous['metrics'] != metrics:
                raise ValueError('Replayed dynamics metrics differ: ' + condition['label'])
            with np.load(root / FOLDER / previous['arrays'], allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[name], value) for name, value in arrays.items()):
                    raise ValueError('Replayed dynamics arrays differ: ' + condition['label'])
            record = previous
        else:
            name = 'case-' + condition['label'] + '.npz'
            np.savez_compressed(root / FOLDER / name, **arrays)
            verify_identity(root, identity)
            record = dict(condition=condition, study_identity_sha256=identity_hash, metrics=metrics,
                          arrays=name, arrays_sha256=sha256(root / FOLDER / name))
            write_json(marker, record)
        records.append(record)
        print(json.dumps(dict(condition=condition['label'], replayed=replay, spectrum=metrics['spectrum'])), flush=True)
    verify_identity(root, identity)
    summary = summary_record(identity, identity_hash, records)
    if replay:
        if read(root / FOLDER / 'summary.json') != summary:
            raise ValueError('Dynamics summary differs from its replayed cases')
        report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), study_identity_sha256=identity_hash,
                      conditions=len(records), exact_metric_and_array_replay=True, threads=1,
                      summary_sha256=sha256(root / FOLDER / 'summary.json'), case_artifact_sha256=case_hashes(root),
                      scope='Same-environment replay of this dynamics diagnostic; not independent training or language evaluation')
        write_json(root / FOLDER / 'replay.json', report)
        verified_results(root)
    else:
        report = summary
        write_json(root / FOLDER / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify'))
    args = parser.parse_args(); root = Path('.').resolve()
    result = prepare(root) if args.action == 'prepare' else run(root, replay=args.action == 'verify')
    print(json.dumps(dict(action=args.action, study_identity_sha256=sha256(root / FOLDER / 'identity.json'),
                          cases=len(result.get('cases', []))), indent=2))


if __name__ == '__main__':
    main()
