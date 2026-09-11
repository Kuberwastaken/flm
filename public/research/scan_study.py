"""Freeze, fit and audit all 36 instruction-transfer conditions before testing.

No test corpus loader or model generation is called here. An explicit protocol,
measured costs and completed priority studies are required for official use.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256, write_json
from .scan import SPLITS
from .scan_conditions import conditions, prepare_condition
from .scan_pilot import Pilot, pilot_conditions
from .scan_train import (Settings, declaration, fit, inventory, optimizer_for,
                         restore_payload, sampling_state, training_lease)

STUDY = Path('runs/scan-study-v1')
REPORTS = Path('reports/scan-runtime')
IDENTITY = REPORTS/'study-identity.json'
SELECTION = REPORTS/'study-selection.json'
PROTOCOL = Path('docs/SCAN-PROTOCOL.md')
PILOT = REPORTS/'timing-pilot.json'
SOURCES = ('scan_study.py', 'scan_evaluate.py', 'scan_test_inputs.py', 'scan_conditions.py', 'scan_train.py', 'scan_runtime.py',
    'scan_task.py', 'scan_inputs.py', 'scan_pilot.py', 'scan.py', 'language_train.py',
    'model.py', 'baselines.py', 'graph.py', 'inference.py', 'tokenizer.py',
    'provenance.py', 'train.py', 'corpus_cache.py')
PILOT_SOURCES = ('scan_pilot.py', 'scan_conditions.py', 'scan_train.py', 'scan_runtime.py', 'scan_inputs.py')
POLICY = dict(maximum_tokens=49, batch_size=32, context_limit=96, threads=4,
    decoding='Greedy argmax over the entire original vocabulary; independent EOS or cap stop',
    checkpoint='Fixed terminal update, with no validation or test checkpoint selection',
    primary='Exact complete action sequence and EOS',
    secondary=['edit distance', 'invalid token rate', 'empty or invalid action sequence rate',
               'cap exhaustion', 'reference action length'],
    scope='Three separate official partitions; no merged training corpus or grammar repair')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def immutable(path, value):
    if path.exists():
        if read(path) != value: raise ValueError('Refusing to replace a frozen SCAN record: '+str(path))
    else:
        write_json(path, value)


def prerequisites(root):
    """Check prior study records; the operator must also check live processes."""
    root = Path(root)
    for scale in ('10m', '100m'):
        for variant in ('flm', 'gru', 'transformer'):
            for seed in (42, 43):
                folder = root/f'runs/babylm-{scale}/{variant}-s{seed}'
                if not (folder/'complete.json').is_file():
                    raise ValueError('Finish the fixed BabyLM queue before instruction-study work')
                complete = read(folder/'complete.json')
                if complete['steps'] != 12000 or complete['best_checkpoint_sha256'] != sha256(folder/'best.pt'):
                    raise ValueError('Priority BabyLM completion identity changed')
    # Selection has explicit priority over additional language-transfer fitting.
    prior = root/'reports/selection-language'
    required = [prior/'study-identity.json', prior/'study-selection.json', prior/'test-summary.json',
                root/'runs/selection-language-evaluation-v1/identity.json']
    if any(not path.is_file() for path in required):
        raise ValueError('Complete the chosen neuron-selection comparison before the instruction study')
    identity, selected, summary, evaluated = map(read, required)
    labels = [row['condition']['label'] for row in identity['conditions']]
    if (not labels or len(set(labels)) != len(labels)
            or [row['condition']['label'] for row in selected['conditions']] != labels
            or set(summary['result_sha256']) != set(labels)
            or selected['study_identity_sha256'] != sha256(required[0])
            or evaluated['study_identity_sha256'] != sha256(required[0])
            or evaluated['selection_sha256'] != sha256(required[1])
            or summary['evaluation_identity_sha256'] != sha256(required[3])):
        raise ValueError('Priority selection inventory or evaluation identity changed')
    for row in selected['conditions']:
        label = row['condition']['label']
        folder = root/'runs/selection-language-v1'/label
        for path, expected in ((folder/'complete.json', row['complete_sha256']),
                (root/row['checkpoint'], row['checkpoint_sha256']),
                (folder/'validation-selection.json', row['validation_selection_sha256']),
                (prior/'test'/label/'result.json', summary['result_sha256'][label])):
            if sha256(path) != expected: raise ValueError('Priority selection result changed')
    return {path.relative_to(root).as_posix(): sha256(path) for path in required}


def verify_pilot(root):
    path = root/PILOT
    if not path.is_file(): raise ValueError('Measure the complete SCAN train-only pilot before choosing a budget')
    pilot = read(path); rows = pilot['conditions']; expected = pilot_conditions()
    if (len(rows) != len(expected) or [row['binding']['condition'] for row in rows] != expected
            or pilot['test_partitions_opened'] is not False or pilot['benchmark_budget_selected'] is not False
            or set(pilot['source_sha256']) != set(PILOT_SOURCES)
            or pilot['torch'] != str(torch.__version__) or pilot['numpy'] != str(np.__version__)):
        raise ValueError('Incomplete or changed SCAN timing inventory')
    for name, digest in pilot['source_sha256'].items():
        if sha256(root/'flm'/name) != digest: raise ValueError('Measured SCAN pilot source changed')
    for row in rows:
        settings = Pilot().settings(row['binding']['condition']['seed'])
        observations = row['observations']
        if (row['settings'] != asdict(settings) or row['pilot'] != asdict(Pilot())
                or row['original_state_unchanged'] is not True or row['checkpoint_written'] is not False
                or row['model_predictions_scored'] is not False or len(observations) != 15
                or [entry['step'] for entry in observations] != list(range(1, 16))
                or [entry['warmup'] for entry in observations] != [True]*3+[False]*12
                or any(type(entry['seconds']) not in (float, int) or not math.isfinite(entry['seconds'])
                       or entry['seconds'] <= 0 for entry in observations)):
            raise ValueError('Invalid SCAN pilot observations or settings')
        timed = observations[3:]
        total = math.fsum(entry['seconds'] for entry in timed)
        if (not math.isclose(row['measured_seconds'], total, rel_tol=1e-12)
                or not math.isclose(row['median_update_seconds'], float(np.median([e['seconds'] for e in timed])), rel_tol=1e-12)
                or not math.isclose(row['measured_updates_per_second'], 12/total, rel_tol=1e-12)):
            raise ValueError('SCAN pilot time arithmetic changed')
        for key in ('examples', 'input_tokens', 'supervised_tokens'):
            if any(type(entry[key]) is not int or entry[key] <= 0 for entry in observations):
                raise ValueError('Invalid SCAN pilot exposure')
            if row['measured_'+key] != sum(entry[key] for entry in timed):
                raise ValueError('SCAN pilot exposure total changed')
    return pilot


def test_metadata(root):
    """Bind existing dataset-card entries, without opening test text or targets."""
    path = root/'data/cards/scan.json'; card = read(path)
    return dict(card_sha256=sha256(path), revision=card['revision'],
        partitions={split: dict(source=SPLITS[split][1],
            source_identity=card['source_files'][SPLITS[split][1]],
            processed_path=f'data/processed/scan/{split}/test.jsonl',
            record=card['partitions'][split]['test']) for split in SPLITS}, payloads_opened=False)


def identity_for(root, settings, pilot, priority):
    settings.validate()
    if settings.sampling_seed != 42: raise ValueError('Use sampling seed 42 in the template; seed 43 is also registered')
    if not (root/PROTOCOL).is_file() or not (root/PROTOCOL).read_text(encoding='utf8').strip():
        raise ValueError('Write the explicit SCAN protocol before freezing the study')
    measured = {row['binding']['condition']['label']: row for row in pilot['conditions']}
    rows = []
    for condition in conditions():
        prepared = prepare_condition(root, condition)
        local_settings = replace(settings, sampling_seed=condition['seed'])
        binding = prepared.training_binding(local_settings)
        declared, lengths, scored = declaration(prepared.model, prepared.records, prepared.lexicon, local_settings, binding)
        pilot_condition = dict(condition, seed=42, label=condition['label'].rsplit('-s', 1)[0]+'-s42')
        observation = measured[pilot_condition['label']]
        for key in ('batch_size', 'threads', 'context_limit'):
            if asdict(local_settings)[key] != observation['settings'][key]:
                raise ValueError('SCAN training dimensions differ from the measured pilot')
        if local_settings.context_limit != POLICY['context_limit']:
            raise ValueError('All instruction models must use the declared common context')
        if condition['seed'] == 42:
            if (observation['binding'] != binding or observation['original_state_sha256'] != declared['initial_state_sha256']
                    or observation['dtype'] != str(next(prepared.model.parameters()).dtype)):
                raise ValueError('SCAN pilot used different training inputs or original tensors')
            pilot_settings = Settings(**observation['settings'])
            sampler = np.random.Generator(np.random.PCG64(pilot_settings.sampling_seed))
            for entry in observation['observations']:
                indices = sampler.integers(0, len(lengths), size=pilot_settings.batch_size, dtype=np.int64)
                expected = dict(examples=pilot_settings.batch_size, input_tokens=int(lengths[indices].sum()),
                                supervised_tokens=int(scored[indices].sum()))
                if any(entry[key] != value for key, value in expected.items()):
                    raise ValueError('SCAN pilot per-update exposure does not replay')
            _, total, digest = sampling_state(pilot_settings, lengths, scored, 15)
            _, warmup, _ = sampling_state(pilot_settings, lengths, scored, 3)
            if digest != observation['sampled_row_sha256']:
                raise ValueError('SCAN pilot sampled row stream changed')
            for key in total:
                if total[key]-warmup[key] != observation['measured_'+key]:
                    raise ValueError('SCAN pilot exposure does not replay on its training rows')
        _, exposure, digest = sampling_state(local_settings, lengths, scored, settings.steps)
        rows.append(dict(condition=condition, base_declaration=declared,
            expected_exposure=exposure, sampled_row_sha256=digest))
    for split in SPLITS:
        for seed in (42, 43):
            group = [row for row in rows if row['condition']['split'] == split and row['condition']['seed'] == seed]
            for key in ('expected_exposure', 'sampled_row_sha256'):
                if any(row[key] != group[0][key] for row in group):
                    raise ValueError('Instruction conditions have unmatched sampled training exposure')
    return dict(format='flm-scan-study-v1', settings=asdict(settings), conditions=rows,
        protocol_sha256=sha256(root/PROTOCOL), pilot_sha256=sha256(root/PILOT), priority_studies=priority,
        sources={name:sha256(root/'flm'/name) for name in SOURCES},
        torch=str(torch.__version__), numpy=str(np.__version__),
        test_metadata=test_metadata(root), evaluation_policy=POLICY,
        selection='Every fixed terminal checkpoint frozen together; no validation or test selection',
        test_gate='All 36 completed payloads and their exact exposure must pass before official test access')


def initialize(root, settings):
    root = Path(root); priority = prerequisites(root); pilot = verify_pilot(root)
    with training_lease(root/STUDY):
        if not (root/IDENTITY).exists() and any((root/STUDY/row['label']).exists() for row in conditions()):
            raise ValueError('Cannot declare SCAN after condition directories already exist')
        identity = identity_for(root, settings, pilot, priority)
        immutable(root/IDENTITY, identity)
    return identity


def verified_context(root):
    if not (root/IDENTITY).is_file(): raise ValueError('Freeze the complete SCAN identity first')
    identity = read(root/IDENTITY)
    if identity.get('format') != 'flm-scan-study-v1': raise ValueError('Unknown SCAN study format')
    expected = identity_for(root, Settings(**identity['settings']), verify_pilot(root), prerequisites(root))
    if identity != expected: raise ValueError('Frozen SCAN study identity changed')
    return identity


def unchanged(root, identity, identity_hash):
    if sha256(root/IDENTITY) != identity_hash: raise ValueError('SCAN identity changed during execution')
    files = {str(PROTOCOL):identity['protocol_sha256'], str(PILOT):identity['pilot_sha256'],
             'data/cards/scan.json':identity['test_metadata']['card_sha256']}
    files.update({str(Path('flm')/name):digest for name, digest in identity['sources'].items()})
    files.update(identity['priority_studies'])
    for name, digest in files.items():
        if sha256(root/name) != digest: raise ValueError('Frozen SCAN source or prerequisite changed: '+name)


def prepared_run(root, row, identity_hash):
    prepared = prepare_condition(root, row['condition'])
    settings = Settings(**row['base_declaration']['settings'])
    binding = prepared.training_binding(settings)
    declared, lengths, scored = declaration(prepared.model, prepared.records, prepared.lexicon, settings, binding)
    if declared != row['base_declaration']: raise ValueError('SCAN prepared run differs from its frozen declaration')
    binding = dict(binding, study_identity_sha256=identity_hash)
    return prepared, settings, binding, lengths, scored


def train_study(root):
    root = Path(root)
    with training_lease(root/STUDY):
        identity = verified_context(root); identity_hash = sha256(root/IDENTITY)
        attempt = root/STUDY/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        for row in identity['conditions']:
            condition = row['condition']
            try:
                unchanged(root, identity, identity_hash)
                prepared, settings, binding, _, _ = prepared_run(root, row, identity_hash)
                print('Training registered instruction condition: '+condition['label'], flush=True)
                result = fit(prepared.model, prepared.records, prepared.lexicon, settings, binding,
                             root/STUDY/condition['label'])
                if result['complete'] is not True or result['exposure'] != row['expected_exposure']:
                    raise ValueError('Instruction fit did not complete the registered exposure')
                unchanged(root, identity, identity_hash)
            except Exception as error:
                write_json(attempt/'failure.json', dict(condition=condition, study_identity_sha256=identity_hash,
                    error_type=type(error).__name__, error=str(error)))
                raise
            write_json(attempt/(condition['label'].replace('/', '--')+'.json'), result)
        if verified_context(root) != identity: raise ValueError('SCAN context changed during training')


def audit_completed(root, row, identity_hash):
    """Restore the terminal weights, optimizer and sampler; perform no inference."""
    prepared, settings, binding, lengths, scored = prepared_run(root, row, identity_hash)
    declared, _, _ = declaration(prepared.model, prepared.records, prepared.lexicon, settings, binding)
    directory = root/STUDY/row['condition']['label']; complete = read(directory/'complete.json')
    saved = inventory(directory, declared)
    if len(saved) != settings.steps//settings.checkpoint_interval or saved[-1]['step'] != settings.steps:
        raise ValueError('Incomplete SCAN checkpoint inventory')
    last = saved[-1]
    # The shared restoration helper also checks graph buffers, full histories,
    # optimizer moments, sampler state and exact row/token exposure.
    previous_rng = torch.get_rng_state()
    try:
        payload = restore_payload(directory/last['checkpoint'], prepared.model,
            optimizer_for(prepared.model, settings), declared, lengths, scored)
    finally:
        torch.set_rng_state(previous_rng)
    if any(bool((state['exp_avg_sq'] < 0).any()) for state in payload['optimizer']['state'].values()):
        raise ValueError('SCAN optimizer second moments cannot be negative')
    expected = dict(declaration=declared, step=settings.steps, checkpoint=last['checkpoint'],
        checkpoint_sha256=last['checkpoint_sha256'], exposure=payload['exposure'],
        selection='Fixed terminal update; no validation or test selection', complete=True)
    if (complete != expected or payload['step'] != settings.steps
            or payload['exposure'] != row['expected_exposure']
            or payload['sampled_row_sha256'] != row['sampled_row_sha256']):
        raise ValueError('Completed SCAN payload or exposure differs from its declaration')
    return dict(condition=row['condition'], checkpoint=(STUDY/row['condition']['label']/last['checkpoint']).as_posix(),
        checkpoint_sha256=last['checkpoint_sha256'], checkpoint_step=settings.steps,
        complete_sha256=sha256(directory/'complete.json'), final_exposure=payload['exposure'],
        sampled_row_sha256=payload['sampled_row_sha256'],
        saved_record_sha256={f"saved-{record['step']:06d}.json":sha256(directory/f"saved-{record['step']:06d}.json") for record in saved})


def freeze_selection(root):
    root = Path(root)
    if not (root/IDENTITY).is_file(): raise ValueError('No frozen SCAN study identity')
    with training_lease(root/STUDY):
        # Reconstruct the inventory from code; never trust a shortened JSON list.
        for condition in conditions():
            if not (root/STUDY/condition['label']/'complete.json').is_file():
                raise ValueError('Registered SCAN condition incomplete: '+condition['label'])
        identity = verified_context(root); identity_hash = sha256(root/IDENTITY); rows = []
        for row in identity['conditions']:
            unchanged(root, identity, identity_hash)
            rows.append(audit_completed(root, row, identity_hash))
        unchanged(root, identity, identity_hash)
        for row in rows:
            directory = root/STUDY/row['condition']['label']
            for path, digest in ((root/row['checkpoint'], row['checkpoint_sha256']),
                                 (directory/'complete.json', row['complete_sha256'])):
                if sha256(path) != digest: raise ValueError('SCAN completion changed during the whole-inventory audit')
            for name, digest in row['saved_record_sha256'].items():
                if sha256(directory/name) != digest: raise ValueError('SCAN saved record changed during audit')
        result = dict(study_identity_sha256=identity_hash, conditions=rows, evaluation_policy=POLICY,
            test_metadata=identity['test_metadata'], test_payloads_opened=False, inference_performed=False,
            selection='All 36 verified fixed terminal checkpoints; no accuracy or transfer result')
        immutable(root/SELECTION, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'train', 'select'))
    parser.add_argument('--settings', type=Path, help='Required JSON Settings for initialization; no implicit budget')
    args = parser.parse_args(); root = Path.cwd()
    if args.action == 'initialize':
        if args.settings is None: parser.error('initialize requires --settings')
        initialize(root, Settings(**read(args.settings)))
    else:
        if args.settings is not None: parser.error('--settings is only valid for initialize')
        (train_study if args.action == 'train' else freeze_selection)(root)


if __name__ == '__main__': main()
