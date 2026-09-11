"""Freeze, fit and select the complete eight-condition learning-rule study.

Initialization requires measured pilot costs and an explicit protocol/budget.
This module never opens test payloads or calculates official test likelihoods.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from .language_learning_inputs import conditions, load_inputs, prepare_condition
from .language_learning_pilot import Pilot, prerequisites
from .language_learning_train import Settings, declaration, fit, replay, training_lease
from .language_learning_validation import load_panel, select_checkpoint
from .provenance import sha256, write_json

STUDY = Path('runs/language-learning-v1')
REPORTS = Path('reports/language-eligibility')
PROTOCOL = Path('docs/LANGUAGE-LEARNING-PROTOCOL.md')
PILOT = REPORTS/'timing-pilot.json'
IDENTITY = REPORTS/'study-identity.json'
SELECTION = REPORTS/'study-selection.json'
SOURCES = ('language_learning_study.py', 'language_learning_inputs.py', 'language_learning_pilot.py',
    'language_learning_train.py', 'language_learning_validation.py', 'language_eligibility.py',
    'embedding_eligibility.py', 'local_learning.py', 'language_train.py', 'model.py', 'train.py',
    'baselines.py', 'graph.py', 'corpus_cache.py', 'tokenizer.py', 'inference.py', 'provenance.py', 'babylm.py',
    'language_learning_test.py', 'corpus_evaluation.py', 'babylm_test.py')
PILOT_SOURCES = ('language_learning_pilot.py', 'language_learning_inputs.py', 'language_learning_train.py',
    'language_eligibility.py', 'embedding_eligibility.py', 'local_learning.py', 'model.py', 'train.py',
    'inference.py', 'corpus_cache.py', 'tokenizer.py', 'provenance.py')


def read(path): return json.loads(path.read_text(encoding='utf8'))


def verify_pilot(root):
    path = root/PILOT
    if not path.is_file(): raise ValueError('Measure the full-window learning-rule pilot before freezing a budget')
    pilot = read(path)
    if (len(pilot['conditions']) != 8 or {row['condition']['label'] for row in pilot['conditions']} !=
            {row['label'] for row in conditions()} or pilot['validation_or_test_payloads_opened'] is not False
            or set(pilot['source_sha256']) != set(PILOT_SOURCES)):
        raise ValueError('Incomplete or invalid learning-rule timing inventory')
    for name, expected in pilot['source_sha256'].items():
        if sha256(root/'flm'/name) != expected: raise ValueError('Measured pilot source changed')
    for row in pilot['conditions']:
        if (row['condition'] not in conditions() or len(row['observations']) != 15
                or row['source_model_unchanged'] is not True or row['disposable_parameters_changed'] is not True
                or row['checkpoint_written'] is not False or row['language_scores_reported'] is not False
                or not np.isfinite(row['measured_seconds']) or row['measured_seconds'] <= 0):
            raise ValueError('Invalid disposable pilot result')
        if ([entry['step'] for entry in row['observations']] != list(range(1, 16))
                or [entry['warmup'] for entry in row['observations']] != [True]*3+[False]*12
                or any(not np.isfinite(entry['seconds']) or entry['seconds'] <= 0 for entry in row['observations'])):
            raise ValueError('Incomplete pilot update observations')
        if row['settings'] != asdict(Pilot().settings(row['condition']['seed'])):
            raise ValueError('Pilot did not measure the declared full-window settings')
        timed = row['observations'][3:]
        if not np.isclose(row['measured_seconds'], sum(entry['seconds'] for entry in timed), rtol=1e-12, atol=1e-12):
            raise ValueError('Pilot timing totals changed')
        if set(row['measured_exposure']) != {'presented_tokens', 'scored_tokens', 'presented_bytes', 'scored_bytes'}:
            raise ValueError('Incomplete pilot exposure inventory')
        for key, value in row['measured_exposure'].items():
            if value != sum(entry[key] for entry in timed): raise ValueError('Pilot exposure totals changed')
    for seed in (42, 43):
        group = [row for row in pilot['conditions'] if row['condition']['seed'] == seed]
        for key in ('sampled_windows_sha256', 'measured_windows_sha256', 'measured_exposure'):
            if any(row[key] != group[0][key] for row in group): raise ValueError('Unmatched pilot text exposure')
    return pilot


def test_metadata(root, lexicon):
    """Bind already prepared metadata; do not open any test token payload."""
    card_path = root/'data/tokenizers/babylm-2026-4096/tokenization-card.json'
    manifest_path = root/'data/processed/babylm-2026-bpe/test/manifest.json'
    card = read(card_path); manifest = read(manifest_path)
    if card['tokenizer_sha256'] != lexicon.sha256 or card['partitions']['test'] != manifest:
        raise ValueError('Prepared test metadata changed')
    return dict(manifest_sha256=sha256(manifest_path), tokenization_card_sha256=sha256(card_path),
        manifest=manifest, payloads_opened=False)


def identity_for(root, inputs, panel, settings, pilot):
    settings.validate(); panel.verify(inputs.lexicon)
    if settings.seed != 42: raise ValueError('Use seed 42 in the template; the study also registers seed 43')
    if settings.score_boundaries: raise ValueError('This learning-rule study excludes boundary targets for every condition')
    protocol = root/PROTOCOL
    if not protocol.is_file() or not protocol.read_text(encoding='utf8').strip():
        raise ValueError('Write the explicit learning-rule protocol before freezing the study')
    rows = []
    for condition in conditions():
        measured = next(row for row in pilot['conditions'] if row['condition'] == condition)
        for key in ('batch', 'sequence', 'warmup', 'threads', 'score_boundaries'):
            if measured['settings'][key] != asdict(settings)[key]:
                raise ValueError('Chosen training dimensions differ from the measured cost pilot')
        model, binding = prepare_condition(inputs, condition)
        local_settings = replace(settings, seed=condition['seed'])
        declared = declaration(model, inputs.documents, inputs.lexicon, local_settings, condition['method'], binding)
        pilot_settings = Settings(**measured['settings'])
        _, total_exposure, digest = replay(inputs.documents, inputs.lexicon.lengths, pilot_settings, pilot_settings.steps)
        warmup_steps = sum(observation['warmup'] for observation in measured['observations'])
        _, warmup_exposure, _ = replay(inputs.documents, inputs.lexicon.lengths, pilot_settings, warmup_steps)
        expected_exposure = {key:value-warmup_exposure[key] for key, value in total_exposure.items()}
        if measured['sampled_windows_sha256'] != digest.hexdigest() or measured['measured_exposure'] != expected_exposure:
            raise ValueError('Measured pilot windows do not replay on the registered training inputs')
        if (measured['initial_state_sha256'] != declared['initial_state_sha256']
                or measured['declaration']['ordered_training_documents_sha256'] != declared['ordered_training_documents_sha256']):
            raise ValueError('Cost pilot used different initial tensors or training documents')
        rows.append(dict(condition=condition, settings=asdict(local_settings), base_binding=binding,
            initial_state_sha256=declared['initial_state_sha256'],
            training_documents_sha256=declared['ordered_training_documents_sha256'],
            trainable_parameters=declared['trainable_parameters'],
            frozen_parameter_names=declared['frozen_parameter_names']))
    for seed in (42, 43):
        group = [row for row in rows if row['condition']['seed'] == seed]
        for key in ('initial_state_sha256', 'training_documents_sha256'):
            if len({row[key] for row in group}) != 1: raise ValueError('Learning rules have unmatched initial tensors or data')
    if pilot['input_binding'] != inputs.binding: raise ValueError('Training inputs differ from measured pilot')
    from .language_learning_test import POLICY
    return dict(format='flm-language-learning-study-v1', settings=asdict(settings), conditions=rows,
        input_binding=inputs.binding, validation_panel_binding=panel.binding,
        test_metadata=test_metadata(root, inputs.lexicon),
        protocol_sha256=sha256(protocol), pilot_sha256=sha256(root/PILOT),
        sources={name:sha256(root/'flm'/name) for name in SOURCES},
        torch=str(torch.__version__), numpy=str(np.__version__),
        evaluation_policy=POLICY, validation_chunk_size=96, selection='Earliest exact minimum validation BPB over every declared checkpoint',
        test_policy='All eight complete runs and validation selections must be frozen before opening test payloads',
        scope='Four training rules on the original graph; no topology, selection-method or behavior result')


def initialize(root, settings):
    root = Path(root); prerequisites(root)
    pilot = verify_pilot(root)
    if not (root/PROTOCOL).is_file(): raise ValueError('Write the explicit learning-rule protocol first')
    inputs = load_inputs(root); panel = load_panel(root, inputs.lexicon)
    identity = identity_for(root, inputs, panel, settings, pilot)
    with training_lease(root/STUDY):
        if (root/IDENTITY).exists():
            if read(root/IDENTITY) != identity: raise ValueError('A different learning-rule study is already frozen')
        else:
            if any((root/STUDY/condition['label']).exists() for condition in conditions()):
                raise ValueError('Cannot freeze a new study after condition directories already exist')
            write_json(root/IDENTITY, identity)
    return identity


def verified_context(root):
    """Reload authoritative inputs and reject changed code, protocol or costs."""
    if not (root/IDENTITY).is_file(): raise ValueError('Freeze the complete study identity before training or selection')
    identity = read(root/IDENTITY)
    if identity.get('format') != 'flm-language-learning-study-v1': raise ValueError('Unknown study format')
    inputs = load_inputs(root); panel = load_panel(root, inputs.lexicon); pilot = verify_pilot(root)
    expected = identity_for(root, inputs, panel, Settings(**identity['settings']), pilot)
    if identity != expected: raise ValueError('Frozen learning-rule study identity changed')
    return identity, inputs, panel


def prepared_run(root, inputs, row):
    condition = row['condition']; model, binding = prepare_condition(inputs, condition)
    if binding != row['base_binding']: raise ValueError('Condition input binding changed')
    return model, dict(binding, study_identity_sha256=sha256(root/IDENTITY)), Settings(**row['settings'])


def train_study(root):
    root = Path(root); prerequisites(root)
    with training_lease(root/STUDY):
        identity, inputs, _ = verified_context(root)
        identity_hash = sha256(root/IDENTITY)
        for row in identity['conditions']:
            current, inputs, _ = verified_context(root)
            if current != identity: raise ValueError('Study context changed between conditions')
            if sha256(root/IDENTITY) != identity_hash: raise ValueError('Study identity changed during training')
            for name, expected in identity['sources'].items():
                if sha256(root/'flm'/name) != expected: raise ValueError('Frozen numerical source changed during training')
            model, binding, settings = prepared_run(root, inputs, row)
            label = row['condition']['label']
            print('Training registered learning-rule condition: '+label, flush=True)
            result = fit(model, inputs.documents, inputs.lexicon, settings, row['condition']['method'],
                         binding, root/STUDY/label)
            if result['complete'] is not True: raise ValueError('Registered condition did not finish')
            current, _, _ = verified_context(root)
            if current != identity: raise ValueError('Study context changed during a condition')
            print('Completed registered learning-rule condition: '+label, flush=True)


def freeze_selection(root):
    """Audit/select every registered run; save all eight together before test access."""
    root = Path(root)
    if not (root/IDENTITY).is_file(): raise ValueError('No frozen learning-rule study identity')
    # Check whole-inventory completion before loading validation or scoring any run.
    for condition in conditions():
        if not (root/STUDY/condition['label']/'complete.json').is_file():
            raise ValueError('Registered learning-rule run incomplete: '+condition['label'])
    with training_lease(root/STUDY):
        identity, inputs, panel = verified_context(root); identity_hash = sha256(root/IDENTITY)
        records = []
        for row in identity['conditions']:
            model, binding, settings = prepared_run(root, inputs, row)
            condition = row['condition']; directory = root/STUDY/condition['label']
            selected = select_checkpoint(model, inputs.documents, inputs.lexicon, settings, condition['method'],
                binding, directory, panel, chunk_size=identity['validation_chunk_size'])
            choice = selected['selected']; complete = read(directory/'complete.json')
            records.append(dict(condition=condition, checkpoint=(STUDY/condition['label']/choice['checkpoint']).as_posix(),
                checkpoint_sha256=choice['checkpoint_sha256'], checkpoint_step=choice['step'],
                validation_bits_per_byte=choice['score']['bits_per_byte'], final_exposure=complete['exposure'],
                validation_selection_sha256=sha256(directory/'validation-selection.json'),
                complete_sha256=sha256(directory/'complete.json')))
        for seed in (42, 43):
            group = [row for row in records if row['condition']['seed'] == seed]
            if any(row['final_exposure'] != group[0]['final_exposure'] for row in group):
                raise ValueError('Completed rules have unequal training exposure')
        result = dict(study_identity_sha256=identity_hash, conditions=records,
            validation_panel_binding=panel.binding, test_payloads_opened=False,
            scope='Whole-inventory completion and validation selection gate; no test likelihoods or comparative quality result')
        # Recheck every selected identity after processing the entire inventory.
        if sha256(root/IDENTITY) != identity_hash: raise ValueError('Study identity changed during selection')
        for name, expected in identity['sources'].items():
            if sha256(root/'flm'/name) != expected: raise ValueError('Frozen source changed during selection')
        for row in records:
            directory = root/STUDY/row['condition']['label']
            for filename, expected in (('complete.json', row['complete_sha256']),
                                       ('validation-selection.json', row['validation_selection_sha256'])):
                if sha256(directory/filename) != expected: raise ValueError('Run records changed during study selection')
            if sha256(root/row['checkpoint']) != row['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed during study selection')
        if (root/SELECTION).exists():
            if read(root/SELECTION) != result: raise ValueError('Existing study selection changed')
        else: write_json(root/SELECTION, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('initialize', 'train', 'select'))
    parser.add_argument('--settings', type=Path, help='Explicit Settings JSON; required for initialize, no default budget')
    args = parser.parse_args()
    if args.operation == 'initialize':
        if args.settings is None: parser.error('initialize requires --settings')
        initialize(Path.cwd(), Settings(**read(args.settings)))
    elif args.operation == 'train': train_study(Path.cwd())
    else: freeze_selection(Path.cwd())
