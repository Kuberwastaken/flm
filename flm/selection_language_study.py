"""Freeze complete neuron-selection groups, fit serially and select validation checkpoints.

No default group or training budget is supplied. Official initialization requires
the completed priority queue, measured original-graph costs and a written protocol.
This coordinator never opens held-out token payloads; its test scorer is pending.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from .language_learning_inputs import load_corpus
from .language_learning_study import test_metadata
from .language_learning_train import Settings, declaration, replay, training_lease
from .language_learning_validation import load_panel, select_checkpoint
from .provenance import sha256, write_json
from .selection_language import GRAPH_SEEDS, TRAINING_SEEDS, load_catalog, prepare_case, fit_case
from .selection_pilot import Pilot, SOURCES as PILOT_SOURCES, prerequisites

STUDY = Path('runs/selection-language-v1')
REPORTS = Path('reports/selection-language')
IDENTITY = REPORTS/'study-identity.json'
SELECTION = REPORTS/'study-selection.json'
PILOT = Path('reports/selection-pilot/timing.json')
PROTOCOL = Path('docs/SELECTION-LANGUAGE-PROTOCOL.md')
SELECTORS = ('candidate', 'contact_ranked', 'uniform_s201', 'uniform_s203', 'uniform_s207',
             'stratified_s201', 'stratified_s203', 'stratified_s207')
SOURCES = tuple(sorted(set(PILOT_SOURCES) | {
    'selection_language_study.py', 'selection_language.py', 'language_learning_inputs.py',
    'language_learning_study.py', 'language_learning_validation.py', 'wiring_controls.py',
    'language_train.py', 'baselines.py', 'graph.py', 'babylm.py'}))


def read(path): return json.loads(path.read_text(encoding='utf8'))


def matrix(catalog, groups):
    """Require every selector and rewire; accept group names, never individual runs."""
    available = {row['candidate'] for row in catalog.entries.values()}
    if (not isinstance(groups, list) or not groups or any(type(g) is not str for g in groups)
            or len(set(groups)) != len(groups) or not set(groups) <= available):
        raise ValueError('Choose unique known candidate groups')
    rows = []
    for group in sorted(groups):
        labels = {f'{group}/{selector}/{suffix}' for selector in SELECTORS
                  for suffix in ('measured', *(f'null{s}' for s in GRAPH_SEEDS))}
        actual = {label for label, row in catalog.entries.items() if row['candidate'] == group}
        if actual != labels: raise ValueError('Candidate group lacks the complete selector/rewire inventory')
        sizes = {catalog.entries[label]['config']['neurons'] for label in labels}
        if len(sizes) != 1: raise ValueError('Selection controls do not match neuron count')
        for seed in TRAINING_SEEDS:
            for label in sorted(labels):
                rows.append(dict(graph=label, seed=seed, label=f'{label}/s{seed}'))
    return rows


def verify_pilot(root, catalog):
    if not (root/PILOT).is_file(): raise ValueError('Measure the full-window selection pilot before freezing a budget')
    pilot = read(root/PILOT); expected = {r['selection']:r for r in catalog.entries.values() if r['graph_seed'] is None}
    rows = pilot['conditions']
    if (len(rows) != len(expected) or {r['selection'] for r in rows} != set(expected)
            or set(pilot['source_sha256']) != set(PILOT_SOURCES)
            or pilot['graph_manifest_sha256'] != catalog.binding['source_manifest_sha256']
            or pilot['binding']['validation_or_test_opened'] is not False
            or pilot['benchmark_budget_selected'] is not False or pilot['training_matrix_frozen'] is not False
            or pilot['torch'] != str(torch.__version__) or pilot['numpy'] != str(np.__version__)):
        raise ValueError('Incomplete or changed selection pilot identity')
    for name, expected_hash in pilot['source_sha256'].items():
        if sha256(root/'flm'/name) != expected_hash: raise ValueError('Measured pilot source changed')
    order = np.random.Generator(np.random.PCG64(519)).permutation(len(expected))
    if pilot['graph_order_seed'] != 519 or [r['selection'] for r in rows] != [sorted(expected)[int(i)] for i in order]:
        raise ValueError('Measured pilot order changed')
    for row in rows:
        entry = expected[row['selection']]; observations = row['observations']
        if (row['pilot'] != asdict(Pilot()) or row['training_settings'] != asdict(Pilot().training_settings())
                or row['update_implementation'] != 'flm.language_learning_train.update:bptt'
                or row['graph_sha256'] != entry['graph_sha256']
                or row['parameter_card']['trainable_parameters'] != entry['trainable_parameters']
                or row['graph_unchanged'] is not True or row['disposable_parameters_updated'] is not True
                or row['checkpoint_written'] is not False or row['language_scores_reported'] is not False
                or row['dtype'] != 'torch.float32' or row['effective_backend'] != 'dense'
                or len(observations) != 15 or [o['step'] for o in observations] != list(range(1,16))
                or [o['warmup'] for o in observations] != [True]*3+[False]*12
                or any(not np.isfinite(o['seconds']) or o['seconds'] <= 0 for o in observations)):
            raise ValueError('Invalid full-window selection pilot condition')
        for key in ('seconds', 'input_tokens', 'supervised_targets', 'input_bytes', 'supervised_bytes'):
            if not np.isclose(row['measured_'+key], sum(o[key] for o in observations[3:]), rtol=1e-12, atol=1e-12):
                raise ValueError('Measured pilot totals changed')
        if any(o['input_tokens'] != 1536 or o['supervised_targets'] != 1280 for o in observations):
            raise ValueError('Pilot token exposure changed')
    if len({r['sampled_windows_sha256'] for r in rows}) != 1: raise ValueError('Pilot windows differ across selections')
    return pilot


def identity_for(root, catalog, corpus, panel, request, pilot):
    if set(request) != {'groups', 'settings', 'rationale', 'cost_allowance_multiplier', 'wall_time_budget_seconds'}:
        raise ValueError('Supply explicit groups, settings, rationale and cost allowance/budget')
    settings = Settings(**request['settings']); settings.validate(); panel.verify(corpus.lexicon)
    if settings.seed != 42 or not settings.score_boundaries:
        raise ValueError('Selection template requires seed 42 and boundary-inclusive training')
    if not isinstance(request['rationale'], str) or not request['rationale'].strip():
        raise ValueError('Record a pre-fit anatomical and computational group-selection rationale')
    for key in ('cost_allowance_multiplier', 'wall_time_budget_seconds'):
        value = request[key]
        if type(value) not in (int,float) or not np.isfinite(value) or value <= 0:
            raise ValueError('Explicit finite positive cost allowance and wall-time budget required')
    if request['cost_allowance_multiplier'] < 1: raise ValueError('Cost allowance cannot discount measured update costs')
    if not (root/PROTOCOL).is_file() or not (root/PROTOCOL).read_text(encoding='utf8').strip():
        raise ValueError('Write the selection-language protocol before freezing the study')
    registry = matrix(catalog, request['groups'])
    measured = {r['selection']:r for r in pilot['conditions']}
    for key in ('partition', 'tokenizer_sha256', 'cache_manifest_sha256', 'cache_files_sha256'):
        if pilot['binding'][key] != corpus.binding[key]: raise ValueError('Pilot corpus differs from the verified training corpus')
    study_path = root/'runs/babylm-10m/study.json'
    if pilot['binding']['study_sha256'] != sha256(study_path): raise ValueError('Pilot reference study changed')
    pilot_settings = Pilot().training_settings()
    for key in ('batch', 'sequence', 'warmup', 'threads', 'score_boundaries'):
        if asdict(settings)[key] != asdict(pilot_settings)[key]: raise ValueError('Training dimensions differ from measured pilot')
    _, all_exposure, sampled = replay(corpus.documents, corpus.lexicon.lengths, pilot_settings, pilot_settings.steps)
    _, warm_exposure, _ = replay(corpus.documents, corpus.lexicon.lengths, pilot_settings, Pilot().warmup_updates)
    mapping = dict(input_tokens='presented_tokens', supervised_targets='scored_tokens',
                   input_bytes='presented_bytes', supervised_bytes='scored_bytes')
    for row in measured.values():
        if row['sampled_windows_sha256'] != sampled.hexdigest(): raise ValueError('Pilot training windows do not replay')
        if any(row['measured_'+key] != all_exposure[value]-warm_exposure[value] for key,value in mapping.items()):
            raise ValueError('Pilot training byte/token exposure does not replay')
    rows = []; learned = {}; nonedge = {}; update_seconds = 0.
    for condition in registry:
        model, binding = prepare_case(catalog, corpus, condition['graph'], condition['seed'])
        local_settings = replace(settings, seed=condition['seed'])
        declared = declaration(model, corpus.documents, corpus.lexicon, local_settings, 'bptt', binding)
        entry = binding['graph']; cost = measured[entry['selection']]
        if entry['graph_seed'] is None and condition['seed'] == 42:
            if cost['initial_state_sha256'] != binding['initial_state_sha256'] or cost['graph_arrays_sha256'] != binding['graph_arrays_sha256']:
                raise ValueError('Pilot used different graph arrays or initial tensors')
        learned.setdefault((entry['selection'], condition['seed']), set()).add(binding['initial_parameters_sha256'])
        nonedge.setdefault((entry['candidate'], condition['seed']), set()).add(binding['initial_nonedge_parameters_sha256'])
        proxy = cost['measured_seconds']/Pilot().measured_updates
        update_seconds += proxy*settings.steps
        rows.append(dict(condition=condition, settings=asdict(local_settings), base_binding=binding,
            training_documents_sha256=declared['ordered_training_documents_sha256'],
            trainable_parameters=declared['trainable_parameters'], original_graph_seconds_per_update=proxy,
            cost_is_proxy_for_untimed_rewire=entry['graph_seed'] is not None))
    if any(len(v) != 1 for v in (*learned.values(), *nonedge.values())):
        raise ValueError('Initial learned parameters are unmatched')
    if len({r['training_documents_sha256'] for r in rows}) != 1: raise ValueError('Training documents differ across conditions')
    planned_seconds = update_seconds*request['cost_allowance_multiplier']
    if planned_seconds > request['wall_time_budget_seconds']: raise ValueError('Complete matrix exceeds the stated planning budget')
    return dict(format='flm-selection-language-study-v1', request=request, conditions=rows,
        catalog_binding=catalog.binding, corpus_binding=corpus.binding, validation_panel_binding=panel.binding,
        test_metadata=test_metadata(root, corpus.lexicon), protocol_sha256=sha256(root/PROTOCOL),
        pilot_sha256=sha256(root/PILOT), sources={n:sha256(root/'flm'/n) for n in SOURCES},
        torch=str(torch.__version__), numpy=str(np.__version__), validation_chunk_size=96,
        cost=dict(update_seconds_using_original_proxies=update_seconds, planned_seconds_with_allowance=planned_seconds,
            limitation='Only original graphs were timed, at seed 42. Rewires and seed 43 use original-graph cost proxies. Loading, validation, checkpoint I/O and long-run effects are unmeasured; the allowance is a planning assumption, not a wall-time guarantee.'),
        selection='Earliest exact minimum validation BPB over every declared checkpoint',
        test_policy='Require every registered condition complete and selected before any held-out token access; official test scorer and inference policy still pending',
        scope='Selection methods differ in edge/parameter counts; within-subset rewires control topology at fixed allocation. Operational truncated candidates, not intact functional circuits or animal behavior.')


def initialize(root, request):
    root = Path(root); prerequisites(root)
    catalog = load_catalog(root); pilot = verify_pilot(root, catalog)
    corpus = load_corpus(root); panel = load_panel(root, corpus.lexicon)
    identity = identity_for(root, catalog, corpus, panel, request, pilot)
    with training_lease(root/STUDY):
        if (root/IDENTITY).exists():
            if read(root/IDENTITY) != identity: raise ValueError('A different selection-language study is already frozen')
        else:
            # Include unselected group directories: a reduced matrix cannot erase an earlier attempt.
            if any(p.name != 'writer.lock' for p in (root/STUDY).iterdir()):
                raise ValueError('Cannot freeze a new study after run files already exist')
            write_json(root/IDENTITY, identity)
    return identity


def verified_context(root):
    if not (root/IDENTITY).is_file(): raise ValueError('Freeze the complete selection study before training or selection')
    identity = read(root/IDENTITY)
    if identity.get('format') != 'flm-selection-language-study-v1': raise ValueError('Unknown selection-study format')
    catalog = load_catalog(root); pilot = verify_pilot(root, catalog)
    corpus = load_corpus(root); panel = load_panel(root, corpus.lexicon)
    expected = identity_for(root, catalog, corpus, panel, identity['request'], pilot)
    if identity != expected: raise ValueError('Frozen selection-language study identity changed')
    return identity, catalog, corpus, panel


def unchanged(root, identity, identity_hash):
    if sha256(root/IDENTITY) != identity_hash: raise ValueError('Study identity changed during execution')
    for name, expected in identity['sources'].items():
        if sha256(root/'flm'/name) != expected: raise ValueError('Frozen source changed during execution')
    if sha256(root/PROTOCOL) != identity['protocol_sha256'] or sha256(root/PILOT) != identity['pilot_sha256']:
        raise ValueError('Frozen protocol or pilot changed during execution')


def prepared_run(catalog, corpus, row, identity_hash):
    model, binding = prepare_case(catalog, corpus, row['condition']['graph'], row['condition']['seed'])
    if binding != row['base_binding']: raise ValueError('Condition input binding changed')
    return model, dict(binding, study=dict(study_identity_sha256=identity_hash)), Settings(**row['settings'])


def train_study(root):
    root = Path(root); prerequisites(root)
    with training_lease(root/STUDY):
        identity, catalog, corpus, _ = verified_context(root); identity_hash = sha256(root/IDENTITY)
        for row in identity['conditions']:
            unchanged(root, identity, identity_hash)
            condition = row['condition']
            print('Training registered selection condition: '+condition['label'], flush=True)
            result = fit_case(catalog, corpus, condition['graph'], Settings(**row['settings']),
                root/STUDY/condition['label'], dict(study_identity_sha256=identity_hash))
            if result['complete'] is not True: raise ValueError('Registered selection condition did not finish')
            unchanged(root, identity, identity_hash)
        # Reopen and reverify corpus/archive/panel metadata at the end as well.
        if verified_context(root)[0] != identity: raise ValueError('Study inputs changed during training')


def freeze_selection(root):
    root = Path(root)
    if not (root/IDENTITY).is_file(): raise ValueError('No frozen selection-language study identity')
    with training_lease(root/STUDY):
        unverified = read(root/IDENTITY)
        # Reconstruct the whole group inventory, ignoring a possibly shortened conditions list.
        catalog = load_catalog(root)
        for condition in matrix(catalog, unverified['request']['groups']):
            if not (root/STUDY/condition['label']/'complete.json').is_file():
                raise ValueError('Registered selection run incomplete: '+condition['label'])
        identity, catalog, corpus, panel = verified_context(root); identity_hash = sha256(root/IDENTITY)
        records = []
        for row in identity['conditions']:
            unchanged(root, identity, identity_hash)
            model, binding, settings = prepared_run(catalog, corpus, row, identity_hash)
            condition = row['condition']; directory = root/STUDY/condition['label']
            selected = select_checkpoint(model, corpus.documents, corpus.lexicon, settings, 'bptt',
                binding, directory, panel, chunk_size=identity['validation_chunk_size'])
            choice = selected['selected']; complete = read(directory/'complete.json')
            records.append(dict(condition=condition, checkpoint=(STUDY/condition['label']/choice['checkpoint']).as_posix(),
                checkpoint_sha256=choice['checkpoint_sha256'], checkpoint_step=choice['step'],
                validation_bits_per_byte=choice['score']['bits_per_byte'], final_exposure=complete['exposure'],
                complete_sha256=sha256(directory/'complete.json'),
                validation_selection_sha256=sha256(directory/'validation-selection.json')))
        for seed in TRAINING_SEEDS:
            group = [r for r in records if r['condition']['seed'] == seed]
            if any(r['final_exposure'] != group[0]['final_exposure'] for r in group): raise ValueError('Selection conditions have unequal text exposure')
        unchanged(root, identity, identity_hash)
        for row in records:
            directory = root/STUDY/row['condition']['label']
            for path, expected in ((root/row['checkpoint'], row['checkpoint_sha256']),
                    (directory/'complete.json', row['complete_sha256']),
                    (directory/'validation-selection.json', row['validation_selection_sha256'])):
                if sha256(path) != expected: raise ValueError('Selected run changed during inventory selection')
        if verified_context(root)[0] != identity: raise ValueError('Study inputs changed during selection')
        result = dict(study_identity_sha256=identity_hash, conditions=records, validation_panel_binding=panel.binding,
            test_payloads_opened=False, scope='Complete inventory and validation selection only; official held-out scorer pending')
        if (root/SELECTION).exists():
            if read(root/SELECTION) != result: raise ValueError('Existing selection-study checkpoint choices changed')
        else: write_json(root/SELECTION, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('initialize', 'train', 'select'))
    parser.add_argument('--request', type=Path, help='Explicit group/settings/rationale/cost JSON for initialization')
    args = parser.parse_args()
    if args.operation == 'initialize':
        if args.request is None: parser.error('initialize requires --request; no default matrix or budget')
        initialize(Path.cwd(), read(args.request))
    elif args.operation == 'train': train_study(Path.cwd())
    else: freeze_selection(Path.cwd())
