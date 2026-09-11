"""Resumable whole-partition generation after all 36 SCAN fits are frozen.

Every output is unconstrained model argmax, including invalid tokens and failed
sequences. This module never trains, selects on test accuracy or repairs output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time

import torch

from .inference import state_hash
from .provenance import sha256, write_json
from .scan_conditions import conditions
from .scan_runtime import greedy_instructions, score_outputs
from .scan_study import (IDENTITY, POLICY, REPORTS, SELECTION, STUDY, freeze_selection,
    immutable, prepared_run, read, unchanged as unchanged_study, verified_context)
from .scan_test_inputs import load_test_partition
from .scan_train import canonical, declaration, optimizer_for, restore_payload, training_lease

EVALUATION = Path('runs/scan-evaluation-v1')
SUMMARY = REPORTS/'test-summary.json'


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unchanged(root, identity, selection, identity_hash, selection_hash):
    unchanged_study(root, identity, identity_hash)
    if sha256(root/SELECTION) != selection_hash:
        raise ValueError('Frozen instruction selection changed during evaluation')
    for row in selection['conditions']:
        folder = root/STUDY/row['condition']['label']
        files = [(root/row['checkpoint'], row['checkpoint_sha256']),
                 (folder/'complete.json', row['complete_sha256'])]
        files += [(folder/name, value) for name, value in row['saved_record_sha256'].items()]
        if any(sha256(path) != expected for path, expected in files):
            raise ValueError('Selected instruction checkpoint or completion changed')


def restore_selected(root, registered, selected, identity_hash):
    prepared, settings, binding, lengths, scored = prepared_run(root, registered, identity_hash)
    declared, _, _ = declaration(prepared.model, prepared.records, prepared.lexicon, settings, binding)
    path = root/selected['checkpoint']
    if sha256(path) != selected['checkpoint_sha256']:
        raise ValueError('Selected instruction payload checksum changed')
    payload = restore_payload(path, prepared.model, optimizer_for(prepared.model, settings),
                              declared, lengths, scored)
    if (payload['step'] != selected['checkpoint_step'] or payload['exposure'] != selected['final_exposure']
            or payload['sampled_row_sha256'] != selected['sampled_row_sha256']
            or any(bool((state['exp_avg_sq'] < 0).any()) for state in payload['optimizer']['state'].values())):
        raise ValueError('Selected instruction terminal state changed')
    return prepared.model, prepared.lexicon


def generate_partition(model, lexicon, records, coverage, directory, run_identity, policy):
    """Persist contiguous publisher-row batches; verify cached traces on resume.

    The caller owns the study/evaluation leases and official membership gate.
    Full row membership is bound here; target actions only enter the scorer.
    """
    if not records or digest(records) != coverage['ordered_rows_sha256']:
        raise ValueError('Generation cohort differs from the verified entire partition')
    if coverage['statistics']['rows'] != len(records):
        raise ValueError('Generation partition row count changed')
    if any(type(policy[key]) is not int or policy[key] < 1
           for key in ('maximum_tokens', 'batch_size', 'context_limit')):
        raise ValueError('Invalid registered instruction generation dimensions')
    directory = Path(directory)
    original_state = state_hash(model)
    identity = dict(binding=run_identity, coverage=coverage, policy=policy,
        model_state_sha256=original_state, tokenizer_sha256=lexicon.sha256,
        batching='Contiguous publisher rows, then equal prefix lengths within each saved batch')
    if not (directory/'identity.json').exists() and list(directory.glob('batch-*.json')):
        raise ValueError('Cannot adopt cached instruction batches without their identity')
    immutable(directory/'identity.json', identity); identity_hash = sha256(directory/'identity.json')
    count = math.ceil(len(records)/policy['batch_size'])
    names = [f'batch-{index:05d}.json' for index in range(count)]
    existing = sorted(path.name for path in directory.glob('batch-*.json'))
    if existing != names[:len(existing)]:
        raise ValueError('Unexpected or noncontiguous instruction batch inventory')
    outputs = []; hashes = {}; seconds = []
    for index, name in enumerate(names):
        start = index*policy['batch_size']; stop = min(len(records), start+policy['batch_size'])
        rows = records[start:stop]
        expected = dict(run_identity_sha256=identity_hash, start=start, stop=stop,
                        ordered_rows_sha256=digest(rows))
        path = directory/name
        if path.exists():
            saved = read(path)
        else:
            started = time.perf_counter()
            generated = greedy_instructions(model, lexicon, [row['command'] for row in rows],
                **{key:policy[key] for key in ('maximum_tokens', 'batch_size', 'context_limit')})
            saved = dict(identity=expected, outputs=generated, seconds=time.perf_counter()-started)
        if (set(saved) != {'identity', 'outputs', 'seconds'} or saved['identity'] != expected
                or type(saved['seconds']) not in (int, float) or not math.isfinite(saved['seconds'])
                or saved['seconds'] < 0):
            raise ValueError('Instruction batch identity or generation time changed')
        score_outputs(rows, saved['outputs'], lexicon, maximum_tokens=policy['maximum_tokens'])
        if state_hash(model) != original_state:
            raise ValueError('Model parameters changed during instruction generation')
        immutable(path, saved)
        hashes[name] = sha256(path); outputs.extend(saved['outputs']); seconds.append(saved['seconds'])
    if (sha256(directory/'identity.json') != identity_hash
            or sorted(path.name for path in directory.glob('batch-*.json')) != names
            or any(sha256(directory/name) != value for name, value in hashes.items())):
        raise ValueError('Instruction generation records changed during evaluation')
    metrics = score_outputs(records, outputs, lexicon, maximum_tokens=policy['maximum_tokens'])
    return dict(run_identity_sha256=identity_hash, coverage=coverage, metrics=metrics,
        batch_record_sha256=hashes, seconds=math.fsum(seconds),
        timing='Generation wall time only; includes retained batches from earlier attempts, not a throughput comparison')


def summarize(results):
    expected = conditions()
    if [row['condition'] for row in results] != expected:
        raise ValueError('Require every registered instruction result in exact order')
    by_label = {row['condition']['label']:row for row in results}
    aggregates = []; comparisons = []
    for split in dict.fromkeys(row['split'] for row in expected):
        cohort = [row for row in results if row['condition']['split'] == split]
        coverage = cohort[0]['coverage']
        reference = [(r['command'], r['reference_actions']) for r in cohort[0]['metrics']['rows']]
        if any(row['coverage'] != coverage or
               [(r['command'], r['reference_actions']) for r in row['metrics']['rows']] != reference
               for row in cohort):
            raise ValueError('Instruction comparisons have unmatched test membership or reference lengths')
        for variant in ('flm', 'gru', 'transformer'):
            for initialization in ('initial', 'wikitext'):
                rows = [by_label[f'{split}/{variant}-{initialization}-s{seed}'] for seed in (42, 43)]
                rates = [row['metrics']['aggregate']['exact_match_rate'] for row in rows]
                aggregates.append(dict(split=split, variant=variant, initialization=initialization,
                    seeds=[42,43], exact_match_rates=rates, mean_exact_match_rate=math.fsum(rates)/2,
                    per_seed=[row['metrics']['aggregate'] for row in rows]))
        for seed in (42, 43):
            contrasts = [('transfer', f'{variant}-wikitext', f'{variant}-initial')
                         for variant in ('flm', 'gru', 'transformer')]
            contrasts += [('architecture', f'flm-{init}', f'{variant}-{init}')
                          for init in ('initial', 'wikitext') for variant in ('gru', 'transformer')]
            for kind, first, second in contrasts:
                a = by_label[f'{split}/{first}-s{seed}']['metrics']['rows']
                b = by_label[f'{split}/{second}-s{seed}']['metrics']['rows']
                first_only = sum(x['exact_match'] and not y['exact_match'] for x,y in zip(a,b))
                second_only = sum(y['exact_match'] and not x['exact_match'] for x,y in zip(a,b))
                comparisons.append(dict(split=split, seed=seed, kind=kind, first=first, second=second,
                    examples=len(a), first_only_correct=first_only, second_only_correct=second_only,
                    both_correct=sum(x['exact_match'] and y['exact_match'] for x,y in zip(a,b)),
                    neither_correct=sum(not x['exact_match'] and not y['exact_match'] for x,y in zip(a,b)),
                    exact_match_rate_difference=(first_only-second_only)/len(a),
                    mean_edit_distance_difference=math.fsum(x['edit_distance']-y['edit_distance'] for x,y in zip(a,b))/len(a)))
    return dict(aggregates=aggregates, paired_comparisons=comparisons,
        interpretation='Positive first-minus-second exact-match difference favors the first condition; negative edit-distance difference favors the first. Two seeds are reported individually and as descriptive means. No significance or independent-command assumption is made. Splits reuse one artificial grammar and are separate experiments. WikiText pretraining adds compute; all conditions share its train-fitted tokenizer. These scores measure symbolic instruction composition, not dialogue or physical execution.')


def score_study(root):
    root = Path(root); previous_threads = torch.get_num_threads()
    attempt = root/EVALUATION/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    active = None
    try:
        with torch.random.fork_rng(devices=[]):
            # Every completion is restored before any official test file is opened.
            selection = freeze_selection(root)
            with training_lease(root/STUDY), training_lease(root/EVALUATION):
                try:
                    identity = verified_context(root)
                    identity_hash = sha256(root/IDENTITY); selection_hash = sha256(root/SELECTION)
                    if (selection != read(root/SELECTION) or selection['study_identity_sha256'] != identity_hash
                            or [r['condition'] for r in selection['conditions']] != conditions()
                            or identity['evaluation_policy'] != POLICY or selection['evaluation_policy'] != POLICY
                            or selection['test_metadata'] != identity['test_metadata']):
                        raise ValueError('Whole-study instruction selection or policy changed')
                    unchanged(root, identity, selection, identity_hash, selection_hash)
                    torch.set_num_threads(POLICY['threads'])
                    partitions = {split:load_test_partition(root, split, identity['test_metadata'])
                                  for split in identity['test_metadata']['partitions']}
                    common = dict(study_identity_sha256=identity_hash, selection_sha256=selection_hash,
                        policy=POLICY, coverage={split:pair[1] for split,pair in partitions.items()},
                        sources=identity['sources'], torch=str(torch.__version__))
                    immutable(root/EVALUATION/'identity.json', common)
                    evaluation_hash = sha256(root/EVALUATION/'identity.json')
                    registered = {row['condition']['label']:row for row in identity['conditions']}
                    results = []; hashes = {}
                    for selected in selection['conditions']:
                        active = selected['condition']; label = active['label']
                        unchanged(root, identity, selection, identity_hash, selection_hash)
                        directory = root/EVALUATION/label
                        report = root/REPORTS/'test'/label/'result.json'
                        # A completed result seals every cached batch before resume.
                        if report.exists():
                            previous = read(report)
                            for name, value in previous['batch_record_sha256'].items():
                                if sha256(directory/name) != value:
                                    raise ValueError('Previously completed instruction batch changed')
                        model, lexicon = restore_selected(root, registered[label], selected, identity_hash)
                        records, coverage = partitions[active['split']]
                        run = dict(evaluation_identity_sha256=evaluation_hash, selected=selected)
                        generated = generate_partition(model, lexicon, records, coverage, directory, run, POLICY)
                        result = dict(condition=active, identity=run, **generated)
                        immutable(report, result); results.append(result); hashes[label] = sha256(report)
                        print('Completed whole-partition instruction generation: '+label, flush=True)
                    unchanged(root, identity, selection, identity_hash, selection_hash)
                    for split, (_,coverage) in partitions.items():
                        if load_test_partition(root, split, identity['test_metadata'])[1] != coverage:
                            raise ValueError('Instruction test payload changed during evaluation')
                    if verified_context(root) != identity or sha256(root/EVALUATION/'identity.json') != evaluation_hash:
                        raise ValueError('Instruction evaluation context changed')
                    for row in results:
                        label = row['condition']['label']; directory = root/EVALUATION/label
                        if (sha256(root/REPORTS/'test'/label/'result.json') != hashes[label]
                                or sha256(directory/'identity.json') != row['run_identity_sha256']
                                or sorted(path.name for path in directory.glob('batch-*.json')) != sorted(row['batch_record_sha256'])
                                or any(sha256(directory/name) != value for name,value in row['batch_record_sha256'].items())):
                            raise ValueError('Instruction result or batch changed before summary')
                    summary = dict(evaluation_identity_sha256=evaluation_hash,
                                   result_sha256=hashes, **summarize(results))
                    immutable(root/SUMMARY, summary)
                    return summary
                except Exception as error:
                    write_json(attempt/'failure.json', dict(condition=active,
                        error_type=type(error).__name__, error=str(error)))
                    raise
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    score_study(Path.cwd())
