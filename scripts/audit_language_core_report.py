"""Recompute computation-control score arithmetic with Python and NumPy only.

This checks supplied article scores and the declared paired comparisons. It
does not authenticate data or checkpoints, retrain models, or rerun inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np


CONTROLS = ('full', 'fixed_dynamics', 'no_lateral', 'no_temporal_state')
SIGN = 'First minus second test BPB; negative favors the first model'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
            'Reported arithmetic differs: ' + label)


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def fingerprint(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def paired(first, second):
    names = sorted(first)
    # Independently calculate bits before resampling, then use fsum for the
    # point estimate. This does not call the training evaluator or its interval.
    bits = np.array([(first[name]['nll'] - second[name]['nll']) / math.log(2) for name in names])
    sizes = np.array([first[name]['bytes'] for name in names], dtype=np.int64)
    draws = np.random.default_rng(31415).integers(0, len(names), size=(10000, len(names)))
    values = bits[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    lower, upper = np.quantile(values, [.025, .975], method='linear')
    return dict(difference_bpb=math.fsum(bits) / int(sizes.sum()), lower_95=float(lower), upper_95=float(upper))


def audit(report):
    require(fingerprint(report.get('study_identity_sha256')) and fingerprint(report.get('selection_sha256')),
            'A study and selection identity are required')
    require(report.get('sign') == SIGN, 'Contrast sign declaration differs')
    conditions = {f'{control}-s{seed}': dict(control=control, seed=seed, reference=control == 'full')
                  for seed in (42, 43) for control in CONTROLS}
    runs = report.get('runs', [])
    require(len(runs) == 8 and {run.get('label') for run in runs} == set(conditions),
            'All eight unique computation conditions are required')
    lookup = {}; coverage = None; shared_inputs = None
    for run in runs:
        label = run['label']; control = conditions[label]['control']
        require(all(run.get(key) == value for key, value in conditions[label].items()), 'Wrong condition identity: ' + label)
        require(run.get('selection_sha256') == report['selection_sha256'], 'Wrong selection: ' + label)
        step = run.get('checkpoint_step')
        require(positive_integer(step) and step % 500 == 0 and 500 <= step <= 6000,
                'Invalid selected update: ' + label)
        counts = dict(allocated_parameters=600003, trainable_parameters=521824 if control == 'fixed_dynamics' else 600003,
                      frozen_parameters=78179 if control == 'fixed_dynamics' else 0)
        require(all(type(run.get(key)) is int and run[key] == value for key, value in counts.items())
                and fingerprint(run.get('checkpoint_sha256')), 'Invalid selected model/count record: ' + label)
        if run['reference']:
            require(fingerprint(run.get('reused_score_sha256')), 'Missing reused reference score identity: ' + label)
        else:
            require('reused_score_sha256' not in run, 'A new control is mislabeled as a reused score: ' + label)
        score = run['score']
        if not run['reference']:
            mechanism = ('Reset both states for every token, including within chunks' if control == 'no_temporal_state'
                         else 'Carry native fast/slow state within each article; reset per article')
            require(score.get('mechanism') == mechanism, 'Scored mechanism declaration differs: ' + label)
        test = run.get('test_cache_sha256'); tokenizer = score.get('tokenizer_sha256'); graph = run.get('graph')
        require(fingerprint(test) and fingerprint(tokenizer) and graph == 'data/graphs/central-1024/graph.npz',
                'Missing or changed test/tokenizer/measured-subset identity: ' + label)
        inputs = (test, tokenizer, graph)
        if shared_inputs is None:
            shared_inputs = inputs
        require(inputs == shared_inputs, 'Test data, tokenizer or graph declaration differs')
        documents = score.get('documents', [])
        require(len(documents) == 60, 'Exactly 60 test article records are required: ' + label)
        by_name = {row['document']: row for row in documents}
        require(len(by_name) == 60 and all(isinstance(name, str) and name for name in by_name),
                'Article identities must be unique and nonempty: ' + label)
        denominators = {}
        for name, row in by_name.items():
            require(positive_integer(row.get('bytes')) and positive_integer(row.get('tokens')),
                    'Invalid article denominator: ' + name)
            nll = row.get('nll')
            require(isinstance(nll, (int, float)) and not isinstance(nll, bool) and math.isfinite(nll) and nll >= 0,
                    'Invalid article likelihood: ' + name)
            same_number(row.get('bits_per_byte'), nll / row['bytes'] / math.log(2), label + '/' + name)
            denominators[name] = (row['bytes'], row['tokens'])
        if coverage is None:
            coverage = denominators
        require(denominators == coverage, 'Article membership or denominators differ: ' + label)
        nll = math.fsum(row['nll'] for row in documents)
        byte_count = sum(row['bytes'] for row in documents); token_count = sum(row['tokens'] for row in documents)
        require(score.get('bytes') == byte_count and score.get('tokens') == token_count,
                'Aggregate denominator differs: ' + label)
        same_number(score.get('nll'), nll, label + '/NLL')
        same_number(score.get('bits_per_byte'), nll / byte_count / math.log(2), label + '/BPB')
        perplexity = score.get('token_perplexity')
        require(isinstance(perplexity, (int, float)) and not isinstance(perplexity, bool)
                and math.isfinite(perplexity) and perplexity > 0, 'Invalid perplexity: ' + label)
        same_number(math.log(perplexity), nll / token_count, label + '/log perplexity')
        lookup[label] = by_name

    primary = []; memory = []
    for seed in (42, 43):
        for control in CONTROLS[1:]:
            primary.append(dict(first='full', second=control, training_seed=seed,
                **paired(lookup[f'full-s{seed}'], lookup[f'{control}-s{seed}'])))
        memory.append(dict(first='no_lateral', second='no_temporal_state', training_seed=seed,
            **paired(lookup[f'no_lateral-s{seed}'], lookup[f'no_temporal_state-s{seed}'])))
    fields = ('first', 'second', 'training_seed')
    for key, expected in [('primary_contrasts', primary), ('independent_unit_memory_contrasts', memory)]:
        published = report.get(key, [])
        indexed = {tuple(row.get(field) for field in fields): row for row in published}
        require(len(published) == len(indexed) == len(expected) and set(indexed) == {
            tuple(row[field] for field in fields) for row in expected}, 'Paired contrast coverage differs: ' + key)
        for row in expected:
            original = indexed[tuple(row[field] for field in fields)]
            require(original.get('replicates') == 10000 and original.get('seed') == 31415
                    and original.get('unit') == 'paired article resampling', 'Bootstrap settings differ: ' + key)
            for metric in ('difference_bpb', 'lower_95', 'upper_95'):
                same_number(original.get(metric), row[metric], key + '/' + str(tuple(row[field] for field in fields)) + '/' + metric)
    means = [dict(control=control, mean_difference_bpb=math.fsum(
                row['difference_bpb'] for row in primary if row['second'] == control) / 2) for control in CONTROLS[1:]]
    published_means = report.get('primary_means', [])
    require(len(published_means) == 3 and {row.get('control') for row in published_means} == set(CONTROLS[1:]),
            'Primary mean coverage differs')
    for expected in means:
        original = next(row for row in published_means if row['control'] == expected['control'])
        require(set(original) == {'control', 'mean_difference_bpb'}, 'Undeclared fields on a descriptive primary mean')
        same_number(original.get('mean_difference_bpb'), expected['mean_difference_bpb'], 'primary mean/' + expected['control'])
    memory_mean = math.fsum(row['difference_bpb'] for row in memory) / 2
    same_number(report.get('independent_unit_memory_mean_difference_bpb'), memory_mean, 'independent-unit memory mean')
    return dict(verified_runs=8, unique_test_articles=60, scored_bytes_per_run=byte_count,
        scored_tokens_per_run=token_count, primary_contrasts=primary, primary_means=means,
        independent_unit_memory_contrasts=memory, independent_unit_memory_mean_difference_bpb=memory_mean,
        study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        comparison_tolerance=dict(relative=1e-9, absolute=1e-12),
        scope='Arithmetic audit of supplied scores; not independent training, test inference or source authentication',
        uncertainty='Eight conditional fitted-pair intervals; shared references/articles and two initializations; no mean confidence interval or topology-by-trainability factorial')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='Complete language computation summary.json')
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.report.read_text(encoding='utf8'))), indent=2))


if __name__ == '__main__':
    main()
