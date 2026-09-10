"""Recompute a complete published topology report using Python and NumPy only.

This audits reported likelihood arithmetic, pairing and bootstrap intervals.
It does not independently authenticate checkpoints, data, training or inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np


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
    # Work in bits and use fsum for the point estimate. This is a separate
    # implementation of the registered ratio of summed NLL to summed bytes.
    bits = np.array([(first[name]['nll']-second[name]['nll']) / math.log(2) for name in names])
    sizes = np.array([first[name]['bytes'] for name in names], dtype=np.int64)
    draws = np.random.default_rng(31415).integers(0, len(names), size=(10000, len(names)))
    values = bits[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    lower, upper = np.quantile(values, [.025, .975], method='linear')
    return dict(difference_bpb=math.fsum(bits)/int(sizes.sum()), lower_95=float(lower), upper_95=float(upper))


def audit(report):
    require(fingerprint(report.get('study_identity_sha256')) and fingerprint(report.get('selection_sha256')),
            'A frozen study and selection identity are required')
    require(report.get('sign') == 'Measured minus control; negative favors the measured fast/slow model',
            'Contrast sign declaration differs')
    conditions = {}
    for seed in (42, 43):
        conditions[f'measured-s{seed}'] = dict(seed=seed, graph_seed=None, reference=True, variant='flm', topology='measured')
        conditions[f'no-slow-s{seed}'] = dict(seed=seed, graph_seed=None, reference=False, variant='no_slow', topology='measured')
        for graph_seed in (101, 103, 107):
            conditions[f'null{graph_seed}-s{seed}'] = dict(seed=seed, graph_seed=graph_seed, reference=False,
                                                       variant='flm', topology='rewired')
    runs = report.get('runs', [])
    require(len(runs) == 10 and {r.get('label') for r in runs} == set(conditions),
            'All ten unique registered conditions are required')
    lookup = {}; coverage = None; shared_test = shared_tokenizer = None
    for run in runs:
        label = run['label']
        require(all(run.get(key) == value for key, value in conditions[label].items()), 'Wrong condition identity: ' + label)
        require(run.get('selection_sha256') == report['selection_sha256'], 'Wrong selection: ' + label)
        step = run.get('checkpoint_step')
        require(positive_integer(step) and step % 500 == 0 and 500 <= step <= 6000,
                'Invalid selected update: ' + label)
        require(run.get('parameters') == 600003 and fingerprint(run.get('checkpoint_sha256')),
                'Invalid selected model record: ' + label)
        score = run['score']; test = run.get('test_cache_sha256'); tokenizer = score.get('tokenizer_sha256')
        require(fingerprint(test) and fingerprint(tokenizer), 'Missing test/tokenizer identity: ' + label)
        if shared_test is None:
            shared_test, shared_tokenizer = test, tokenizer
        require((test, tokenizer) == (shared_test, shared_tokenizer), 'Test data or tokenizer identity differs')
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
        byte_count = sum(row['bytes'] for row in documents)
        token_count = sum(row['tokens'] for row in documents)
        require(score.get('bytes') == byte_count and score.get('tokens') == token_count, 'Aggregate denominator differs: ' + label)
        same_number(score.get('nll'), nll, label + '/NLL')
        same_number(score.get('bits_per_byte'), nll / byte_count / math.log(2), label + '/BPB')
        perplexity = score.get('token_perplexity')
        require(isinstance(perplexity, (int, float)) and math.isfinite(perplexity) and perplexity > 0,
                'Invalid perplexity: ' + label)
        same_number(math.log(perplexity), nll / token_count, label + '/log perplexity')
        lookup[label] = by_name

    topology = []; slow = []
    for seed in (42, 43):
        for graph_seed in (101, 103, 107):
            topology.append(dict(training_seed=seed, graph_seed=graph_seed,
                **paired(lookup[f'measured-s{seed}'], lookup[f'null{graph_seed}-s{seed}'])))
        slow.append(dict(training_seed=seed, **paired(lookup[f'measured-s{seed}'], lookup[f'no-slow-s{seed}'])))
    for key, expected, fields in [('topology_contrasts', topology, ('training_seed', 'graph_seed')),
                                  ('slow_state_contrasts', slow, ('training_seed',))]:
        published = report.get(key, [])
        indexed = {tuple(row.get(f) for f in fields): row for row in published}
        require(len(published) == len(indexed) == len(expected) and set(indexed) == {
            tuple(row[f] for f in fields) for row in expected}, 'Paired contrast coverage differs: ' + key)
        for row in expected:
            original = indexed[tuple(row[f] for f in fields)]
            require(original.get('replicates') == 10000 and original.get('seed') == 31415
                    and original.get('unit') == 'paired article resampling',
                    'Bootstrap settings differ: ' + key)
            for metric in ('difference_bpb', 'lower_95', 'upper_95'):
                same_number(original.get(metric), row[metric], key + '/' + str(tuple(row[f] for f in fields)) + '/' + metric)
    for field, values, key in [('graph_seed', (101, 103, 107), 'topology_by_graph'),
                                ('training_seed', (42, 43), 'topology_by_training_seed')]:
        rows = report.get(key, [])
        require(len(rows) == len(values) and {row.get(field) for row in rows} == set(values), 'Mean coverage differs: ' + key)
        for row in rows:
            selected = [pair['difference_bpb'] for pair in topology if pair[field] == row[field]]
            same_number(row.get('mean_difference_bpb'), math.fsum(selected)/len(selected), key + '/' + str(row[field]))
    for key, rows in [('topology_mean_difference_bpb', topology), ('slow_state_mean_difference_bpb', slow)]:
        same_number(report.get(key), math.fsum(row['difference_bpb'] for row in rows)/len(rows), key)
    return dict(verified_runs=10, unique_test_articles=60, scored_bytes_per_run=byte_count,
                scored_tokens_per_run=token_count, topology_contrasts=topology, slow_state_contrasts=slow,
                study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
                scope='Arithmetic audit of supplied scores; not independent training, test inference or source authentication',
                uncertainty='Intervals condition on each fitted pair; six topology contrasts share two measured references')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='Complete published language-topology-results.json')
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.report.read_text(encoding='utf8'))), indent=2))


if __name__ == '__main__':
    main()
