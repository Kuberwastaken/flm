"""Construct node-count and annotation-stratified controls for KC candidates.

All conditions share one explicit source population. These are selection
controls, not degree-matched topology controls or trained language models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .provenance import sha256, write_json

POPULATION_CLASSES = ('ascending_neuron', 'cb_intrinsic', 'descending_neuron',
                      'ol_intrinsic', 'visual_centrifugal', 'visual_projection')
SEEDS = (201, 203, 207)


def controls(matrix, metadata, ids, signs, candidate_indices):
    n = len(ids)
    if matrix.shape != (n, n) or len(metadata) != n or signs.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError('Invalid source dimensions or IDs')
    eligible = np.array([row[2] in POPULATION_CLASSES for row in metadata])
    population = np.flatnonzero(eligible)
    population = population[np.argsort(ids[population], kind='stable')]
    selected = np.asarray(candidate_indices)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or len(selected) < 1 or np.any(selected < 0) or np.any(selected >= n)
            or len(np.unique(selected)) != len(selected) or not eligible[selected].all()):
        raise ValueError('Candidate must be distinct cells in the shared source population')
    central = matrix[population][:, population]
    # Exact signed integer ranking prevents unsigned negation and float ties.
    strength = (np.asarray(central.sum(axis=0, dtype=np.int64)).ravel()
                + np.asarray(central.sum(axis=1, dtype=np.int64)).ravel())
    ranked = population[np.lexsort((ids[population], -strength))]
    strata = [(row[2], row[3], int(signs[i])) for i, row in enumerate(metadata)]
    wanted = Counter(strata[i] for i in selected)
    buckets = {key: np.array([i for i in population if strata[i] == key], dtype=np.int64) for key in sorted(wanted)}
    result = {'contact_ranked': ranked[:len(selected)]}
    for seed in SEEDS:
        rng = np.random.Generator(np.random.PCG64(seed))
        result[f'uniform_s{seed}'] = rng.choice(population, len(selected), replace=False)
        rng = np.random.Generator(np.random.PCG64(seed))
        result[f'stratified_s{seed}'] = np.concatenate([rng.choice(buckets[key], wanted[key], replace=False) for key in sorted(wanted)])
    return {name: np.array(sorted(indices, key=lambda i: int(ids[i])), dtype=np.int64) for name, indices in result.items()}


def describe(matrix, metadata, ids, signs, selected, reference, incoming, outgoing):
    compact = matrix[selected][:, selected].tocoo()
    inside = int(compact.data.sum(dtype=np.uint64))
    fast = signs[selected][compact.col] != 0
    total_in = int(incoming[selected].sum(dtype=np.uint64))
    total_out = int(outgoing[selected].sum(dtype=np.uint64))
    strata = Counter((metadata[i][2], metadata[i][3], int(signs[i])) for i in selected)
    return dict(neurons=len(selected), inside_raw_edges=compact.nnz, inside_raw_contacts=inside,
        inside_fast_edges=int(fast.sum()), inside_fast_contacts=int(compact.data[fast].sum(dtype=np.uint64)),
        incoming_cut_fraction=(total_in-inside)/total_in if total_in else None,
        outgoing_cut_fraction=(total_out-inside)/total_out if total_out else None,
        overlap_with_candidate=len(set(int(ids[i]) for i in selected) & set(int(ids[i]) for i in reference)),
        strata=[dict(superclass=key[0], side=key[1], fast_sign=key[2], neurons=value) for key, value in sorted(strata.items())])


def audit(source, candidates_path, destination):
    candidate_report = json.loads(candidates_path.with_name('summary.json').read_text(encoding='utf8'))
    if sha256(candidates_path) != candidate_report['tables'][candidates_path.name]:
        raise ValueError('Candidate identities changed')
    if sha256(Path('flm/circuit_selection.py')) != candidate_report['audit_source_sha256']:
        raise ValueError('Candidate rule source changed')
    hashes = {name: sha256(source/name) for name in candidate_report['runtime_source_sha256']}
    if hashes != candidate_report['runtime_source_sha256']:
        raise ValueError('Source graph changed')
    metadata = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    load = lambda name: np.load(source/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    offsets, sources, counts, signs, ids = (load(name) for name in ('offsets','sources','counts','fast_sign','body_ids'))
    if len(metadata) != len(ids) or not np.array_equal(ids, np.array([row[0] for row in metadata])):
        raise ValueError('Metadata identities disagree')
    matrix = csr_matrix((counts, sources, offsets), shape=(len(ids), len(ids)), copy=False)
    incoming = np.asarray(matrix.sum(axis=1, dtype=np.uint64)).ravel()
    outgoing = np.asarray(matrix.sum(axis=0, dtype=np.uint64)).ravel()
    index = {int(body): i for i, body in enumerate(ids)}
    candidates = json.loads(candidates_path.read_text(encoding='utf8'))
    memberships, records = {}, {}
    for candidate, body_ids in candidates.items():
        reference = np.array([index[int(body)] for body in body_ids], dtype=np.int64)
        selections = dict(candidate=reference, **controls(matrix, metadata, ids, signs, reference))
        records[candidate] = {}
        for label, selected in selections.items():
            memberships[candidate+'/'+label] = sorted(int(ids[i]) for i in selected)
            record = describe(matrix, metadata, ids, signs, selected, reference, incoming, outgoing)
            if label.startswith('stratified') and record['strata'] != records[candidate]['candidate']['strata']:
                raise AssertionError('Stratum counts changed')
            if record['neurons'] != len(reference):
                raise AssertionError('Node budget changed')
            records[candidate][label] = record
        # Independent sparse submatrix accounting must reproduce earlier scan.
        original, repeated = candidate_report['candidates'][candidate], records[candidate]['candidate']
        for field in ('connections','contacts'):
            for kind in ('raw','fast'):
                key = 'inside_'+kind+('_edges' if field == 'connections' else '_contacts')
                if original['inside_'+kind][field] != repeated[key]:
                    raise AssertionError('Candidate edge scan mismatch')
        for field in ('incoming_cut_fraction','outgoing_cut_fraction'):
            if original[field] != repeated[field]:
                raise AssertionError('Candidate boundary mismatch')
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'control-body-ids.json', memberships)
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        status='Untrained selector inventories; within-subset rewiring and language protocol not yet frozen',
        audit_source_sha256=sha256(Path(__file__)), runtime_source_sha256=hashes,
        candidate_summary_sha256=sha256(candidates_path.with_name('summary.json')),
        candidate_body_ids_sha256=sha256(candidates_path), population_classes=list(POPULATION_CLASSES),
        population_neurons=sum(row[2] in POPULATION_CLASSES for row in metadata),
        population_rule='Common union of the six source superclasses found across all eight candidate inventories; body-ID-sorted; no exclusion of candidate cells',
        ranked_rule='Descending exact integer in+out raw contacts within the common eligible population; self contacts count twice; body-ID tie break',
        random_rule='NumPy PCG64 without replacement; uniform matches N only; stratified additionally matches superclass, source side and fast sign counts',
        random_seeds=list(SEEDS), selection_count=len(memberships), candidates=records,
        control_body_ids_sha256=sha256(destination/'control-body-ids.json'),
        limitations=['The eligible population is broader than the historic cb_intrinsic-only ranking, to include every superclass appearing in either threshold inventory.',
            'Node count or annotation strata do not match degrees, density, trainable parameters, biological function or compute.',
            'Within-subset degree/sign/weight-preserving rewiring is still required for a topology comparison.',
            'Candidate families, hemispheres and shared random seeds are not independent biological or training replications.',
            'All eight anatomy candidates and seven controls each are retained; this is not a declaration of 64 language fits.'])
    write_json(destination/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/processed/connectome'))
    parser.add_argument('--candidates', type=Path, default=Path('reports/circuit-selection/candidate-body-ids.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/selection-controls'))
    args = parser.parse_args()
    report = audit(args.source, args.candidates, args.output)
    print('Verified', report['selection_count'], 'node selections from', report['population_neurons'], 'eligible cells')


if __name__ == '__main__': main()
