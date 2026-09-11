"""Anatomy-only feasibility audit for a future neuron-selection comparison.

Named families are candidate inventories, not validated functional circuits.
No graph is substituted into a model and no language data is opened here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from .provenance import sha256, write_json

FAMILIES = ('KC', 'MBON', 'PAM', 'PPL1', 'APL', 'DPM', 'EPG', 'PEN', 'PEG')
MB_FAMILIES = FAMILIES[:6]
RANDOM_SEEDS = (201, 203, 207)


def family(name):
    return next((prefix for prefix in FAMILIES if name.startswith(prefix)), 'other')


def measure(offsets, sources, counts, signs, masks, family_ids, *, chunk_rows=1024):
    n = len(signs)
    if (not n or offsets.shape != (n+1,) or offsets[0] != 0 or offsets[-1] != len(sources)
            or len(sources) != len(counts) or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or not np.isin(signs, [-1, 0, 1]).all() or family_ids.shape != (n,)
            or any(x.shape != (n,) or x.dtype != np.bool_ for x in masks.values())
            or not masks or type(chunk_rows) is not int or chunk_rows < 1
            or not all(np.issubdtype(x.dtype, np.integer) for x in (offsets, sources, counts, family_ids))
            or np.any(family_ids < 0) or np.any(family_ids > len(FAMILIES))):
        raise ValueError('Invalid selection audit inputs')
    groups = ('inside_raw', 'inside_fast', 'incoming_boundary', 'outgoing_boundary')
    records = {name: {key: dict(connections=0, contacts=0) for key in groups} for name in masks}
    size = len(FAMILIES) + 1
    cross = np.zeros((size, size), dtype=np.uint64)
    cross_fast = np.zeros_like(cross)
    for start in range(0, n, chunk_rows):
        end = min(n, start+chunk_rows)
        a, b = int(offsets[start]), int(offsets[end])
        pre = np.asarray(sources[a:b], dtype=np.int64)
        post = np.repeat(np.arange(start, end), np.diff(offsets[start:end+1]).astype(np.int64))
        if np.any(counts[a:b] <= 0):
            raise ValueError('Invalid edge source or contact count')
        contacts = np.asarray(counts[a:b], dtype=np.uint64)
        if np.any(pre < 0) or np.any(pre >= n):
            raise ValueError('Invalid edge source or contact count')
        fast = signs[pre] != 0
        # Exact integer accumulation, avoiding floating weighted bincount sums.
        np.add.at(cross, (family_ids[pre], family_ids[post]), contacts)
        np.add.at(cross_fast, (family_ids[pre[fast]], family_ids[post[fast]]), contacts[fast])
        for name, selected in masks.items():
            ps, qs = selected[pre], selected[post]
            inside = ps & qs
            selections = (inside, inside & fast, ~ps & qs, ps & ~qs)
            for key, chosen in zip(groups, selections):
                records[name][key]['connections'] += int(chosen.sum())
                records[name][key]['contacts'] += int(contacts[chosen].sum(dtype=np.uint64))
    for record in records.values():
        inside = record['inside_raw']['contacts']
        incoming = record['incoming_boundary']['contacts']
        outgoing = record['outgoing_boundary']['contacts']
        record['incoming_cut_fraction'] = incoming/(inside+incoming) if inside+incoming else None
        record['outgoing_cut_fraction'] = outgoing/(inside+outgoing) if inside+outgoing else None
        record['inside_contacts_removed_by_fast_sign'] = inside-record['inside_fast']['contacts']
        record['inside_fast_fraction'] = record['inside_fast']['contacts']/inside if inside else None
    return records, cross, cross_fast


def candidates(metadata, ids, original_indices, budget=1024):
    eligible = np.array([row[2] == 'cb_intrinsic' for row in metadata])
    if budget < 1 or int(eligible.sum()) < budget:
        raise ValueError('Insufficient eligible population for requested budget')
    original = np.zeros(len(ids), dtype=bool); original[original_indices] = True
    mb = np.array([family(row[1]) in MB_FAMILIES for row in metadata])
    compass = np.array([family(row[1]) in ('EPG', 'PEN', 'PEG') for row in metadata])
    masks = dict(frozen_connectivity_1024=original, named_mb_bilateral=mb,
                 named_mb_left=mb & np.array([r[3] == 'L' for r in metadata]),
                 named_mb_right=mb & np.array([r[3] == 'R' for r in metadata]),
                 named_compass_families=compass)
    population = np.flatnonzero(eligible)
    population = population[np.argsort(ids[population], kind='stable')]
    for seed in RANDOM_SEEDS:
        indices = np.random.Generator(np.random.PCG64(seed)).choice(population, budget, replace=False)
        mask = np.zeros(len(ids), dtype=bool); mask[indices] = True
        masks[f'uniform_{budget}_s{seed}'] = mask
    return masks


def audit(source, graph_path, output):
    card = json.loads(graph_path.with_name('graph-card.json').read_text(encoding='utf8'))
    if sha256(graph_path) != card['graph_sha256']:
        raise ValueError('Frozen reference graph changed')
    hashes = {name: sha256(source/name) for name in card['source_file_sha256']}
    if hashes != card['source_file_sha256']:
        raise ValueError('Acquired source files changed')
    metadata = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    load = lambda name: np.load(source/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    offsets, sources, counts, signs, ids = (load(name) for name in ('offsets', 'sources', 'counts', 'fast_sign', 'body_ids'))
    if len(metadata) != len(ids) or not np.array_equal(ids, np.array([row[0] for row in metadata])):
        raise ValueError('Metadata and CSR source identities disagree')
    with np.load(graph_path, allow_pickle=False) as graph:
        original = graph['source_indices'].copy()
        if not np.array_equal(ids[original], graph['body_ids']):
            raise ValueError('Frozen subset source IDs changed')
    masks = candidates(metadata, ids, original)
    family_labels = (*FAMILIES, 'other')
    encoded = np.array([family_labels.index(family(row[1])) for row in metadata])
    records, cross, fast = measure(offsets, sources, counts, signs, masks, encoded)
    selected_ids = {}
    for name, selected in masks.items():
        rows = [metadata[i] for i in np.flatnonzero(selected)]
        records[name].update(neurons=int(selected.sum()), families=dict(Counter(family(r[1]) for r in rows)),
            sides=dict(Counter(r[3] for r in rows)), classes=dict(Counter(r[2] for r in rows)),
            fast_signs={str(k): int(v) for k, v in zip(*np.unique(signs[selected], return_counts=True))},
            exceeds_1024_budget=int(selected.sum()) > 1024)
        selected_ids[name] = sorted(int(i) for i in ids[selected])
    # This independently computed reference must reproduce the published audit.
    reference = records['frozen_connectivity_1024']
    if reference['neurons'] != 1024 or reference['inside_fast']['connections'] != card['edges']:
        raise ValueError('Reference neuron/edge counts do not reproduce the frozen model')
    family_inventory = {}
    for i, label in enumerate(FAMILIES):
        indices = np.flatnonzero(encoded == i)
        family_inventory[label] = dict(neurons=len(indices),
            sides=dict(Counter(metadata[j][3] for j in indices)),
            source_transmitter_labels=dict(Counter(metadata[j][4] for j in indices)),
            fast_signs={str(k): int(v) for k, v in zip(*np.unique(signs[indices], return_counts=True))})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'candidate-body-ids.json', selected_ids)
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        scope='Anatomy-only descriptive feasibility; named cell families are not validated functional circuits. No trained comparison.',
        source_neurons=len(ids), source_connections=len(sources), source_file_sha256=hashes,
        source_graph_sha256=sha256(graph_path), audit_source_sha256=sha256(Path(__file__)),
        candidates=records, family_inventory=family_inventory,
        family_cross_contacts=dict(order=list(family_labels), orientation='row presynaptic, column postsynaptic',
                                   raw=cross.tolist(), retained_fast=fast.tolist()),
        candidate_body_ids_sha256=sha256(output/'candidate-body-ids.json'),
        random_selection=dict(seeds=list(RANDOM_SEEDS), generator='NumPy PCG64', population='All cb_intrinsic, sorted by body ID',
                              rule='1024 uniformly sampled distinct neurons; neuron count only matched, not degrees, signs, side or parameter count'),
        named_selection=dict(mushroom_body_prefixes=list(MB_FAMILIES), compass_prefixes=['EPG','PEN','PEG'],
                             rule='Literal case-sensitive type prefixes, optionally side L/R; no connectivity expansion or language scores'),
        limitations=['The acquired source is a runtime derivative of MaleCNS, not a new audit of all publisher synapse data.',
                     'Named inventories omit external partners and do not establish intact circuits or functional preservation.',
                     'Source transmitter labels and fast-sign rules are modeling inputs, not ground-truth physiology.',
                     'Different induced edge counts imply different trainable parameter counts under the current model.',
                     'Candidate sizes differ; this audit is not a matched language experiment.'])
    write_json(output/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/processed/connectome'))
    parser.add_argument('--graph', type=Path, default=Path('data/graphs/central-1024/graph.npz'))
    parser.add_argument('--output', type=Path, default=Path('reports/selection-feasibility'))
    args = parser.parse_args()
    report = audit(args.source, args.graph, args.output)
    for name, row in report['candidates'].items():
        print(name, row['neurons'], row['inside_fast']['connections'], row['incoming_cut_fraction'], row['outgoing_cut_fraction'])


if __name__ == '__main__':
    main()
