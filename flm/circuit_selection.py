"""Operational KC-centered candidate selection, prior to language fitting.

These are literature-motivated body-level subsets, not intact or validated
biological circuits. The threshold selects members, not induced-graph edges.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .provenance import sha256, write_json
from .selection_feasibility import FAMILIES, family, measure

SEED_TYPES = ('KCg-d', 'KCg-m')
SIDES = ('L', 'R')
THRESHOLDS = (1, 5)
AUXILIARY = ('PAM', 'PPL1', 'APL', 'DPM')
RULE_VERSION = 'kc-centered-one-step-v1'


def select(matrix, metadata, signs, seed_type, side, threshold):
    n = len(metadata)
    if (matrix.shape != (n, n) or signs.shape != (n,) or seed_type not in SEED_TYPES
            or side not in SIDES or threshold not in THRESHOLDS
            or not np.isin(signs, [-1, 0, 1]).all()):
        raise ValueError('Invalid circuit selection inputs')
    types = [r[1] for r in metadata]
    families = np.array([family(name) for name in types])
    seed = np.array([r[1] == seed_type and r[3] == side for r in metadata])
    if not seed.any():
        raise ValueError('Empty KC seed population')
    inputs = np.zeros(n, dtype=bool)
    incoming = matrix[seed].tocoo()
    inputs[incoming.col[incoming.data >= threshold]] = True
    inputs &= ~np.isin(families, ('KC', 'MBON', *AUXILIARY)) & (signs != 0)
    outputs = np.zeros(n, dtype=bool)
    outgoing = matrix[:, seed].tocoo()
    outputs[outgoing.row[outgoing.data >= threshold]] = True
    outputs &= families == 'MBON'
    # Auxiliary membership uses either direction, at any acquired positive count.
    # No side filter: cross-side partners remain explicit in the resulting card.
    anchors = seed | outputs
    auxiliary = np.zeros(n, dtype=bool)
    auxiliary[matrix[anchors].tocoo().col] = True
    auxiliary[matrix[:, anchors].tocoo().row] = True
    auxiliary &= np.isin(families, AUXILIARY)
    return dict(seed_kc=seed, input_partner=inputs, output_mbon=outputs, auxiliary_partner=auxiliary)


def audit(source, destination):
    card = json.loads(Path('data/graphs/central-1024/graph-card.json').read_text(encoding='utf8'))
    hashes = {name: sha256(source/name) for name in card['source_file_sha256']}
    if hashes != card['source_file_sha256']:
        raise ValueError('Acquired runtime files changed')
    metadata = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    load = lambda name: np.load(source/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    offsets, sources, counts, signs, ids = (load(name) for name in ('offsets', 'sources', 'counts', 'fast_sign', 'body_ids'))
    if len(metadata) != len(ids) or not np.array_equal(ids, np.array([r[0] for r in metadata])):
        raise ValueError('Runtime metadata identities disagree')
    matrix = csr_matrix((counts, sources, offsets), shape=(len(ids), len(ids)), copy=False)
    if not matrix.has_canonical_format or np.any(counts <= 0):
        raise ValueError('Source must have unique sorted positive body-pair entries')
    masks, memberships = {}, {}
    for seed_type in SEED_TYPES:
        for side in SIDES:
            for threshold in THRESHOLDS:
                name = f'{seed_type}-{side}-t{threshold}'
                roles = select(matrix, metadata, signs, seed_type, side, threshold)
                masks[name] = np.logical_or.reduce(list(roles.values()))
                memberships[name] = roles
    labels = (*FAMILIES, 'other')
    family_ids = np.array([labels.index(family(r[1])) for r in metadata])
    boundaries, _, _ = measure(offsets, sources, counts, signs, masks, family_ids)
    destination.mkdir(parents=True, exist_ok=True)
    member_rows, selected_ids = [], {}
    for name, selected in masks.items():
        roles = memberships[name]
        indices = np.flatnonzero(selected)
        record = boundaries[name]
        record.update(neurons=len(indices), role_counts={role: int(mask.sum()) for role, mask in roles.items()},
            superclasses=dict(Counter(metadata[i][2] for i in indices)),
            sides=dict(Counter(metadata[i][3] for i in indices)),
            fast_signs=dict(Counter(str(int(signs[i])) for i in indices)),
            seed_type=name.rsplit('-', 2)[0], seed_side=name.rsplit('-', 2)[1],
            membership_threshold=int(name.rsplit('t', 1)[1]))
        selected_ids[name] = sorted(int(ids[i]) for i in indices)
        # Anatomy-only node identity/role export; no learned values.
        for i in sorted(indices, key=lambda i: int(ids[i])):
            member_rows.append(dict(candidate=name, body_id=int(ids[i]), type=metadata[i][1],
                superclass=metadata[i][2], side=metadata[i][3], fast_sign=int(signs[i]),
                role=';'.join(role for role, mask in roles.items() if mask[i])))
        # Quantify how much selected output/input partners depend on excluded cells.
        for role, mask in roles.items():
            internal_in = int(matrix[mask][:, selected].sum(dtype=np.uint64))
            all_in = int(matrix[mask].sum(dtype=np.uint64))
            internal_out = int(matrix[selected][:, mask].sum(dtype=np.uint64))
            all_out = int(matrix[:, mask].sum(dtype=np.uint64))
            record.setdefault('role_boundaries', {})[role] = dict(
                incoming_contacts=all_in, incoming_retained_raw=internal_in,
                outgoing_contacts=all_out, outgoing_retained_raw=internal_out,
                incoming_cut_fraction=(all_in-internal_in)/all_in if all_in else None,
                outgoing_cut_fraction=(all_out-internal_out)/all_out if all_out else None)
    with (destination/'members.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(member_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(member_rows)
    write_json(destination/'candidate-body-ids.json', selected_ids)
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), rule_version=RULE_VERSION,
        status='Candidate graphs measured; no final language condition selection or fits',
        audit_source_sha256=sha256(Path(__file__)), runtime_source_sha256=hashes,
        boundary_audit_source_sha256=sha256(Path('flm/selection_feasibility.py')),
        seed_types=list(SEED_TYPES), seed_sides=list(SIDES), edge_thresholds=list(THRESHOLDS),
        rule=dict(seed='All cells of the exact KC type and source side; no rank truncation',
            inputs='All non-KC/non-MBON/non-auxiliary fast-sign-nonzero cells with a qualifying edge onto a seed KC',
            outputs='All MBON-prefix cells receiving a qualifying edge from a seed KC',
            auxiliary='All PAM/PPL1/APL/DPM-prefix cells connected in either direction to a seed KC or selected MBON, at any positive count',
            stopping='One input/output expansion, then one auxiliary expansion; no recursion or size cap',
            side='Only seeds are side-filtered; connected partners may have any side or superclass',
            edges='Induced graph retains all acquired positive body pairs between selected cells, even below the membership threshold; fast calculation separately removes zero-sign sources'),
        candidates=boundaries,
        tables={name: sha256(destination/name) for name in ('members.csv', 'candidate-body-ids.json')},
        limitations=['An operational literature-motivated body-level selector, not an intact visual/olfactory circuit certification.',
            'The input-partner label describes a connection, not a verified sensory modality or calyx compartment.',
            'DPM transmitter identity is unresolved in the acquired annotations; its auxiliary role is a hypothesis.',
            'Large shared APL/DAN/MBON arbors can pull in cells with extensive external dependencies.',
            'Other KC types and upstream input partners are excluded even when strongly connected.',
            'Different candidate sizes require budget-matched comparator construction; current 1024 results are historical only.',
            'Zero-sign auxiliary sources remain ineffective in the current fast recurrent model.',
            'No selected node set is fitted, used for language selection or substituted into the browser.'])
    write_json(destination/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/processed/connectome'))
    parser.add_argument('--output', type=Path, default=Path('reports/circuit-selection'))
    args = parser.parse_args()
    report = audit(args.source, args.output)
    for name, row in report['candidates'].items():
        print(name, row['neurons'], row['role_counts'], row['incoming_cut_fraction'], row['outgoing_cut_fraction'])


if __name__ == '__main__': main()
