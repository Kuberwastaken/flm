"""Body-level pathway coverage for candidate mushroom-body selection, without fitting.

CSR rows are postsynaptic. Group membership is annotation-based; contacts and
two-hop existence do not certify compartments, signal flow or preserved function.
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

from .provenance import sha256, write_json

GROUPS = ('ALPN', 'KC', 'MBON', 'PAM', 'PPL1', 'APL', 'DPM', 'other')
THRESHOLDS = (1, 5)


def encode_groups(metadata, annotation_rows):
    annotations = {int(row['body_id']): row for row in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise ValueError('Duplicate annotation body IDs')
    labels = []
    for row in metadata:
        annotation = annotations.get(int(row[0]))
        family = next((name for name in GROUPS[1:-1] if row[1].startswith(name)), 'other')
        if annotation is not None:
            if annotation['runtime_type'] != row[1] or annotation['publisher_type'] != row[1]:
                raise ValueError('Annotation type mismatch')
            if annotation['curated_class'] == 'ALPN':
                if family != 'other':
                    raise ValueError('Overlapping ALPN and named family membership')
                family = 'ALPN'
        elif family != 'other':
            raise ValueError('Missing named-family annotation')
        labels.append(GROUPS.index(family))
    return np.asarray(labels, dtype=np.int64)


def measure(offsets, sources, counts, signs, groups, *, chunk_rows=1024):
    n, k = len(signs), len(GROUPS)
    if (n < 1 or offsets.shape != (n+1,) or offsets[0] != 0 or offsets[-1] != len(sources)
            or counts.shape != sources.shape or groups.shape != (n,)
            or any(not np.issubdtype(x.dtype, np.integer) for x in (offsets, sources, counts, groups))
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or not np.isin(signs, [-1, 0, 1]).all() or np.any(groups < 0) or np.any(groups >= k)
            or type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError('Invalid pathway audit inputs')
    # Per-cell partner-group totals preserve direction and permit exact coverage.
    totals = {name: np.zeros((n, k), dtype=np.uint64) for name in
              ('incoming_contacts', 'outgoing_contacts', 'incoming_edges', 'outgoing_edges',
               'incoming_fast_contacts', 'outgoing_fast_contacts')}
    threshold_partners = {t: {d: np.zeros((n, k), dtype=np.uint64) for d in ('incoming', 'outgoing')}
                          for t in THRESHOLDS}
    for start in range(0, n, chunk_rows):
        end = min(n, start+chunk_rows)
        a, b = int(offsets[start]), int(offsets[end])
        pre = np.asarray(sources[a:b], dtype=np.int64)
        post = np.repeat(np.arange(start, end), np.diff(offsets[start:end+1]).astype(np.int64))
        if np.any(pre < 0) or np.any(pre >= n) or np.any(counts[a:b] <= 0):
            raise ValueError('Invalid source or contact count')
        contacts = np.asarray(counts[a:b], dtype=np.uint64)
        gp, gq = groups[pre], groups[post]
        fast = signs[pre] != 0
        for direction, indices in [('incoming', (post, gp)), ('outgoing', (pre, gq))]:
            np.add.at(totals[direction+'_contacts'], indices, contacts)
            np.add.at(totals[direction+'_edges'], indices, 1)
            np.add.at(totals[direction+'_fast_contacts'], (indices[0][fast], indices[1][fast]), contacts[fast])
            for threshold in THRESHOLDS:
                keep = contacts >= threshold
                np.add.at(threshold_partners[threshold][direction], (indices[0][keep], indices[1][keep]), 1)
    # Check conservation by both independently accumulated directions.
    for suffix in ('contacts', 'edges', 'fast_contacts'):
        outgoing = np.array([totals['outgoing_'+suffix][groups == g].sum(axis=0) for g in range(k)])
        incoming = np.array([totals['incoming_'+suffix][groups == g].sum(axis=0) for g in range(k)]).T
        if not np.array_equal(outgoing, incoming):
            raise AssertionError('Directed contact conservation failed')
    return totals, threshold_partners


def pathway_coverage(groups, threshold_partners, selected):
    kc = selected & (groups == GROUPS.index('KC'))
    records = {}
    for threshold, partners in threshold_partners.items():
        has_input = partners['incoming'][:, GROUPS.index('ALPN')] > 0
        has_output = partners['outgoing'][:, GROUPS.index('MBON')] > 0
        records[str(threshold)] = dict(kenyon_cells=int(kc.sum()),
            with_alpn_input=int((kc & has_input).sum()), with_mbon_output=int((kc & has_output).sum()),
            with_both=int((kc & has_input & has_output).sum()),
            without_either=int((kc & ~has_input & ~has_output).sum()))
    return records


def write_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def audit(source, annotation_path, output):
    card = json.loads(Path('data/graphs/central-1024/graph-card.json').read_text(encoding='utf8'))
    hashes = {name: sha256(source/name) for name in card['source_file_sha256']}
    if hashes != card['source_file_sha256']:
        raise ValueError('Acquired runtime files changed')
    annotation_summary = annotation_path.with_name('summary.json')
    annotation_report = json.loads(annotation_summary.read_text(encoding='utf8'))
    if sha256(annotation_path) != annotation_report['tables'][annotation_path.name]:
        raise ValueError('Publisher reconciliation table changed')
    with annotation_path.open(encoding='utf8', newline='') as handle:
        annotation_rows = list(csv.DictReader(handle))
    metadata = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    load = lambda name: np.load(source/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    offsets, sources, counts, signs, ids = (load(name) for name in ('offsets', 'sources', 'counts', 'fast_sign', 'body_ids'))
    if len(metadata) != len(ids) or not np.array_equal(ids, np.array([r[0] for r in metadata])):
        raise ValueError('Runtime metadata IDs disagree')
    groups = encode_groups(metadata, annotation_rows)
    if Counter(GROUPS[g] for g in groups)['ALPN'] != annotation_report['alpn_inventory']['neurons']:
        raise ValueError('ALPN inventory mismatch')
    totals, partners = measure(offsets, sources, counts, signs, groups)
    group_rows, pair_rows = [], []
    for g, name in enumerate(GROUPS):
        selected = groups == g
        group_rows.append(dict(group=name, neurons=int(selected.sum()),
            sides=dict(Counter(metadata[i][3] for i in np.flatnonzero(selected))),
            fast_signs=dict(Counter(str(int(signs[i])) for i in np.flatnonzero(selected)))))
        for h, target in enumerate(GROUPS):
            row = dict(pre=name, post=target)
            for key in ('contacts', 'edges', 'fast_contacts'):
                row[key] = int(totals['outgoing_'+key][selected, h].sum())
            row['removed_fast_contacts'] = row['contacts']-row['fast_contacts']
            for threshold in THRESHOLDS:
                row[f'pre_cells_ge_{threshold}'] = int((partners[threshold]['outgoing'][selected, h] > 0).sum())
                row[f'post_cells_ge_{threshold}'] = int((partners[threshold]['incoming'][groups == h, g] > 0).sum())
            pair_rows.append(row)
    # Every external type is exported, including weak connections; no top-k selection.
    external = {}
    for i in np.flatnonzero(groups == GROUPS.index('other')):
        for g, name in enumerate(GROUPS[:-1]):
            for direction, array in [('into_family', 'outgoing'), ('from_family', 'incoming')]:
                contacts = int(totals[array+'_contacts'][i, g])
                if not contacts:
                    continue
                key = (name, direction, metadata[i][1], metadata[i][2], metadata[i][3])
                if key not in external:
                    external[key] = dict(group=name, direction=direction, external_type=key[2],
                        external_superclass=key[3], external_side=key[4], partner_cells=0, contacts=0,
                        fast_contacts=0, edges=0)
                row = external[key]; row['partner_cells'] += 1; row['contacts'] += contacts
                row['fast_contacts'] += int(totals[array+'_fast_contacts'][i, g])
                row['edges'] += int(totals[array+'_edges'][i, g])
    external_rows = sorted(external.values(), key=lambda r: (r['group'], r['direction'], -r['contacts'], r['external_type'], r['external_superclass'], r['external_side']))
    # Exact KC type/side strata, not hand-picked examples.
    kc_strata = sorted({(r[1], r[3]) for i, r in enumerate(metadata) if groups[i] == GROUPS.index('KC')})
    coverage = []
    for name, side in kc_strata:
        selected = np.array([r[1] == name and r[3] == side for r in metadata])
        for threshold, row in pathway_coverage(groups, partners, selected).items():
            coverage.append(dict(type=name, side=side, minimum_contacts_per_edge=int(threshold), **row))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'directed-groups.csv', pair_rows)
    write_csv(output/'external-partners.csv', external_rows)
    write_csv(output/'kenyon-coverage.csv', coverage)
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        scope='Anatomy-only body-level pathway coverage; no functional certification or language fits',
        audit_source_sha256=sha256(Path(__file__)), runtime_source_sha256=hashes,
        annotation_summary_sha256=sha256(annotation_summary), annotation_table_sha256=sha256(annotation_path),
        groups=group_rows, source_connections=len(sources),
        all_source_contacts=sum(row['contacts'] for row in pair_rows),
        group_rule='ALPN uses publisher curated class; KC/MBON/PAM/PPL1/APL/DPM use literal case-sensitive type prefixes; all others grouped separately',
        edge_thresholds=list(THRESHOLDS), threshold_rule='Minimum contacts on each directed body pair, not summed group input; sensitivity audit only',
        kc_pathway_coverage=pathway_coverage(groups, partners, np.ones(len(ids), dtype=bool)),
        orientation='pre to post; CSR rows postsynaptic',
        external_partner_rule='All observed other-group type/superclass/side strata connected to each focal family; not a selected input interface',
        tables={name: sha256(output/name) for name in ('directed-groups.csv', 'external-partners.csv', 'kenyon-coverage.csv')},
        limitations=['Contacts are acquired runtime body-pair totals, not compartment-resolved publisher synapses.',
            'ALPN-to-KC-to-MBON existence is anatomical two-hop coverage, not evidence of signal transmission or learning.',
            'ALPN coverage omits non-ALPN sensory pathways; its absence is not proof that a KC lacks sensory input.',
            'Two thresholds describe robustness; neither defines a final functional selection rule.',
            'Fast-sign removal describes the existing model, not absent biological dopamine signaling.',
            'External partners exclude the seven named groups and are not a complete list of each group boundary.',
            'No new subset, model dynamics or language protocol is selected by this audit.'])
    write_json(output/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/processed/connectome'))
    parser.add_argument('--annotations', type=Path, default=Path('reports/publisher-annotations/candidate-annotations.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/selection-pathways'))
    args = parser.parse_args()
    report = audit(args.source, args.annotations, args.output)
    print(json.dumps(report['kc_pathway_coverage'], indent=2))


if __name__ == '__main__': main()
