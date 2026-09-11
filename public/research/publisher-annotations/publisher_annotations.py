"""Reconcile the frozen runtime's body IDs with pinned publisher annotations."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import requests

from .provenance import sha256, write_json
from .selection_feasibility import FAMILIES, family

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
ANNOTATIONS = 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
TRANSMITTERS = 'body-neurotransmitters-male-cns-v1.0.feather'


def acquire(card_path, cache):
    """Replay published object-generation and SHA-256 pins; never refresh them."""
    card = json.loads(card_path.read_text(encoding='utf8'))
    if card.get('complete') is not True or [r['filename'] for r in card['files']] != [ANNOTATIONS, TRANSMITTERS]:
        raise ValueError('Complete two-file annotation acquisition card required')
    cache.mkdir(parents=True, exist_ok=True)
    for record in card['files']:
        name = record['filename']; generation = record['generation']
        expected = BASE + name + '?generation=' + generation
        if not generation.isdigit() or record['url'] != BASE+name or record['pinned_url'] != expected:
            raise ValueError('Unexpected annotation source or object generation')
        path = cache/name
        if path.exists() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']:
            continue
        temporary = path.with_suffix('.part'); size = 0
        with requests.get(expected, headers={'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            if response.headers.get('x-goog-generation') != generation:
                raise ValueError('Publisher object generation changed')
            with temporary.open('wb') as handle:
                for block in response.iter_content(65536):
                    size += len(block)
                    if size > record['bytes']:
                        raise ValueError('Annotation download exceeds pinned size')
                    handle.write(block)
        if size != record['bytes'] or sha256(temporary) != record['sha256']:
            raise ValueError('Annotation download checksum mismatch')
        temporary.replace(path)
    return card


def text(value):
    return '' if value is None else str(value)


def index_unique(records, key):
    result = {}
    for row in records:
        identity = row[key]
        if type(identity) is not int or identity < 0 or identity in result:
            raise ValueError('Publisher body IDs must be unique nonnegative integers')
        result[identity] = row
    return result


def compare(runtime, annotations, transmitters):
    """Join by identity, preserving missingness and distinct evidence fields."""
    annotation = index_unique(annotations, 'bodyId')
    nt = index_unique(transmitters, 'body')
    if len({row[0] for row in runtime}) != len(runtime):
        raise ValueError('Duplicate runtime body IDs')
    counts = Counter(); joined = []
    for raw in runtime:
        if len(raw) < 6 or type(raw[0]) is not int or raw[5] not in (-1, 0, 1):
            raise ValueError('Malformed runtime metadata')
        identity, kind, superclass, side, transmitter, sign = raw[:6]
        a, t = annotation.get(identity), nt.get(identity)
        counts['runtime_neurons'] += 1
        counts['annotation_present'] += a is not None
        counts['transmitter_present'] += t is not None
        if a is not None:
            counts['type_exact_matches'] += text(kind) == text(a.get('type'))
            counts['superclass_exact_matches'] += text(superclass) == text(a.get('superclass'))
            counts['soma_side_exact_matches'] += text(side) == text(a.get('somaSide'))
            derived = a.get('somaSide') or a.get('rootSide') or ''
            counts['soma_then_root_side_matches'] += text(side) == text(derived)
        if t is not None:
            counts['consensus_nt_matches'] += text(transmitter) == text(t.get('consensus_nt'))
            counts['body_predicted_nt_matches'] += text(transmitter) == text(t.get('predicted_nt'))
            derived_sign = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}.get(t.get('consensus_nt'), 0)
            counts['consensus_fast_sign_rule_matches'] += sign == derived_sign
            counts['ground_truth_field_nonempty'] += bool(t.get('ground_truth'))
            counts['ground_truth_matches_when_present'] += bool(t.get('ground_truth')) and text(transmitter) == text(t['ground_truth'])
        else:
            counts['missing_transmitter_with_zero_fast_sign'] += sign == 0
        a, t = a or {}, t or {}
        joined.append(dict(body_id=identity, runtime_type=kind, publisher_type=text(a.get('type')),
            runtime_superclass=superclass, curated_class=text(a.get('class')), curated_subclass=text(a.get('subclass')),
            instance=text(a.get('instance')), hemibrain_type=text(a.get('hemibrainType')),
            runtime_side=side, soma_side=text(a.get('somaSide')), root_side=text(a.get('rootSide')),
            runtime_nt=transmitter, predicted_nt=text(t.get('predicted_nt')),
            predicted_nt_confidence=t.get('predicted_nt_confidence'), ground_truth=text(t.get('ground_truth')),
            celltype_predicted_nt=text(t.get('celltype_predicted_nt')), consensus_nt=text(t.get('consensus_nt')),
            fast_sign=sign, annotation_present=identity in annotation, transmitter_present=identity in nt))
    return dict(counts), joined


def subset_table(path, key, body_ids, columns):
    table = feather.read_table(path, memory_map=True, use_threads=False)
    if pc.count_distinct(table[key]).as_py() != table.num_rows or table[key].null_count:
        raise ValueError('Publisher table body identities are not unique and complete')
    selected = table.select(columns).filter(pc.is_in(table[key], value_set=pa.array(body_ids)))
    return selected.to_pylist(), dict(rows=table.num_rows, columns=table.column_names,
                                      matched_runtime_rows=selected.num_rows)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to publish')
    with path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def audit(card_path, cache, source, output):
    card = json.loads(card_path.read_text(encoding='utf8'))
    if not card['complete']:
        raise ValueError('Both publisher tables must be acquired')
    for record in card['files']:
        if sha256(cache/record['filename']) != record['sha256']:
            raise ValueError('Publisher annotation file changed')
    graph_card = json.loads(Path('data/graphs/central-1024/graph-card.json').read_text(encoding='utf8'))
    for name in ('neurons.json.gz', 'body_ids.npy', 'fast_sign.npy'):
        if sha256(source/name) != graph_card['source_file_sha256'][name]:
            raise ValueError('Runtime metadata identity changed')
    runtime = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    ids = np.load(source/'body_ids.npy', allow_pickle=False)
    signs = np.load(source/'fast_sign.npy', allow_pickle=False)
    if not np.array_equal(ids, [r[0] for r in runtime]) or not np.array_equal(signs, [r[5] for r in runtime]):
        raise ValueError('Runtime metadata and computation arrays disagree')
    annotation_fields = ['bodyId', 'type', 'superclass', 'somaSide', 'rootSide', 'instance', 'class', 'subclass', 'hemibrainType']
    nt_fields = ['body', 'predicted_nt', 'predicted_nt_confidence', 'ground_truth', 'celltype_predicted_nt', 'consensus_nt']
    annotations, annotation_schema = subset_table(cache/ANNOTATIONS, 'bodyId', ids, annotation_fields)
    transmitters, nt_schema = subset_table(cache/TRANSMITTERS, 'body', ids, nt_fields)
    counts, joined = compare(runtime, annotations, transmitters)
    candidates = [r for r in joined if family(r['runtime_type']) != 'other' or r['curated_class'] == 'ALPN']
    missing = [r for r in joined if not r['transmitter_present']]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'candidate-annotations.csv', candidates)
    write_csv(output/'missing-transmitter-rows.csv', missing)
    families = {}
    for prefix in FAMILIES:
        rows = [r for r in candidates if family(r['runtime_type']) == prefix]
        families[prefix] = dict(neurons=len(rows), ground_truth_field_nonempty=sum(bool(r['ground_truth']) for r in rows),
            runtime_nt=dict(Counter(r['runtime_nt'] for r in rows)), consensus_nt=dict(Counter(r['consensus_nt'] for r in rows)),
            curated_classes=dict(Counter(r['curated_class'] for r in rows)),
            instances_nonempty=sum(bool(r['instance']) for r in rows))
    alpn = [r for r in candidates if r['curated_class'] == 'ALPN']
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), scope='Body-ID reconciliation; no graph replacement or circuit validation',
        counts=counts, publisher_annotations=annotation_schema, publisher_transmitters=nt_schema,
        family_inventory=families, candidate_rows=len(candidates),
        alpn_inventory=dict(neurons=len(alpn), superclasses=dict(Counter(r['runtime_superclass'] for r in alpn)),
                            subclasses=dict(Counter(r['curated_subclass'] for r in alpn))),
        missing_transmitter_runtime_labels=dict(Counter(r['runtime_nt'] for r in missing)),
        acquisition_card_sha256=sha256(card_path), audit_source_sha256=sha256(Path(__file__)),
        runtime_source_sha256={name: graph_card['source_file_sha256'][name] for name in ('neurons.json.gz','body_ids.npy','fast_sign.npy')},
        tables={name: sha256(output/name) for name in ('candidate-annotations.csv','missing-transmitter-rows.csv')},
        limitations=['The publisher ground_truth column is preserved as source annotation, not independently established here.',
                     'Consensus, per-body predictions and cell-type predictions are different fields; none is silently substituted for another.',
                     'Soma/root side and instance text do not specify every synaptic compartment or prove functional circuit membership.',
                     'The larger publisher tables include segments outside the acquired runtime; row counts are not comparable whole-neuron population estimates.'])
    write_json(output/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, default=Path('data/cards/malecns-annotations.json'))
    parser.add_argument('--cache', type=Path, default=Path('data/raw/malecns-annotations-v1'))
    parser.add_argument('--source', type=Path, default=Path('data/processed/connectome'))
    parser.add_argument('--output', type=Path, default=Path('reports/publisher-annotations'))
    parser.add_argument('--acquire', action='store_true')
    args = parser.parse_args()
    if args.acquire: acquire(args.card, args.cache)
    report = audit(args.card, args.cache, args.source, args.output)
    print(json.dumps(report['counts'], indent=2))


if __name__ == '__main__': main()
