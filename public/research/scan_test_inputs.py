"""Verify an entire official SCAN test partition against frozen metadata.

The study evaluator owns the all-condition checkpoint gate and calls this only
after it passes. Source grammar interpretation verifies acquired targets; it is
never used to generate, filter, select or repair a model output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import scan
from .provenance import sha256
from .scan import BASE, FILES, REVISION, SPLITS, parse, statistics
from .scan_train import canonical


def load_test_partition(root, split, frozen_metadata):
    if split not in SPLITS:
        raise ValueError('Unknown official SCAN test split')
    root = Path(root); card_path = root/'data/cards/scan.json'
    if sha256(card_path) != frozen_metadata['card_sha256']:
        raise ValueError('SCAN test card differs from the pre-training identity')
    card = json.loads(card_path.read_text(encoding='utf8'))
    inventory = {name: dict(url=BASE+name, bytes=size, sha256=digest)
                 for name, (size, digest) in FILES.items()}
    parser_hash = sha256(Path(scan.__file__))
    expected_metadata = dict(card_sha256=sha256(card_path), revision=REVISION,
        partitions={key: dict(source=names[1], source_identity=inventory[names[1]],
            processed_path=f'data/processed/scan/{key}/test.jsonl',
            record=card['partitions'][key]['test']) for key, names in SPLITS.items()},
        payloads_opened=False)
    if (card['revision'] != REVISION or card['source_files'] != inventory
            or card['parser_sha256'] != parser_hash or frozen_metadata != expected_metadata):
        raise ValueError('Frozen SCAN test inventory, source revision or parser changed')
    # All paths come from code, never a path supplied by the metadata document.
    source = SPLITS[split][1]
    raw = root/'data/raw/scan'/REVISION/source
    size, digest = FILES[source]
    payload = raw.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Pinned SCAN test source changed')
    source_rows = parse(payload, source)
    path = root/f'data/processed/scan/{split}/test.jsonl'
    payload = path.read_bytes(); record = card['partitions'][split]['test']
    if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
        raise ValueError('Prepared SCAN test file changed')
    rows = [json.loads(line) for line in payload.decode('utf8').splitlines()]
    if rows != source_rows:
        raise ValueError('Test rows, order, multiplicity or source metadata differ from the pinned source')
    measured = json.loads(json.dumps(statistics(rows)))
    if record != dict(measured, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()):
        raise ValueError('Test partition statistics differ from the acquired card')
    # Recheck the files after parsing so the returned coverage cannot mix versions.
    if (sha256(card_path) != frozen_metadata['card_sha256'] or sha256(raw) != digest
            or sha256(path) != record['sha256']):
        raise ValueError('SCAN test files changed during verification')
    return rows, dict(format='flm-scan-test-input-v1', split=split, partition='test',
        frozen_metadata=frozen_metadata, source=source, source_sha256=digest,
        processed_path=path.relative_to(root).as_posix(), processed_sha256=record['sha256'],
        ordered_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest(), statistics=measured,
        parser_sha256=parser_hash, loader_sha256=sha256(Path(__file__)),
        coverage='Every publisher test row in original order, with source line and multiplicity',
        training_partition_opened=False)
