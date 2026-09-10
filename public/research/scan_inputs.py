"""Load an official SCAN training partition with its exact provenance and weights.

Only the chosen training source and processed training records are opened. No
download, test partition, model, prediction or experiment budget is involved.
The grammar parser is used solely to verify source integrity, never predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import scan
from .provenance import sha256
from .scan import BASE, FILES, REVISION, SPLITS, parse, statistics


def load_training_partition(root, split):
    if split not in SPLITS:
        raise ValueError('Unknown official SCAN split')
    root = Path(root)
    card_path = root / 'data/cards/scan.json'
    card = json.loads(card_path.read_text(encoding='utf8'))
    source_inventory = {name: dict(url=BASE + name, bytes=size, sha256=digest)
                        for name, (size, digest) in FILES.items()}
    parser_hash = sha256(Path(scan.__file__))
    if (card['revision'] != REVISION or card['source_files'] != source_inventory
            or card['parser_sha256'] != parser_hash):
        raise ValueError('SCAN source revision, inventory or parser changed')
    name = SPLITS[split][0]
    raw = root / 'data/raw/scan' / REVISION / name
    size, digest = FILES[name]
    if raw.stat().st_size != size or sha256(raw) != digest:
        raise ValueError('Pinned SCAN training source changed')
    source_rows = parse(raw.read_bytes(), name)
    processed = root / 'data/processed/scan' / split / 'train.jsonl'
    recorded = card['partitions'][split]['train']
    if processed.stat().st_size != recorded['bytes'] or sha256(processed) != recorded['sha256']:
        raise ValueError('Prepared SCAN training file changed')
    rows = [json.loads(line) for line in processed.read_text(encoding='utf8').splitlines()]
    if rows != source_rows:
        raise ValueError('Training rows, order, multiplicity or source metadata differ from the pinned source')
    # JSON object keys are strings in the acquired card's histogram fields.
    measured = json.loads(json.dumps(statistics(rows)))
    if recorded != dict(measured, bytes=processed.stat().st_size, sha256=sha256(processed)):
        raise ValueError('Training partition statistics differ from the acquired card')
    return rows, dict(format='flm-scan-training-input-v1', split=split, partition='train',
        revision=REVISION, source=name, source_sha256=digest,
        processed_path=processed.relative_to(root).as_posix(), processed_sha256=recorded['sha256'],
        card_sha256=sha256(card_path), parser_sha256=parser_hash,
        loader_sha256=sha256(Path(__file__)), rows=len(rows),
        unique_commands=measured['unique_commands'], repeated_rows=measured['repeated_rows'],
        order='Exact publisher file order, including every repeated row and original source line',
        test_partition_opened=False)
