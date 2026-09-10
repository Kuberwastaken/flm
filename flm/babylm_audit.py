"""Audit substantial exact/normalized line overlap without publishing corpus text."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from .provenance import sha256, write_json

PARTITIONS = ('train-10m', 'train-100m', 'validation', 'test')


def normalized_key(text):
    normalized = ' '.join(text.split()).casefold(); words = len(normalized.split())
    if words < 8 or len(normalized) < 40: return None
    return hashlib.sha256(normalized.encode('utf8')).digest(), words


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('CREATE TABLE IF NOT EXISTS lines (digest BLOB PRIMARY KEY, words INTEGER NOT NULL, c0 INTEGER NOT NULL, c1 INTEGER NOT NULL, c2 INTEGER NOT NULL, c3 INTEGER NOT NULL) WITHOUT ROWID')
    db.execute('CREATE TABLE IF NOT EXISTS completed (path TEXT PRIMARY KEY, sha256 TEXT, record TEXT)')
    return db


def add_file(db, path, partition, expected_hash):
    old = db.execute('SELECT sha256,record FROM completed WHERE path=?', (str(path),)).fetchone()
    if old:
        if old[0] != expected_hash: raise ValueError('Completed audit input changed')
        return json.loads(old[1])
    if sha256(path) != expected_hash: raise ValueError(f'Source changed: {path}')
    index = PARTITIONS.index(partition); counts = [0] * 4; counts[index] = 1
    statement = ('INSERT INTO lines VALUES (?,?,?,?,?,?) ON CONFLICT(digest) DO UPDATE SET '
                 + f'c{index}=c{index}+1')
    batch = []; candidate_lines = candidate_words = total_lines = 0
    with db:
        with path.open(encoding='utf8', newline='') as stream:
            for text in stream:
                total_lines += 1; key = normalized_key(text)
                if key is None: continue
                digest, words = key; batch.append((digest, words, *counts))
                candidate_lines += 1; candidate_words += words
                if len(batch) >= 10000: db.executemany(statement, batch); batch.clear()
        if batch: db.executemany(statement, batch)
        record = dict(path=str(path), partition=partition, total_lines=total_lines,
            substantial_lines=candidate_lines, substantial_words=candidate_words)
        db.execute('INSERT INTO completed VALUES (?,?,?)', (str(path), expected_hash, json.dumps(record)))
    return record


def summarize(db):
    unique = db.execute('SELECT COUNT(*) FROM lines').fetchone()[0]; comparisons = []
    for a in range(4):
        for b in range(a + 1, 4):
            rows = db.execute(f'SELECT COUNT(*),COALESCE(SUM(c{a}),0),COALESCE(SUM(c{b}),0),COALESCE(SUM(c{b}*words),0) FROM lines WHERE c{a}>0 AND c{b}>0').fetchone()
            comparisons.append(dict(first=PARTITIONS[a], second=PARTITIONS[b], shared_unique_normalized_lines=rows[0],
                occurrences_in_first=rows[1], occurrences_in_second=rows[2], words_in_second_matching_first=rows[3]))
    return dict(unique_substantial_normalized_lines=unique, comparisons=comparisons)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, default=Path('runs/babylm/line-audit.sqlite'))
    a = p.parse_args(); card_path = Path('data/cards/babylm-2026-acquisition.json')
    card = json.loads(card_path.read_text(encoding='utf8')); db = connect(a.database); records = []
    try:
        for record in card['files']:
            result = add_file(db, Path(record['path']), record['partition'], record['sha256']); records.append(result)
            print(json.dumps(result), flush=True)
        report = dict(acquisition_card_sha256=sha256(card_path), files=records, **summarize(db),
            normalization='Collapse whitespace and casefold; compare SHA-256 of complete lines with at least 8 words and 40 normalized characters.',
            limits='This detects substantial normalized whole-line overlap. It does not establish book, conversation or speaker disjointness, and it is not a near-duplicate paragraph audit. Common phrases can still match. The official training partitions remain unchanged.',
            test_usage='Text identity audit only; no model test loss or model selection.')
        write_json(Path('data/cards/babylm-2026-overlap.json'), report)
    finally: db.close()


if __name__ == '__main__': main()
