"""Lossless, source-indexed BabyLM blocks and bounded-memory token caches."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import numpy as np
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from .babylm import COMPONENTS
from .babylm_audit import normalized_key
from .provenance import sha256, write_json
from .tokenizer import Lexicon, OFFSET

BLOCK_BYTES = 16384


def blocks(path, target=BLOCK_BYTES):
    """Yield complete lines without changing, discarding or duplicating a byte."""
    if target < 1: raise ValueError('Positive block target required')
    start = end = 0; first_line = 1; lines = []
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if lines and end - start + len(line) > target:
                yield b''.join(lines), start, end, first_line, number - 1
                start = end; first_line = number; lines = []
            lines.append(line); end += len(line)
    if lines: yield b''.join(lines), start, end, first_line, number


def tokenizer_train(records, output):
    output.mkdir(parents=True, exist_ok=True); path = output / 'tokenizer.json'
    inputs = {r['component']: r['sha256'] for r in records}
    card_path = output / 'tokenizer-card.json'
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding='utf8'))
        if card['training_sources'] != inputs or card['tokenizer_sha256'] != sha256(path):
            raise ValueError('Existing tokenizer identity differs')
        return Lexicon(path)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096 - OFFSET, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[], show_progress=False)
    iterator = (payload.decode('utf8') for record in records for payload, *_ in blocks(Path(record['path'])))
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    tokenizer.save(str(path)); lexicon = Lexicon(path); lexicon.export(output / 'browser-tokenizer.json')
    write_json(card_path, dict(algorithm='Byte-level BPE', implementation=f'tokenizers {tokenizers.__version__}',
        vocabulary=lexicon.vocabulary, tokenizer_sha256=lexicon.sha256, training_sources=inputs,
        fitted_partition='train-10m', validation_or_test_used_for_training=False,
        normalization='none', prefix_space=False, all_bytes_retained=True, block_target_bytes=BLOCK_BYTES,
        boundaries=dict(bos=0, eos=1, offset=2), encoding_audit='Every encoded block is checked for exact byte reconstruction.'))
    return lexicon


def overlap_count(payload, db):
    count = 0
    for line in io.StringIO(payload.decode('utf8'), newline=''):
        key = normalized_key(line)
        if key is not None:
            row = db.execute('SELECT c0,c1 FROM lines WHERE digest=?', (key[0],)).fetchone()
            count += bool(row and (row[0] or row[1]))
    return count


def encode_partition(records, output, lexicon, db=None, audit_sha=None):
    output.mkdir(parents=True, exist_ok=True); manifest_path = output / 'manifest.json'
    identity = dict(tokenizer_sha256=lexicon.sha256, sources={r['component']: r['sha256'] for r in records},
        block_target_bytes=BLOCK_BYTES, overlap_audit_sha256=audit_sha)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf8'))
        if old['identity'] != identity: raise ValueError('Existing cache has different inputs')
        for name, digest in old['files'].items():
            if sha256(output / name) != digest: raise ValueError(f'Existing cache changed: {name}')
        return old
    offsets = [0]; component_stats = {}; total_bytes = 0
    with (output / 'tokens.u16').open('wb') as token_file, (output / 'blocks.jsonl').open('w', encoding='utf8', newline='\n') as metadata:
        for source in records:
            stats = dict(blocks=0, text_tokens=0, utf8_bytes=0, overlapping_blocks=0, overlapping_block_bytes=0, overlapping_lines=0)
            for payload, start, end, first, last in blocks(Path(source['path'])):
                tokens = np.asarray(lexicon.encode(payload.decode('utf8'), boundaries=True), dtype='<u2')
                if b''.join(lexicon.pieces[int(t)] for t in tokens) != payload:
                    raise ValueError('Encoding changed the original bytes')
                overlap = overlap_count(payload, db) if db is not None else 0
                block = dict(id=f"{source['partition']}/{source['component']}/{source['sha256'][:16]}/{start}-{end}",
                    component=source['component'], source_sha256=source['sha256'], byte_start=start, byte_end=end,
                    first_line=first, last_line=last, text_sha256=hashlib.sha256(payload).hexdigest(),
                    token_start=offsets[-1], token_end=offsets[-1] + len(tokens), utf8_bytes=len(payload),
                    overlapping_training_lines=overlap, overlap_filtered_eligible=overlap == 0 if db is not None else None)
                token_file.write(tokens.tobytes()); offsets.append(block['token_end'])
                metadata.write(json.dumps(block) + '\n'); total_bytes += len(payload)
                stats['blocks'] += 1; stats['text_tokens'] += len(tokens) - 2; stats['utf8_bytes'] += len(payload)
                stats['overlapping_blocks'] += overlap > 0
                stats['overlapping_block_bytes'] += len(payload) if overlap else 0
                stats['overlapping_lines'] += overlap
            if stats['utf8_bytes'] != source['utf8_bytes']: raise ValueError('Cache did not preserve the complete source')
            component_stats[source['component']] = stats
            print(json.dumps(dict(partition=source['partition'], component=source['component'], **stats)), flush=True)
    np.save(output / 'offsets.npy', np.asarray(offsets, dtype='<u8'), allow_pickle=False)
    card = dict(format='flm-mmap-tokens-v1', dtype='<u2', identity=identity,
        blocks=len(offsets) - 1, text_tokens=offsets[-1] - 2 * (len(offsets) - 1), utf8_bytes=total_bytes,
        components=component_stats, files={name: sha256(output / name) for name in ('tokens.u16', 'offsets.npy', 'blocks.jsonl')},
        boundary_semantics='Artificial whole-line blocks within a component file; true document and speaker boundaries unknown.',
        roundtrip='Every block reconstructs its exact source bytes; every source byte is retained once.')
    write_json(manifest_path, card); return card


def validation_panel(cache):
    records = [json.loads(line) for line in (cache / 'blocks.jsonl').read_text(encoding='utf8').splitlines()]
    selected = []
    for component in COMPONENTS:
        candidates = [r for r in records if r['component'] == component]
        candidates.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        if len(candidates) < 8: raise ValueError('Validation component has fewer than eight blocks')
        selected.extend(candidates[:8])
    return dict(selection='Eight blocks per component, sorted by SHA-256 of stable block ID; no losses used.',
        cache_manifest_sha256=sha256(cache / 'manifest.json'), target_tokens_per_block=1024,
        ids=[r['id'] for r in selected], overlapping_blocks=sum(not r['overlap_filtered_eligible'] for r in selected),
        components={c: [r['id'] for r in selected if r['component'] == c] for c in COMPONENTS})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/processed/babylm-2026-bpe'))
    p.add_argument('--tokenizer', type=Path, default=Path('data/tokenizers/babylm-2026-4096'))
    p.add_argument('--database', type=Path, default=Path('runs/babylm/line-audit.sqlite'))
    a = p.parse_args(); acquisition = Path('data/cards/babylm-2026-acquisition.json')
    audit_path = Path('data/cards/babylm-2026-overlap.json')
    card = json.loads(acquisition.read_text(encoding='utf8')); audit = json.loads(audit_path.read_text(encoding='utf8'))
    if audit['acquisition_card_sha256'] != sha256(acquisition): raise ValueError('Audit acquisition identity changed')
    for record in card['files']:
        if sha256(Path(record['path'])) != record['sha256']: raise ValueError('Raw source changed')
    db = sqlite3.connect(a.database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for record in card['files']:
            completed = db.execute('SELECT sha256 FROM completed WHERE path=?', (record['path'],)).fetchone()
            if completed != (record['sha256'],): raise ValueError('Incomplete or stale overlap database')
        lexicon = tokenizer_train([r for r in card['files'] if r['partition'] == 'train-10m'], a.tokenizer)
        reports = {}
        for partition in ('train-10m', 'validation', 'test', 'train-100m'):
            reports[partition] = encode_partition([r for r in card['files'] if r['partition'] == partition],
                a.output / partition, lexicon, db if partition in ('validation', 'test') else None,
                sha256(audit_path) if partition in ('validation', 'test') else None)
            if partition == 'validation':
                write_json(a.tokenizer / 'validation-panel.json', validation_panel(a.output / partition))
        write_json(a.tokenizer / 'tokenization-card.json', dict(dataset='BabyLM 2026 English',
            tokenizer_sha256=lexicon.sha256, acquisition_card_sha256=sha256(acquisition), partitions=reports,
            test_usage='Source identity, overlap and lossless encoding only; no test losses or checkpoint selection.'))
    finally: db.close()


if __name__ == '__main__': main()
