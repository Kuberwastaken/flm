"""Validate and memory-map a portable source-indexed token corpus."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .provenance import sha256


def read_mmap(path, tokenizer_sha256=None):
    path = Path(path); manifest = json.loads((path / 'manifest.json').read_text(encoding='utf8'))
    if manifest['format'] != 'flm-mmap-tokens-v1' or manifest['dtype'] != '<u2':
        raise ValueError('Unsupported token cache')
    if tokenizer_sha256 is not None and manifest['identity']['tokenizer_sha256'] != tokenizer_sha256:
        raise ValueError('Cache tokenizer mismatch')
    if set(manifest['files']) != {'tokens.u16', 'offsets.npy', 'blocks.jsonl'}:
        raise ValueError('Cache file inventory is incomplete')
    for name, digest in manifest['files'].items():
        if name not in ('tokens.u16', 'offsets.npy', 'blocks.jsonl') or sha256(path / name) != digest:
            raise ValueError('Cache file changed or has an invalid name')
    offsets = np.load(path / 'offsets.npy', mmap_mode='r', allow_pickle=False)
    tokens = np.memmap(path / 'tokens.u16', mode='r', dtype='<u2')
    records = [json.loads(line) for line in (path / 'blocks.jsonl').read_text(encoding='utf8').splitlines()]
    if len(records) != manifest['blocks']: raise ValueError('Cache block count mismatch')
    if len(offsets) != len(records) + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(tokens):
        raise ValueError('Cache offsets do not cover the complete token file')
    documents = []
    for i, record in enumerate(records):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if end <= start + 1 or (record['token_start'], record['token_end']) != (start, end):
            raise ValueError('Invalid block token range')
        if int(tokens[start]) != 0 or int(tokens[end - 1]) != 1: raise ValueError('Missing block boundary')
        documents.append((record['id'], tokens[start:end]))
    return documents, records, manifest
