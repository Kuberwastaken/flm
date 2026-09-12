"""Small, versioned Colab corpora; no teacher weights or generated corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .provenance import sha256, write_json

ROLE_TOKENS = ['<|system|>', '<|user|>', '<|assistant|>', '<|end|>', '<|pad|>']


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def partition(key):
    bucket = int(digest(key)[:8], 16) % 100
    return 'test' if bucket < 5 else 'validation' if bucket < 10 else 'train'


def tokenizer_from_repo(root: Path):
    from tokenizers import Tokenizer
    source = root / 'data/tokenizers/babylm-2026-4096/tokenizer.json'
    card = json.loads(source.with_name('tokenizer-card.json').read_text())
    if sha256(source) != card['tokenizer_sha256']:
        raise ValueError('Tokenizer hash mismatch')
    tokenizer = Tokenizer.from_file(str(source))
    # The legacy runtime reserves IDs 0/1 and offsets native BPE IDs by two.
    # This independent tokenizer instead allocates real, non-colliding specials.
    tokenizer.add_special_tokens(['<|bos|>', '<|eos|>'] + ROLE_TOKENS)
    return tokenizer


def windows(tokens, labels, length):
    """Adjacent windows share the boundary token, never a scored target."""
    if len(tokens) != len(labels) or length < 1:
        raise ValueError('Invalid sequence/window')
    for start in range(0, len(tokens)-1, length):
        t, y = tokens[start:start+length+1], labels[start:start+length+1]
        if len(t) > 1 and any(v != -100 for v in y[1:]):
            yield dict(tokens=t, labels=y)


def conversation(tokenizer, messages):
    tokens, labels = [tokenizer.token_to_id('<|bos|>')], [-100]
    for message in messages:
        role = message['role']
        if role not in ('system', 'user', 'assistant'):
            raise ValueError('Unknown role')
        marker = tokenizer.token_to_id(f'<|{role}|>')
        text = tokenizer.encode(message['content'], add_special_tokens=False).ids
        piece = [marker] + text + [tokenizer.token_to_id('<|end|>')]
        tokens.extend(piece)
        # Predict assistant content and its end token, never the assistant role marker.
        labels.extend([-100] + piece[1:] if role == 'assistant' else [-100] * len(piece))
    return tokens, labels


def oasst_branches(rows):
    """Full valid parent chains; split by tree before branch/window extraction."""
    eligible = {r['message_id']: r for r in rows
                if r.get('lang') == 'en' and r.get('review_result') is True
                and r.get('deleted') is False and r.get('synthetic') is False
                and (r.get('role') != 'assistant' or r.get('rank') == 0)}
    for end in sorted(eligible):
        if eligible[end]['role'] != 'assistant':
            continue
        chain, seen, key = [], set(), end
        while key is not None:
            if key in seen or key not in eligible:
                chain = []; break
            seen.add(key); row = eligible[key]; chain.append(row); key = row.get('parent_id')
        if not chain:
            continue
        chain.reverse()
        if any(row['message_tree_id'] != chain[0]['message_tree_id'] for row in chain):
            continue
        roles = [r['role'] for r in chain]
        if roles != ['prompter' if i % 2 == 0 else 'assistant' for i in range(len(chain))]:
            continue
        yield dict(tree_id=chain[0]['message_tree_id'], message_id=end,
                   messages=[dict(role='user' if r['role'] == 'prompter' else 'assistant', content=r['text']) for r in chain])


def prepare(root: Path, output: Path, *, kind='fineweb', length=128, max_documents=10000, max_train_tokens=4_000_000):
    """Resolve revisions before acquisition; retain only local raw/prepared data."""
    from datasets import load_dataset
    if kind not in ('fineweb', 'oasst'):
        raise ValueError('Unknown corpus')
    if output.exists():
        raise ValueError('Data version already exists; reuse it or choose a new directory')
    output.mkdir(parents=True)
    tokenizer = tokenizer_from_repo(root)
    tokenizer.save(str(output / 'tokenizer.json'))
    repo = 'HuggingFaceFW/fineweb-edu' if kind == 'fineweb' else 'OpenAssistant/oasst1'
    pinned = json.loads((root / 'data/sources/colab-v1.json').read_text())
    revision = next(r['revision'] for r in pinned['sources'] if r['repository'] == repo)
    config = 'sample-10BT' if kind == 'fineweb' else None
    source = dict(repository=repo, revision=revision, config=config,
                  license='ODC-By-1.0; underlying web content rights remain' if kind == 'fineweb' else 'Apache-2.0',
                  provenance='Classifier-selected web documents; human authorship not guaranteed' if kind == 'fineweb' else 'Reviewed English, non-synthetic, rank-0 assistant paths',
                  split='SHA256 document content hash / OASST tree ID: 90% train, 5% dev, 5% test',
                  dedup=('Whitespace-normalized exact documents' if kind == 'fineweb' else 'Exact serialized conversation branches, not individual messages') + '; no near-dedup or benchmark decontamination claim',
                  tokenizer='Existing BabyLM train-10m native BPE plus seven explicit boundary/role/padding tokens; no new fitting or legacy ID offset')
    write_json(output / 'source.json', source)
    parts = {name: [] for name in ('train', 'validation', 'test')}
    manifests, seen, train_tokens = [], set(), 0
    if kind == 'fineweb':
        records = load_dataset(repo, name=config, revision=revision, split='train', streaming=True)
    else:
        # Use only upstream train trees; our test remains unused by the notebook.
        upstream = load_dataset(repo, revision=revision, split='train')
        records = oasst_branches(upstream)
    for index, row in enumerate(records):
        if index >= max_documents or train_tokens >= max_train_tokens:
            break
        if kind == 'fineweb':
            text = row['text']
            if len(text) < 200 or len(text) > 20000 or row.get('score', 3) < 3:
                continue
            canonical = re.sub(r'\s+', ' ', text).strip()
            text_hash = digest(canonical)
            split_key = text_hash
            tokens = [tokenizer.token_to_id('<|bos|>')] + tokenizer.encode(text, add_special_tokens=False).ids + [tokenizer.token_to_id('<|eos|>')]
            labels = tokens[:]
            source_id = row.get('id', text_hash)
        else:
            canonical = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
            text_hash = digest(canonical)
            split_key = row['tree_id']
            tokens, labels = conversation(tokenizer, row['messages'])
            source_id = row['message_id']
        if text_hash in seen:
            continue
        seen.add(text_hash)
        split = partition(split_key)
        records_here = list(windows(tokens, labels, length))
        parts[split].extend(records_here)
        scored = sum(sum(v != -100 for v in r['labels'][1:]) for r in records_here)
        if split == 'train':
            train_tokens += scored
        manifests.append(dict(source_id=source_id, split_key=split_key, text_sha256=text_hash, split=split,
                              tokens=len(tokens), scored_tokens=scored, windows=len(records_here)))
        if len(manifests) % 1000 == 0:
            print(f'{len(manifests)} retained documents/branches; {train_tokens:,} train targets', flush=True)
    if any(not part for part in parts.values()):
        raise ValueError('Insufficient documents for all three partitions; prepare a new, larger data version')
    for name, records_here in parts.items():
        with (output / f'{name}.jsonl').open('w', encoding='utf8') as handle:
            for row in records_here:
                handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    with (output / 'documents.jsonl').open('w', encoding='utf8') as handle:
        for row in manifests:
            handle.write(json.dumps(row) + '\n')
    card = dict(source=source, kind=kind, length=length, vocabulary=tokenizer.get_vocab_size(),
                pad=tokenizer.token_to_id('<|pad|>'), bos=tokenizer.token_to_id('<|bos|>'), eos=tokenizer.token_to_id('<|eos|>'),
                documents_or_branches=len(manifests),
                counts={name: len(part) for name, part in parts.items()}, train_scored_tokens=train_tokens,
                repeated_ancestor_messages_possible=kind == 'oasst',
                files={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output / 'data-card.json', card)
    return card


def load_prepared(path: Path):
    card = json.loads((path / 'data-card.json').read_text())
    # Hash test bytes for integrity without selecting on their scores or reading their text.
    for name, expected in card['files'].items():
        if sha256(path / name) != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
    train, validation = [[json.loads(line) for line in (path / f'{name}.jsonl').read_text().splitlines()]
                         for name in ('train', 'validation')]
    return train, validation, card
