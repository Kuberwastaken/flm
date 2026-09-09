"""Lossless train-only byte-pair tokenization with a portable browser vocabulary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from .provenance import sha256, write_json

BOS, EOS, OFFSET = 0, 1, 2


def byte_characters():
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = printable.copy(); extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte); codepoints.append(256 + extra); extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(printable, codepoints)}


class Lexicon:
    def __init__(self, path):
        self.path = Path(path); self.tokenizer = Tokenizer.from_file(str(path))
        self.sha256 = sha256(self.path)
        vocab = self.tokenizer.get_vocab(); inverse = {char: byte for byte, char in byte_characters().items()}
        self.pieces = [b'', b''] + [b''] * len(vocab)
        for token, index in vocab.items(): self.pieces[index + OFFSET] = bytes(inverse[char] for char in token)
        self.lengths = np.asarray([len(piece) for piece in self.pieces], dtype=np.int64)
        self.vocabulary = len(self.pieces)

    def encode(self, text, boundaries=False):
        values = [token + OFFSET for token in self.tokenizer.encode(text, add_special_tokens=False).ids]
        return [BOS] + values + [EOS] if boundaries else values

    def decode(self, values):
        return b''.join(self.pieces[int(value)] for value in values).decode('utf8', errors='strict')

    def export(self, path):
        raw = json.loads(self.path.read_text(encoding='utf8')); vocab = raw['model']['vocab']
        pairs = [[vocab[left] + OFFSET, vocab[right] + OFFSET, vocab[left + right] + OFFSET]
                 for left, right in raw['model']['merges']]
        write_json(path, dict(format='flm-byte-bpe-v1', tokenizer_sha256=self.sha256,
            bos=BOS, eos=EOS, vocabulary=self.vocabulary, pieces=[list(piece) for piece in self.pieces],
            byte_ids=[vocab[byte_characters()[byte]] + OFFSET for byte in range(256)], merges=pairs,
            pretokenizer='GPT-2 ByteLevel regular expression; no prefix space; no normalization',
            special_token_rule='Boundary IDs are inserted by the caller only. Literal boundary-marker text remains ordinary text.'))


def train_tokenizer(data, output, vocabulary=4096):
    if vocabulary < 258: raise ValueError('Vocabulary must include all bytes and two boundary IDs')
    source = data / 'train.jsonl'; documents = [json.loads(line) for line in source.read_text(encoding='utf8').splitlines()]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocabulary - OFFSET, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[], show_progress=False)
    tokenizer.train_from_iterator((document['text'] for document in documents), trainer=trainer, length=len(documents))
    output.mkdir(parents=True, exist_ok=True); path = output / 'tokenizer.json'; tokenizer.save(str(path))
    lexicon = Lexicon(path)
    for document in documents:
        if lexicon.decode(lexicon.encode(document['text'])) != document['text']: raise ValueError('Training text did not roundtrip')
    lexicon.export(output / 'browser-tokenizer.json')
    write_json(output / 'tokenizer-card.json', dict(algorithm='Byte-level BPE', implementation=f'tokenizers {tokenizers.__version__}',
        vocabulary=lexicon.vocabulary, train_sha256=sha256(source), tokenizer_sha256=sha256(path),
        training_documents=len(documents), normalization='none', prefix_space=False, all_bytes_retained=True,
        boundaries=dict(bos=BOS, eos=EOS, offset=OFFSET), validation_or_test_used_for_training=False,
        training_roundtrip='Every training document decodes to exactly its original UTF-8 string',
        source='https://huggingface.co/docs/tokenizers/api/pre-tokenizers'))
    return lexicon


def cache_split(data, output, split, lexicon):
    path = data / f'{split}.jsonl'; documents = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    chunks = []; offsets = [0]; sizes = []; ids = []
    for document in documents:
        tokens = np.asarray(lexicon.encode(document['text'], boundaries=True), dtype=np.int32)
        size = len(document['text'].encode())
        if int(lexicon.lengths[tokens].sum()) != size or lexicon.decode(tokens) != document['text']:
            raise ValueError('Token cache changed the source text')
        chunks.append(tokens); offsets.append(offsets[-1] + len(tokens)); sizes.append(size); ids.append(document['id'])
    output.mkdir(parents=True, exist_ok=True); destination = output / f'{split}.npz'
    np.savez_compressed(destination, tokens=np.concatenate(chunks), offsets=np.asarray(offsets, dtype=np.int64),
                        raw_bytes=np.asarray(sizes, dtype=np.int64), ids=np.asarray(ids))
    return dict(documents=len(ids), tokens=offsets[-1] - 2 * len(ids), utf8_bytes=sum(sizes),
        bytes_per_token=sum(sizes) / (offsets[-1] - 2 * len(ids)), cache_sha256=sha256(destination), source_sha256=sha256(path))


def read_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        tokens = archive['tokens'].astype(np.int64); offsets = archive['offsets']; ids = archive['ids']
    return [(str(identity), tokens[offsets[i]:offsets[i + 1]]) for i, identity in enumerate(ids)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/wikitext2'))
    parser.add_argument('--output', type=Path, default=Path('data/tokenizers/wikitext2-4096'))
    parser.add_argument('--cache', type=Path, default=Path('data/processed/wikitext2-bpe'))
    parser.add_argument('--vocabulary', type=int, default=4096)
    args = parser.parse_args(); lexicon = train_tokenizer(args.data, args.output, args.vocabulary)
    partitions = {split: cache_split(args.data, args.cache, split, lexicon) for split in ('train', 'validation', 'test')}
    write_json(args.output / 'tokenization-card.json', dict(tokenizer_sha256=lexicon.sha256, partitions=partitions,
        test_usage='Encoded after fitting the train-only tokenizer for reproducible scoring; no test losses or model selection performed'))
    print(json.dumps(partitions, indent=2))


if __name__ == '__main__': main()
