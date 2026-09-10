import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
from flm.babylm_prepare import blocks, encode_partition
from flm.babylm_audit import connect, add_file
from flm.corpus_cache import read_mmap
from flm.provenance import sha256
from flm.tokenizer import Lexicon
from flm.train import Sampler
from flm.language_train import load_training_data, evaluate, construct


class BabyLMPreparationTests(unittest.TestCase):
    def test_blocks_retain_unicode_crlf_blank_lines_and_oversized_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'text'; payload = ('café\r\n\r\n' + 'a' * 50 + '\n日本語').encode()
            path.write_bytes(payload); pieces = list(blocks(path, 12))
            self.assertEqual(b''.join(r[0] for r in pieces), payload)
            self.assertEqual(pieces[0][1], 0); self.assertEqual(pieces[-1][2], len(payload))
            for a, b in zip(pieces, pieces[1:]): self.assertEqual(a[2], b[1])
            self.assertEqual(pieces[-1][3:], (4, 4))

    def test_verified_mmap_overlap_roundtrip_and_sampler_match_int64(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'input.txt'
            text = 'The little brown bird returns to the garden every morning.\r\n' * 50
            path.write_bytes(text.encode()); db = connect(root / 'audit.sqlite')
            add_file(db, path, 'train-10m', sha256(path))
            lexicon = Lexicon('data/tokenizers/wikitext2-4096/tokenizer.json')
            record = dict(component='synthetic', partition='validation', path=str(path), sha256=sha256(path), utf8_bytes=path.stat().st_size)
            cache = root / 'cache'; card = encode_partition([record], cache, lexicon, db, 'synthetic-audit')
            documents, metadata, _ = read_mmap(cache, lexicon.sha256)
            self.assertIsInstance(documents[0][1], np.memmap)
            self.assertEqual(lexicon.decode(documents[0][1]), text)
            self.assertEqual(card['components']['synthetic']['overlapping_lines'], 50)
            self.assertFalse(metadata[0]['overlap_filtered_eligible'])
            narrow = Sampler(documents, 42, 96); wide = Sampler([(i, d.astype(np.int64)) for i, d in documents], 42, 96)
            for _ in range(4):
                a = narrow.sample(4, 'cpu'); b = wide.sample(4, 'cpu')
                np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
            with self.assertRaises(ValueError): read_mmap(cache, 'wrong-tokenizer')
            train, validation, identities = load_training_data(root, lexicon, training_cache=cache, validation_cache=cache)
            model = construct('gru', None, lexicon.vocabulary, 42)
            narrow_score = evaluate(model, validation, lexicon, token_limit=120, unit='block')
            wide_score = evaluate(model, [(i, d.astype(np.int64)) for i, d in validation], lexicon, token_limit=120, unit='block')
            self.assertEqual(narrow_score['bits_per_byte'], wide_score['bits_per_byte'])
            self.assertIn('per block', narrow_score['protocol'])
            del train, validation
            del documents, narrow, wide
            # Tampering metadata is caught before any token is returned.
            with (cache / 'blocks.jsonl').open('a') as stream: stream.write('\n')
            with self.assertRaises(ValueError): read_mmap(cache)
            db.close()


if __name__ == '__main__': unittest.main()
