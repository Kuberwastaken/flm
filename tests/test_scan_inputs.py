import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flm import scan
from flm.provenance import sha256, write_json
from flm.scan import BASE, REVISION, SPLITS, parse, statistics
from flm.scan_inputs import load_training_partition


class ScanInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.split = 'add_primitive_jump'
        self.name = SPLITS[self.split][0]
        self.raw = self.root / 'data/raw/scan' / REVISION / self.name
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b'IN: jump OUT: I_JUMP\nIN: walk OUT: I_WALK\nIN: jump OUT: I_JUMP\n')
        self.rows = parse(self.raw.read_bytes(), self.name)
        self.files = {self.name: (self.raw.stat().st_size, sha256(self.raw))}
        self.processed = self.root / 'data/processed/scan' / self.split / 'train.jsonl'
        self.processed.parent.mkdir(parents=True)
        self.card_path = self.root / 'data/cards/scan.json'
        self.card = dict(revision=REVISION, parser_sha256=sha256(Path(scan.__file__)),
            source_files={self.name: dict(url=BASE + self.name, bytes=self.raw.stat().st_size,
                                        sha256=sha256(self.raw))}, partitions={self.split: {}})
        self.write_rows(self.rows)
        self.patcher = patch('flm.scan_inputs.FILES', self.files)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def write_rows(self, rows):
        self.processed.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        self.card['partitions'][self.split]['train'] = dict(statistics(rows),
            bytes=self.processed.stat().st_size, sha256=sha256(self.processed))
        write_json(self.card_path, self.card)

    def test_preserves_repeated_rows_and_needs_no_test_or_universe_file(self):
        actual, binding = load_training_partition(self.root, self.split)
        self.assertEqual(actual, self.rows)
        self.assertEqual([r['source_line'] for r in actual], [1, 2, 3])
        self.assertEqual(binding['rows'], 3)
        self.assertEqual(binding['unique_commands'], 2)
        self.assertEqual(binding['repeated_rows'], 1)
        self.assertFalse(binding['test_partition_opened'])
        self.assertEqual(list((self.root / 'data/raw/scan' / REVISION).rglob('*.txt')), [self.raw])

    def test_rehashed_deduplication_order_or_metadata_changes_still_fail(self):
        changed = copy.deepcopy(self.rows); changed[0]['source_line'] = 7
        for rows in (self.rows[:2], list(reversed(self.rows)), changed):
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, 'multiplicity or source metadata'):
                    load_training_partition(self.root, self.split)

    def test_rejects_raw_bytes_even_when_the_changed_command_is_valid(self):
        self.raw.write_bytes(self.raw.read_bytes().replace(b'walk OUT: I_WALK', b'look OUT: I_LOOK'))
        with self.assertRaisesRegex(ValueError, 'Pinned SCAN training source'):
            load_training_partition(self.root, self.split)

    def test_rejects_stale_processed_hash_and_altered_card(self):
        original = self.processed.read_bytes()
        self.processed.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Prepared SCAN training file'):
            load_training_partition(self.root, self.split)
        self.processed.write_bytes(original)
        self.card['parser_sha256'] = '0' * 64; write_json(self.card_path, self.card)
        with self.assertRaisesRegex(ValueError, 'parser changed'):
            load_training_partition(self.root, self.split)

    def test_unknown_split_fails_before_reading_any_file(self):
        with self.assertRaisesRegex(ValueError, 'Unknown official SCAN split'):
            load_training_partition(self.root / 'missing', '../test')


if __name__ == '__main__': unittest.main()
