"""Test-partition provenance checks use artificial publisher files only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flm import scan
from flm.provenance import sha256, write_json
from flm.scan import BASE, REVISION, SPLITS, parse, statistics
from flm.scan_study import test_metadata as frozen_metadata
from flm.scan_test_inputs import load_test_partition


class ScanTestInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.split = 'simple'; self.files = {}
        self.card = dict(revision=REVISION, parser_sha256=sha256(Path(scan.__file__)),
                         source_files={}, partitions={})
        for split, names in SPLITS.items():
            source = names[1]; path = self.root/'data/raw/scan'/REVISION/source
            path.parent.mkdir(parents=True)
            path.write_bytes(b'IN: jump OUT: I_JUMP\nIN: look OUT: I_LOOK\nIN: jump OUT: I_JUMP\n')
            rows = parse(path.read_bytes(), source)
            self.files[source] = (path.stat().st_size, sha256(path))
            self.card['source_files'][source] = dict(url=BASE+source, bytes=path.stat().st_size, sha256=sha256(path))
            processed = self.root/f'data/processed/scan/{split}/test.jsonl'
            processed.parent.mkdir(parents=True)
            processed.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
            self.card['partitions'][split] = dict(test=dict(statistics(rows),
                bytes=processed.stat().st_size, sha256=sha256(processed)))
        self.card_path = self.root/'data/cards/scan.json'; write_json(self.card_path, self.card)
        self.frozen = frozen_metadata(self.root)
        self.raw = self.root/'data/raw/scan'/REVISION/SPLITS[self.split][1]
        self.processed = self.root/f'data/processed/scan/{self.split}/test.jsonl'
        self.rows = parse(self.raw.read_bytes(), SPLITS[self.split][1])
        patcher = patch('flm.scan_test_inputs.FILES', self.files)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_every_row_retained_without_training_or_other_test_payloads(self):
        for split, names in SPLITS.items():
            if split != self.split:
                (self.root/'data/raw/scan'/REVISION/names[1]).unlink()
                (self.root/f'data/processed/scan/{split}/test.jsonl').unlink()
        rows, coverage = load_test_partition(self.root, self.split, self.frozen)
        self.assertEqual(rows, self.rows)
        self.assertEqual([r['source_line'] for r in rows], [1,2,3])
        self.assertEqual(coverage['statistics']['repeated_rows'], 1)
        self.assertFalse(coverage['training_partition_opened'])
        self.assertEqual(list((self.root/'data/processed/scan').rglob('train.jsonl')), [])

    def test_rehashed_reorder_deduplication_and_source_metadata_fail(self):
        changed = copy.deepcopy(self.rows); changed[0]['source_line'] = 9
        for rows in (list(reversed(self.rows)), self.rows[:2], changed):
            with self.subTest(rows=rows):
                self.processed.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
                self.card['partitions'][self.split]['test'] = dict(statistics(rows),
                    bytes=self.processed.stat().st_size, sha256=sha256(self.processed))
                write_json(self.card_path, self.card)
                with self.assertRaisesRegex(ValueError, 'multiplicity or source metadata'):
                    load_test_partition(self.root, self.split, frozen_metadata(self.root))

    def test_changed_source_or_processed_bytes_fail(self):
        original = self.raw.read_bytes(); self.raw.write_bytes(original.replace(b'jump', b'walk'))
        with self.assertRaisesRegex(ValueError, 'Pinned SCAN test source'):
            load_test_partition(self.root, self.split, self.frozen)
        self.raw.write_bytes(original); self.processed.write_bytes(self.processed.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Prepared SCAN test file'):
            load_test_partition(self.root, self.split, self.frozen)

    def test_changed_card_is_rejected_before_test_access(self):
        self.card['revision'] = 'different'; write_json(self.card_path, self.card)
        self.raw.unlink(); self.processed.unlink()
        with self.assertRaisesRegex(ValueError, 'pre-training identity'):
            load_test_partition(self.root, self.split, self.frozen)

    def test_forged_path_statistics_or_partition_inventory_fail(self):
        path = copy.deepcopy(self.frozen)
        path['partitions']['simple']['processed_path'] = '../../outside.jsonl'
        incomplete = copy.deepcopy(self.frozen); del incomplete['partitions']['length']
        for value in (path, incomplete):
            with self.assertRaisesRegex(ValueError, 'test inventory'):
                load_test_partition(self.root, self.split, value)
        self.card['partitions'][self.split]['test']['actions'] += 1
        write_json(self.card_path, self.card)
        with self.assertRaisesRegex(ValueError, 'statistics differ'):
            load_test_partition(self.root, self.split, frozen_metadata(self.root))

    def test_unknown_split_fails_without_reading_files(self):
        with self.assertRaisesRegex(ValueError, 'Unknown official SCAN test split'):
            load_test_partition(self.root/'absent', '../test', {})


if __name__ == '__main__': unittest.main()
