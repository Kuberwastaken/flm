import json
import tempfile
from pathlib import Path
import unittest
from flm.language_test import freeze_selection, paired_interval, verify_inputs
from flm.provenance import sha256, write_json
from flm.runtime_benchmark import state_storage
import torch


class ReportTests(unittest.TestCase):
    def test_test_inputs_reject_changed_data_and_unselected_smoothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol = {}; partitions = {}
            for key, name in [('tokenizer_sha256', 'tokenizers/wikitext2-4096/tokenizer.json'),
                              ('graph_sha256', 'graphs/central-1024/graph.npz')]:
                path = root / 'data' / name; path.parent.mkdir(parents=True); path.write_bytes(b'fixture')
                protocol[key] = sha256(path)
            for split in ('train', 'validation', 'test'):
                partitions[split] = {}
                for suffix, folder, key in [('.npz', 'wikitext2-bpe', 'cache_sha256'), ('.jsonl', 'wikitext2', 'source_sha256')]:
                    path = root / 'data/processed' / folder / (split + suffix)
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((split + suffix).encode())
                    partitions[split][key] = sha256(path)
                if split != 'test': protocol[split + '_cache_sha256'] = partitions[split]['cache_sha256']
            write_json(root / 'data/tokenizers/wikitext2-4096/tokenization-card.json',
                       dict(tokenizer_sha256=protocol['tokenizer_sha256'], partitions=partitions))
            tuning = dict(context_bytes=4, smoothing=4., train_sha256=partitions['train']['source_sha256'],
                validation_sha256=partitions['validation']['source_sha256'],
                validation={str(n): dict(bits_per_byte=5.-n) for n in (.25, 1., 4.)})
            tuning_path = root / 'tuning.json'; write_json(tuning_path, tuning)
            _, inputs = verify_inputs(protocol, tuning_path, root)
            self.assertEqual(inputs['test_cache_sha256'], partitions['test']['cache_sha256'])
            tuning['smoothing'] = 1.; write_json(tuning_path, tuning)
            with self.assertRaisesRegex(ValueError, 'validation selection'): verify_inputs(protocol, tuning_path, root)
            tuning['smoothing'] = 4.; write_json(tuning_path, tuning)
            (root / 'data/processed/wikitext2-bpe/test.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Acquired data changed'): verify_inputs(protocol, tuning_path, root)

    def test_state_accounting_distinguishes_views_from_owned_storage(self):
        backing = torch.zeros(100)
        sliced = backing[:10]
        result = state_storage((10, [sliced, backing[10:20]]))
        self.assertEqual(result['logical_tensor_bytes'], 80)
        self.assertEqual(result['allocated_tensor_bytes'], 400)
        owned = state_storage([sliced.clone()])
        self.assertEqual(owned['allocated_tensor_bytes'], 40)

    def test_test_selection_fails_closed_when_a_registered_run_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'incomplete'): freeze_selection(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_paired_bootstrap_is_exact_for_a_constant_byte_normalized_gap(self):
        a = [dict(document=str(i), bytes=n, nll=n * 2.) for i, n in enumerate([10, 100, 1000])]
        b = [dict(document=str(i), bytes=n, nll=n) for i, n in enumerate([10, 100, 1000])]
        import math
        result = paired_interval(a, b, 1000)
        for key in ('difference_bpb', 'lower_95', 'upper_95'): self.assertAlmostEqual(result[key], 1 / math.log(2))
        self.assertEqual(paired_interval(a, a, 1000)['upper_95'], 0)
        with self.assertRaises(ValueError): paired_interval(a, b[:-1])
        with self.assertRaises(ValueError): paired_interval(a, a + [a[0]])


if __name__ == '__main__': unittest.main()
