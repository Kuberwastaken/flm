import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from flm.inference import state_hash
from flm.scan_conditions import Prepared
from flm.scan_pilot import Pilot, measure, pilot_conditions, run
from flm.scan_train import canonical, sampling_state
from flm.scan_task import encode_example
from test_scan_runtime import models, RECORDS
from test_scan_train import FixtureLexicon
import hashlib
import numpy as np


def prepared(index):
    model = models()[index]; records = copy.deepcopy(RECORDS); lexicon = FixtureLexicon()
    return Prepared(model, records, lexicon, dict(condition=dict(seed=42),
        language_source=dict(initial_state_sha256=state_hash(model), tokenizer_sha256=lexicon.sha256),
        ordered_rows_sha256=hashlib.sha256(canonical(records)).hexdigest()))


class ScanPilotTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_inventory_keeps_every_split_architecture_and_source_at_seed_42(self):
        rows = pilot_conditions()
        self.assertEqual(len(rows), 18)
        self.assertEqual({r['seed'] for r in rows}, {42})
        self.assertEqual(len({r['label'] for r in rows}), 18)

    def test_disposable_updates_preserve_source_and_rng_and_count_only_timed_exposure(self):
        pilot = Pilot(1, 2, 2, 1)
        for index in range(3):
            source = prepared(index); original_hash = state_hash(source.model)
            before_rng = torch.get_rng_state().clone(); source_mode = source.model.training
            result = measure(source, pilot)
            self.assertEqual(state_hash(source.model), original_hash)
            self.assertEqual(source.model.training, source_mode)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            self.assertTrue(all(p.grad is None for p in source.model.parameters()))
            self.assertEqual(len(result['observations']), 3)
            self.assertEqual(result['measured_examples'], 4)
            self.assertEqual([r['warmup'] for r in result['observations']], [True, False, False])
            examples = [encode_example(r, source.lexicon) for r in source.records]
            lengths = np.array([len(r['input']) for r in examples]); scored = np.array([r['supervised_tokens'] for r in examples])
            _, exposure, digest = sampling_state(pilot.settings(42), lengths, scored, 3)
            self.assertEqual(result['sampled_row_sha256'], digest)
            self.assertEqual(sum(r['input_tokens'] for r in result['observations']), exposure['input_tokens'])
            self.assertEqual(sum(r['supervised_tokens'] for r in result['observations']), exposure['supervised_tokens'])
            self.assertFalse(result['checkpoint_written'])

    def test_nonfinite_loss_restores_rng_threads_and_original_weights(self):
        source = prepared(0); before = state_hash(source.model); rng = torch.get_rng_state().clone()
        with patch('flm.scan_pilot.batch_loss', return_value=torch.tensor(float('nan'))):
            with self.assertRaisesRegex(FloatingPointError, 'pilot loss'): measure(source, Pilot(1, 1, 2, 1))
        self.assertEqual(state_hash(source.model), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(torch.get_num_threads(), 1)

    def test_unfinished_priority_queues_refuse_before_data_or_model_preparation(self):
        with tempfile.TemporaryDirectory() as temporary, patch('flm.scan_pilot.prepare_condition') as prepare:
            with self.assertRaisesRegex(ValueError, 'Priority'): run(Path(temporary))
            prepare.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_failed_attempt_is_retained_and_completed_pilot_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f'runs/language-core-v1/{c}-s{s}/complete.json'
                     for c in ('fixed_dynamics', 'no_lateral', 'no_temporal_state') for s in (42, 43)]
            names += [f'runs/babylm-{scale}/{v}-s{s}/complete.json'
                      for scale in ('10m', '100m') for v in ('flm', 'gru', 'transformer') for s in (42, 43)]
            for name in names:
                p = root / name; p.parent.mkdir(parents=True); p.write_text('{}')
            cases = [dict(label='fixture/first'), dict(label='fixture/second')]
            with patch('flm.scan_pilot.pilot_conditions', return_value=cases), \
                    patch('flm.scan_pilot.prepare_condition'), \
                    patch('flm.scan_pilot.measure', side_effect=[{'fixture': True}, ValueError('Synthetic failure')]):
                with self.assertRaisesRegex(ValueError, 'Synthetic failure'): run(root)
            attempts = [p for p in (root / 'runs/scan-timing-pilot').iterdir() if p.is_dir()]
            self.assertEqual(len(attempts), 1)
            failure = json.loads((attempts[0] / 'failure.json').read_text())
            self.assertEqual(failure['completed_conditions'], 1)
            self.assertTrue((attempts[0] / 'fixture--first.json').exists())
            self.assertFalse((root / 'reports/scan-runtime/timing-pilot.json').exists())
            with patch('flm.scan_pilot.pilot_conditions', return_value=cases), \
                    patch('flm.scan_pilot.prepare_condition') as prepare, \
                    patch('flm.scan_pilot.measure', return_value={'fixture': True}):
                result = run(root); self.assertTrue(result.is_file())
                prepare.reset_mock()
                with self.assertRaisesRegex(ValueError, 'already recorded'): run(root)
                prepare.assert_not_called()
            self.assertTrue((attempts[0] / 'failure.json').exists())
            self.assertEqual(list(root.rglob('*.pt')), [])


if __name__ == '__main__': unittest.main()
