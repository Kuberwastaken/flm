"""Small artificial training runs only; no acquired SCAN partitions or weights."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm.provenance import sha256, write_json
from flm.scan_task import prompt
from flm.scan_train import Settings, fit, training_lease
from test_scan import ByteLexicon
from test_scan_runtime import RECORDS, models


class FixtureLexicon(ByteLexicon):
    sha256 = 'a' * 64


SETTINGS = Settings(steps=6, batch_size=2, learning_rate=.002, final_learning_rate=.0002,
    warmup_steps=1, weight_decay=.01, gradient_clip=1., checkpoint_interval=3, sampling_seed=77, threads=1)
BINDING = dict(study='Artificial software fixture', condition='unit test; not a registered SCAN fit')


def payload(directory, step=6):
    return torch.load(directory / f'checkpoint-{step:06d}.pt', weights_only=True, map_location='cpu')


class ScanTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix='flm-scan-training-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def equal_tree(self, first, second):
        self.assertEqual(type(first), type(second))
        if isinstance(first, torch.Tensor):
            self.assertEqual(first.dtype, second.dtype)
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first: self.equal_tree(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second): self.equal_tree(a, b)
        else:
            self.assertEqual(first, second)

    def test_committed_resume_matches_uninterrupted_state_optimizer_rng_and_history_for_all_models(self):
        for index in range(3):
            uninterrupted = self.root / f'whole-{index}'
            resumed = self.root / f'resumed-{index}'
            fit(models()[index], RECORDS, FixtureLexicon(), SETTINGS, BINDING, uninterrupted)
            partial = fit(models()[index], RECORDS, FixtureLexicon(), SETTINGS, BINDING, resumed, until=3)
            self.assertFalse(partial['complete']); self.assertFalse((resumed / 'complete.json').exists())
            result = fit(models()[index], RECORDS, FixtureLexicon(), SETTINGS, BINDING, resumed)
            self.assertTrue(result['complete'])
            self.equal_tree(payload(uninterrupted), payload(resumed))

    def test_interruption_after_payload_before_marker_replays_from_last_committed_boundary(self):
        whole = self.root / 'whole'; interrupted = self.root / 'interrupted'
        fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, whole)
        fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, interrupted, until=3)
        original = write_json
        def crash(path, value):
            if path.name == 'saved-000006.json':
                raise RuntimeError('Simulated loss of process before commit marker')
            return original(path, value)
        with patch('flm.scan_train.write_json', side_effect=crash), self.assertRaisesRegex(RuntimeError, 'Simulated'):
            fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, interrupted)
        self.assertTrue((interrupted / 'checkpoint-000006.pt').is_file())
        self.assertFalse((interrupted / 'saved-000006.json').exists())
        self.assertFalse((interrupted / 'complete.json').exists())
        fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, interrupted)
        self.equal_tree(payload(whole), payload(interrupted))

    def test_sampled_exposure_counts_real_tokens_and_retains_duplicate_row_weight(self):
        records = [RECORDS[0], RECORDS[2], RECORDS[0]]
        fit(models()[0], records, FixtureLexicon(), SETTINGS, BINDING, self.root)
        saved = payload(self.root)
        rng = np.random.Generator(np.random.PCG64(SETTINGS.sampling_seed))
        seen = []
        for _ in range(SETTINGS.steps): seen.extend(rng.integers(0, len(records), SETTINGS.batch_size).tolist())
        expected = dict(examples=len(seen),
            input_tokens=sum(1+len(prompt(records[index]['command']).encode())+len(records[index]['actions']) for index in seen),
            supervised_tokens=sum(1+len(records[index]['actions']) for index in seen))
        self.assertEqual(saved['exposure'], expected)
        self.assertEqual(saved['sampler_rng'], rng.bit_generator.state)
        self.assertEqual(saved['declaration']['training_rows'], 3)

    def test_changed_initialization_rows_tokenizer_binding_or_schedule_reject_resume(self):
        fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, self.root, until=3)
        changed_model = models()[0]
        with torch.no_grad(): next(changed_model.parameters()).add_(.01)
        changed_lexicon = FixtureLexicon(); changed_lexicon.sha256 = 'b' * 64
        cases = [
            (changed_model, RECORDS, FixtureLexicon(), SETTINGS, BINDING),
            (models()[0], RECORDS[::-1], FixtureLexicon(), SETTINGS, BINDING),
            (models()[0], RECORDS + RECORDS[:1], FixtureLexicon(), SETTINGS, BINDING),
            (models()[0], RECORDS, changed_lexicon, SETTINGS, BINDING),
            (models()[0], RECORDS, FixtureLexicon(), SETTINGS, dict(condition='another condition')),
            (models()[0], RECORDS, FixtureLexicon(), replace(SETTINGS, steps=9), BINDING)]
        for arguments in cases:
            with self.assertRaisesRegex(ValueError, 'declaration changed'):
                fit(*arguments, self.root)

    def test_changed_committed_bytes_fail_before_checkpoint_loading(self):
        fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, self.root, until=3)
        checkpoint = self.root / 'checkpoint-000003.pt'
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
        with patch('flm.scan_train.torch.load') as loader, self.assertRaises(ValueError):
            fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, self.root)
        loader.assert_not_called()

    def test_rehashed_payload_cannot_change_graph_optimizer_or_sampling_state(self):
        for index, change in enumerate(('graph', 'optimizer', 'sampling', 'history')):
            folder = self.root / str(index)
            fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, folder, until=3)
            saved = payload(folder, 3)
            if change == 'graph': saved['model']['base_weight'][0] += .1
            if change == 'optimizer': saved['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
            if change == 'sampling': saved['exposure']['examples'] += 1
            if change == 'history': saved['history'].pop()
            checkpoint = folder / 'checkpoint-000003.pt'; torch.save(saved, checkpoint)
            marker = folder / 'saved-000003.json'
            record = json.loads(marker.read_text()); record['checkpoint_sha256'] = sha256(checkpoint)
            write_json(marker, record)
            with self.subTest(change=change), self.assertRaises(ValueError):
                fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, folder)

    def test_writer_lease_excludes_a_second_writer_and_releases_after_failure(self):
        with training_lease(self.root):
            with self.assertRaisesRegex(RuntimeError, 'Another instruction-training writer'):
                fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, self.root)
        self.assertTrue((self.root / 'writer.lock').exists())
        result = fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, self.root)
        self.assertTrue(result['complete'])

    def test_invalid_configuration_or_context_creates_no_training_directory(self):
        for settings in (replace(SETTINGS, steps=5), replace(SETTINGS, learning_rate=float('nan')),
                         replace(SETTINGS, context_limit=4), replace(SETTINGS, threads=0)):
            directory = self.root / 'must-not-exist'
            with self.assertRaises(ValueError):
                fit(models()[0], RECORDS, FixtureLexicon(), settings, BINDING, directory)
            self.assertFalse(directory.exists())

    def test_nonfinite_final_update_cannot_be_committed_or_marked_complete(self):
        original = torch.optim.AdamW.step
        steps = 0
        def corrupt_final(optimizer, *args, **kwargs):
            nonlocal steps
            result = original(optimizer, *args, **kwargs)
            steps += 1
            if steps == 6:
                with torch.no_grad(): optimizer.param_groups[0]['params'][0].fill_(float('nan'))
            return result
        with patch('torch.optim.AdamW.step', new=corrupt_final), self.assertRaisesRegex(FloatingPointError, 'cannot be committed'):
            fit(models()[0], RECORDS, FixtureLexicon(), SETTINGS, BINDING, self.root)
        self.assertTrue((self.root / 'saved-000003.json').exists())
        self.assertFalse((self.root / 'saved-000006.json').exists())
        self.assertFalse((self.root / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
