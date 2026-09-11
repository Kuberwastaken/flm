"""Tiny disposable fixtures; no timing claim about the actual language model."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from flm.inference import state_hash
from flm.language_learning_inputs import conditions
from flm.language_learning_pilot import Pilot, measure, ordered_conditions, prerequisites, run
from flm.language_learning_train import configure
from flm.train import Sampler
from test_language_learning_train import DOCUMENTS, Lexicon, BINDING
from test_language_eligibility import fixture

PILOT = Pilot(1, 2, 2, 4, 1, 1)


class LearningPilotTests(unittest.TestCase):
    def model(self, condition):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(condition['seed']); model = fixture()
        configure(model, condition['method'])
        return model

    def test_all_rules_preserve_sources_rng_and_threads_with_matched_actual_updates(self):
        before_rng = torch.get_rng_state().clone(); threads = torch.get_num_threads()
        results = []
        for condition in conditions():
            model = self.model(condition); original = state_hash(model)
            result = measure(model, DOCUMENTS, Lexicon(), condition, BINDING, PILOT)
            self.assertEqual(state_hash(model), original)
            self.assertTrue(result['source_model_unchanged']); self.assertTrue(result['disposable_parameters_changed'])
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertEqual(torch.get_num_threads(), threads)
            self.assertFalse(result['language_scores_reported']); self.assertFalse(result['checkpoint_written'])
            self.assertFalse(result['peak_memory_measured'])
            self.assertTrue(all('loss' not in row for row in result['observations']))
            if condition['method'] == 'fixed_core': self.assertTrue(result['fixed_core_preserved'])
            if condition['method'] in ('bptt', 'fixed_core'):
                self.assertTrue(all(row['persistent_trace_bytes'] == 0 for row in result['observations']))
            else: self.assertTrue(all(row['persistent_trace_bytes'] > 0 for row in result['observations']))
            results.append(result)
        for seed in (42, 43):
            group = [r for r in results if r['condition']['seed'] == seed]
            for key in ('initial_state_sha256', 'sampled_windows_sha256', 'measured_windows_sha256'):
                self.assertEqual(len({r[key] for r in group}), 1)

    def test_exposure_and_window_digests_match_independent_masked_arithmetic(self):
        condition = conditions()[0]
        result = measure(self.model(condition), DOCUMENTS, Lexicon(), condition, BINDING, PILOT)
        sampler = Sampler(DOCUMENTS, 42, 4); total = hashlib.sha256(); measured = hashlib.sha256()
        expected = dict(presented_tokens=0, scored_tokens=0, presented_bytes=0, scored_bytes=0)
        for step in range(3):
            x, y = sampler.sample(2, 'cpu'); xn = x.numpy(); yn = y.numpy()
            for array in (xn, yn):
                total.update(array.astype('<i8').tobytes())
                if step: measured.update(array.astype('<i8').tobytes())
            if step:
                mask = yn >= 2; mask[:, :1] = False
                expected['presented_tokens'] += xn.size; expected['scored_tokens'] += int(mask.sum())
                expected['presented_bytes'] += int(Lexicon.lengths[xn].sum())
                expected['scored_bytes'] += int(Lexicon.lengths[yn[mask]].sum())
        self.assertEqual(result['sampled_windows_sha256'], total.hexdigest())
        self.assertEqual(result['measured_windows_sha256'], measured.hexdigest())
        self.assertEqual(result['measured_exposure'], expected)
        self.assertEqual([r['warmup'] for r in result['observations']], [True, False, False])
        self.assertAlmostEqual(result['measured_input_tokens_per_second'], expected['presented_tokens']/result['measured_seconds'])

    def test_invalid_settings_and_update_failure_restore_caller_state(self):
        condition = conditions()[0]; model = self.model(condition); original = state_hash(model)
        for bad in (replace(PILOT, batch=0), replace(PILOT, measured_updates=True), replace(PILOT, context_warmup=4)):
            with self.assertRaises(ValueError): measure(model, DOCUMENTS, Lexicon(), condition, BINDING, bad)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            measure(model, DOCUMENTS, Lexicon(), dict(method='unknown', seed=42, label='unknown-s42'), BINDING, PILOT)
        rng = torch.get_rng_state().clone(); threads = torch.get_num_threads()
        with patch('flm.language_learning_pilot.update', side_effect=FloatingPointError('Fixture failure')):
            with self.assertRaisesRegex(FloatingPointError, 'Fixture failure'):
                measure(model, DOCUMENTS, Lexicon(), condition, BINDING, PILOT)
        self.assertEqual(state_hash(model), original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state())); self.assertEqual(torch.get_num_threads(), threads)

    def test_pilot_order_and_priority_gate(self):
        self.assertEqual(ordered_conditions(), ordered_conditions())
        self.assertEqual({r['label'] for r in ordered_conditions()}, {r['label'] for r in conditions()})
        self.assertEqual(len(ordered_conditions()), 8); self.assertNotEqual(ordered_conditions(), conditions())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('flm.language_learning_pilot.load_inputs') as loader:
                with self.assertRaisesRegex(ValueError, 'Priority BabyLM'): run(root)
                loader.assert_not_called(); self.assertEqual(list(root.iterdir()), [])
            for scale in ('10m', '100m'):
                for variant in ('flm', 'gru', 'transformer'):
                    for seed in (42, 43):
                        target = root/f'runs/babylm-{scale}/{variant}-s{seed}'
                        target.mkdir(parents=True); (target/'best.pt').write_bytes(b'fixture')
                        (target/'complete.json').write_text(json.dumps(dict(steps=12000,
                            best_checkpoint_sha256=hashlib.sha256(b'fixture').hexdigest())))
            prerequisites(root)
            (target/'best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity changed'): prerequisites(root)

    def execute(self, root, effect=None):
        def prepared(inputs, condition): return self.model(condition), BINDING
        def small(model, docs, lexicon, condition, binding):
            return measure(model, docs, lexicon, condition, binding, PILOT)
        with patch('flm.language_learning_pilot.prerequisites'), \
             patch('flm.language_learning_pilot.load_inputs', return_value=SimpleNamespace(
                 documents=DOCUMENTS, lexicon=Lexicon(), binding=BINDING)), \
             patch('flm.language_learning_pilot.prepare_condition', side_effect=prepared), \
             patch('flm.language_learning_pilot.measure', side_effect=effect or small), patch('builtins.print'):
            return run(root)

    def test_runner_preserves_failed_attempt_and_refuses_completed_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = Path(__file__).resolve().parents[1]/'flm'
            (root/'flm').mkdir()
            for path in source.glob('*.py'): shutil.copyfile(path, root/'flm'/path.name)
            count = 0
            def fail_second(model, docs, lexicon, condition, binding):
                nonlocal count
                count += 1
                if count == 2: raise RuntimeError('Synthetic second-condition failure')
                return measure(model, docs, lexicon, condition, binding, PILOT)
            with self.assertRaisesRegex(RuntimeError, 'second-condition'): self.execute(root, fail_second)
            failed = list((root/'runs/language-learning-timing-pilot').glob('*/failure.json'))
            self.assertEqual(len(failed), 1)
            self.assertEqual(json.loads(failed[0].read_text())['completed_conditions'], 1)
            self.assertFalse((root/'reports/language-eligibility/timing-pilot.json').exists())
            destination = self.execute(root); result = json.loads(destination.read_text())
            self.assertEqual(len(result['conditions']), 8)
            self.assertFalse(result['training_protocol_frozen']); self.assertFalse(result['validation_or_test_payloads_opened'])
            with self.assertRaisesRegex(ValueError, 'already recorded'): self.execute(root)
            self.assertTrue(failed[0].is_file()); self.assertEqual(list(root.rglob('*.pt')), [])

    def test_runner_rejects_mismatched_within_seed_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); seen = set()
            def mismatch(model, docs, lexicon, condition, binding):
                result = measure(model, docs, lexicon, condition, binding, PILOT)
                if condition['seed'] in seen: result['sampled_windows_sha256'] = 'f'*64
                seen.add(condition['seed'])
                return result
            with self.assertRaisesRegex(ValueError, 'Unmatched'): self.execute(root, mismatch)
            self.assertEqual(len(list(root.rglob('failure.json'))), 1)
            self.assertFalse((root/'reports/language-eligibility/timing-pilot.json').exists())


if __name__ == '__main__': unittest.main()
