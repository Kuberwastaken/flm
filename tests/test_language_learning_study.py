"""Temporary synthetic studies; no official protocol, cost result or corpus fit."""
from contextlib import ExitStack
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from flm.language_learning_inputs import conditions
from flm.language_learning_pilot import Pilot, measure
from flm.language_learning_train import configure
from flm.language_learning_study import (IDENTITY, PILOT, PILOT_SOURCES, PROTOCOL, REPORTS,
    SELECTION, SOURCES, STUDY, freeze_selection, initialize, test_metadata as metadata,
    train_study, verified_context, verify_pilot)
from test_language_learning_train import DOCUMENTS, Lexicon, SETTINGS
from test_language_learning_validation import panel
from test_language_eligibility import fixture

SETTINGS = replace(SETTINGS, seed=42)
TINY = Pilot(1, 2, 2, 4, 1, 1)


class LearningStudyTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = Path(__file__).resolve().parents[1]/'flm'; (self.root/'flm').mkdir()
        for name in SOURCES: shutil.copyfile(source/name, self.root/'flm'/name)
        self.inputs = SimpleNamespace(documents=DOCUMENTS, lexicon=Lexicon(), binding=dict(fixture=True))
        self.write(PROTOCOL, 'Synthetic fixture protocol; not an official training budget.', raw=True)
        manifest = dict(fixture=True, files={'ids.bin': 'test payload deliberately absent'})
        self.write(Path('data/processed/babylm-2026-bpe/test/manifest.json'), manifest)
        self.write(Path('data/tokenizers/babylm-2026-4096/tokenization-card.json'),
                   dict(tokenizer_sha256=Lexicon.sha256, partitions=dict(test=manifest)))
        self.pilot = dict(input_binding=self.inputs.binding, conditions=[])
        for condition in conditions():
            model, binding = self.prepare(self.inputs, condition)
            self.pilot['conditions'].append(measure(model, DOCUMENTS, Lexicon(), condition, binding, TINY))
        self.write(PILOT, self.pilot)

    def write(self, relative, value, raw=False):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value), encoding='utf8')

    def prepare(self, inputs, condition):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(condition['seed']); model = fixture()
        configure(model, condition['method'])
        return model, dict(inputs.binding, condition=condition)

    def patched(self):
        stack = ExitStack()
        stack.enter_context(patch('flm.language_learning_study.prerequisites'))
        stack.enter_context(patch('flm.language_learning_study.verify_pilot', return_value=self.pilot))
        stack.enter_context(patch('flm.language_learning_study.load_inputs', return_value=self.inputs))
        stack.enter_context(patch('flm.language_learning_study.load_panel', side_effect=lambda *args: panel()))
        stack.enter_context(patch('flm.language_learning_study.prepare_condition', side_effect=self.prepare))
        stack.enter_context(patch('builtins.print'))
        return stack

    def test_official_initialization_refuses_priority_and_missing_costs_without_loading_inputs(self):
        with patch('flm.language_learning_study.load_inputs') as loader:
            with self.assertRaisesRegex(ValueError, 'Priority BabyLM'): initialize(self.root, SETTINGS)
            loader.assert_not_called()
            with patch('flm.language_learning_study.prerequisites'):
                # Replace only the test-owned path, not any real project result.
                (self.root/PILOT).unlink()
                with self.assertRaisesRegex(ValueError, 'Measure the full-window'): initialize(self.root, SETTINGS)
                loader.assert_not_called()
        self.assertFalse((self.root/IDENTITY).exists()); self.assertFalse((self.root/STUDY).exists())

    def test_identity_registers_all_eight_before_fitting_and_is_immutable(self):
        with self.patched():
            identity = initialize(self.root, SETTINGS)
            self.assertEqual(len(identity['conditions']), 8)
            self.assertEqual({row['condition']['label'] for row in identity['conditions']}, {c['label'] for c in conditions()})
            self.assertFalse(any((self.root/STUDY/c['label']).exists() for c in conditions()))
            self.assertEqual(initialize(self.root, SETTINGS), identity)
            with self.assertRaisesRegex(ValueError, 'different learning-rule study'):
                initialize(self.root, replace(SETTINGS, steps=6))
            altered = copy.deepcopy(identity); altered['conditions'].pop(); self.write(IDENTITY, altered)
            with self.assertRaisesRegex(ValueError, 'identity changed'): verified_context(self.root)

    def test_existing_condition_directory_prevents_retroactive_freeze(self):
        (self.root/STUDY/conditions()[0]['label']).mkdir(parents=True)
        with self.patched():
            with self.assertRaisesRegex(ValueError, 'already exist'): initialize(self.root, SETTINGS)
        self.assertFalse((self.root/IDENTITY).exists())

    def test_complete_tiny_study_trains_selects_and_revalidates_all_eight_without_test_payloads(self):
        with self.patched():
            initialize(self.root, SETTINGS); train_study(self.root)
            result = freeze_selection(self.root)
            self.assertEqual(len(result['conditions']), 8); self.assertFalse(result['test_payloads_opened'])
            self.assertEqual(freeze_selection(self.root), result)
            for seed in (42, 43):
                rows = [r for r in result['conditions'] if r['condition']['seed'] == seed]
                self.assertTrue(all(r['final_exposure'] == rows[0]['final_exposure'] for r in rows))
                self.assertTrue(all(r['final_exposure']['presented_tokens'] == 32 for r in rows))
            self.assertFalse((self.root/'data/processed/babylm-2026-bpe/test/ids.bin').exists())
            self.assertTrue((self.root/SELECTION).exists())
            # Changing a selected payload is rejected even when a cached selection exists.
            chosen = self.root/result['conditions'][-1]['checkpoint']; chosen.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint or declaration changed'): freeze_selection(self.root)

    def test_incomplete_inventory_refuses_before_validation_input_loading(self):
        with self.patched(): initialize(self.root, SETTINGS)
        self.write(STUDY/conditions()[0]['label']/'complete.json', dict(complete=True))
        with patch('flm.language_learning_study.load_panel') as loader:
            with self.assertRaisesRegex(ValueError, 'run incomplete'): freeze_selection(self.root)
            loader.assert_not_called()
        self.assertFalse((self.root/SELECTION).exists())

    def test_changed_code_protocol_and_test_metadata_invalidate_frozen_identity(self):
        with self.patched():
            initialize(self.root, SETTINGS)
            for target in (PROTOCOL, Path('flm/model.py')):
                path = self.root/target; original = path.read_bytes(); path.write_bytes(original+b'\n# altered')
                with self.assertRaisesRegex(ValueError, 'identity changed'): verified_context(self.root)
                path.write_bytes(original)
            manifest_path = Path('data/processed/babylm-2026-bpe/test/manifest.json')
            self.write(manifest_path, dict(changed=True))
            with self.assertRaisesRegex(ValueError, 'test metadata changed'): verified_context(self.root)

    def test_cost_gate_rejects_incomplete_settings_source_and_arithmetic(self):
        baseline = dict(conditions=[], validation_or_test_payloads_opened=False,
            source_sha256={name:hashlib.sha256((self.root/'flm'/name).read_bytes()).hexdigest() for name in PILOT_SOURCES})
        for condition in conditions():
            observations = [dict(step=i, warmup=i <= 3, seconds=1., presented_tokens=1536, scored_tokens=1280,
                                 presented_bytes=4500, scored_bytes=3750)
                            for i in range(1, 16)]
            baseline['conditions'].append(dict(condition=condition, settings=asdict(Pilot().settings(condition['seed'])),
                observations=observations, source_model_unchanged=True, disposable_parameters_changed=True,
                checkpoint_written=False, language_scores_reported=False, measured_seconds=12.,
                sampled_windows_sha256=f'{condition["seed"]:064x}', measured_windows_sha256=f'{condition["seed"]:064x}',
                measured_exposure=dict(presented_tokens=18432, scored_tokens=15360, presented_bytes=54000, scored_bytes=45000)))
        self.write(PILOT, baseline); verify_pilot(self.root)
        for change in (lambda r:r['conditions'].pop(), lambda r:r['source_sha256'].clear(),
                       lambda r:r['conditions'][0]['settings'].update(sequence=3),
                       lambda r:r['conditions'][0].update(measured_seconds=99),
                       lambda r:r['conditions'][0]['measured_exposure'].update(presented_tokens=99)):
            bad = copy.deepcopy(baseline); change(bad); self.write(PILOT, bad)
            with self.assertRaises(ValueError): verify_pilot(self.root)

    def test_test_metadata_binding_requires_no_token_payload_and_detects_card_changes(self):
        result = metadata(self.root, Lexicon()); self.assertFalse(result['payloads_opened'])
        path = Path('data/tokenizers/babylm-2026-4096/tokenization-card.json')
        self.write(path, dict(tokenizer_sha256='b'*64, partitions=dict(test={})))
        with self.assertRaisesRegex(ValueError, 'metadata changed'): metadata(self.root, Lexicon())


if __name__ == '__main__': unittest.main()
