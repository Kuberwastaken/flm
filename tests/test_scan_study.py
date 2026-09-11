"""Whole-study lifecycle tests use small artificial models and training rows only."""
import copy
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm import scan_study as study
from flm.inference import state_hash
from flm.provenance import sha256, write_json
from flm.scan_conditions import Prepared, conditions
from flm.scan_pilot import Pilot, measure, pilot_conditions
from flm.scan_train import Settings, canonical, fit
from test_scan_runtime import models, RECORDS
from test_scan_train import FixtureLexicon

SETTINGS = Settings(6, 2, .002, .0002, 1, .01, 1., 3, 42, 1)
SMALL_PILOT = Pilot(batch_size=2, threads=1)
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def prepared_fixture(root, condition):
    model = models()[('flm', 'gru', 'transformer').index(condition['variant'])]
    # Distinct, deterministic fixtures, not claimed pretrained checkpoints.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.001 * (condition['seed'] - 42) + (0.003 if condition['initialization'] == 'wikitext' else 0))
    rows = copy.deepcopy(RECORDS + [RECORDS[0]])
    lexicon = FixtureLexicon()
    binding = dict(format='artificial-scan-study-fixture', condition=condition,
        language_source=dict(initial_state_sha256=state_hash(model), tokenizer_sha256=lexicon.sha256),
        ordered_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest(), training_input={'fixture': True})
    return Prepared(model, rows, lexicon, binding)


def patches(root):
    stack = ExitStack()
    stack.enter_context(patch.object(study, 'prepare_condition', side_effect=prepared_fixture))
    stack.enter_context(patch.object(study, 'Pilot', return_value=SMALL_PILOT))
    stack.enter_context(patch.object(study, 'prerequisites', return_value={'prior.json': sha256(root/'prior.json')}))
    return stack


class ScanStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix='flm-scan-study-fixture-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.baseline = Path(cls.temporary.name)/'baseline'; cls.baseline.mkdir()
        root = cls.baseline
        (root/'flm').mkdir()
        for name in study.SOURCES: shutil.copyfile(SOURCE_ROOT/'flm'/name, root/'flm'/name)
        write_json(root/'prior.json', {'scope': 'Artificial priority-study fixture'})
        (root/study.PROTOCOL).parent.mkdir(parents=True)
        (root/study.PROTOCOL).write_text('Synthetic lifecycle test; not an official study budget.\n', encoding='utf8')
        write_json(root/'data/cards/scan.json', dict(revision='fixture',
            source_files={names[1]:dict(sha256='a'*64, bytes=1) for names in study.SPLITS.values()},
            partitions={split:dict(test=dict(sha256='b'*64, bytes=1, examples=1)) for split in study.SPLITS}))
        # Actual timed disposable fixture updates; their timings stay in temp files.
        with patches(root):
            observations = [measure(prepared_fixture(root, condition), SMALL_PILOT) for condition in pilot_conditions()]
            write_json(root/study.PILOT, dict(conditions=observations,
                source_sha256={name:sha256(root/'flm'/name) for name in study.PILOT_SOURCES},
                torch=str(torch.__version__), numpy=str(np.__version__),
                test_partitions_opened=False, benchmark_budget_selected=False))
            study.initialize(root, SETTINGS)
            with patch('builtins.print'): study.train_study(root)

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(prefix='flm-scan-study-case-')
        self.addCleanup(self.temporary_case.cleanup)
        self.root = Path(self.temporary_case.name)/'study'
        shutil.copytree(self.baseline, self.root)
        self.context = patches(self.root); self.addCleanup(self.context.close)
        self.identity = study.read(self.root/study.IDENTITY)

    def test_all_36_actual_fixture_fits_pass_terminal_gate_without_any_test_files(self):
        result = study.freeze_selection(self.root)
        self.assertEqual([r['condition'] for r in result['conditions']], conditions())
        self.assertEqual(len(result['conditions']), 36)
        self.assertFalse(result['test_payloads_opened']); self.assertFalse(result['inference_performed'])
        self.assertEqual(study.freeze_selection(self.root), result)
        for split in study.SPLITS:
            self.assertFalse((self.root/f'data/processed/scan/{split}/test.jsonl').exists())
            for seed in (42, 43):
                rows = [r for r in result['conditions'] if r['condition']['split'] == split and r['condition']['seed'] == seed]
                self.assertTrue(all(r['final_exposure'] == rows[0]['final_exposure'] for r in rows))
                self.assertEqual(len({r['sampled_row_sha256'] for r in rows}), 1)

    def test_gate_reconstructs_inventory_before_preparation_even_if_identity_is_shortened(self):
        missing = conditions()[-1]
        (self.root/study.STUDY/missing['label']/'complete.json').unlink()
        identity = copy.deepcopy(self.identity); identity['conditions'].pop()
        write_json(self.root/study.IDENTITY, identity)
        with patch.object(study, 'verified_context') as context:
            with self.assertRaisesRegex(ValueError, 'condition incomplete'): study.freeze_selection(self.root)
            context.assert_not_called()
        self.assertFalse((self.root/study.SELECTION).exists())

    def test_identity_cannot_be_replaced_by_a_new_budget_or_shortened_matrix(self):
        self.assertEqual(study.initialize(self.root, SETTINGS), self.identity)
        with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
            study.initialize(self.root, replace(SETTINGS, steps=12))
        with self.assertRaisesRegex(ValueError, 'sampling seed 42'):
            study.initialize(self.root, replace(SETTINGS, sampling_seed=43))
        changed = copy.deepcopy(self.identity); changed['conditions'].pop()
        write_json(self.root/study.IDENTITY, changed)
        with self.assertRaisesRegex(ValueError, 'identity changed'): study.verified_context(self.root)

    def test_changed_sources_and_protocol_are_rejected_before_fitting(self):
        (self.root/'flm/scan_runtime.py').write_text('# changed fixture\n', encoding='utf8')
        with patch.object(study, 'fit') as fitter:
            with self.assertRaisesRegex(ValueError, 'pilot source changed'): study.train_study(self.root)
            fitter.assert_not_called()
        shutil.copyfile(SOURCE_ROOT/'flm/scan_runtime.py', self.root/'flm/scan_runtime.py')
        (self.root/study.PROTOCOL).write_text('changed protocol\n', encoding='utf8')
        with patch.object(study, 'fit') as fitter:
            with self.assertRaisesRegex(ValueError, 'identity changed'): study.train_study(self.root)
            fitter.assert_not_called()

    def test_existing_fit_directories_prevent_a_new_posthoc_identity(self):
        (self.root/study.IDENTITY).unlink()
        with patch.object(study, 'prepare_condition') as prepare:
            with self.assertRaisesRegex(ValueError, 'condition directories already exist'):
                study.initialize(self.root, SETTINGS)
            prepare.assert_not_called()

    def test_missing_priority_queue_refuses_initialization_before_pilot_or_data(self):
        self.context.close()
        with patch.object(study, 'verify_pilot') as pilot, patch.object(study, 'prepare_condition') as prepare:
            with self.assertRaisesRegex(ValueError, 'fixed BabyLM queue'): study.initialize(self.root, SETTINGS)
            pilot.assert_not_called(); prepare.assert_not_called()

    def test_duplicate_pilot_condition_and_forged_exposure_are_rejected(self):
        original = study.read(self.root/study.PILOT)
        bad = copy.deepcopy(original); bad['conditions'][-1] = bad['conditions'][0]
        write_json(self.root/study.PILOT, bad)
        with self.assertRaisesRegex(ValueError, 'timing inventory'): study.verify_pilot(self.root)
        bad = copy.deepcopy(original)
        bad['conditions'][0]['observations'][3]['input_tokens'] += 1
        bad['conditions'][0]['measured_input_tokens'] += 1
        write_json(self.root/study.PILOT, bad)
        pilot = study.verify_pilot(self.root)  # Totals alone are not sufficient.
        with self.assertRaisesRegex(ValueError, 'per-update exposure'):
            study.identity_for(self.root, SETTINGS, pilot, self.identity['priority_studies'])

    def rewrite_payload(self, row, change):
        directory = self.root/study.STUDY/row['condition']['label']
        path = directory/'checkpoint-000006.pt'
        payload = torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location='cpu')
        change(payload); torch.save(payload, path)
        for name in ('saved-000006.json', 'complete.json'):
            record = study.read(directory/name); record['checkpoint_sha256'] = sha256(path)
            write_json(directory/name, record)

    def test_rehashed_nonfinite_checkpoint_cannot_pass_complete_metadata(self):
        row = self.identity['conditions'][0]
        def corrupt(payload): next(iter(payload['model'].values())).flatten()[0] = float('nan')
        self.rewrite_payload(row, corrupt)
        with self.assertRaisesRegex(ValueError, 'finiteness changed'):
            study.audit_completed(self.root, row, sha256(self.root/study.IDENTITY))
        self.assertFalse((self.root/study.SELECTION).exists())

    def test_rehashed_sampler_change_is_detected_and_audit_does_not_change_rng(self):
        row = self.identity['conditions'][0]
        self.rewrite_payload(row, lambda payload: payload.update(sampled_row_sha256='f'*64))
        # The fixture constructor has its own deterministic initialization; protect
        # that separately so this assertion isolates checkpoint restoration.
        prepared, settings, binding, lengths, scored = study.prepared_run(self.root, row, sha256(self.root/study.IDENTITY))
        before = torch.get_rng_state().clone()
        with patch.object(study, 'prepared_run', return_value=(prepared, settings, binding, lengths, scored)):
            with self.assertRaisesRegex(ValueError, 'sampled rows, RNG or exact exposure'):
                study.audit_completed(self.root, row, sha256(self.root/study.IDENTITY))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_rehashed_negative_optimizer_second_moment_is_not_valid_completion(self):
        row = self.identity['conditions'][0]
        def corrupt(payload): next(iter(payload['optimizer']['state'].values()))['exp_avg_sq'].flatten()[0] = -1
        self.rewrite_payload(row, corrupt)
        with self.assertRaisesRegex(ValueError, 'second moments cannot be negative'):
            study.audit_completed(self.root, row, sha256(self.root/study.IDENTITY))

    def test_failure_is_retained_and_resumption_preserves_uninterrupted_payloads(self):
        # Remove only this test's independent temporary fit outputs.
        study_root = (self.root/study.STUDY).resolve()
        self.assertTrue(study_root.is_relative_to(self.root.resolve()))
        shutil.rmtree(study_root)
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                fit(*args, **kwargs, until=3)
                raise RuntimeError('synthetic interruption at a committed boundary')
            return fit(*args, **kwargs)
        with patch.object(study, 'fit', side_effect=interrupted), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'): study.train_study(self.root)
        failures = list((self.root/study.STUDY/'attempts').glob('*/failure.json'))
        self.assertEqual(len(failures), 1)
        self.assertEqual(study.read(failures[0])['condition'], conditions()[1])
        with patch('builtins.print'): study.train_study(self.root)
        result = study.freeze_selection(self.root)
        self.assertEqual(len(result['conditions']), 36); self.assertTrue(failures[0].exists())
        for row in result['conditions']:
            path = Path(row['checkpoint'])
            a = torch.load(self.root/path, weights_only=True, map_location='cpu')
            b = torch.load(self.baseline/path, weights_only=True, map_location='cpu')
            for name in a['model']: self.assertTrue(torch.equal(a['model'][name], b['model'][name]))
            for key in ('history', 'sampler_rng', 'exposure', 'sampled_row_sha256', 'declaration'):
                self.assertEqual(a[key], b[key])

    def test_late_completion_mutation_prevents_any_whole_study_selection(self):
        original = study.audit_completed
        first = self.root/study.STUDY/conditions()[0]['label']/'complete.json'
        def mutate_after_last(root, row, digest):
            result = original(root, row, digest)
            if row['condition'] == conditions()[-1]: first.write_bytes(first.read_bytes()+b' ')
            return result
        with patch.object(study, 'audit_completed', side_effect=mutate_after_last):
            with self.assertRaisesRegex(ValueError, 'completion changed'): study.freeze_selection(self.root)
        self.assertFalse((self.root/study.SELECTION).exists())


if __name__ == '__main__': unittest.main()
