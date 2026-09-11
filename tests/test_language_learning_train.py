"""Tiny artificial token streams; no acquired corpus or official study fitting."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from flm.language_learning_train import (METHODS, Settings, configure, fit, learning_rate,
                                         optimizer_for, score_mask, training_lease, update)
from flm.local_learning import CORE_PARAMETERS
from flm.provenance import sha256, write_json
from test_language_eligibility import fixture

SETTINGS = Settings(steps=4, batch=2, sequence=4, warmup=1, learning_rate=.002,
    final_learning_rate=.0002, lr_warmup_updates=1, weight_decay=.01,
    gradient_clip=1., checkpoint_interval=2, seed=84, threads=1)
BINDING = dict(study='Artificial runner fixture; no corpus or benchmark result')
DOCUMENTS = [('a', np.array([0, 2, 2, 3, 4, 5, 2, 7, 1])),
             ('b', np.array([0, 8, 7, 3, 4, 6, 3, 2, 8, 1]))]


class Lexicon:
    vocabulary = 9
    sha256 = 'a'*64
    lengths = np.array([0, 0, 1, 2, 3, 1, 4, 2, 3])


def initial():
    torch.manual_seed(98)
    return fixture()


def payload(folder, step=4):
    return torch.load(folder/f'checkpoint-{step:06d}.pt', weights_only=True)


class LanguageLearningTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix='flm-language-rule-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fit(self, folder, method='eligibility', **kwargs):
        return fit(initial(), DOCUMENTS, Lexicon(), SETTINGS, method, BINDING, folder, **kwargs)

    def equal_tree(self, first, second):
        self.assertEqual(type(first), type(second))
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.equal_tree(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.equal_tree(a, b)
        else:
            self.assertEqual(first, second)

    def test_all_four_conditions_resume_exact_model_optimizer_rng_history_and_exposure(self):
        results = []
        for method in METHODS:
            with self.subTest(method=method):
                whole, resumed = self.root/f'whole-{method}', self.root/f'resume-{method}'
                self.fit(whole, method)
                partial = self.fit(resumed, method, until=2)
                self.assertFalse(partial['complete'])
                self.assertFalse((resumed/'complete.json').exists())
                final = self.fit(resumed, method)
                self.assertTrue(final['complete'])
                self.equal_tree(payload(whole), payload(resumed))
                self.assertEqual(self.fit(resumed, method), final)
                results.append(payload(whole))
        for result in results[1:]:
            self.assertEqual(result['sampled_token_sha256'], results[0]['sampled_token_sha256'])
            self.assertEqual(result['exposure'], results[0]['exposure'])
            self.assertEqual(result['declaration']['initial_state_sha256'], results[0]['declaration']['initial_state_sha256'])

    def test_fixed_core_is_unchanged_but_embedding_and_readout_learn(self):
        folder = self.root/'fixed'
        self.fit(folder, 'fixed_core')
        saved = payload(folder)
        before = initial().state_dict()
        for name in CORE_PARAMETERS:
            torch.testing.assert_close(saved['model'][name], before[name], atol=0, rtol=0)
        for name in ('embedding.weight', 'readout.weight'):
            self.assertFalse(torch.equal(saved['model'][name], before[name]))
        self.assertEqual(set(saved['declaration']['frozen_parameter_names']), set(CORE_PARAMETERS))
        self.assertNotIn('input.weight', saved['declaration']['optimizer_parameter_names'])

    def test_bptt_update_matches_direct_masked_objective_and_adamw_without_parameter_projection(self):
        model = initial()
        with torch.no_grad():
            model.edge_log_gain[0] = 3.4
        exact = copy.deepcopy(model)
        configure(model, 'bptt'); configure(exact, 'bptt')
        opt = optimizer_for(model, SETTINGS); reference_opt = optimizer_for(exact, SETTINGS)
        x = torch.tensor([[0, 2, 3, 4], [0, 3, 4, 5]])
        y = torch.tensor([[2, 3, 4, 1], [3, 4, 5, 1]])
        result = update(model, opt, x, y, SETTINGS, 'bptt', 1)
        reference_opt.param_groups[0]['lr'] = learning_rate(SETTINGS, 1)
        logits, _ = exact(x)
        losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').reshape_as(y)
        objective = losses[:, 1:3].mean()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(exact.parameters(), 1.)
        reference_opt.step()
        self.assertEqual(result['loss'], float(objective.detach()))
        for p, q in zip(model.parameters(), exact.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        self.assertGreater(float(model.edge_log_gain[0].detach()), 3)
        self.assertEqual(int(score_mask(y, SETTINGS).sum()), 4)
        self.assertEqual(int(score_mask(y, replace(SETTINGS, score_boundaries=True)).sum()), 6)

    def test_uncommitted_payload_is_ignored_after_interruption(self):
        whole, interrupted = self.root/'whole', self.root/'interrupted'
        self.fit(whole)
        self.fit(interrupted, until=2)
        original = write_json
        def crash(path, value):
            if path.name == 'saved-000004.json':
                raise RuntimeError('Simulated crash before commit marker')
            return original(path, value)
        with patch('flm.language_learning_train.write_json', side_effect=crash), self.assertRaisesRegex(RuntimeError, 'Simulated'):
            self.fit(interrupted)
        self.assertTrue((interrupted/'checkpoint-000004.pt').exists())
        self.assertFalse((interrupted/'saved-000004.json').exists())
        self.fit(interrupted)
        self.equal_tree(payload(whole), payload(interrupted))

    def test_changed_data_settings_and_initial_model_cannot_resume(self):
        folder = self.root/'bound'
        self.fit(folder, until=2)
        changed = copy.deepcopy(DOCUMENTS)
        changed[0][1][3] = 8
        for docs, settings, model in ((changed, SETTINGS, initial()),
                (DOCUMENTS, replace(SETTINGS, score_boundaries=True), initial()),
                (DOCUMENTS, SETTINGS, fixture())):
            with self.assertRaisesRegex(ValueError, 'declaration changed'):
                fit(model, docs, Lexicon(), settings, 'eligibility', BINDING, folder)

    def test_payload_tampering_rejected_even_with_updated_file_checksum(self):
        cases = ('moments', 'exposure', 'sampler', 'frozen', 'buffer', 'history')
        for case in cases:
            with self.subTest(case=case):
                folder = self.root/case
                self.fit(folder, 'fixed_core', until=2)
                saved = payload(folder, 2)
                if case == 'moments': saved['optimizer']['state'][0]['exp_avg'].flatten()[0] = float('nan')
                elif case == 'exposure': saved['exposure']['scored_bytes'] += 1
                elif case == 'sampler': saved['sampler_rng']['state']['state'] += 1
                elif case == 'frozen': saved['model']['input.weight'][0, 0] += .01
                elif case == 'buffer': saved['model']['base_weight'][0] += .01
                else: saved['history'][0]['learning_rate'] += .001
                checkpoint = folder/'checkpoint-000002.pt'
                torch.save(saved, checkpoint)
                marker = json.loads((folder/'saved-000002.json').read_text())
                marker['checkpoint_sha256'] = sha256(checkpoint)
                write_json(folder/'saved-000002.json', marker)
                with self.assertRaises(ValueError):
                    self.fit(folder, 'fixed_core')
                self.assertFalse((folder/'complete.json').exists())

    def test_empty_masks_fail_without_resampling_and_live_writer_excludes_second_writer(self):
        with self.assertRaisesRegex(ValueError, 'no scored'):
            score_mask(torch.zeros(2, 4, dtype=torch.long), SETTINGS)
        with training_lease(self.root/'locked'):
            with self.assertRaisesRegex(RuntimeError, 'writer'):
                self.fit(self.root/'locked')
        self.assertFalse((self.root/'locked'/'declaration.json').exists())
        with self.assertRaisesRegex(ValueError, 'checkpoint boundary'):
            self.fit(self.root/'invalid-stop', until=3)


if __name__ == '__main__':
    unittest.main()
