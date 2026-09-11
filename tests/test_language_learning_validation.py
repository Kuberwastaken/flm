"""Artificial training and validation records; no acquired-corpus fitting."""
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from flm.inference import state_hash
from flm.language_learning_train import fit, METHODS
from flm.language_learning_validation import (Panel, check_score, document_identity,
                                              earliest_minimum, select_checkpoint)
from test_language_learning_train import DOCUMENTS, Lexicon, BINDING, SETTINGS
from test_language_eligibility import fixture

SETTINGS = replace(SETTINGS, seed=42)
VALIDATION = [('validation-a', np.array([0, 2, 3, 4, 5, 6, 7, 1])),
              ('validation-b', np.array([0, 8, 7, 6, 3, 4, 2, 1]))]


def initial():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return fixture()


def panel():
    docs = copy.deepcopy(VALIDATION)
    return Panel(docs, dict(partition='validation', fixture=True, coverage=document_identity(docs, Lexicon())))


class LearningValidationTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def train(self, method='bptt', until=None):
        folder = self.root/method
        fit(initial(), DOCUMENTS, Lexicon(), SETTINGS, method, BINDING, folder, until=until)
        return folder

    def select(self, folder, method='bptt', **kwargs):
        return select_checkpoint(initial(), DOCUMENTS, Lexicon(), SETTINGS, method, BINDING, folder, kwargs.pop('panel', panel()), **kwargs)

    def test_all_four_methods_select_complete_checkpoints_and_preserve_initial_state(self):
        for method in METHODS:
            folder = self.train(method); source = initial(); before = state_hash(source)
            rng = torch.get_rng_state().clone(); threads = torch.get_num_threads()
            result = select_checkpoint(source, DOCUMENTS, Lexicon(), SETTINGS, method, BINDING, folder, panel(), chunk_size=2)
            self.assertEqual(state_hash(source), before); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(torch.get_num_threads(), threads)
            self.assertEqual([r['step'] for r in result['observations']], [2, 4])
            expected = min(result['observations'], key=lambda row: (row['score']['bits_per_byte'], row['step']))
            self.assertEqual(result['selected'], expected); self.assertFalse(result['test_payloads_opened'])
            self.assertEqual(self.select(folder, method, chunk_size=2), result)

    def test_likelihoods_match_whole_block_autograd_oracle_and_exact_byte_denominators(self):
        folder = self.train(); result = self.select(folder, chunk_size=2)
        for record in result['observations']:
            model = initial(); model.load_state_dict(torch.load(folder/record['checkpoint'], weights_only=True)['model'])
            total = 0.; tokens = byte_count = 0
            for _, values in VALIDATION:
                x = torch.tensor(values[:-1]).unsqueeze(0); targets = torch.tensor(values[1:])
                logits, _ = model(x); mask = targets >= 2
                nll = F.cross_entropy(logits.squeeze(0), targets, reduction='none')[mask].sum()
                total += float(nll.detach()); tokens += int(mask.sum())
                byte_count += int(Lexicon.lengths[targets[mask].numpy()].sum())
            score = record['score']
            self.assertAlmostEqual(score['nll'], total, places=10)
            self.assertEqual(score['tokens'], tokens); self.assertEqual(score['bytes'], byte_count)
            self.assertAlmostEqual(score['bits_per_byte'], total/byte_count/math.log(2), places=10)

    def test_earliest_exact_tie_and_complete_cadence_are_required(self):
        rows = [dict(step=2, score=dict(bits_per_byte=2.)), dict(step=4, score=dict(bits_per_byte=2.))]
        self.assertEqual(earliest_minimum(rows, SETTINGS)['step'], 2)
        for bad in (rows[::-1], rows[:1], rows+rows[-1:]):
            with self.assertRaisesRegex(ValueError, 'Every declared'): earliest_minimum(bad, SETTINGS)
        rows[1]['score']['bits_per_byte'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid'): earliest_minimum(rows, SETTINGS)

    def test_incomplete_training_and_changed_endpoint_fail_before_scoring(self):
        folder = self.train(until=2)
        with patch('flm.language_learning_validation.evaluate') as evaluator:
            with self.assertRaisesRegex(ValueError, 'must complete'): self.select(folder)
            evaluator.assert_not_called()
        self.train()
        path = folder/'complete.json'; completed = json.loads(path.read_text()); completed['exposure']['scored_bytes'] += 1
        path.write_text(json.dumps(completed))
        with patch('flm.language_learning_validation.evaluate') as evaluator:
            with self.assertRaisesRegex(ValueError, 'exposure changed'): self.select(folder)
            evaluator.assert_not_called()
        self.assertFalse((folder/'validation-selection.json').exists())

    def test_rehashed_invalid_optimizer_payload_is_rejected_before_validation(self):
        folder = self.train(); path = folder/'checkpoint-000004.pt'
        saved = torch.load(path, weights_only=True); saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(1)
        torch.save(saved, path); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('saved-000004.json', 'complete.json'):
            target = folder/name; record = json.loads(target.read_text()); record['checkpoint_sha256'] = digest
            target.write_text(json.dumps(record))
        with patch('flm.language_learning_validation.evaluate') as evaluator:
            with self.assertRaisesRegex(ValueError, 'optimizer moments'): self.select(folder)
            evaluator.assert_not_called()

    def test_panel_changes_overlap_and_inconsistent_cached_scores_are_rejected(self):
        folder = self.train(); selected = self.select(folder)
        bad = panel(); bad.documents[0][1][2] = 8
        with self.assertRaisesRegex(ValueError, 'panel identity changed'): self.select(folder, panel=bad)
        duplicate = Panel(copy.deepcopy(DOCUMENTS), dict(partition='validation', coverage=document_identity(DOCUMENTS, Lexicon())))
        with self.assertRaisesRegex(ValueError, 'IDs overlap'): self.select(folder, panel=duplicate)
        path = folder/'validation-selection.json'; damaged = copy.deepcopy(selected)
        damaged['observations'][0]['score']['bytes'] += 1; path.write_text(json.dumps(damaged))
        with self.assertRaisesRegex(ValueError, 'denominators changed'): self.select(folder)
        path.write_text(json.dumps(selected))
        with self.assertRaisesRegex(ValueError, 'identity changed'): self.select(folder, chunk_size=3)

    def test_scoring_failure_restores_rng_threads_and_writes_no_selection(self):
        folder = self.train(); rng = torch.get_rng_state().clone(); threads = torch.get_num_threads()
        with patch('flm.language_learning_validation.evaluate', side_effect=RuntimeError('Synthetic scoring failure')):
            with self.assertRaisesRegex(RuntimeError, 'scoring failure'): self.select(folder)
        self.assertTrue(torch.equal(rng, torch.get_rng_state())); self.assertEqual(torch.get_num_threads(), threads)
        self.assertFalse((folder/'validation-selection.json').exists())

    def test_score_validator_rejects_changed_block_order_and_nonfinite_arithmetic(self):
        folder = self.train(); score = self.select(folder)['selected']['score']; coverage = panel().binding['coverage']
        for mutate in (lambda s:s['documents'].reverse(), lambda s:s.update(nll=float('inf')),
                       lambda s:s.update(token_perplexity=float('nan')),
                       lambda s:s['documents'][0].update(bits_per_byte=999)):
            bad = copy.deepcopy(score); mutate(bad)
            with self.assertRaises(ValueError): check_score(bad, coverage)


if __name__ == '__main__': unittest.main()
