"""Small payload/score fixtures only; never load a corpus or fit a model."""
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm.babylm_test import validation_coverage, verify_completed_payloads
from flm.provenance import sha256, write_json


class BabyLMSelectionPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.folder = self.root / 'run'; self.folder.mkdir()
        self.lexicon = SimpleNamespace(sha256='tokenizer')
        self.protocol = dict(steps=12000)
        self.card = dict(trainable_parameters=1)
        self.model = SimpleNamespace(parameter_card=lambda: self.card)
        for name in ('best.pt', 'last.pt'):
            (self.folder / name).write_bytes(name.encode())
        for step in range(500, 12001, 500):
            nll = 25 - step / 500
            row = dict(document='block', tokens=2, bytes=4, nll=nll)
            write_json(self.folder / f'validation-{step:06d}.json', dict(
                tokenizer_sha256='tokenizer', documents=[row], tokens=2, bytes=4,
                nll=nll, bits_per_byte=nll / 4 / math.log(2)))
        self.selected = dict(checkpoint='run/best.pt', checkpoint_sha256=sha256(self.folder / 'best.pt'),
            seed=42, variant='gru', training_source_commit='source', selection_validation_bpb=1 / 4 / math.log(2))
        self.payload = dict(config=dict(variant='gru'), step=12000, best=self.selected['selection_validation_bpb'],
            model=dict(weight=torch.ones(1)), run=dict(protocol=self.protocol, seed=42,
                source_commit='source', test_set_used_for_training=False, parameter_card=self.card,
                exposure=dict(presented_tokens=18432000, presented_bytes=100, scored_bytes=80)))
        self.best = copy.deepcopy(self.payload); self.final = copy.deepcopy(self.payload)
        self.best['_file_sha256'] = sha256(self.folder / 'best.pt')
        self.final['_file_sha256'] = sha256(self.folder / 'last.pt')

    def verify(self):
        with patch('flm.babylm_test.restore', side_effect=[(self.model, self.best), (self.model, self.final)]):
            return verify_completed_payloads(self.root, self.selected, self.protocol,
                self.lexicon, [('block', 2, 4)])

    def test_binds_final_payload_and_every_validation_observation(self):
        result = self.verify()
        self.assertEqual(result['checkpoint_step'], 12000)
        self.assertEqual(result['final_checkpoint_sha256'], sha256(self.folder / 'last.pt'))
        self.assertEqual(len(result['validation_record_sha256']), 24)
        self.assertEqual(result['final_exposure']['presented_tokens'], 18432000)

    def test_rejects_incomplete_wrong_or_nonfinite_payloads(self):
        original = copy.deepcopy(self.final)
        cases = (
            ('step', lambda p: p.update(step=11500)),
            ('seed', lambda p: p['run'].update(seed=43)),
            ('protocol', lambda p: p['run'].update(protocol=dict(steps=10000))),
            ('card', lambda p: p['run'].update(parameter_card={})),
            ('weights', lambda p: p['model'].update(weight=torch.tensor([float('nan')]))),
            ('exposure', lambda p: p['run']['exposure'].update(presented_tokens=0)),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                self.final = copy.deepcopy(original); mutate(self.final)
                with self.assertRaises(ValueError): self.verify()

    def test_rejects_nonminimal_selection_and_changed_panel_counts(self):
        path = self.folder / 'validation-000500.json'
        write_json(path, dict(tokenizer_sha256='tokenizer', documents=[dict(
            document='block', tokens=2, bytes=4, nll=.1)], tokens=2, bytes=4,
            nll=.1, bits_per_byte=.1 / 4 / math.log(2)))
        with self.assertRaisesRegex(ValueError, 'minimum-validation'): self.verify()
        self.best['step'] = 500
        self.best['best'] = self.final['best'] = self.selected['selection_validation_bpb'] = .1 / 4 / math.log(2)
        self.assertEqual(self.verify()['checkpoint_step'], 500)
        with patch('flm.babylm_test.restore', side_effect=[(self.model, self.best), (self.model, self.final)]):
            with self.assertRaisesRegex(ValueError, 'coverage'):
                verify_completed_payloads(self.root, self.selected, self.protocol,
                    self.lexicon, [('block', 3, 4)])

    def test_panel_counts_use_exact_prefix_and_exclude_boundary_tokens(self):
        lexicon = SimpleNamespace(lengths=np.array([0, 0, 1, 2, 4]))
        documents = [('a', np.array([0, 2, 3, 4, 1])), ('b', np.array([0, 4, 1]))]
        panel = dict(ids=['a', 'b'], target_tokens_per_block=2)
        self.assertEqual(validation_coverage(documents, panel, lexicon, 4), [('a', 2, 3), ('b', 1, 4)])
        with self.assertRaisesRegex(ValueError, 'panel'):
            validation_coverage(documents, dict(ids=['a', 'a'], target_tokens_per_block=2), lexicon, 4)

    def test_validation_ties_select_the_earlier_checkpoint(self):
        last_score = (self.folder / 'validation-012000.json').read_bytes()
        (self.folder / 'validation-000500.json').write_bytes(last_score)
        with self.assertRaisesRegex(ValueError, 'earliest minimum-validation'): self.verify()
        self.best['step'] = 500
        self.assertEqual(self.verify()['checkpoint_step'], 500)


if __name__ == '__main__': unittest.main()
