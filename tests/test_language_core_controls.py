import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from flm.language_core_controls import (CONTROLS, FROZEN_DYNAMICS, CoreControlConfig,
                                        LanguageCoreControl, construct)
from flm.language_train import construct as original_construct, restore
from flm.provenance import sha256
from flm.train import Sampler, save_checkpoint
from test_model import fixture


class LanguageCoreControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.graph = Path(self.directory.name) / 'graph.npz'
        np.savez(self.graph, **fixture())
        self.tokens = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]])

    def model(self, control):
        return construct(control, self.graph, 32, 42)

    def test_every_control_has_identical_initial_named_parameters_and_rng(self):
        reference = original_construct('flm', self.graph, 32, 42)
        expected_rng = torch.get_rng_state()
        for control in CONTROLS:
            model = self.model(control)
            self.assertEqual(dict(reference.named_parameters()).keys(), dict(model.named_parameters()).keys())
            for name, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), (control, name))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        actual = self.model('full')
        self.assertTrue(torch.equal(reference(self.tokens)[0], actual(self.tokens)[0]))

    def test_memoryless_control_ignores_prefix_and_supplied_state_at_every_token(self):
        model = self.model('no_temporal_state').double()
        tokens = torch.tensor([[0, 2, 3, 7], [5, 6, 8, 7]])
        logits, state = model(tokens)
        # Different GEMM row positions can differ at floating-point roundoff.
        torch.testing.assert_close(logits[0, -1], logits[1, -1], rtol=0, atol=1e-14)
        changed_prefix = tokens.clone()
        changed_prefix[:, :-1] = tokens.flip(0)[:, :-1]
        # At the SAME matrix position, changing only the prefix is bit-identical.
        self.assertTrue(torch.equal(logits[:, -1], model(changed_prefix)[0][:, -1]))
        hostile_state = (torch.full((2, 16), float('nan'), dtype=torch.float64),
                         torch.full((2, 16), float('inf'), dtype=torch.float64))
        actual, end = model(tokens, hostile_state)
        self.assertTrue(torch.equal(logits, actual))
        for a, b in zip(state, end):
            self.assertTrue(torch.equal(a, b))
        first, partial = model(tokens[:, :2]); rest, final = model(tokens[:, 2:], partial)
        torch.testing.assert_close(logits, torch.cat((first, rest), 1), rtol=0, atol=1e-14)
        for a, b in zip(state, final):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-14)

    def test_memoryless_matches_full_model_reset_separately_for_every_token(self):
        model = self.model('no_temporal_state').double()
        oracle = self.model('full').double()
        actual = model(self.tokens)[0]
        expected = torch.cat([oracle(self.tokens[:, t:t+1])[0] for t in range(4)], 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        actual.square().mean().backward(); expected.square().mean().backward()
        for name, parameter in model.named_parameters():
            original_gradient = oracle.get_parameter(name).grad
            if name in ('edge_log_gain', 'recurrent_logit'):
                self.assertIsNone(parameter.grad)
                self.assertEqual(int(torch.count_nonzero(original_gradient)), 0)
            else:
                torch.testing.assert_close(parameter.grad, original_gradient, rtol=1e-10, atol=1e-14)

    def test_no_lateral_removes_edges_but_keeps_temporal_dependence(self):
        model = self.model('no_lateral')
        oracle = original_construct('no_recurrence', self.graph, 32, 42)
        expected = oracle(self.tokens)[0]
        self.assertTrue(torch.equal(expected, model(self.tokens)[0]))
        with torch.no_grad():
            model.base_weight.fill_(float('nan'))
            model.edge_log_gain.fill_(2)
            model.recurrent_logit.fill_(7)
        self.assertTrue(torch.equal(expected, model(self.tokens)[0]))
        state = model.initial_state(2)
        changed, _ = model(self.tokens, (state[0]+.4, state[1]-.2))
        self.assertGreater(float((expected[:, -1]-changed[:, -1]).detach().abs().max()), 1e-5)

    def test_only_memoryless_loss_has_no_path_to_earlier_input_drives(self):
        for control in CONTROLS:
            model = self.model(control)
            drives = []
            handle = model.input.register_forward_hook(lambda _m, _a, output: drives.append(output))
            try:
                logits, _ = model(self.tokens)
                gradient, = torch.autograd.grad(logits[:, -1].square().mean(), drives[0])
            finally:
                handle.remove()
            prefix_nonzero = int(torch.count_nonzero(gradient[:, :-1]))
            if control == 'no_temporal_state':
                self.assertEqual(prefix_nonzero, 0)
            else:
                self.assertGreater(prefix_nonzero, 0, control)
            self.assertGreater(int(torch.count_nonzero(gradient[:, -1])), 0)

    def test_fixed_dynamics_survive_optimizer_steps_and_corruption_is_rejected(self):
        model = self.model('fixed_dynamics')
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(self.tokens)[0].flatten(0, 1), self.tokens.roll(-1, 1).flatten()).backward()
            self.assertTrue(all(model.get_parameter(name).grad is None for name in FROZEN_DYNAMICS))
            optimizer.step()
            model.verify_frozen_dynamics()
        self.assertFalse(torch.equal(initial['input.weight'], model.input.weight))
        self.assertFalse(torch.equal(initial['embedding.weight'], model.embedding.weight))
        saved = copy.deepcopy(model.state_dict())
        fresh = self.model('fixed_dynamics')
        fresh.verify_frozen_dynamics(saved); fresh.load_state_dict(saved)
        fresh.verify_frozen_dynamics()
        for name in FROZEN_DYNAMICS:
            corrupt = copy.deepcopy(saved)
            corrupt[name].reshape(-1)[0] += .1
            with self.assertRaisesRegex(ValueError, 'differs from initialization'):
                fresh.verify_frozen_dynamics(corrupt)
        corrupt = copy.deepcopy(saved); del corrupt['alpha_logit']
        with self.assertRaisesRegex(ValueError, 'differs from initialization'):
            fresh.verify_frozen_dynamics(corrupt)
        fresh.alpha_logit.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'made trainable'):
            fresh.verify_frozen_dynamics()

    def test_parameter_card_distinguishes_freezing_unused_and_zero_gradients(self):
        for control in CONTROLS:
            model = self.model(control)
            card = model.parameter_card()
            self.assertEqual(card['allocated_parameters'], sum(p.numel() for p in model.parameters()))
            frozen = sum(model.get_parameter(n).numel() for n in FROZEN_DYNAMICS) if control == 'fixed_dynamics' else 0
            self.assertEqual(card['frozen_parameters'], frozen)
            self.assertEqual(card['trainable_parameters'], card['allocated_parameters']-frozen)
            probe = model.gradient_probe(model(self.tokens)[0].square().mean())
            disconnected = {name for name, row in probe['groups'].items() if not row['connected']}
            self.assertEqual(disconnected, {'edge_log_gain', 'recurrent_logit'}
                             if control in ('no_lateral', 'no_temporal_state') else set())
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        model = self.model('full')
        one = model.gradient_probe(model(self.tokens[:, :1])[0].square().mean())
        self.assertTrue(one['groups']['edge_log_gain']['connected'])
        self.assertEqual(one['groups']['edge_log_gain']['nonzero_entries'], 0)

    def test_old_checkpoint_loader_cannot_silently_restore_a_different_mechanism(self):
        from types import SimpleNamespace
        for control in CONTROLS:
            model = self.model(control)
            self.assertEqual(asdict(model.config)['control'], control)
            path = Path(self.directory.name) / (control+'.pt')
            optimizer = torch.optim.AdamW(model.parameters())
            sampler = Sampler([('fixture', np.arange(30) % 32)], 42, 4)
            save_checkpoint(path, model, optimizer, sampler, 0, 9., dict(seed=42,
                tokenizer_sha256='fixture', graph_sha256=sha256(self.graph)))
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                restore(path, self.graph, SimpleNamespace(vocabulary=32, sha256='fixture'))

    def test_invalid_or_ambiguous_control_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.model('memory')
        with self.assertRaisesRegex(ValueError, 'disagree'):
            LanguageCoreControl(fixture(), CoreControlConfig(neurons=16, pools=8, control='no_temporal_state'))


if __name__ == '__main__':
    unittest.main()
