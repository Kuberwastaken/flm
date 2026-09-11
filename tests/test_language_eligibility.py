import copy
import io
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from flm.language_eligibility import language_window_gradients
from flm.model import Config, FLM


def fixture(coupled=True, tied=True):
    n = 4
    row, col = np.indices((n, n)).reshape(2, -1) if coupled else (np.arange(n), np.arange(n))
    graph = dict(body_ids=np.arange(n), row=row, col=col,
                 weight=np.asarray([(-1 if j % 2 else 1) * (1 + i + j) for i, j in zip(row, col)], dtype=np.float32),
                 pool=np.array([0, 0, 1, 2]))
    return FLM(graph, Config(neurons=n, pools=3, embedding=3, vocabulary=9,
                            backend='dense', tied_readout=tied)).double()


def reference(model, tokens, targets, mask, *, local=False, history=True, state=None):
    """Independent autograd oracle: alter temporal derivative, not forward value."""
    if not local:
        logits, state = model(tokens, state)
    else:
        h, s = model.initial_state(len(tokens)) if state is None else tuple(x.detach() for x in state)
        w, alpha, beta, gain = model.constants()
        logits = []
        for t in range(tokens.shape[1]):
            if not history:
                h, s = h.detach(), s.detach()
            # Recurrent weights retain their full instantaneous derivative.
            # The previous fast state retains only its self-neuron derivative.
            incoming = F.linear(h.detach(), w) + (h - h.detach()) * w.diag().detach()
            z = torch.tanh(model.input(model.embedding(tokens[:, t])) + gain * incoming)
            h = (1 - alpha) * h + alpha * z
            s = (1 - beta) * s + beta * h
            encoded = model.readout(model.norm(torch.cat((model.pool(h), model.pool(s)), -1)))
            logits.append(F.linear(encoded, model.embedding.weight, model.output_bias)
                          if model.config.tied_readout else encoded)
        logits, state = torch.stack(logits, 1), (h, s)
    losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape_as(mask)
    loss = losses[mask].mean()
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return loss.detach(), logits.detach(), state, dict(zip(dict(model.named_parameters()), gradients))


class LanguageEligibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(84)
        self.tokens = torch.tensor([[2, 2, 5, 3, 2, 4], [6, 3, 6, 3, 7, 6]])
        self.targets = torch.tensor([[2, 5, 3, 2, 4, 1], [3, 6, 3, 7, 6, 1]])
        self.mask = torch.tensor([[False, False, True, True, False, True],
                                  [False, True, False, True, True, True]])

    def compare(self, actual, expected):
        loss, logits, state, gradients = expected
        self.assertAlmostEqual(actual.loss, float(loss), places=13)
        torch.testing.assert_close(actual.logits, logits, atol=1e-13, rtol=1e-12)
        for first, second in zip(actual.state, state):
            torch.testing.assert_close(first, second, atol=1e-13, rtol=1e-12)
        self.assertEqual(set(actual.gradients), set(gradients))
        for name, value in actual.gradients.items():
            torch.testing.assert_close(value, gradients[name], atol=2e-12, rtol=2e-10, msg=name)
            self.assertFalse(value.requires_grad)

    def test_all_self_only_parameters_match_masked_bptt_tied_and_untied(self):
        for tied in (True, False):
            with self.subTest(tied=tied):
                model = fixture(coupled=False, tied=tied)
                state = tuple(torch.randn(2, 4, dtype=torch.float64) * .2 for _ in range(2))
                actual = language_window_gradients(model, self.tokens, self.targets, self.mask, state=state)
                self.compare(actual, reference(model, self.tokens, self.targets, self.mask, state=state))
                self.assertEqual(actual.scored_tokens, int(self.mask.sum()))

    def test_coupled_gradients_match_independent_diagonal_autograd_and_differ_from_bptt(self):
        model = fixture()
        actual = language_window_gradients(model, self.tokens, self.targets, self.mask)
        self.compare(actual, reference(model, self.tokens, self.targets, self.mask, local=True))
        exact = reference(model, self.tokens, self.targets, self.mask)
        torch.testing.assert_close(actual.logits, exact[1], atol=1e-13, rtol=1e-12)
        self.assertGreater(float((actual.gradients['embedding.weight'] - exact[3]['embedding.weight']).abs().max()), 1e-5)
        self.assertGreater(float((actual.gradients['edge_log_gain'] - exact[3]['edge_log_gain']).abs().max()), 1e-6)

    def test_history_removal_preserves_states_but_changes_temporal_credit(self):
        model = fixture()
        actual = language_window_gradients(model, self.tokens, self.targets, self.mask, history=False)
        self.compare(actual, reference(model, self.tokens, self.targets, self.mask, local=True, history=False))
        traced = language_window_gradients(model, self.tokens, self.targets, self.mask)
        torch.testing.assert_close(actual.logits, traced.logits, atol=0, rtol=0)
        self.assertFalse(torch.allclose(actual.gradients['input.weight'], traced.gradients['input.weight']))
        self.assertLess(actual.persistent_trace_bytes, traced.persistent_trace_bytes)

    def test_unscored_prefix_still_contributes_input_credit(self):
        model = fixture(coupled=False, tied=False)
        tokens = torch.tensor([[8, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 1]])
        mask = torch.tensor([[False, False, False, True]])
        traced = language_window_gradients(model, tokens, targets, mask)
        instant = language_window_gradients(model, tokens, targets, mask, history=False)
        self.assertGreater(float(traced.gradients['embedding.weight'][8].abs().sum()), 1e-7)
        self.assertEqual(float(instant.gradients['embedding.weight'][8].abs().sum()), 0)
        self.compare(traced, reference(model, tokens, targets, mask))

    def test_normalization_floor_and_gain_clamps_match_instantaneous_autograd(self):
        model = fixture()
        with torch.no_grad():
            model.base_weight[:4].mul_(1e-12)
            model.edge_log_gain.copy_(torch.linspace(-4, 4, len(model.row), dtype=torch.float64))
        state = tuple(torch.randn(2, 4, dtype=torch.float64) for _ in range(2))
        tokens, targets = self.tokens[:, :1], self.targets[:, :1]
        mask = torch.ones_like(tokens, dtype=torch.bool)
        result = language_window_gradients(model, tokens, targets, mask, state=state)
        self.compare(result, reference(model, tokens, targets, mask, state=state))
        outside = (model.edge_log_gain < -3) | (model.edge_log_gain > 3)
        self.assertTrue((result.gradients['edge_log_gain'][outside] == 0).all())

    def test_readonly_call_and_optimizer_checkpoint_resume(self):
        model = fixture()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = language_window_gradients(model, self.tokens, self.targets, self.mask)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
            self.assertTrue((p.grad == 1).all())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)

        def update(instance, opt, gradients):
            opt.zero_grad(set_to_none=True)
            for name, p in instance.named_parameters():
                p.grad = gradients[name].clone()
            torch.nn.utils.clip_grad_norm_(instance.parameters(), 1.)
            opt.step()
            with torch.no_grad():
                instance.edge_log_gain.clamp_(-3, 3)

        update(model, optimizer, result.gradients)
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = copy.deepcopy(model)
        restored.load_state_dict(saved['model'])
        resumed = torch.optim.AdamW(restored.parameters(), lr=.001)
        resumed.load_state_dict(saved['optimizer'])
        for instance, opt in ((model, optimizer), (restored, resumed)):
            # Every call starts fresh eligibility after the optimizer changed A.
            next_result = language_window_gradients(instance, self.tokens, self.targets, self.mask)
            update(instance, opt, next_result.gradients)
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        for p in model.parameters():
            self.assertEqual(int(optimizer.state[p]['step']), 2)

    def test_invalid_or_unsupported_windows_are_rejected_without_grad_mutation(self):
        model = fixture()
        with self.assertRaisesRegex(ValueError, 'at least one'):
            language_window_gradients(model, self.tokens, self.targets, torch.zeros_like(self.mask))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            language_window_gradients(model, self.tokens, self.targets, self.mask.float())
        with self.assertRaisesRegex(ValueError, 'outside'):
            language_window_gradients(model, self.tokens + 9, self.targets, self.mask)
        with self.assertRaisesRegex(ValueError, 'states'):
            language_window_gradients(model, self.tokens, self.targets, self.mask, state=(torch.zeros(2, 3),) * 2)
        model.config.variant = 'no_slow'
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            language_window_gradients(model, self.tokens, self.targets, self.mask)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
