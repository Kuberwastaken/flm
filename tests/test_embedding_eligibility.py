import unittest
import torch
from torch.nn import functional as F
from flm.embedding_eligibility import EmbeddingEligibility, trace_accounting


class EmbeddingEligibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(93)

    def test_occurrences_match_dense_local_traces_with_repeated_tokens(self):
        b, n, v, d, t = 2, 3, 7, 4, 6
        a = torch.randn(n, d, dtype=torch.float64)
        trace = EmbeddingEligibility(a, v, b, t)
        eh = torch.zeros(b, n, v, d, dtype=a.dtype)
        es = torch.zeros_like(eh)
        tokens = torch.tensor([[2, 2, 4, 2, 5, 4], [6, 3, 3, 6, 3, 1]])
        beta = torch.rand(n, dtype=a.dtype)
        for step in range(t):
            j, direct = torch.randn(2, b, n, dtype=a.dtype)
            eh = j[:, :, None, None] * eh
            for row in range(b):
                eh[row, :, tokens[row, step], :] += direct[row, :, None] * a
            es = (1 - beta)[None, :, None, None] * es + beta[None, :, None, None] * eh
            trace.step(tokens[:, step], j, direct, beta)
            lh, ls = torch.randn(2, b, n, dtype=a.dtype)
            expected = (lh[:, :, None, None] * eh + ls[:, :, None, None] * es).sum((0, 1))
            torch.testing.assert_close(trace.gradient((lh, ls), a), expected, atol=1e-12, rtol=1e-12)
        self.assertFalse(trace.fast.requires_grad)
        self.assertEqual(trace.storage_bytes()['fast'] + trace.storage_bytes()['slow'], 2*b*n*t*8)

    def _tied_gradient(self, coupled):
        b, n, v, d, t = 2, 3, 7, 4, 5
        a = torch.randn(n, d, dtype=torch.float64) * .3
        embedding = torch.randn(v, d, dtype=a.dtype, requires_grad=True)
        readout = torch.randn(2*n, d, dtype=a.dtype) * .4
        weight = torch.diag(torch.tensor([.3, -.2, .1], dtype=a.dtype))
        if coupled: weight[0, 1] = .8; weight[2, 0] = -.7
        alpha = torch.tensor([.3, .6, .8], dtype=a.dtype)
        beta = torch.tensor([.03, .07, .09], dtype=a.dtype)
        tokens = torch.tensor([[2, 2, 4, 3, 2], [5, 3, 5, 5, 6]])
        targets = torch.tensor([[2, 4, 3, 2, 6], [3, 5, 5, 6, 1]])
        h = torch.zeros(b, n, dtype=a.dtype); s = torch.zeros_like(h)
        losses = []; states = []; derivatives = []
        for step in range(t):
            z = torch.tanh(embedding[tokens[:, step]] @ a.T + h @ weight.T)
            direct = alpha * (1 - z.square())
            derivatives.append((1-alpha+direct*weight.diag(), direct))
            h = (1-alpha)*h + alpha*z; s = (1-beta)*s + beta*h
            states.append((h.detach(), s.detach()))
            logits = (torch.cat((h, s), -1) @ readout) @ embedding.T
            losses.append(F.cross_entropy(logits, targets[:, step]) / t)
        exact = torch.autograd.grad(sum(losses), embedding)[0]
        trace = EmbeddingEligibility(a, v, b, t)
        local = torch.zeros_like(embedding); direct_only = torch.zeros_like(embedding)
        for step, ((h, s), (j, direct)) in enumerate(zip(states, derivatives)):
            trace.step(tokens[:, step], j, direct, beta)
            h.requires_grad_(True); s.requires_grad_(True)
            loss = F.cross_entropy((torch.cat((h, s), -1) @ readout) @ embedding.T, targets[:, step]) / t
            lh, ls, output = torch.autograd.grad(loss, (h, s, embedding))
            local += output + trace.gradient((lh, ls), a)
            direct_only += output
        return exact, local, direct_only

    def test_tied_output_and_input_gradients_sum_to_bptt_without_cross_neuron_paths(self):
        exact, local, direct_only = self._tied_gradient(False)
        torch.testing.assert_close(local, exact, atol=1e-12, rtol=1e-12)
        self.assertGreater(float((exact-direct_only).abs().max()), 1e-3)

    def test_coupled_network_retains_approximation_error(self):
        exact, local, _ = self._tied_gradient(True)
        self.assertGreater(float((exact-local).abs().max()), 1e-4)

    def test_window_contract_and_accounting(self):
        a = torch.ones(3, 4, dtype=torch.float64)
        trace = EmbeddingEligibility(a, 7, 2, 1)
        x = torch.ones(2, 3, dtype=a.dtype)
        with self.assertRaisesRegex(ValueError, 'outside'): trace.step(torch.tensor([0, 7]), x, x, a[:, 0])
        self.assertEqual(trace.used, 0)
        trace.step(torch.tensor([0, 6]), x, x, a[:, 0])
        with self.assertRaisesRegex(ValueError, 'capacity'): trace.step(torch.tensor([0, 6]), x, x, a[:, 0])
        with self.assertRaisesRegex(ValueError, 'changed'): trace.gradient((x, x), a + .01)
        sizes = trace_accounting()
        self.assertEqual(sizes['teacher_forced_state_parameters'], 570723)
        self.assertEqual(sizes['dense_embedding_trace_bytes'], 48 * 1024**3)
        self.assertEqual(sizes['occurrence_trace_bytes'], 12 * 1024**2)


if __name__ == '__main__': unittest.main()
