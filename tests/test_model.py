import unittest

import numpy as np
import torch

from flm.graph import rewire
from flm.model import Config, FLM, encode


def fixture():
    n = 16
    rows, cols = [], []
    for j in range(n):
        for delta in (1, 3, 7):
            rows.append(j); cols.append((j + delta) % n)
    signs = np.array([1, -1] * (n // 2), dtype=np.int8)
    return dict(row=np.array(rows), col=np.array(cols), weight=signs[cols].astype(np.float32) / 3,
                source_sign=signs, body_ids=np.arange(n), pool=np.arange(n) // 2)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, backend="dense"):
        torch.manual_seed(7)
        return FLM(fixture(), Config(neurons=16, pools=8, embedding=8, backend=backend))

    def test_future_tokens_cannot_change_earlier_predictions(self):
        m = self.model().eval()
        a, _ = m(torch.tensor([[256, 97, 98, 99]]))
        b, _ = m(torch.tensor([[256, 97, 120, 121]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])

    def test_streaming_matches_whole_prefix(self):
        m = self.model().eval(); t = torch.tensor([[256, 97, 98, 99]])
        full, state = m(t)
        first, partial = m(t[:, :2]); rest, end = m(t[:, 2:], partial)
        torch.testing.assert_close(full, torch.cat((first, rest), 1))
        torch.testing.assert_close(state[0], end[0]); torch.testing.assert_close(state[1], end[1])

    def test_reset_isolates_documents(self):
        m = self.model().eval(); t = torch.tensor([[256, 97]])
        expected, _ = m(t); m(torch.tensor([[99, 100, 101]])); actual, _ = m(t)
        torch.testing.assert_close(expected, actual)

    def test_sparse_dense_forward_and_gradients_match(self):
        a = self.model("dense"); b = self.model("sparse"); b.load_state_dict(a.state_dict())
        tokens = torch.tensor([[256, 97, 98], [256, 100, 99]])
        x, _ = a(tokens); y, _ = b(tokens)
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        x.square().mean().backward(); y.square().mean().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)

    def test_edge_gradient_matches_finite_difference(self):
        m = self.model().double(); tokens = torch.tensor([[256, 97, 98]])
        m(tokens)[0].square().mean().backward()
        analytic = m.edge_log_gain.grad[0].item(); epsilon = 1e-5
        with torch.no_grad():
            m.edge_log_gain[0] += epsilon; plus = m(tokens)[0].square().mean().item()
            m.edge_log_gain[0] -= 2 * epsilon; minus = m(tokens)[0].square().mean().item()
            m.edge_log_gain[0] += epsilon
        self.assertAlmostEqual(analytic, (plus - minus) / (2 * epsilon), delta=1e-7)

    def test_signs_and_normalization_survive_training_parameters(self):
        m = self.model()
        with torch.no_grad(): m.edge_log_gain.normal_(0, 4)
        w = m.constants()[0]
        self.assertTrue(torch.equal(w[m.row, m.col].sign(), m.base_weight.sign()))
        torch.testing.assert_close(w.abs().sum(1), torch.ones(16))

    def test_rewire_preserves_degrees_and_changes_topology(self):
        g = fixture(); row, col = rewire(g["row"], g["col"], g["source_sign"], 3)
        self.assertTrue(np.array_equal(np.bincount(col), np.bincount(g["col"])))
        self.assertTrue(np.array_equal(row, g["row"]))
        self.assertTrue(np.array_equal(g["source_sign"][col], g["source_sign"][g["col"]]))
        self.assertEqual(len(set(zip(row, col))), len(row))
        self.assertNotEqual(set(zip(row, col)), set(zip(g["row"], g["col"])))

    def test_byte_tokenizer_roundtrip(self):
        text = "fly café 🪰"
        self.assertEqual(bytes(encode(text)).decode(), text)


if __name__ == "__main__":
    unittest.main()
