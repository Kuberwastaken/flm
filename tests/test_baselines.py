import unittest
import torch
from flm.baselines import GRU, Transformer


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_parameter_counts_are_within_two_percent_of_flm(self):
        for model in (GRU(), Transformer()):
            self.assertLess(abs(model.parameter_card()['trainable_parameters'] / 228069 - 1), .02)

    def test_future_tokens_cannot_change_past(self):
        for model in (GRU(), Transformer()):
            model.eval(); x = torch.randint(0, 256, (2, 120)); y = x.clone(); y[:, 65:] = 40
            a, _ = model(x); b, _ = model(y)
            torch.testing.assert_close(a[:, :65], b[:, :65], atol=2e-6, rtol=2e-5)

    def test_streaming_matches_full_sequence_beyond_attention_window(self):
        for model in (GRU(), Transformer()):
            model.eval(); x = torch.randint(0, 256, (2, 220))
            with torch.no_grad():
                full, _ = model(x); pieces = []; state = None
                for start in range(0, 220, 17):
                    y, state = model(x[:, start:start + 17], state); pieces.append(y)
            torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-5)

    def test_finite_gradients_reach_recurrent_or_attention_parameters(self):
        for model in (GRU(), Transformer()):
            x = torch.randint(0, 256, (2, 24)); y, _ = model(x)
            torch.nn.functional.cross_entropy(y.flatten(0, 1), x.flatten()).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad); self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__': unittest.main()
