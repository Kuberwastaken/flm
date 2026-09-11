"""Transfer-interface invariants on synthetic graphs, without food-task fitting."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from flm.model import FLM, Config
from flm.food_core import FoodCore


def tiny():
    graph = dict(body_ids=np.arange(4), row=np.array([0, 1, 2, 3, 0]), col=np.array([1, 2, 3, 0, 3]),
        weight=np.array([.3, -.4, .6, .2, -.1]), pool=np.array([0, 1, 0, 1]))
    return FLM(graph, Config(neurons=4, pools=2, embedding=3, vocabulary=9, tied_readout=True)).double()


class FoodCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(61)

    def test_native_lexical_forward_is_preserved(self):
        model = tiny(); bridge = FoodCore(model, 711)
        tokens = torch.tensor([[0, 3, 2, 4, 1], [2, 2, 5, 6, 7]])
        logits, state = model(tokens)
        features, actual = bridge.encode_projected(model.embedding(tokens))
        reconstructed = F.linear(model.readout(features), model.embedding.weight, model.output_bias)
        torch.testing.assert_close(reconstructed, logits, atol=0, rtol=0)
        for a, b in zip(actual, state): torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_adapters_match_across_different_cores_without_mutating_source_or_rng(self):
        model = tiny(); before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone(); first = FoodCore(model, 711)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        with torch.no_grad(): model.alpha_logit.add_(.7)
        second = FoodCore(model, 711)
        for name in ('sensor', 'action'):
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                self.assertTrue(torch.equal(a, b))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any(p.requires_grad for p in first.core.parameters()))
        self.assertTrue(all(torch.equal(first.core.state_dict()[k], v) for k, v in before.items()))

    def test_state_carries_across_chunks_and_resets_explicitly(self):
        bridge = FoodCore(tiny(), 711); x = torch.rand(2, 7, 6, dtype=torch.float64)
        whole, whole_state, _ = bridge(x)
        a, state, _ = bridge(x[:, :3]); b, state, _ = bridge(x[:, 3:], state)
        torch.testing.assert_close(torch.cat((a, b), 1), whole, atol=1e-14, rtol=1e-14)
        for u, v in zip(state, whole_state): torch.testing.assert_close(u, v, atol=1e-14, rtol=1e-14)
        reset, _, _ = bridge(x[:, 3:])
        self.assertGreater(float((b-reset).abs().max().detach()), 1e-5)

    def test_frozen_core_still_transmits_adapter_gradients_through_time(self):
        bridge = FoodCore(tiny(), 711); x = torch.rand(1, 5, 6, dtype=torch.float64)
        initial = {k: v.clone() for k, v in bridge.core.state_dict().items()}
        loss = bridge(x)[0][0, -1, 0]; loss.backward()
        analytic = float(bridge.sensor.weight.grad[0, 0]); epsilon = 1e-6
        with torch.no_grad():
            original = float(bridge.sensor.weight[0, 0])
            bridge.sensor.weight[0, 0] = original+epsilon; plus = float(bridge(x)[0][0, -1, 0])
            bridge.sensor.weight[0, 0] = original-epsilon; minus = float(bridge(x)[0][0, -1, 0])
            bridge.sensor.weight[0, 0] = original
        self.assertAlmostEqual(analytic, (plus-minus)/(2*epsilon), places=7)
        self.assertGreater(abs(analytic), 1e-6)
        self.assertTrue(all(p.grad is None for p in bridge.core.parameters()))
        before = bridge.sensor.weight.detach().clone()
        # A single synthetic optimizer step verifies the trainability boundary.
        torch.optim.SGD((p for p in bridge.parameters() if p.requires_grad), lr=.01).step()
        self.assertFalse(torch.equal(before, bridge.sensor.weight))
        self.assertTrue(all(torch.equal(bridge.core.state_dict()[k], v) for k, v in initial.items()))

    def test_invalid_inputs_and_state_fail_before_inference(self):
        bridge = FoodCore(tiny(), 711)
        for x in (torch.zeros(1, 0, 6, dtype=torch.float64), torch.zeros(1, 2, 5, dtype=torch.float64),
                  torch.full((1, 2, 6), float('nan'), dtype=torch.float64), torch.ones(1, 2, 6),
                  torch.full((1, 2, 6), 1.1, dtype=torch.float64)):
            with self.assertRaises(ValueError): bridge(x)
        with self.assertRaises(ValueError): bridge(torch.zeros(1, 2, 6, dtype=torch.float64), (torch.zeros(2, 4),)*2)
        with self.assertRaises(ValueError): FoodCore(tiny(), True)


if __name__ == '__main__': unittest.main()
