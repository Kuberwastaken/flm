import copy
import unittest
import numpy as np
import torch
from flm.local_learning import ChoiceFLM, CORE_PARAMETERS, edge_direct, learning_step, trace_bytes
from test_model import fixture


class LocalLearningTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(17)

    def test_normalized_edge_direct_derivative_matches_finite_differences(self):
        model = ChoiceFLM(fixture()).double(); n = model.config.neurons
        previous = torch.randn(3, n, dtype=torch.float64); weight = model.constants()[0]
        actual = edge_direct(weight, model.row, model.col, previous, previous @ weight.T)
        epsilon = 1e-6
        with torch.no_grad():
            for e in range(len(model.row)):
                model.edge_log_gain[e] += epsilon; plus = previous @ model.constants()[0].T
                model.edge_log_gain[e] -= 2 * epsilon; minus = previous @ model.constants()[0].T
                model.edge_log_gain[e] += epsilon
                torch.testing.assert_close(actual[:, e], (plus - minus)[:, model.row[e]] / (2 * epsilon), atol=2e-9, rtol=1e-7)

    def test_forward_traces_match_full_gradients_without_cross_neuron_temporal_paths(self):
        graph = fixture(); n = len(graph['body_ids'])
        graph.update(row=np.arange(n), col=np.arange(n), weight=np.ones(n, dtype=np.float32))
        model = ChoiceFLM(graph).double(); sensory = torch.randn(3, 19, 4, dtype=torch.float64)
        targets = torch.tensor([0, 1, 0])
        loss = torch.nn.functional.cross_entropy(model(sensory)[0], targets); loss.backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True); state, traces = model.forward_eligibility(sensory)
        self.assertTrue(all(not t.requires_grad for pair in traces.values() for t in pair))
        state = tuple(value.requires_grad_(True) for value in state)
        torch.nn.functional.cross_entropy(model.decision(state), targets).backward()
        model.assign_core_gradients(tuple(value.grad for value in state), traces)
        for name, p in model.named_parameters(): torch.testing.assert_close(p.grad, expected[name], atol=1e-10, rtol=1e-9, msg=name)

    def test_state_parity_trace_size_and_history_removal(self):
        model = ChoiceFLM(fixture()).double(); sensory = torch.randn(2, 13, 4, dtype=torch.float64)
        _, exact = model(sensory); local, traces = model.forward_eligibility(sensory)
        instant, instant_traces = model.forward_eligibility(sensory, history=False)
        for a, b, c in zip(exact, local, instant):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12); torch.testing.assert_close(a, c, atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.allclose(traces['input.weight'][0], instant_traces['input.weight'][0]))
        _, longer = model.forward_eligibility(torch.cat([sensory, sensory], dim=1))
        self.assertEqual(trace_bytes(traces), trace_bytes(longer))
        self.assertTrue(all(not t.requires_grad for pair in traces.values() for t in pair))

    def test_reservoir_freezes_core_and_reward_updates_are_reproducible(self):
        model = ChoiceFLM(fixture()); before = {name: p.detach().clone() for name, p in model.named_parameters()}
        sensory = torch.randn(4, 9, 4); target = torch.tensor([0, 1, 0, 1])
        learning_step(model, sensory, target, torch.optim.SGD(model.parameters(), lr=.03), 'reservoir')
        for name, p in model.named_parameters():
            if name in CORE_PARAMETERS: torch.testing.assert_close(p, before[name], atol=0, rtol=0)
        self.assertFalse(torch.equal(model.readout.weight, before['readout.weight']))
        a, b = copy.deepcopy(model), copy.deepcopy(model)
        for instance in (a, b):
            learning_step(instance, sensory, target, torch.optim.SGD(instance.parameters(), lr=.03), 'reward', torch.Generator().manual_seed(19))
        for p, q in zip(a.parameters(), b.parameters()): torch.testing.assert_close(p, q, atol=0, rtol=0)
        self.assertFalse(torch.equal(a.input.weight, model.input.weight))


if __name__ == '__main__': unittest.main()
