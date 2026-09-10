"""Synthetic fixtures for the later instruction-transfer runtime, not SCAN scores."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from flm.baselines import BaselineConfig, GRU, Transformer
from flm.model import Config, FLM
from flm.scan_runtime import batch_loss, greedy_instructions, instruction_batch, masked_nll, score_outputs
from flm.scan_task import action_ids, encode_example, score_actions
from test_model import fixture
from test_scan import ByteLexicon


RECORDS = [dict(command='walk', actions=['I_WALK']),
           dict(command='look twice', actions=['I_LOOK', 'I_LOOK']),
           dict(command='jump left after walk twice', actions=['I_WALK', 'I_WALK', 'I_TURN_LEFT', 'I_JUMP'])]


def models():
    torch.manual_seed(31)
    return [FLM(fixture(), Config(neurons=16, pools=8, embedding=8, vocabulary=258,
                                  tied_readout=True)).double(),
            GRU(BaselineConfig('gru', embedding=8, hidden=12, vocabulary=258, tied_readout=True)).double(),
            Transformer(BaselineConfig('transformer', embedding=8, width=12, heads=3,
                layers=1, window=96, vocabulary=258, tied_readout=True)).double()]


class ToySequenceModel(nn.Module):
    """Scripted logits solely for runtime tests; never a model or experiment export."""
    def __init__(self, nonfinite=False):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocabulary=258, window=96)
        self.starts = []; self.nonfinite = nonfinite

    def forward(self, tokens, state=None):
        if state is None:
            self.starts.append(tokens.clone())
            codes = tokens[:, 5].tolist()  # First command character in the independent byte-code fixture.
            step = 0
        else:
            codes, step = state
            assert tokens.shape[1] == 1
        codec = action_ids(ByteLexicon())
        sequences = {ord('j')+2: [codec['I_JUMP'], 1], ord('r')+2: [1],
                     ord('l')+2: [codec['I_LOOK'], 0, 1], ord('w')+2: [codec['I_WALK']] * 20}
        logits = torch.zeros(len(codes), tokens.shape[1], 258)
        for row, code in enumerate(codes):
            sequence = sequences[code]
            chosen = sequence[min(step, len(sequence)-1)]
            logits[row, -1, chosen] = float('nan') if self.nonfinite else 10.
        return logits, (codes, step + 1)


class ScanRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_padding_preserves_repeats_and_only_complete_action_eos_targets(self):
        records = [RECORDS[0], RECORDS[2], RECORDS[0]]
        batch = instruction_batch(records, ByteLexicon())
        self.assertEqual(batch['examples'], 3)
        self.assertEqual(batch['supervised_tokens'], 2 + 5 + 2)
        self.assertTrue(torch.equal(batch['input'][0], batch['input'][2]))
        for index, record in enumerate(records):
            example = encode_example(record, ByteLexicon())
            length = batch['lengths'][index]
            self.assertEqual(batch['input'][index, :length].tolist(), example['input'])
            self.assertEqual(batch['loss_mask'][index, :length].tolist(), example['loss_mask'])
            self.assertFalse(batch['loss_mask'][index, length:].any())
            self.assertEqual(batch['target'][index, length-1].item(), 1)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            instruction_batch(records, ByteLexicon(), context_limit=10)

    def test_masked_objective_uses_supervised_token_weighting_and_no_other_logit_gradients(self):
        batch = instruction_batch(RECORDS, ByteLexicon())
        torch.manual_seed(3)
        logits = torch.randn(*batch['input'].shape, 258, dtype=torch.float64, requires_grad=True)
        actual = masked_nll(logits, batch['target'], batch['loss_mask'])
        expected = sum(F.cross_entropy(logits[i][mask], batch['target'][i][mask], reduction='sum')
                       for i, mask in enumerate(batch['loss_mask'])) / batch['supervised_tokens']
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[~batch['loss_mask']]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[batch['loss_mask']]).item(), 0)
        with self.assertRaises(ValueError):
            masked_nll(logits, batch['target'], torch.zeros_like(batch['loss_mask']))

    def test_padded_batch_loss_and_parameter_gradients_match_independent_examples_for_all_architectures(self):
        batch = instruction_batch(RECORDS, ByteLexicon())
        for model in models():
            with self.subTest(model=type(model).__name__):
                loss = batch_loss(model, batch); loss.backward()
                gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                individual = []
                for record in RECORDS:
                    single = instruction_batch([record], ByteLexicon())
                    individual.append(batch_loss(model, single) * single['supervised_tokens'])
                independent = sum(individual) / batch['supervised_tokens']
                independent.backward()
                torch.testing.assert_close(loss, independent, atol=1e-11, rtol=1e-11)
                for name, p in model.named_parameters():
                    torch.testing.assert_close(p.grad, gradients[name], atol=1e-10, rtol=1e-9)

    def test_unscored_instruction_positions_still_receive_temporal_credit(self):
        batch = instruction_batch(RECORDS[:1], ByteLexicon())
        for model in models():
            activations = []
            def retain(module, inputs, output):
                output.retain_grad(); activations.append(output)
            handle = model.embedding.register_forward_hook(retain)
            try:
                batch_loss(model, batch).backward()
            finally:
                handle.remove()
            self.assertEqual(len(activations), 1)
            prefix = ~batch['loss_mask']
            self.assertGreater(float(activations[0].grad[prefix].abs().sum()), 0., type(model).__name__)

    def test_generation_keeps_caller_order_invalid_ids_independent_eos_and_caps(self):
        model = ToySequenceModel()
        commands = ['walk', 'jump', 'run', 'look', 'jump']
        with patch('flm.scan.interpret', side_effect=AssertionError('Grammar oracle must not generate outputs')):
            results = greedy_instructions(model, ByteLexicon(), commands, maximum_tokens=4, batch_size=3)
        self.assertEqual([r['command'] for r in results], commands)
        codec = action_ids(ByteLexicon())
        self.assertEqual([r['tokens'] for r in results], [
            [codec['I_WALK']] * 4, [codec['I_JUMP'], 1], [1], [codec['I_LOOK'], 0, 1], [codec['I_JUMP'], 1]])
        self.assertEqual([r['stop_reason'] for r in results], ['length', 'eos', 'eos', 'eos', 'eos'])
        self.assertEqual(results[3]['actions'], ['I_LOOK', 'INVALID_TOKEN:0'])
        self.assertFalse(score_actions(['I_WALK'], results[0]['actions'], terminated=results[0]['terminated'])['exact_match'])
        self.assertFalse(score_actions(['I_RUN'], results[2]['actions'], terminated=results[2]['terminated'])['exact_match'])
        self.assertTrue(score_actions(['I_JUMP'], results[1]['actions'], terminated=results[1]['terminated'])['exact_match'])
        starts = [row.tolist() for call in model.starts for row in call]
        self.assertCountEqual(starts, [r['prefix_tokens'] for r in results])

    def test_batched_incremental_generation_matches_independent_whole_prefix_for_all_architectures(self):
        for model in models():
            state = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
            results = greedy_instructions(model, ByteLexicon(), ['jump', 'run', 'look', 'jump'], maximum_tokens=6)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()))
            with torch.inference_mode():
                for result in results:
                    expected = []
                    for _ in range(6):
                        tokens = torch.tensor([result['prefix_tokens'] + expected])
                        logits, _ = model(tokens, None)
                        chosen = int(logits[0, -1].argmax())
                        expected.append(chosen)
                        if chosen == 1: break
                    self.assertEqual(result['tokens'], expected, type(model).__name__)

    def test_invalid_generation_limits_and_nonfinite_outputs_fail_without_changing_training_mode(self):
        model = ToySequenceModel()
        for limits in ({'maximum_tokens': True}, {'batch_size': 0}, {'context_limit': 4}):
            with self.assertRaises(ValueError):
                greedy_instructions(model, ByteLexicon(), ['jump'], **limits)
        model = ToySequenceModel(nonfinite=True)
        with self.assertRaises(FloatingPointError):
            greedy_instructions(model, ByteLexicon(), ['jump'], maximum_tokens=4)
        self.assertTrue(model.training)

    def test_complete_supplied_cohort_counts_every_failure_with_exact_denominators(self):
        records = [dict(command=c, actions=['I_' + c.upper()]) for c in ('walk', 'jump', 'run', 'look')]
        outputs = greedy_instructions(ToySequenceModel(), ByteLexicon(), [r['command'] for r in records], maximum_tokens=4)
        report = score_outputs(records, outputs, ByteLexicon(), maximum_tokens=4)
        total = report['aggregate']
        self.assertEqual(total['examples'], 4)
        self.assertEqual(total['exact_match_rate'], .25)
        self.assertEqual(total['total_edit_distance'], 5)
        self.assertEqual(total['total_reference_actions'], 4)
        self.assertEqual(total['edits_per_reference_action'], 1.25)
        self.assertEqual(total['invalid_token_output_rate'], .25)
        self.assertEqual(total['invalid_or_empty_action_output_rate'], .5)
        self.assertEqual(total['eos_termination_rate'], .75)
        self.assertEqual(total['cap_exhaustion_rate'], .25)
        self.assertEqual(report['by_reference_length'], [dict(reference_length=1, **total)])

    def test_altered_decoding_early_truncation_or_changed_command_cannot_enter_scores(self):
        records = [dict(command='look', actions=['I_LOOK']), dict(command='walk', actions=['I_WALK'])]
        outputs = greedy_instructions(ToySequenceModel(), ByteLexicon(), [r['command'] for r in records], maximum_tokens=4)
        for change in ('decoded', 'shortened', 'command', 'cap', 'missing'):
            altered = copy.deepcopy(outputs)
            if change == 'decoded': altered[0]['actions'] = ['I_LOOK']
            if change == 'shortened':
                altered[1]['tokens'].pop(); altered[1]['actions'].pop()
            if change == 'command': altered[0]['command'] = 'jump'
            if change == 'cap': altered[0]['maximum_tokens'] = 5
            if change == 'missing': altered.pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_outputs(records, altered, ByteLexicon(), maximum_tokens=4)


if __name__ == '__main__':
    unittest.main()
