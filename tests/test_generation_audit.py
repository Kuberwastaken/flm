import unittest
from types import SimpleNamespace
import torch
from flm.generation_audit import generate, repetition, SETTINGS


class Scripted(torch.nn.Module):
    def __init__(self, scores): super().__init__(); self.scores = scores
    def forward(self, x, state=None):
        state = 0 if state is None else state
        return torch.tensor(self.scores[min(state, len(self.scores) - 1)]).expand(*x.shape, -1).clone(), state + 1


class GenerationAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lexicon = SimpleNamespace(pieces=[b'', b'', b'a', b'\x00', b'\xc3', b'\xa9'], encode=lambda _: [2])

    def test_generation_filters_controls_stops_on_eos_and_retains_exact_tokens(self):
        model = Scripted([[200., -100., -100., 100., 50., -100.], [-100., -100., -100., -100., -100., 50.], [-100., 50., -100., -100., -100., -100.]])
        model.train()
        result = generate(model, self.lexicon, 'fixture', 17, dict(SETTINGS, top_k=1))
        self.assertEqual(result['tokens'], [4, 5]); self.assertEqual(result['continuation'], 'é')
        self.assertEqual(result['termination'], 'eos'); self.assertFalse(result['invalid_utf8'])
        self.assertTrue(model.training)

    def test_reproducibility_reset_length_limit_and_invalid_utf8_are_explicit(self):
        model = Scripted([[-100., -100., 1., 100., 1., 1.]])
        options = dict(SETTINGS, maximum_tokens=17)
        a = generate(model, self.lexicon, 'fixture', 17, options)
        b = generate(model, self.lexicon, 'fixture', 17, options)
        self.assertEqual(a, b); self.assertEqual(a['generated_tokens'], 17)
        self.assertNotIn(3, a['tokens']); self.assertEqual(a['termination'], 'maximum_tokens')
        model = Scripted([[-100., -100., -100., -100., 50., -100.]])
        invalid = generate(model, self.lexicon, 'fixture', 17, dict(SETTINGS, maximum_tokens=1, top_k=1))
        self.assertEqual(invalid['tokens'], [4]); self.assertTrue(invalid['invalid_utf8'])
        self.assertEqual(invalid['bytes'], 1)

    def test_repetition_has_defined_short_sequence_behavior(self):
        self.assertIsNone(repetition([])['repeated_token_fourgram_fraction'])
        self.assertEqual(repetition([])['longest_identical_token_run'], 0)
        self.assertEqual(repetition([2] * 8)['longest_identical_token_run'], 8)
        self.assertAlmostEqual(repetition([2] * 8)['repeated_token_fourgram_fraction'], .8)
        self.assertEqual(repetition(list(range(9)))['repeated_token_fourgram_fraction'], 0.)


if __name__ == '__main__': unittest.main()
