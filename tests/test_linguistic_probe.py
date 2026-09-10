import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from flm.language_train import construct
from flm.linguistic_probe import sentence_nll, aggregate
from test_model import fixture


class LinguisticProbeTests(unittest.TestCase):
    def test_right_padding_matches_separate_sentence_scores_for_all_models(self):
        torch.set_num_threads(1)
        lexicon = SimpleNamespace(encode=lambda text: [ord(c) % 20 + 2 for c in text])
        sentences = ['a', 'short text', 'a much longer sentence in this fixture', 'x']
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.npz'; np.savez(graph, **fixture())
            for variant in ('flm', 'gru', 'transformer'):
                model = construct(variant, graph, 30, 42).double()
                single = sentence_nll(model, sentences, lexicon, batch_size=1)
                batched = sentence_nll(model, sentences, lexicon, batch_size=4)
                np.testing.assert_allclose(single, batched, atol=1e-10, rtol=0)
                with self.assertRaises(ValueError): sentence_nll(model, [''], lexicon)

    def test_macro_paradigm_accuracy_does_not_overweight_larger_groups(self):
        records = [dict(paradigm='a', category='one', good_nll=1., bad_nll=2., correct=True)]
        records += [dict(paradigm='b', category='one', good_nll=2., bad_nll=2., correct=False) for _ in range(3)]
        report = aggregate(records)
        self.assertEqual(report['macro_paradigm_accuracy'], .5)
        self.assertEqual(next(r for r in report['groups'] if r['kind'] == 'category')['ties'], 3)


if __name__ == '__main__': unittest.main()
