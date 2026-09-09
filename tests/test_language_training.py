import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from flm.language_train import construct, evaluate, restore
from flm.provenance import sha256
from flm.train import Sampler, save_checkpoint
from test_model import fixture


class Uniform(torch.nn.Module):
    def forward(self, x, state=None):
        return torch.zeros(*x.shape, 5), None


class LanguageTrainingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_evaluation_scores_text_tokens_and_exact_utf8_byte_lengths(self):
        lexicon = SimpleNamespace(vocabulary=5, lengths=np.array([0, 0, 1, 2, 4]), sha256='fixture')
        docs = [('one', np.array([0, 2, 3, 4, 1])), ('two', np.array([0, 4, 1]))]
        model = Uniform(); model.train()
        result = evaluate(model, docs, lexicon, chunk_size=2)
        self.assertTrue(model.training)
        self.assertEqual(result['tokens'], 4); self.assertEqual(result['bytes'], 11)
        self.assertAlmostEqual(result['bits_per_byte'], 4 * math.log2(5) / 11, places=6)
        self.assertAlmostEqual(result['token_perplexity'], 5, places=6)
        self.assertEqual([x['bytes'] for x in result['documents']], [7, 4])
        subset = evaluate(model, docs, lexicon, token_limit=3, chunk_size=1)
        self.assertEqual(subset['tokens'], 3); self.assertEqual(subset['bytes'], 7)
        with self.assertRaises(ValueError): evaluate(model, [], lexicon)
        with self.assertRaises(ValueError): evaluate(model, [('empty', np.array([0, 1]))], lexicon)

    def test_checkpoint_resume_preserves_next_optimizer_update_and_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.npz'; np.savez(graph, **fixture())
            lexicon = SimpleNamespace(vocabulary=270, sha256='fixture-tokenizer')
            for variant in ('flm', 'gru', 'transformer'):
                model = construct(variant, graph, 270, 42)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
                sampler = Sampler([('one', np.arange(150) % 270)], 42, 24)
                x, y = sampler.sample(2, 'cpu')
                loss = torch.nn.functional.cross_entropy(model(x)[0].flatten(0, 1), y.flatten())
                loss.backward(); optimizer.step()
                path = Path(directory) / f'{variant}.pt'
                save_checkpoint(path, model, optimizer, sampler, 1, 3., dict(seed=42,
                    tokenizer_sha256=lexicon.sha256, graph_sha256=sha256(graph)))
                resumed, saved = restore(path, graph, lexicon)
                self.assertEqual(saved['_file_sha256'], sha256(path))
                second = Sampler([('one', np.arange(150) % 270)], 99, 24)
                second.rng.bit_generator.state = saved['sampler_rng']
                x, y = sampler.sample(2, 'cpu'); rx, ry = second.sample(2, 'cpu')
                torch.testing.assert_close(x, rx); torch.testing.assert_close(y, ry)
                restored_optimizer = torch.optim.AdamW(resumed.parameters()); restored_optimizer.load_state_dict(saved['optimizer'])
                for m, opt in ((model, optimizer), (resumed, restored_optimizer)):
                    opt.zero_grad(); torch.nn.functional.cross_entropy(m(x)[0].flatten(0, 1), y.flatten()).backward(); opt.step()
                torch.testing.assert_close(model(x)[0], resumed(x)[0], atol=0, rtol=0)
                with self.assertRaises(ValueError): restore(path, graph, SimpleNamespace(vocabulary=270, sha256='wrong'))

    def test_article_scoring_is_independent_of_order_and_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.npz'; np.savez(graph, **fixture())
            lexicon = SimpleNamespace(vocabulary=270, sha256='fixture', lengths=np.r_[0, 0, np.ones(268, dtype=np.int64)])
            docs = [('one', np.r_[0, np.arange(140) + 2, 1]), ('two', np.r_[0, np.arange(150) + 5, 1])]
            for variant in ('flm', 'gru', 'transformer'):
                model = construct(variant, graph, 270, 42).double()
                a = evaluate(model, docs, lexicon, chunk_size=13)
                b = evaluate(model, list(reversed(docs)), lexicon, chunk_size=96)
                self.assertAlmostEqual(a['nll'], b['nll'], places=9)
                self.assertEqual(a['bytes'], 290)

    def test_transformer_cache_owns_only_its_bounded_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.npz'; np.savez(graph, **fixture())
            model = construct('transformer', graph, 270, 42).eval()
            with torch.no_grad():
                _, state = model(torch.arange(220).unsqueeze(0))
                for chunk in (1, 96, 17):
                    _, state = model(torch.arange(chunk).unsqueeze(0), state)
                    for key, value in state[1]:
                        for tensor in (key, value):
                            self.assertEqual(tensor.shape[2], 95)
                            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())


if __name__ == '__main__': unittest.main()
