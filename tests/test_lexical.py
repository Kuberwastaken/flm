import json
from pathlib import Path
import unittest
import numpy as np
import torch
from flm.model import FLM, Config, load_graph
from flm.baselines import GRU, Transformer, BaselineConfig
from flm.tokenizer import Lexicon, byte_characters, read_cache


class LexicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_byte_mapping_is_complete_and_invertible(self):
        mapping = byte_characters()
        self.assertEqual(set(mapping), set(range(256))); self.assertEqual(len(set(mapping.values())), 256)

    def test_trained_tokenizer_roundtrips_unicode_whitespace_and_literal_markers(self):
        lexicon = Lexicon('data/tokenizers/wikitext2-4096/tokenizer.json')
        for text in ['Hello, world!', '  spaces\n\tand tabs  ', 'café 日本語 🪰 हिन्दी', '<|bos|> [BOS] [EOS]', '\x00\r\n', '']:
            tokens = lexicon.encode(text, boundaries=True)
            self.assertEqual(lexicon.decode(tokens), text)
            self.assertEqual(int(lexicon.lengths[tokens].sum()), len(text.encode()))
            self.assertTrue(all(x >= 2 for x in tokens[1:-1]))

    def test_portable_vocabulary_reconstructs_every_merge(self):
        portable = json.loads(Path('data/tokenizers/wikitext2-4096/browser-tokenizer.json').read_text())
        self.assertEqual(portable['vocabulary'], 4096)
        for left, right, result in portable['merges']:
            self.assertEqual(portable['pieces'][left] + portable['pieces'][right], portable['pieces'][result])

    def test_tied_models_match_parameter_budget_and_preserve_streaming(self):
        graph = load_graph('data/graphs/central-1024/graph.npz')
        models = [FLM(graph, Config(embedding=96, vocabulary=4096, tied_readout=True)),
            GRU(BaselineConfig('gru', embedding=96, hidden=200, vocabulary=4096, tied_readout=True)),
            Transformer(BaselineConfig('transformer', embedding=96, width=108, heads=6, vocabulary=4096, tied_readout=True))]
        for model in models:
            self.assertLess(abs(model.parameter_card()['trainable_parameters'] / 600003 - 1), .02)
            tokens = torch.randint(2, 4096, (1, 20)); model.eval()
            whole, _ = model(tokens); first, state = model(tokens[:, :9]); rest, _ = model(tokens[:, 9:], state)
            torch.testing.assert_close(whole, torch.cat((first, rest), dim=1), atol=2e-6, rtol=2e-5)
            whole.square().mean().backward()
            self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
            self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)


if __name__ == '__main__': unittest.main()
