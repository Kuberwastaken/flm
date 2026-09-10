import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from flm.corpus_evaluation import evaluate_batch, aggregate, component_summary, paired_block_interval
from flm.language_train import construct, evaluate
from test_model import fixture


class CorpusEvaluationTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_ragged_batches_match_independent_long_block_scores_for_all_models(self):
        lexicon = SimpleNamespace(vocabulary=270, sha256='fixture', lengths=np.r_[0, 0, np.arange(268) % 4 + 1])
        docs = [(str(i), np.r_[0, np.arange(length) % 268 + 2, 1].astype(np.uint16))
                for i, length in enumerate((1, 39, 214, 257))]
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.npz'; np.savez(graph, **fixture())
            for variant in ('flm', 'gru', 'transformer'):
                model = construct(variant, graph, 270, 42).double(); model.train()
                expected = evaluate(model, docs, lexicon, chunk_size=17)
                actual = evaluate_batch(model, docs, lexicon, chunk_size=96)
                self.assertTrue(model.training)
                for a, b in zip(actual['documents'], expected['documents']):
                    self.assertEqual((a['document'], a['bytes'], a['tokens']), (b['document'], b['bytes'], b['tokens']))
                    self.assertAlmostEqual(a['nll'], b['nll'], places=9, msg=variant)
                reversed_result = evaluate_batch(model, docs[::-1], lexicon, chunk_size=31)
                self.assertAlmostEqual(aggregate(reversed_result['documents'])['nll'], expected['nll'], places=9)

    def test_overlap_filter_drops_complete_blocks_and_keeps_component_counts(self):
        records = [dict(document=str(i), component=c, bytes=n, tokens=n, nll=n * 2., overlap_filtered_eligible=keep)
                   for i, (c, n, keep) in enumerate([('speech', 10, True), ('speech', 20, False), ('book', 30, False)])]
        result = component_summary(records)
        self.assertEqual(result['all']['official']['bytes'], 60)
        self.assertEqual(result['all']['overlap_filtered']['bytes'], 10)
        self.assertEqual(result['all']['excluded_bytes'], 50)
        self.assertEqual(result['speech']['excluded_blocks'], 1)
        self.assertIsNone(result['book']['overlap_filtered'])

    def test_stratified_paired_intervals_pair_ids_and_handle_exact_gaps(self):
        a = [dict(document=str(i), component=str(i % 2), bytes=n, tokens=n, nll=n * 2.)
             for i, n in enumerate([10, 100, 30, 1000])]
        b = [dict(r, nll=r['bytes']) for r in a][::-1]
        result = paired_block_interval(a, b, 257)
        for key in ('difference_bpb', 'lower_95', 'upper_95'):
            self.assertAlmostEqual(result[key], 1 / math.log(2))
        self.assertEqual(paired_block_interval(a, a, 257)['upper_95'], 0)
        with self.assertRaises(ValueError): paired_block_interval(a, b[:-1])
        with self.assertRaises(ValueError): paired_block_interval(a, b + [b[0]])
        with self.assertRaises(ValueError): paired_block_interval(a, [dict(r, component='wrong') for r in b])


if __name__ == '__main__': unittest.main()
