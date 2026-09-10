import copy
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from flm.language_topology import conditions
from flm.language_topology_test import freeze_selection, score_study, summarize, validate_score


def fixture_score(delta=0):
    rows = [dict(document=name, nll=(2 + delta)*size*math.log(2), bytes=size, tokens=2,
        bits_per_byte=2 + delta) for name, size in [('a', 3), ('b', 5)]]
    return dict(documents=rows, bytes=8, tokens=4, nll=sum(r['nll'] for r in rows),
        bits_per_byte=2+delta, token_perplexity=math.exp(sum(r['nll'] for r in rows)/4), tokenizer_sha256='fixture')


class LanguageTopologyEvaluationTests(unittest.TestCase):
    def test_incomplete_control_rejected_before_any_test_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('flm.language_topology_test.read_json', return_value={}), \
                 patch('flm.language_topology_test.verify_source_identity'), \
                 patch('flm.language_topology_test.read_cache') as reader:
                with self.assertRaisesRegex(ValueError, 'incomplete'): score_study(Path(directory))
                reader.assert_not_called()

    def test_duplicate_missing_and_tampered_article_scores_fail(self):
        lexicon = SimpleNamespace(lengths=np.array([0, 0, 1, 2, 3]), sha256='fixture')
        documents = [('a', np.array([0, 2, 3, 1])), ('b', np.array([0, 3, 4, 1]))]
        score = fixture_score(); validate_score(score, documents, lexicon)
        for mutate in (lambda s: s['documents'].append(s['documents'][0]),
                       lambda s: s['documents'].pop(),
                       lambda s: s['documents'][0].update(bytes=4),
                       lambda s: s['documents'][0].update(nll=float('nan')),
                       lambda s: s.update(bits_per_byte=0),
                       lambda s: s.update(token_perplexity=1)):
            wrong = copy.deepcopy(score); mutate(wrong)
            with self.assertRaises(ValueError): validate_score(wrong, documents, lexicon)

    def test_crossed_summary_preserves_pair_sign_and_separate_variation(self):
        results = []
        for condition in conditions():
            delta = (condition['graph_seed'] - 100)/100 if condition['graph_seed'] else .2 if condition['variant'] == 'no_slow' else 0
            results.append(dict(**condition, score=fixture_score(delta)))
        report = summarize(results)
        self.assertEqual(len(report['topology_contrasts']), 6)
        self.assertEqual(len(report['topology_by_graph']), 3)
        self.assertEqual(len(report['topology_by_training_seed']), 2)
        self.assertAlmostEqual(report['topology_mean_difference_bpb'], -(.01+.03+.07)/3)
        self.assertAlmostEqual(report['slow_state_mean_difference_bpb'], -.2)
        for row in report['topology_contrasts']:
            self.assertLess(row['difference_bpb'], 0)
            self.assertAlmostEqual(row['difference_bpb'], row['lower_95'])
        with self.assertRaisesRegex(ValueError, 'ten unique'): summarize(results[:-1])
        with self.assertRaisesRegex(ValueError, 'ten unique'): summarize(results+[results[0]])


if __name__ == '__main__': unittest.main()
