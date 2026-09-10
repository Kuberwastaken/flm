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

    def test_completion_markers_cannot_bypass_checkpoint_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in conditions():
                marker = root / condition['output'] / 'complete.json'
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text('{}')
            with patch('flm.language_topology_test.read_json', return_value={}), \
                 patch('flm.language_topology_test.verify_source_identity'), \
                 patch('flm.language_topology_test.Lexicon'), \
                 patch('flm.language_topology_test.verify_complete', side_effect=ValueError('Invalid saved checkpoint')), \
                 patch('flm.language_topology_test.read_cache') as reader:
                with self.assertRaisesRegex(ValueError, 'Invalid saved checkpoint'):
                    score_study(root)
                reader.assert_not_called()
                self.assertFalse((root / 'reports/language-topology/selection.json').exists())

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

    def test_mixed_article_effects_use_byte_weighting_and_the_matching_seed(self):
        # Two documents contain 1 and 9 bytes. Values below are per-byte losses,
        # chosen so unweighted article means and mixed-up seeds give wrong signs.
        article_bpb = {
            'measured-s42': (1, 3), 'measured-s43': (5, 1),
            'null101-s42': (2, 2), 'null101-s43': (4, 2),
            'null103-s42': (3, 3), 'null103-s43': (6, .5),
            'null107-s42': (4, 2), 'null107-s43': (4, 1),
            'no-slow-s42': (1, 4), 'no-slow-s43': (5, .5)}
        results = []
        for condition in reversed(conditions()):
            losses = article_bpb[condition['label']]
            documents = [dict(document=name, bytes=size, nll=loss*size*math.log(2))
                         for name, size, loss in zip(('short', 'long'), (1, 9), losses)]
            if condition['seed'] == 43 or condition['graph_seed'] == 103:
                documents.reverse()
            results.append(dict(**condition, score={'documents': documents}))
        report = summarize(results)
        expected = {(42, 101): .8, (42, 103): -.2, (42, 107): .6,
                    (43, 101): -.8, (43, 103): .35, (43, 107): .1}
        for row in report['topology_contrasts']:
            self.assertAlmostEqual(row['difference_bpb'], expected[row['training_seed'], row['graph_seed']])
        self.assertAlmostEqual(report['topology_mean_difference_bpb'], 17/120)
        for row in report['topology_by_graph']:
            self.assertAlmostEqual(row['mean_difference_bpb'], {101: 0, 103: .075, 107: .35}[row['graph_seed']])
        for row in report['topology_by_training_seed']:
            self.assertAlmostEqual(row['mean_difference_bpb'], {42: .4, 43: -7/60}[row['training_seed']])
        for row in report['slow_state_contrasts']:
            self.assertAlmostEqual(row['difference_bpb'], {42: -.9, 43: .45}[row['training_seed']])
        self.assertAlmostEqual(report['slow_state_mean_difference_bpb'], -.225)
        # Resampling two paired articles can attain each single-article effect.
        first = report['topology_contrasts'][0]
        self.assertAlmostEqual(first['lower_95'], -1)
        self.assertAlmostEqual(first['upper_95'], 1)
        self.assertEqual(first['replicates'], 10000)
        self.assertEqual(first['seed'], 31415)


if __name__ == '__main__': unittest.main()
