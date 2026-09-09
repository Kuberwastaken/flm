import unittest
import numpy as np
from flm.ngram import NGram


class NgramTests(unittest.TestCase):
    def test_all_backoff_levels_are_normalized_and_unseen_bytes_possible(self):
        model = NGram(3).fit([('one', np.array([256, 97, 98, 97, 99, 257]))])
        for history in ([], [256], [97], [256, 97, 98], [42, 43, 44]):
            for smoothing in (.25, 1., 4.):
                probabilities = [model.probability(x, history, smoothing) for x in range(258)]
                self.assertAlmostEqual(sum(probabilities), 1., places=12)
                self.assertTrue(all(x > 0 for x in probabilities))

    def test_documents_do_not_create_boundary_ngrams(self):
        model = NGram(3).fit([('one', np.array([256, 97, 257])), ('two', np.array([256, 98, 257]))])
        self.assertNotIn((97, 257, 256), model.tables[3])
        result = model.evaluate([('probe', np.array([256, 97, 98, 257]))])
        self.assertEqual(result['bytes'], 2)


if __name__ == '__main__': unittest.main()
