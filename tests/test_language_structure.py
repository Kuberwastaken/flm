import unittest
import numpy as np

from flm.language_structure import node_statistics, pair_audit, overlap_matrix


class LanguageStructureTests(unittest.TestCase):
    def graph(self):
        return dict(body_ids=np.arange(4), row=np.array([0, 1, 2, 3]),
                    col=np.array([1, 0, 3, 2]), weight=np.array([.2, .3, .5, .7]),
                    source_sign=np.ones(4), pool=np.arange(4), positions=np.zeros((4, 3)),
                    contacts=np.array([2, 3, 5, 7]))

    def test_preserved_degrees_do_not_imply_preserved_outgoing_strength(self):
        measured = self.graph()
        control = {k: v.copy() for k, v in measured.items()}
        control['col'] = np.array([2, 3, 0, 1])
        audit = pair_audit(measured, control)
        self.assertTrue(all(audit['preserved'].values()))
        self.assertEqual(set(audit['maximum_absolute_node_differences'].values()), {0})
        self.assertEqual(audit['outgoing_absolute_weight_difference']['changed_nodes'], 4)
        self.assertAlmostEqual(audit['outgoing_absolute_weight_difference']['maximum_absolute'], .5)
        self.assertEqual(overlap_matrix([measured, control]), [[1, 0], [0, 1]])
        control['weight'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'fixed array'):
            pair_audit(measured, control)

    def test_statistics_use_directed_edges_and_source_signs(self):
        graph = self.graph()
        graph['source_sign'][1] = -1
        graph['weight'][0] = -.2
        stats = node_statistics(graph)
        self.assertEqual(stats['negative_in_degree'].tolist(), [1, 0, 0, 0])
        self.assertEqual(stats['positive_in_degree'].tolist(), [0, 1, 1, 1])
        np.testing.assert_array_equal(stats['outgoing_absolute_weight'], [.3, .2, .7, .5])
        with self.assertRaisesRegex(ValueError, 'equal edge counts'):
            overlap_matrix([graph, dict(graph, row=np.array([0]), col=np.array([1]))])


if __name__ == '__main__':
    unittest.main()
