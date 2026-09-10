"""Independent hand-counted graph checks for the subset's published audit."""
import unittest

import numpy as np

from flm.subset_audit import measure


class SubsetAuditTests(unittest.TestCase):
    def fixture(self):
        # Rows are postsynaptic. Selected nodes are 0 and 2; node 2 has sign 0.
        # Included self edges and connections to ineligible node 4 distinguish
        # total boundary accounting from the eligible-population ranking.
        return dict(
            offsets=np.array([0, 4, 5, 8, 9, 11]),
            sources=np.array([0, 1, 2, 4, 0, 0, 2, 3, 1, 2, 4]),
            counts=np.array([2, 5, 3, 7, 19, 11, 13, 17, 29, 23, 31]),
            eligible=np.array([True, True, True, True, False]),
            selected=np.array([True, False, True, False, False]),
            signs=np.array([1, -1, 0, 1, -1]))

    def test_directed_boundary_and_zero_sign_losses_are_distinct(self):
        groups, _, retained, maximum = measure(**self.fixture())
        self.assertEqual(groups, {
            'inside': {'connections': 4, 'contacts': 29},
            'outside_to_selected': {'connections': 3, 'contacts': 29},
            'selected_to_outside': {'connections': 2, 'contacts': 42},
            'outside': {'connections': 2, 'contacts': 60},
            'inside_active': {'connections': 2, 'contacts': 13},
            'inside_zero_sign': {'connections': 2, 'contacts': 16}})
        self.assertEqual([part.tolist() for part in retained], [[0, 2], [0, 0], [2, 11]])
        self.assertEqual(maximum, 31)

    def test_ranking_and_node_tables_have_correct_direction_and_denominator(self):
        _, node, _, _ = measure(**self.fixture())
        expected = {
            'central_in_contacts': [10, 19, 41, 29, 0],
            'central_out_contacts': [32, 34, 16, 17, 0],
            'full_in_contacts': [17, 19, 41, 29, 54],
            'full_out_contacts': [32, 34, 39, 17, 38],
            'inside_in_contacts': [5, 0, 24, 0, 0],
            'inside_out_contacts': [13, 0, 16, 0, 0],
            'boundary_in_contacts': [12, 0, 17, 0, 0],
            'boundary_out_contacts': [19, 0, 23, 0, 0],
            'active_in_contacts': [2, 0, 11, 0, 0],
            'active_out_contacts': [13, 0, 0, 0, 0],
            'full_in_connections': [4, 1, 3, 1, 2],
            'full_out_connections': [3, 2, 3, 1, 2],
            'boundary_in_connections': [2, 0, 1, 0, 0],
            'boundary_out_connections': [1, 0, 1, 0, 0]}
        self.assertEqual({name: values.tolist() for name, values in node.items()}, expected)

    def test_chunk_boundaries_and_empty_rows_cannot_change_accounting(self):
        fixture = self.fixture()
        reference = measure(**fixture, chunk_rows=100)
        for size in (1, 2, 3):
            actual = measure(**fixture, chunk_rows=size)
            self.assertEqual(actual[0], reference[0])
            for name in reference[1]:
                np.testing.assert_array_equal(actual[1][name], reference[1][name])
            for a, b in zip(actual[2], reference[2]):
                np.testing.assert_array_equal(a, b)
        fixture['offsets'] = np.array([0, 0, 0, 0, 0, 0])
        fixture['sources'] = np.array([], dtype=np.int64)
        fixture['counts'] = np.array([], dtype=np.int64)
        groups, node, retained, maximum = measure(**fixture, chunk_rows=1)
        self.assertTrue(all(row == {'connections': 0, 'contacts': 0} for row in groups.values()))
        self.assertTrue(all(not row.any() for row in node.values()))
        self.assertTrue(all(len(row) == 0 for row in retained))
        self.assertEqual(maximum, 0)

    def test_malformed_inputs_fail_before_publication(self):
        mutations = [('offsets', 2, 12), ('sources', 1, 5), ('sources', 1, -1),
                     ('counts', 1, 0), ('counts', 1, -3), ('signs', 1, 2)]
        for name, index, value in mutations:
            fixture = self.fixture(); fixture[name][index] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                measure(**fixture)
        fixture = self.fixture(); fixture['counts'] = fixture['counts'].astype(float)
        with self.assertRaises(ValueError):
            measure(**fixture)


if __name__ == '__main__':
    unittest.main()
