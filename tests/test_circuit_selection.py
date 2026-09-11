import unittest

import numpy as np
from scipy.sparse import csr_matrix

from flm.circuit_selection import select


class CircuitSelectionTests(unittest.TestCase):
    def fixture(self):
        names = ['KCg-d','KCg-d','KCg-m','input','weak','MBON01','PAM01',
                 'APL','DPM','upstream','KCg-d','zero','PPL201','MBON02']
        sides = ['L','R','L','R','L','R','L','R','R','L','L','L','L','L']
        metadata = [[100+i, name, 'cb_intrinsic', sides[i]] for i, name in enumerate(names)]
        signs = np.array([1,1,1,1,1,-1,0,-1,0,1,1,0,0,1])
        # Ordered triples are PRE, POST, count, converted explicitly to post-row CSR.
        edges = [(3,0,5),(4,0,3),(4,10,3),(0,5,7),(6,5,1),(0,7,1),
                 (8,0,1),(9,3,9),(11,0,8),(12,0,8),(13,0,9),(2,0,10)]
        matrix = csr_matrix(([w for pre, post, w in edges],
                             ([post for pre, post, w in edges], [pre for pre, post, w in edges])),
                            shape=(len(names),len(names)))
        return matrix, metadata, signs

    def test_complete_seed_directed_roles_auxiliary_feedback_and_no_recursion(self):
        matrix, metadata, signs = self.fixture()
        roles = select(matrix, metadata, signs, 'KCg-d', 'L', 5)
        expected = dict(seed_kc=[0,10], input_partner=[3], output_mbon=[5], auxiliary_partner=[6,7,8])
        for role, indices in expected.items():
            self.assertEqual(np.flatnonzero(roles[role]).tolist(), indices)
        selected = np.logical_or.reduce(list(roles.values()))
        self.assertFalse(selected[1])  # Other side's KC excluded; input on other side retained.
        self.assertTrue(selected[3])
        self.assertFalse(selected[9])  # No recursive inclusion of input partner's input.
        self.assertFalse(selected[12])  # PPL2 is not silently treated as PPL1.
        self.assertFalse(selected[13])  # MBON feedback alone is not the output inclusion rule.
        self.assertEqual(sum(int(mask.sum()) for mask in roles.values()), int(selected.sum()))

    def test_threshold_is_per_pair_and_does_not_change_complete_seed_population(self):
        matrix, metadata, signs = self.fixture()
        one = select(matrix, metadata, signs, 'KCg-d', 'L', 1)
        five = select(matrix, metadata, signs, 'KCg-d', 'L', 5)
        self.assertTrue(one['input_partner'][4])
        self.assertFalse(five['input_partner'][4])  # 3+3 across two seeds is not >=5 per pair.
        np.testing.assert_array_equal(one['seed_kc'], five['seed_kc'])
        np.testing.assert_array_equal(one['auxiliary_partner'], five['auxiliary_partner'])
        # No caller matrix/sign/metadata mutation.
        copy, original_metadata, original_signs = self.fixture()
        self.assertEqual((matrix != copy).nnz, 0)
        np.testing.assert_array_equal(signs, original_signs)
        self.assertEqual(metadata, original_metadata)

    def test_absent_seed_and_unsupported_rule_rejected(self):
        matrix, metadata, signs = self.fixture()
        for seed_type, side, threshold in [('KCg-m','R',5),('KC','L',5),('KCg-d','both',5),('KCg-d','L',3)]:
            with self.assertRaises(ValueError): select(matrix, metadata, signs, seed_type, side, threshold)
        with self.assertRaises(ValueError): select(matrix, metadata, signs[:-1], 'KCg-d','L',5)


if __name__ == '__main__': unittest.main()
