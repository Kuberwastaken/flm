from collections import Counter
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from flm.selection_controls import SEEDS, controls, describe


class SelectionControlTests(unittest.TestCase):
    def fixture(self):
        ids = np.array([40,10,30,20,60,50,90,70])
        metadata = [[int(body), 'type', 'cb_intrinsic' if i < 4 else 'visual_projection' if i < 7 else 'motor_neuron',
                     'L' if i < 4 else 'R'] for i, body in enumerate(ids)]
        signs = np.array([1,1,1,1,-1,-1,-1,1])
        # Include a self edge (counted twice), a zero-strength eligible cell and
        # a very strong excluded motor connection that must not affect ranking.
        matrix = csr_matrix(([5,7,5,5,1000],([0,1,2,3,7],[0,2,1,4,0])),shape=(8,8))
        return matrix, metadata, ids, signs

    def test_ranking_matches_literal_eligible_strength_and_id_ties(self):
        matrix, metadata, ids, signs = self.fixture()
        result = controls(matrix, metadata, ids, signs, np.array([0,4,5]))
        eligible = list(range(7))
        strength = {i: sum(int(matrix[i,j])+int(matrix[j,i]) for j in eligible) for i in eligible}
        ranked = sorted(eligible, key=lambda i: (-strength[i], int(ids[i])))[:3]
        self.assertEqual(set(result['contact_ranked']), set(ranked))
        self.assertEqual(strength[0], 10)
        self.assertNotIn(7, result['contact_ranked'])
        self.assertNotIn(6, result['contact_ranked'])  # Unsigned negation must not rank zero first.

    def test_strata_budget_reproducibility_and_input_permutation_invariance(self):
        matrix, metadata, ids, signs = self.fixture()
        selected = np.array([0,4,5])
        first = controls(matrix, metadata, ids, signs, selected)
        second = controls(matrix, metadata, ids, signs, selected)
        strata = lambda chosen: Counter((metadata[i][2],metadata[i][3],int(signs[i])) for i in chosen)
        for name, chosen in first.items():
            self.assertEqual(len(chosen), len(set(chosen.tolist())))
            self.assertEqual(len(chosen), 3)
            self.assertTrue(set(chosen) <= set(range(7)))
            np.testing.assert_array_equal(chosen, second[name])
            if name.startswith('stratified'): self.assertEqual(strata(chosen), strata(selected))
        perm = np.array([5,2,7,0,3,6,4,1]); inverse = np.argsort(perm)
        shuffled = controls(matrix[perm][:,perm], [metadata[i] for i in perm], ids[perm], signs[perm], inverse[selected])
        for name in first:
            np.testing.assert_array_equal(ids[first[name]], ids[perm][shuffled[name]])
        self.assertEqual(len(first), 1+2*len(SEEDS))

    def test_boundaries_match_literal_pairs_and_invalid_candidates_rejected(self):
        matrix, metadata, ids, signs = self.fixture()
        selected = np.array([0,1,2])
        incoming = np.asarray(matrix.sum(axis=1)).ravel()
        outgoing = np.asarray(matrix.sum(axis=0)).ravel()
        result = describe(matrix, metadata, ids, signs, selected, selected, incoming, outgoing)
        inside = sum(int(matrix[i,j]) for i in selected for j in selected)
        all_in = sum(int(matrix[i,j]) for i in selected for j in range(8))
        all_out = sum(int(matrix[i,j]) for i in range(8) for j in selected)
        self.assertEqual(result['inside_raw_contacts'],inside)
        self.assertEqual(result['incoming_cut_fraction'],(all_in-inside)/all_in)
        self.assertEqual(result['outgoing_cut_fraction'],(all_out-inside)/all_out)
        for bad in (np.array([7]),np.array([0,0]),np.array([-1]),np.array([8]),np.array([],dtype=int)):
            with self.assertRaises(ValueError): controls(matrix,metadata,ids,signs,bad)


if __name__ == '__main__': unittest.main()
