import unittest
import numpy as np

from flm.selection_feasibility import FAMILIES, RANDOM_SEEDS, candidates, family, measure


class SelectionFeasibilityTests(unittest.TestCase):
    def test_chunked_contacts_match_literal_edges_including_sign_zero_and_boundary(self):
        offsets = np.array([0, 3, 5, 6, 8])
        sources = np.array([0, 1, 3, 0, 2, 1, 0, 2])
        contacts = np.array([2, 5, 7, 11, 13, 17, 19, 23])
        signs = np.array([1, 0, -1, 1])
        families = np.array([0, 1, 2, len(FAMILIES)])
        masks = dict(pair=np.array([True, True, False, False]),
                     single=np.array([False, False, True, False]),
                     full=np.ones(4, dtype=bool))
        for chunk in (1, 3, 10):
            records, raw, fast = measure(offsets, sources, contacts, signs, masks, families, chunk_rows=chunk)
            expected_raw = np.zeros_like(raw); expected_fast = np.zeros_like(fast)
            for post in range(4):
                for edge in range(offsets[post], offsets[post+1]):
                    pre, count = sources[edge], contacts[edge]
                    expected_raw[families[pre], families[post]] += count
                    if signs[pre]: expected_fast[families[pre], families[post]] += count
            np.testing.assert_array_equal(raw, expected_raw)
            np.testing.assert_array_equal(fast, expected_fast)
            for name, selected in masks.items():
                expected = {key: dict(connections=0, contacts=0) for key in
                            ('inside_raw', 'inside_fast', 'incoming_boundary', 'outgoing_boundary')}
                for post in range(4):
                    for edge in range(offsets[post], offsets[post+1]):
                        pre, count = sources[edge], int(contacts[edge])
                        matched = dict(inside_raw=selected[pre] and selected[post],
                            inside_fast=selected[pre] and selected[post] and signs[pre] != 0,
                            incoming_boundary=not selected[pre] and selected[post],
                            outgoing_boundary=selected[pre] and not selected[post])
                        for key, include in matched.items():
                            if include:
                                expected[key]['connections'] += 1; expected[key]['contacts'] += count
                for key in expected: self.assertEqual(records[name][key], expected[key])
            self.assertEqual(records['full']['incoming_cut_fraction'], 0)
            self.assertEqual(records['pair']['inside_contacts_removed_by_fast_sign'], 5)

    def test_named_inventory_keeps_all_matches_and_random_is_count_only_reproducible(self):
        types = ['KCg-m', 'MBON01', 'PAM01', 'PPL101', 'PPL201', 'APL', 'DPM', 'EPG', 'PEN_a', 'PEG', 'other', 'other']
        metadata = [[100+i, name, 'cb_intrinsic' if i < 11 else 'descending_neuron', 'L' if i % 2 else 'R']
                    for i, name in enumerate(types)]
        ids = np.array([row[0] for row in metadata])
        first = candidates(metadata, ids, np.array([0, 4, 8]), budget=3)
        second = candidates(metadata, ids, np.array([0, 4, 8]), budget=3)
        self.assertEqual(int(first['named_mb_bilateral'].sum()), 6)
        np.testing.assert_array_equal(first['named_mb_left'] | first['named_mb_right'], first['named_mb_bilateral'])
        self.assertFalse(first['named_mb_bilateral'][4])  # PPL2 is not silently included as PPL1.
        self.assertEqual(int(first['named_compass_families'].sum()), 3)
        for seed in RANDOM_SEEDS:
            name = f'uniform_3_s{seed}'
            self.assertEqual(int(first[name].sum()), 3)
            self.assertFalse(first[name][-1])
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(family('KC'), 'KC')
        self.assertEqual(family('kc'), 'other')

    def test_invalid_sources_counts_masks_and_empty_denominators(self):
        offsets = np.array([0, 1, 1])
        sources = np.array([1]); signs = np.array([1, 0]); families = np.array([0, 1])
        masks = dict(empty=np.zeros(2, dtype=bool))
        for counts in (np.array([0]), np.array([-1])):
            with self.assertRaises(ValueError): measure(offsets, sources, counts, signs, masks, families)
        with self.assertRaises(ValueError): measure(offsets, np.array([2]), np.array([1]), signs, masks, families)
        with self.assertRaises(ValueError): measure(offsets, sources, np.array([1]), signs, dict(bad=np.ones(2)), families)
        result, _, _ = measure(offsets, sources, np.array([1]), signs, masks, families)
        self.assertIsNone(result['empty']['incoming_cut_fraction'])
        self.assertIsNone(result['empty']['inside_fast_fraction'])


if __name__ == '__main__': unittest.main()
