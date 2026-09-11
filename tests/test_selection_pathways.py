import unittest

import numpy as np

from flm.selection_pathways import GROUPS, THRESHOLDS, encode_groups, measure, pathway_coverage


class PathwayTests(unittest.TestCase):
    def test_directed_counts_and_thresholds_match_literal_enumeration(self):
        # Two ALPN inputs of 3 each do not make either body pair pass threshold 5.
        offsets = np.array([0, 1, 1, 4, 6, 7, 9])
        sources = np.array([5, 0, 1, 4, 2, 3, 2, 2, 4])
        counts = np.array([11, 3, 3, 13, 7, 5, 17, 19, 23])
        groups = np.array([0, 0, 1, 2, 3, 7])
        signs = np.array([1, 1, 1, -1, 0, -1])
        for chunk in (1, 4, 10):
            totals, partners = measure(offsets, sources, counts, signs, groups, chunk_rows=chunk)
            expected = {name: np.zeros_like(value) for name, value in totals.items()}
            expected_partners = {t: {d: np.zeros_like(a) for d, a in rows.items()} for t, rows in partners.items()}
            for post in range(6):
                for edge in range(offsets[post], offsets[post+1]):
                    pre, weight = int(sources[edge]), int(counts[edge])
                    for direction, cell, group in [('incoming', post, groups[pre]), ('outgoing', pre, groups[post])]:
                        expected[direction+'_contacts'][cell, group] += weight
                        expected[direction+'_edges'][cell, group] += 1
                        if signs[pre] != 0:
                            expected[direction+'_fast_contacts'][cell, group] += weight
                        for threshold in THRESHOLDS:
                            if weight >= threshold:
                                expected_partners[threshold][direction][cell, group] += 1
            for key in expected:
                np.testing.assert_array_equal(expected[key], totals[key])
            for threshold in THRESHOLDS:
                for direction in ('incoming', 'outgoing'):
                    np.testing.assert_array_equal(expected_partners[threshold][direction], partners[threshold][direction])
            coverage = pathway_coverage(groups, partners, np.ones(6, dtype=bool))
            self.assertEqual(coverage['1']['with_both'], 1)
            self.assertEqual(coverage['5']['with_both'], 0)
            self.assertEqual(coverage['5']['with_mbon_output'], 1)
            self.assertEqual(totals['incoming_fast_contacts'][2, 3], 0)
            self.assertEqual(totals['incoming_contacts'][2, 3], 13)
            empty = pathway_coverage(groups, partners, np.zeros(6, dtype=bool))
            self.assertEqual(empty['1']['kenyon_cells'], 0)

    def test_annotation_identity_membership_and_missingness(self):
        names = ['foo', 'KCg-m', 'MBON01', 'PAM01', 'PPL101', 'PPL201', 'APL', 'DPM', 'EPG']
        metadata = [[10+i, name] for i, name in enumerate(names)]
        rows = [dict(body_id=str(r[0]), runtime_type=r[1], publisher_type=r[1],
                     curated_class='ALPN' if i == 0 else '') for i, r in enumerate(metadata)]
        expected = [0, 1, 2, 3, 4, 7, 5, 6, 7]
        np.testing.assert_array_equal(encode_groups(metadata, list(reversed(rows))), expected)
        with self.assertRaises(ValueError): encode_groups(metadata, rows+rows[:1])
        with self.assertRaises(ValueError): encode_groups(metadata, rows[:1]+rows[2:])
        altered = [dict(row) for row in rows]; altered[1]['curated_class'] = 'ALPN'
        with self.assertRaises(ValueError): encode_groups(metadata, altered)
        altered[1]['curated_class'] = ''; altered[1]['publisher_type'] = 'changed'
        with self.assertRaises(ValueError): encode_groups(metadata, altered)

    def test_invalid_inputs_fail_before_reporting(self):
        offsets = np.array([0, 1, 1]); sources = np.array([1]); counts = np.array([3])
        signs = np.array([1, 0]); groups = np.array([0, 1])
        for bad in (np.array([-1]), np.array([2])):
            with self.assertRaises(ValueError): measure(offsets, bad, counts, signs, groups)
        for bad in (np.array([0]), np.array([-3]), np.array([1.2])):
            with self.assertRaises(ValueError): measure(offsets, sources, bad, signs, groups)
        with self.assertRaises(ValueError): measure(offsets, sources, counts, signs, np.array([0, len(GROUPS)]))
        with self.assertRaises(ValueError): measure(offsets, sources, counts, signs, groups, chunk_rows=0)


if __name__ == '__main__': unittest.main()
