import unittest

from flm.publisher_annotations import compare


class PublisherAnnotationTests(unittest.TestCase):
    def test_join_is_by_id_and_preserves_consensus_prediction_and_missingness(self):
        runtime = [[1,'KC','cb_intrinsic','L','acetylcholine',1],
                   [2,'DPM','cb_intrinsic','R','dopamine',0],
                   [3,'other','cb_intrinsic','','nan',0]]
        annotation = [dict(bodyId=2,type='DPM',superclass='cb_intrinsic',rootSide='R',somaSide=None),
                      dict(bodyId=1,type='KC',superclass='cb_intrinsic',rootSide='R',somaSide='L'),
                      dict(bodyId=3,type='other',superclass='cb_intrinsic')]
        nt = [dict(body=2,predicted_nt='dopamine',consensus_nt='dopamine',ground_truth=None),
              dict(body=1,predicted_nt='gaba',consensus_nt='acetylcholine',ground_truth='acetylcholine')]
        counts, rows = compare(runtime, annotation, nt)
        self.assertEqual(counts['annotation_present'], 3)
        self.assertEqual(counts['transmitter_present'], 2)
        self.assertEqual(counts['type_exact_matches'], 3)
        self.assertEqual(counts['soma_side_exact_matches'], 2)
        self.assertEqual(counts['soma_then_root_side_matches'], 3)
        self.assertEqual(counts['consensus_nt_matches'], 2)
        self.assertEqual(counts['body_predicted_nt_matches'], 1)
        self.assertEqual(counts['consensus_fast_sign_rule_matches'], 2)
        self.assertEqual(counts['missing_transmitter_with_zero_fast_sign'], 1)
        self.assertEqual(counts['ground_truth_field_nonempty'], 1)
        self.assertEqual(rows[0]['predicted_nt'], 'gaba')
        self.assertEqual(rows[0]['consensus_nt'], 'acetylcholine')
        self.assertEqual(rows[1]['ground_truth'], '')
        self.assertFalse(rows[2]['transmitter_present'])
        self.assertEqual(rows[2]['runtime_nt'], 'nan')
        self.assertEqual(rows[2]['consensus_nt'], '')

    def test_duplicate_and_malformed_identities_are_rejected(self):
        runtime = [[1,'KC','cb_intrinsic','L','acetylcholine',1]]
        with self.assertRaisesRegex(ValueError, 'unique'):
            compare(runtime, [dict(bodyId=1),dict(bodyId=1)], [])
        with self.assertRaisesRegex(ValueError, 'unique'):
            compare(runtime, [], [dict(body=1),dict(body=1)])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compare(runtime+runtime, [], [])
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            compare([[1,'KC','cb_intrinsic','L','acetylcholine',2]], [], [])

    def test_missing_annotation_is_not_counted_as_an_exact_blank_match(self):
        counts, rows = compare([[1,'','','','nan',0]], [], [])
        self.assertEqual(counts['annotation_present'], 0)
        self.assertEqual(counts.get('type_exact_matches',0), 0)
        self.assertFalse(rows[0]['annotation_present'])
        self.assertFalse(rows[0]['transmitter_present'])


if __name__ == '__main__': unittest.main()
