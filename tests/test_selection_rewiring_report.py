import copy
import hashlib
import unittest

from scripts.selection_rewiring_report import group,summarize,verify_manifest


def fixture():
    name='KCg-d-L-t5/candidate'
    source=dict(graphs={name:dict(config=dict(neurons=6),edges=6)})
    original=dict(neurons=6,edges=6,strong_component_sizes=[6],reciprocal_off_diagonal_fraction=0)
    records={}
    for seed,overlap in zip((101,103,107),(.1,.2,.3)):
        records[name+f'/null{seed}']=dict(status='complete',report=dict(seed=seed,accepted_swaps=60,proposed_swaps=100,
            acceptance_fraction=.6,original=original,randomized=dict(neurons=6,edges=6,strong_component_sizes=[2,2,2],
            reciprocal_off_diagonal_fraction=1),original_edge_overlap_fraction=overlap,unchanged_endpoint_slots_fraction=0))
    return source,dict(planned=3,complete=3,failed=0,records=records)


class SelectionRewiringReportTests(unittest.TestCase):
    def test_archive_hash_and_semantic_local_manifest_allow_only_newline_differences(self):
        archived=b'{\n  "complete": 3\n}\n'; local=archived.replace(b'\n',b'\r\n')
        expected=hashlib.sha256(archived).hexdigest()
        self.assertEqual(verify_manifest(archived,local,expected),dict(complete=3))
        with self.assertRaises(ValueError): verify_manifest(archived,local.replace(b'3',b'2'),expected)
        with self.assertRaises(ValueError): verify_manifest(archived,local,'0'*64)

    def test_one_selection_point_preserves_all_seed_values_and_scc_denominators(self):
        result=summarize(*fixture()); row=result['selections'][0]
        self.assertEqual((len(result['cases']),result['plotted_selections']),(3,1))
        self.assertEqual(row['original_edge_overlap_percent'],dict(mean=20.,minimum=10.,maximum=30.))
        self.assertEqual(row['original_largest_scc_percent'],100)
        self.assertAlmostEqual(row['rewired_largest_scc_percent']['mean'],100/3)
        self.assertAlmostEqual(row['edge_density_percent'],100/6)
        self.assertEqual(group('case/stratified_s203'),'stratified')
        with self.assertRaises(ValueError): group('case/unknown')

    def test_failures_remain_cases_without_a_partial_seed_average(self):
        source,manifest=fixture(); manifest['records']['KCg-d-L-t5/candidate/null103']=dict(status='failed',reason='Fixture failure')
        manifest.update(complete=2,failed=1)
        result=summarize(source,manifest)
        self.assertEqual(result['plotted_selections'],0)
        self.assertEqual(len(result['cases']),3)
        self.assertEqual(result['cases'][1]['failure_reason'],'Fixture failure')
        self.assertNotIn('original_edge_overlap_percent',result['selections'][0])
        self.assertEqual(result['groups']['candidate']['failed_controls'],1)

    def test_missing_cases_changed_original_diagnostics_and_bad_totals_fail(self):
        source,manifest=fixture()
        changed=copy.deepcopy(manifest); changed['records'].pop('KCg-d-L-t5/candidate/null101')
        with self.assertRaisesRegex(ValueError,'inventory'): summarize(source,changed)
        changed=copy.deepcopy(manifest)
        changed['records']['KCg-d-L-t5/candidate/null101']['report']['original']=dict(neurons=6,edges=6,
            strong_component_sizes=[3,3],reciprocal_off_diagonal_fraction=0)
        with self.assertRaisesRegex(ValueError,'Original diagnostics'): summarize(source,changed)
        changed=copy.deepcopy(manifest); changed['complete']=2
        with self.assertRaisesRegex(ValueError,'totals'): summarize(source,changed)
        changed=copy.deepcopy(manifest); changed['records']['KCg-d-L-t5/candidate/null101']['report']['randomized']['neurons']=7
        with self.assertRaisesRegex(ValueError,'dimensions'): summarize(source,changed)


if __name__=='__main__': unittest.main()
