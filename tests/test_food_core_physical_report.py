"""Reporting semantics for contact, censoring and retained simulation failures."""
from pathlib import Path
import tempfile
import unittest
from scripts.food_core_physical_report import outcome_rows, main


def record(status='complete', sugar=1., censored=False):
    metrics=dict(first_contact_s=None if censored else .8,
        first_contact_sources=[] if censored else ['a'], first_contact_sugar=None if censored else sugar,
        contact_latency_censored=censored, censor_time_s=2. if censored else None,
        final_thorax_position_mm=[9.,1.,1.])
    return dict(model_id='fixture',case=dict(label='synthetic'),status=status,
        observations=201,physics_steps=20000,metrics=metrics,
        failure=dict(type='FixtureFailure',message='Deliberate test') if status=='failed' else None)


class PhysicalReportTests(unittest.TestCase):
    def test_all_outcomes_retained_without_treating_censor_time_as_contact(self):
        trials=[record(),record(sugar=0.),record(censored=True),record(status='failed')]
        rows=outcome_rows(dict(trials=trials))
        self.assertEqual(len(rows),4)
        self.assertEqual([r['outcome'] for r in rows],['sugar contact','neutral contact','no source contact','simulation failure'])
        self.assertIsNone(rows[2]['first_contact_s']); self.assertEqual(rows[2]['censor_time_s'],2.)
        self.assertEqual(rows[3]['first_contact_s'],.8)
        self.assertEqual(rows[3]['status'],'failed')

    def test_empty_failure_is_not_discarded_or_invented_as_a_timeout(self):
        failed=record(status='failed'); failed.update(metrics=None,observations=0,physics_steps=0)
        row=outcome_rows(dict(trials=[failed]))[0]
        for key in ('first_contact_s','first_contact_sugar','censor_time_s','latency_censored','final_x_mm'):
            self.assertIsNone(row[key])
        self.assertEqual(row['outcome'],'simulation failure')
        self.assertIn('FixtureFailure',row['failure'])

    def test_missing_completed_metrics_and_unknown_status_are_rejected(self):
        invalid=record(); invalid['metrics']=None
        with self.assertRaises(ValueError): outcome_rows(dict(trials=[invalid]))
        invalid=record(); invalid['status']='skipped'
        with self.assertRaises(ValueError): outcome_rows(dict(trials=[invalid]))

    def test_report_refuses_an_unfinished_inventory_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'complete physical inventory'):
                main(root,root/'missing-bundle')
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__': unittest.main()
