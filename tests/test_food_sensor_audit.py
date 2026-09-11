"""Fault injection into the supplied physical sensor calibration record."""
import copy
import json
from pathlib import Path
import unittest
from scripts.audit_food_sensor_probe import audit


class FoodAuditTests(unittest.TestCase):
    def setUp(self):
        self.record=json.loads((Path(__file__).resolve().parents[1]/'reports/food-sensors/visible-odor-probe.json').read_text(encoding='utf8'))

    def test_independent_scalar_replay_covers_every_observation(self):
        result=audit(self.record)
        self.assertEqual(result['observations_recomputed'],9)
        values=self.record['geometry_counterfactuals']['cases'][0]['observation']['sensory']
        self.assertTrue(all(v>0 for v in values[:4])); self.assertEqual(values[4:],[0.,0.])

    def test_invented_signal_wrong_body_origin_and_contact_mask_are_rejected(self):
        for mutate in (
            lambda r:r['frames'][0]['observation']['sensory'].__setitem__(0,.9),
            lambda r:r['frames'][0]['antennae_mm'][0].__setitem__(0,99.),
            lambda r:r['frames'][0]['observation']['contact_mask'][0].__setitem__(0,True),
            lambda r:r['geometry_counterfactuals']['cases'][0]['observation']['sensory'].__setitem__(4,1.)):
            bad=copy.deepcopy(self.record); mutate(bad)
            with self.assertRaises(ValueError): audit(bad)

    def test_missing_case_changed_clock_or_claimed_learning_are_rejected(self):
        for mutate in (
            lambda r:r['geometry_counterfactuals']['cases'].pop(),
            lambda r:r['frames'][1].update(time_s=.011),
            lambda r:r.update(neural_policy_used=True),
            lambda r:r['frames'][0]['qpos'].__setitem__(0,float('nan'))):
            bad=copy.deepcopy(self.record); mutate(bad)
            with self.assertRaises(ValueError): audit(bad)


if __name__=='__main__': unittest.main()
