"""Fault injection in synthetic physical records; no simulated outcome is claimed."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scripts.audit_food_approach import audit_trial
from scripts.audit_food_sensor_probe import replay


class ApproachAuditTests(unittest.TestCase):
    def setUp(self):
        # A fixed real pose is tiled into an explicitly synthetic, nonmoving
        # record. At t=1s one foot is placed in the patch for one sample only.
        source = json.loads((Path(__file__).resolve().parents[1]/'reports/food-sensors/visible-odor-probe.json').read_bytes())
        self.names = source['body_names']; frame = source['frames'][0]; n = 201
        field = copy.deepcopy(source['field'])
        for item in field['sources']:
            item['contact_radius_mm'] = .75; item['contact_height_mm'] = .25
        self.declared = dict(fields={'fixture': field}, odor_body_origins=source['odor_body_origins'], contact_body_origins=source['contact_body_origins'])
        body = np.tile(np.asarray(frame['body_positions_mm']), (n, 1, 1))
        body[100, self.names.index('lf_tarsus5')] = [8., 3., .1]
        antenna = [self.names.index(name) for name in source['odor_body_origins']]
        feet = [self.names.index(name) for name in source['contact_body_origins']]
        observations = [replay(field, row[antenna].tolist(), row[feet].tolist(), False) for row in body]
        sensory = np.array([r['sensory'] for r in observations])
        a, b = sensory[0, :2]; turn = min(.4, max(-.4, 24*(a-b)/(a+b+1e-12)))
        drive = np.tile([.8-turn, .8+turn], (n, 1)); drive[100:] = 0
        self.arrays = dict(time_s=np.arange(n)*.01, body_positions_mm=body,
            thorax_quaternion_wxyz=np.tile([1., 0., 0., 0.], (n, 1)), qpos=np.tile(frame['qpos'], (n, 1)),
            sensory=sensory, raw_odor=np.array([r['raw_odor'] for r in observations]),
            contact_mask=np.array([r['contact_mask'] for r in observations]), descending_signal=drive)
        thorax = body[0, self.names.index('c_thorax')]
        metrics = dict(first_contact_s=1., first_contact_sources=['a'], first_contact_sugar=1.,
            contact_latency_censored=False, censor_time_s=None, all_contacted_sources=['a'],
            minimum_thorax_planar_distance_mm={s['name']: float(np.linalg.norm(thorax[:2]-np.array(s['position_mm'][:2]))) for s in field['sources']},
            final_thorax_position_mm=thorax.tolist(), minimum_thorax_height_mm=float(thorax[2]), minimum_thorax_up_z=1.)
        self.record = dict(case=dict(label='fixture', mode='odor', missing_odor=False), observations=n,
            body_names=self.names, status='complete', physics_steps=20000, failure=None, metrics=metrics)

    def test_sensor_command_latch_and_contact_latency_replay(self):
        self.assertEqual(audit_trial(self.declared, self.record, self.arrays), 201)
        self.assertEqual(self.arrays['sensory'][101, 5], 0.)
        np.testing.assert_array_equal(self.arrays['descending_signal'][101], [0., 0.])

    def test_changed_action_sensor_geometry_or_clock_is_rejected(self):
        for key, index, value in (
            ('descending_signal', (101, 0), .8), ('sensory', (0, 4), 1.),
            ('body_positions_mm', (0, self.names.index('l_funiculus'), 0), 500.),
            ('contact_mask', (0, 0, 0), True), ('time_s', 1, .015)):
            arrays = {k: a.copy() for k, a in self.arrays.items()}; arrays[key][index] = value
            with self.assertRaises(ValueError): audit_trial(self.declared, self.record, arrays)

    def test_invented_success_or_missing_physical_budget_is_rejected(self):
        for change in (
            lambda r: r['metrics'].update(first_contact_s=.5),
            lambda r: r['metrics'].update(first_contact_sources=['b']),
            lambda r: r['metrics'].update(contact_latency_censored=True),
            lambda r: r.update(physics_steps=10000),
            lambda r: r.update(observations=200)):
            record = copy.deepcopy(self.record); change(record)
            with self.assertRaises(ValueError): audit_trial(self.declared, record, self.arrays)

    def test_failed_empty_trial_is_preserved_without_claiming_success(self):
        record = dict(case=self.record['case'], observations=0, body_names=[], status='failed',
            physics_steps=0, failure=dict(type='Fixture', message='Deliberate setup failure'), metrics=None)
        arrays = {k: np.array([]) for k in self.arrays}
        self.assertEqual(audit_trial(self.declared, record, arrays), 0)


if __name__ == '__main__': unittest.main()
