"""Fault injection in synthetic records, not additional body simulations."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from scripts.audit_food_core_physical import audit_trial
from scripts.audit_food_sensor_probe import replay


def tiny_core():
    # Deliberately simple recurrent fixture, unrelated to the released checkpoints.
    rng = np.random.default_rng(182)
    def random(shape): return rng.normal(0, .1, shape).astype(np.float32)
    return FoodCoreRuntime(dict(recurrent=np.array([[0, .4], [-.2, 0]], dtype=np.float32),
        alpha=np.array([.3, .6], dtype=np.float32), beta=np.array([.01, .07], dtype=np.float32),
        gain=np.array(.7, dtype=np.float32), input_weight=random((2,3)), input_bias=random((2,)),
        pool_index=np.array([0,1], dtype=np.int64), pool_sizes=np.ones(2, dtype=np.float32),
        norm_weight=np.ones(4, dtype=np.float32), norm_bias=np.zeros(4, dtype=np.float32),
        norm_epsilon=np.array(1e-5, dtype=np.float32), sensor_weight=random((3,6)), sensor_bias=random((3,)),
        action_weight=random((3,4)), action_bias=random((3,)), body_ids=np.array([100,200], dtype=np.int64)))


class PhysicalCoreAuditTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        source = json.loads((root/'reports/food-sensors/visible-odor-probe.json').read_bytes())
        names = source['body_names']; frame = source['frames'][0]; field = copy.deepcopy(source['field'])
        for item in field['sources']: item['contact_radius_mm'] = .75; item['contact_height_mm'] = .25
        n = 201; body = np.tile(np.asarray(frame['body_positions_mm']), (n,1,1))
        body[100, names.index('lf_tarsus5')] = [8., 3., .1]
        antennae = [names.index(name) for name in source['odor_body_origins']]
        feet = [names.index(name) for name in source['contact_body_origins']]
        observations = [replay(field, row[antennae].tolist(), row[feet].tolist(), False) for row in body]
        sensory = np.array([r['sensory'] for r in observations], dtype=np.float32)
        core = tiny_core(); records = {name: [] for name in ('fast','slow','features','logits','probabilities')}
        commands = []
        for i, x in enumerate(sensory):
            logits, probabilities, features = core.step(x)
            for name, value in (('fast',core.fast), ('slow',core.slow), ('features',features), ('logits',logits), ('probabilities',probabilities)):
                records[name].append(value.copy())
            turn = .4*(float(probabilities[2])-float(probabilities[0]))
            commands.append([.8-turn, .8+turn] if i < 100 else [0., 0.])
        self.arrays = dict(time_s=np.arange(n)*.01, body_positions_mm=body,
            thorax_quaternion_wxyz=np.tile([1.,0.,0.,0.], (n,1)), qpos=np.tile(frame['qpos'], (n,1)),
            sensory=sensory, raw_odor=np.array([r['raw_odor'] for r in observations]),
            contact_mask=np.array([r['contact_mask'] for r in observations]), descending_signal=np.array(commands),
            **{name:np.array(values) for name,values in records.items()})
        self.declared = dict(fields={'fixture':field}, odor_body_origins=source['odor_body_origins'], contact_body_origins=source['contact_body_origins'])
        thorax = body[0, names.index('c_thorax')]
        metrics = dict(first_contact_s=1., first_contact_sources=['a'], first_contact_sugar=1.,
            contact_latency_censored=False, censor_time_s=None, all_contacted_sources=['a'],
            minimum_thorax_planar_distance_mm={s['name']:float(np.linalg.norm(thorax[:2]-np.array(s['position_mm'][:2]))) for s in field['sources']},
            final_thorax_position_mm=thorax.tolist(), minimum_thorax_height_mm=float(thorax[2]), minimum_thorax_up_z=1.)
        self.record = dict(case=dict(label='fixture', missing_odor=False), observations=n, body_names=names,
            status='complete', physics_steps=20000, failure=None, metrics=metrics)

    def test_all_neural_states_and_latched_command_replay(self):
        checked = audit_trial(self.declared, self.record, self.arrays, tiny_core())
        self.assertEqual(checked['observations'], 201)
        self.assertTrue(all(value == 0 for value in checked['maximum_core_error'].values()))
        self.assertTrue(np.any(self.arrays['fast'][100] != self.arrays['fast'][101]))
        np.testing.assert_array_equal(self.arrays['descending_signal'][101], [0,0])

    def test_corrupted_neural_sensor_action_or_clock_records_fail(self):
        for name, index, delta in (('fast',(13,0),.1), ('slow',(33,1),.1), ('features',(3,0),.1),
            ('logits',(7,0),.1), ('probabilities',(4,1),.1), ('sensory',(0,0),.1),
            ('descending_signal',(101,0),.8), ('time_s',1,.003)):
            changed = {key:value.copy() for key,value in self.arrays.items()}; changed[name][index] += delta
            with self.assertRaises((ValueError, AssertionError)):
                audit_trial(self.declared, self.record, changed, tiny_core())

    def test_fabricated_success_truncated_budget_and_wrong_dtype_fail(self):
        for change in (lambda r:r['metrics'].update(first_contact_s=.5),
                       lambda r:r.update(physics_steps=10000), lambda r:r.update(observations=200)):
            record = copy.deepcopy(self.record); change(record)
            with self.assertRaises(ValueError): audit_trial(self.declared, record, self.arrays, tiny_core())
        arrays = dict(self.arrays); arrays['fast'] = arrays['fast'].astype(np.float64)
        with self.assertRaises(ValueError): audit_trial(self.declared, self.record, arrays, tiny_core())

    def test_empty_simulation_failure_is_retained(self):
        record = dict(case=self.record['case'], observations=0, body_names=[], status='failed',
            physics_steps=0, failure=dict(type='Fixture', message='Synthetic setup failure'), metrics=None)
        checked = audit_trial(self.declared, record, {name:np.array([]) for name in self.arrays}, tiny_core())
        self.assertEqual(checked['observations'], 0)


if __name__ == '__main__': unittest.main()
