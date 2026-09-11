"""Mechanism checks for the scripted reference, not behavior success tests."""
import unittest
import numpy as np
from flm.food_approach import OdorApproach, cases


class FoodApproachTests(unittest.TestCase):
    def test_calibrated_turn_direction_and_bound(self):
        left = OdorApproach().step([.3, .1, 0., 0., 0., 0.])
        np.testing.assert_allclose(left, [.4, 1.2], atol=1e-15)
        right = OdorApproach().step([.1, .3, 0., 0., 0., 0.])
        np.testing.assert_array_equal(right, left[::-1])
        np.testing.assert_array_equal(OdorApproach().step([0.]*6), [.8, .8])
        np.testing.assert_array_equal(OdorApproach('straight').step([.3, .1, 0., 0., 0., 0.]), [.8, .8])

    def test_reward_and_other_odor_cannot_steer_this_reference(self):
        a = OdorApproach().step([.2, .199, 0., 0., 0., 0.])
        b = OdorApproach().step([.2, .199, 1., .7, 1., 0.])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, [.8-24*.001/(.399+1e-12), .8+24*.001/(.399+1e-12)], atol=1e-14)

    def test_source_contact_stops_and_latches_even_without_sugar(self):
        policy = OdorApproach()
        np.testing.assert_array_equal(policy.step([.3, .1, 0., 0., 0., 1.]), [0., 0.])
        np.testing.assert_array_equal(policy.step([.3, .1, 0., 0., 0., 0.]), [0., 0.])
        self.assertTrue(policy.contacted)

    def test_complete_counterfactual_inventory_and_field_invariance(self):
        inventory = cases()
        self.assertEqual(len(inventory), 7)
        self.assertEqual(len({c.label for c in inventory}), 7)
        self.assertEqual(inventory[-1].repeat_of, inventory[0].label)
        antennae = np.array([[1., .1, 1.], [1., -.1, 1.]])
        feet = np.array([[8., 3., .1]]*6)
        observed = [c.field().observe(antennae, feet, missing_odor=c.missing_odor) for c in inventory]
        for index in (2, 3):
            np.testing.assert_array_equal(observed[0]['sensory'][:4], observed[index]['sensory'][:4])
            self.assertEqual(observed[index]['sensory'][4], 0.)
            self.assertEqual(observed[index]['sensory'][5], 1.)
        self.assertEqual(observed[0]['sensory'][4], 1.)
        np.testing.assert_array_equal(observed[4]['sensory'][:4], np.zeros(4))
        np.testing.assert_array_equal(observed[4]['sensory'][4:], [1., 1.])
        np.testing.assert_allclose(observed[1]['sensory'][:2], observed[0]['sensory'][1::-1])
        self.assertEqual(inventory[0].field().card(), inventory[-1].field().card())

    def test_malformed_sensor_vectors_are_rejected(self):
        for values in ([0.]*5, [0.]*7, [[0.]*6], [float('nan')]*6, [float('inf')]*6, [-.1]*6, [1.1]*6):
            with self.assertRaises(ValueError): OdorApproach().step(values)
        with self.assertRaises(ValueError): OdorApproach('target-coordinates')


if __name__ == '__main__': unittest.main()
