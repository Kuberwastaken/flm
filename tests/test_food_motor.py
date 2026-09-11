import unittest
import numpy as np
from flm.food_motor import FoodMotor


class FoodMotorTests(unittest.TestCase):
    def test_calibrated_direction_and_probability_mixtures(self):
        motor = FoodMotor(); sensory = np.zeros(6)
        for probabilities, expected in (([1, 0, 0], [1.2, .4]), ([0, 1, 0], [.8, .8]),
                                        ([0, 0, 1], [.4, 1.2]), ([.2, .3, .5], [.68, .92])):
            np.testing.assert_allclose(motor.step(sensory, probabilities), expected, atol=1e-15, rtol=0)

    def test_stop_is_any_source_contact_not_sugar_and_is_latched(self):
        motor = FoodMotor()
        np.testing.assert_array_equal(motor.step([0, 0, 0, 0, 1, 0], [0, 1, 0]), [.8, .8])
        np.testing.assert_array_equal(motor.step([0, 0, 0, 0, 0, 1], [0, 0, 1]), [0, 0])
        np.testing.assert_array_equal(motor.step(np.zeros(6), [1, 0, 0]), [0, 0])
        np.testing.assert_array_equal(FoodMotor().step(np.zeros(6), [0, 1, 0]), [.8, .8])

    def test_invalid_call_does_not_latch_contact(self):
        for bad in ([0, 0, 0], [1, 1, 1], [-1, 1, 1], [float('nan'), 0, 1], [0, 1]):
            motor = FoodMotor()
            with self.assertRaises(ValueError): motor.step([0, 0, 0, 0, 0, 1], bad)
            self.assertFalse(motor.contacted)
        for bad in ([0]*5, [2]*6, [float('inf')]*6):
            with self.assertRaises(ValueError): FoodMotor().step(bad, [0, 1, 0])

    def test_convex_action_mixture_bounds_and_odor_independence(self):
        rng = np.random.default_rng(937)
        for _ in range(100):
            probabilities = rng.dirichlet(np.ones(3))
            sensory = rng.random(6); sensory[5] = 0
            actual = FoodMotor().step(sensory, probabilities)
            expected = probabilities @ np.array([[1.2, .4], [.8, .8], [.4, 1.2]])
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
            self.assertTrue(np.all((actual >= .4) & (actual <= 1.2)))
            np.testing.assert_array_equal(actual, FoodMotor().step(np.zeros(6), probabilities))


if __name__ == '__main__': unittest.main()
