import unittest
import numpy as np
from flm.behavior_study import episode_batch, reversed_mapping


class BehaviorStudyTests(unittest.TestCase):
    def test_cues_are_balanced_then_absent_during_delay_and_query(self):
        sensory, labels = episode_batch(np.random.default_rng(17), 8, 12)
        self.assertEqual(sensory.shape, (8, 15, 4))
        self.assertEqual(int(labels.sum()), 4)
        np.testing.assert_array_equal(sensory[:, 0, :2].argmax(-1), labels)
        np.testing.assert_array_equal(sensory[:, 1, :2].argmax(-1), labels)
        self.assertTrue(np.all(sensory[:, 2:, :2] == 0))
        self.assertTrue(np.all(sensory[:, :-1, 3] == 0)); self.assertTrue(np.all(sensory[:, -1, 3] == 1))
        same, same_labels = episode_batch(np.random.default_rng(17), 8, 12)
        np.testing.assert_array_equal(sensory, same); np.testing.assert_array_equal(labels, same_labels)
        zero, _ = episode_batch(np.random.default_rng(17), 8, 12, zero_distractor=True)
        self.assertTrue(np.all(zero[:, :, 2] == 0))

    def test_reversal_occurs_only_at_declared_training_boundaries(self):
        self.assertEqual([reversed_mapping(i) for i in (0, 1, 300, 301, 600, 601, 900)], [False, False, False, True, True, False, False])


if __name__ == '__main__': unittest.main()
