import math
import unittest
import numpy as np

from experiments.embodiment.feedback import FeedbackController, bearing_error, target_at, cases


class RememberCue:
    def __init__(self):
        self.resets = 0; self.fast = np.zeros(2); self.slow = np.zeros(2)

    def reset(self):
        self.resets += 1; self.fast[:] = 0; self.slow[:] = 0

    def step(self, sensory):
        self.fast += sensory[:2]; self.slow = self.fast.copy()
        logits = self.fast.copy(); probabilities = np.exp(logits); probabilities /= probabilities.sum()
        return logits, probabilities


class FeedbackTests(unittest.TestCase):
    def test_world_bearing_wrap_and_pose_influence(self):
        self.assertAlmostEqual(bearing_error([0,0,1],0,[1,1]),math.pi/4)
        self.assertAlmostEqual(bearing_error([0,0,1],math.pi,[-1,-.01]),math.atan(.01))
        self.assertAlmostEqual(bearing_error([0,2,1],0,[1,1]),-math.pi/4)
        with self.assertRaises(ValueError): bearing_error([1,1,1],0,[1,1])

    def test_target_switch_is_independent_of_feedback_mode(self):
        np.testing.assert_array_equal(target_at('switch',.9999),[40,20])
        np.testing.assert_array_equal(target_at('switch',1),[40,-20])
        self.assertEqual(len(cases()),27)
        self.assertEqual(len({r['label'] for r in cases()}),27)

    def test_decision_has_exact_delay_and_uses_sampled_not_current_error(self):
        model=RememberCue(); loop=FeedbackController(model,'positive','live',[0,0,1],0)
        for frame in range(11):
            record, decision=loop.frame(frame*.005,[0,0,1],0 if frame==0 else math.pi)
            np.testing.assert_array_equal(record['sensory'], [1,0,0,0] if frame<2 else ([0,0,0,1] if frame==10 else [0,0,0,0]))
            if frame<10:
                self.assertIsNone(decision); np.testing.assert_array_equal(loop.signal,[1,1])
        self.assertEqual(decision['time_s'],.05)
        self.assertEqual(decision['sampled']['time_s'],0)
        self.assertEqual(decision['action'],0)
        np.testing.assert_array_equal(loop.signal,[.4,1.2])
        self.assertEqual(model.resets,1)
        loop.frame(.055,[0,0,1],math.pi)
        self.assertEqual(loop.sample['cue'],1); self.assertEqual(model.resets,2)

    def test_frozen_pose_never_reads_new_pose_but_sees_new_target(self):
        loop=FeedbackController(None,'switch','frozen',[0,0,1],0)
        for frame in range(221):
            loop.frame(frame*.005,[30,40,1],math.pi)
            if frame==198: self.assertEqual(loop.sample['cue'],0)
        self.assertEqual(loop.sample['cue'],1)
        self.assertEqual(loop.sample['position_mm'],[0,0,1])
        self.assertEqual(loop.sample['yaw_rad'],0)
        self.assertEqual(loop.sample['target_mm'],[40,-20])

    def test_deadband_and_initial_signal_match_scripted_and_neural_controllers(self):
        yaw=math.atan2(20,40)
        loops=[FeedbackController(model,'positive','live',[0,0,1],yaw) for model in (None,RememberCue())]
        for loop in loops:
            for frame in range(11): _, decision=loop.frame(frame*.005,[0,0,1],yaw if frame==0 else 2.)
            self.assertTrue(decision['straight_deadband'])
            np.testing.assert_array_equal(loop.signal,[1,1])

    def test_missing_frames_are_rejected(self):
        loop=FeedbackController(None,'positive','live',[0,0,1],0)
        with self.assertRaises(ValueError): loop.frame(.005,[0,0,1],0)


if __name__=='__main__': unittest.main()
