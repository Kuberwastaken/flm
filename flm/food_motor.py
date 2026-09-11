"""Declared probability-to-walking interface, independent of source locations."""
import numpy as np

BASE_DRIVE = .8
MAX_TURN = .4


class FoodMotor:
    def __init__(self):
        self.contacted = False

    def step(self, sensory, probabilities):
        sensory = np.asarray(sensory, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if (sensory.shape != (6,) or not np.isfinite(sensory).all()
                or np.any((sensory < 0) | (sensory > 1))):
            raise ValueError('Six finite normalized sensory channels required')
        if (probabilities.shape != (3,) or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))
                or not np.isclose(probabilities.sum(), 1., atol=1e-6, rtol=0)):
            raise ValueError('Three normalized right/straight/left probabilities required')
        self.contacted = self.contacted or bool(sensory[5] >= .5)
        turn = MAX_TURN * float(probabilities[2] - probabilities[0])
        return np.zeros(2, dtype=np.float64) if self.contacted else np.array(
            [BASE_DRIVE - turn, BASE_DRIVE + turn], dtype=np.float64)


def motor_card():
    return dict(action_order=['turn_right', 'straight', 'turn_left'], base_drive=BASE_DRIVE,
        maximum_turn=MAX_TURN, formula='turn=0.4*(p_left-p_right); drive=[0.8-turn,0.8+turn]',
        stop='Latch source_contact >= 0.5; thereafter both drives are zero',
        core_after_contact='Continue one recurrent transition per sample, even after the motor stop latches',
        probability_sampling=False, location_or_reward_identity_input=False,
        scope='Engineered continuous motor mapping and contact stop, not learned walking or feeding.')
