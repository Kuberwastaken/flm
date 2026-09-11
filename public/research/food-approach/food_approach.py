"""Declared scripted odor-approach reference; no model, fitting or hidden target.

Only the six FoodField sensor channels enter the policy. Source geometry and
reward identity belong to the environment and diagnostic scorer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from flm.food_sensors import FoodField, Source

CONTROL_STEP_S = .01
DURATION_S = 2.
GAIT_SEED = 17
BASE_DRIVE = .8
STEERING_GAIN = 24.
MAX_TURN = .4
CONTRAST_EPSILON = 1e-12


@dataclass(frozen=True)
class Case:
    label: str
    a_side: int = 1
    a_sugar: float = 1.
    b_sugar: float = 0.
    missing_odor: bool = False
    mode: str = 'odor'
    repeat_of: str | None = None

    def field(self):
        if self.a_side not in (-1, 1): raise ValueError('Source side must be -1 or 1')
        return FoodField((
            Source('a', (8., float(3*self.a_side), 0.), (5., 4., 2.), (1., 0.), self.a_sugar, .75, .25),
            Source('b', (8., float(-3*self.a_side), 0.), (5., 4., 2.), (0., 1.), self.b_sugar, .75, .25)))


def cases():
    return (
        Case('odor-a-left'),
        Case('odor-a-right', a_side=-1),
        Case('odor-a-left-reversed', a_sugar=0., b_sugar=1.),
        Case('odor-a-left-neutral', a_sugar=0.),
        Case('odor-a-left-missing', missing_odor=True),
        Case('straight-a-left', mode='straight'),
        Case('odor-a-left-repeat', repeat_of='odor-a-left'))


class OdorApproach:
    """A hardwired preference for odor A, with a latched any-source stop.

    The calibrated gait turns toward positive y when its second drive exceeds
    its first. Equal or absent odor gives straight walking, without search.
    Sugar and odor B are deliberately unused by this reference policy.
    """
    def __init__(self, mode='odor'):
        if mode not in ('odor', 'straight'): raise ValueError('Unknown scripted policy')
        self.mode = mode
        self.contacted = False

    def step(self, sensory):
        values = np.asarray(sensory, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Six finite normalized sensory channels required')
        self.contacted = self.contacted or bool(values[5] >= .5)
        if self.contacted: return np.zeros(2, dtype=np.float64)
        left, right = values[:2]
        contrast = (left-right) / (left+right+CONTRAST_EPSILON)
        turn = float(np.clip(STEERING_GAIN*contrast, -MAX_TURN, MAX_TURN)) if self.mode == 'odor' else 0.
        return np.array([BASE_DRIVE-turn, BASE_DRIVE+turn], dtype=np.float64)


def policy_card():
    return dict(name='Scripted bilateral odor-A approach with latched source-contact stop',
        used_channels=['odor_a_left', 'odor_a_right', 'source_contact'],
        unused_channels=['odor_b_left', 'odor_b_right', 'sugar_contact'],
        control_step_s=CONTROL_STEP_S, base_drive=BASE_DRIVE, steering_gain=STEERING_GAIN,
        maximum_turn=MAX_TURN, contrast_epsilon=CONTRAST_EPSILON,
        formula='contrast=(A_left-A_right)/(A_left+A_right+epsilon); turn=clip(gain*contrast,-maximum_turn,maximum_turn); drive=[base-turn,base+turn]',
        stop='Latch any-source contact >= 0.5; thereafter both descending drives are zero',
        missing_odor='Equal zero odor yields equal drives until any-source contact',
        coordinates_or_reward_identity_input=False, training_updates=0)
