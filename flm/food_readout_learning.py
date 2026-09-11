"""Episodic reward-modulated softmax readout; no recurrent or body dynamics updates.

The full-history score trace gives the usual terminal-reward likelihood-ratio
estimator. Decaying or removing that history is an explicit, biased control.
All decisions in an episode use the same parameters. Uniform sampling inputs
are supplied by the caller, permitting matched random numbers across policies.
This module declares no physical training budget or food-learning result.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Rule:
    learning_rate: float
    trace_decay: float
    baseline_rate: float
    maximum_decisions: int

    def validate(self):
        for value in (self.learning_rate, self.trace_decay, self.baseline_rate):
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError('Finite numeric learning-rule parameters required')
        if self.learning_rate <= 0 or not 0 <= self.trace_decay <= 1 or not 0 <= self.baseline_rate <= 1:
            raise ValueError('Invalid learning rate, trace decay or baseline rate')
        if type(self.maximum_decisions) is not int or self.maximum_decisions < 1:
            raise ValueError('Positive integer decision budget required')


def finite_array(value, shape, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Invalid finite array: '+name)
    return array


class EpisodicReadout:
    """A three-action readout with an episode-lagged scalar reward baseline.

    Only weight/bias change. The caller owns recurrent state, physical sampling,
    rewards, source identities and episode reset. Failed episodes must be kept
    by the caller and aborted here, never recoded as successful rewards.
    """
    def __init__(self, weight, bias, rule):
        if type(rule) is not Rule: raise ValueError('An explicit Rule is required')
        rule.validate()
        source = np.asarray(weight)
        if source.ndim != 2 or source.shape[0] != 3 or source.shape[1] < 1:
            raise ValueError('Three actions and a nonempty feature vector required')
        self.rule = rule
        self.weight = finite_array(weight, source.shape, 'weight')
        self.bias = finite_array(bias, (3,), 'bias')
        self.baseline = 0.
        self.completed_episodes = 0; self.aborted_episodes = 0
        self.active = False; self.decisions = 0
        self.trace_weight = np.zeros_like(self.weight); self.trace_bias = np.zeros_like(self.bias)

    def probabilities(self, features):
        x = finite_array(features, (self.weight.shape[1],), 'features')
        with np.errstate(over='ignore', invalid='ignore'):
            logits = self.weight @ x + self.bias
        if not np.isfinite(logits).all(): raise FloatingPointError('Nonfinite policy logits')
        probability = np.exp(logits-logits.max()); probability /= probability.sum()
        return probability

    def begin(self):
        if self.active: raise ValueError('Finish or explicitly abort the active episode first')
        self.trace_weight.fill(0); self.trace_bias.fill(0); self.decisions = 0; self.active = True

    def act(self, features, uniform):
        if not self.active: raise ValueError('Begin an episode before accumulating decisions')
        if self.decisions >= self.rule.maximum_decisions: raise ValueError('Declared decision budget exceeded')
        if type(uniform) not in (float, int) or not np.isfinite(uniform) or not 0 <= uniform < 1:
            raise ValueError('Supply an explicit uniform draw in [0,1)')
        x = finite_array(features, (self.weight.shape[1],), 'features')
        probability = self.probabilities(x)
        # The final bin absorbs summation roundoff; zero-probability bins are skipped.
        cumulative = np.cumsum(probability); cumulative[-1] = 1.
        action = int(np.searchsorted(cumulative, uniform, side='right'))
        score = -probability; score[action] += 1.
        with np.errstate(over='ignore', invalid='ignore'):
            trace_weight = self.rule.trace_decay*self.trace_weight + np.outer(score, x)
            trace_bias = self.rule.trace_decay*self.trace_bias + score
        if not np.isfinite(trace_weight).all() or not np.isfinite(trace_bias).all():
            raise FloatingPointError('Nonfinite action eligibility')
        self.trace_weight = trace_weight; self.trace_bias = trace_bias; self.decisions += 1
        return dict(action=action, probabilities=probability.copy(), decision=self.decisions)

    def finish(self, reward):
        if not self.active or self.decisions == 0: raise ValueError('A nonempty active episode is required')
        if type(reward) not in (float, int) or not np.isfinite(reward) or not -1 <= reward <= 1:
            raise ValueError('A finite terminal reward in [-1,1] is required')
        # Baseline from earlier episodes is independent of this episode's actions.
        advantage = float(reward-self.baseline)
        with np.errstate(over='ignore', invalid='ignore'):
            weight = self.weight + self.rule.learning_rate*advantage*self.trace_weight
            bias = self.bias + self.rule.learning_rate*advantage*self.trace_bias
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise FloatingPointError('Nonfinite reward update; episode remains available for inspection')
        result = dict(reward=float(reward), baseline_before=self.baseline, advantage=advantage,
            decisions=self.decisions, trace_decay=self.rule.trace_decay,
            maximum_absolute_parameter_change=float(max(np.abs(weight-self.weight).max(), np.abs(bias-self.bias).max())))
        self.weight = weight; self.bias = bias
        self.baseline += self.rule.baseline_rate*advantage
        self.completed_episodes += 1; self.active = False
        self.trace_weight.fill(0); self.trace_bias.fill(0); self.decisions = 0
        result['baseline_after'] = self.baseline
        return result

    def abort(self):
        if not self.active: raise ValueError('No active episode to abort')
        result = dict(decisions=self.decisions, updated=False)
        self.active = False; self.decisions = 0; self.aborted_episodes += 1
        self.trace_weight.fill(0); self.trace_bias.fill(0)
        return result

    def state(self):
        return dict(format='flm-episodic-food-readout-v1',rule=asdict(self.rule),
            weight=self.weight.tolist(),bias=self.bias.tolist(),baseline=self.baseline,
            completed_episodes=self.completed_episodes,aborted_episodes=self.aborted_episodes,
            active=self.active,decisions=self.decisions,trace_weight=self.trace_weight.tolist(),trace_bias=self.trace_bias.tolist())

    @classmethod
    def restore(cls, state):
        expected = {'format','rule','weight','bias','baseline','completed_episodes','aborted_episodes',
                    'active','decisions','trace_weight','trace_bias'}
        if set(state) != expected or state['format'] != 'flm-episodic-food-readout-v1':
            raise ValueError('Unknown readout state inventory')
        result = cls(state['weight'], state['bias'], Rule(**state['rule']))
        for name in ('completed_episodes','aborted_episodes','decisions'):
            if type(state[name]) is not int or state[name] < 0: raise ValueError('Invalid episode counter')
        if (type(state['active']) is not bool or type(state['baseline']) not in (float,int)
                or not np.isfinite(state['baseline']) or not -1 <= state['baseline'] <= 1
                or state['decisions'] > result.rule.maximum_decisions):
            raise ValueError('Invalid baseline or active-episode state')
        result.trace_weight = finite_array(state['trace_weight'],result.weight.shape,'trace_weight')
        result.trace_bias = finite_array(state['trace_bias'],result.bias.shape,'trace_bias')
        if ((not state['active'] or state['decisions'] == 0)
                and (state['decisions'] != 0 or np.any(result.trace_weight) or np.any(result.trace_bias))):
            raise ValueError('Inactive or empty episode must have zero eligibility')
        for name in ('baseline','completed_episodes','aborted_episodes','active','decisions'): setattr(result,name,state[name])
        return result
