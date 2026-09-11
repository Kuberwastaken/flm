"""Sampled-action food episodes over a frozen sensory/core runtime.

This is the controller/environment boundary, not an official physical study.
No source coordinates, task phase or reward labels enter action selection.
Serialized controller state does not checkpoint a physical simulator or gait.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from .food_readout_learning import EpisodicReadout
from .food_motor import BASE_DRIVE, MAX_TURN


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        value = np.asarray(array)
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode('utf8'))
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


@dataclass(frozen=True)
class Clock:
    physics_timestep_s: float
    steps_per_action: int
    maximum_physics_steps: int

    def validate(self):
        if (type(self.physics_timestep_s) not in (int,float) or not np.isfinite(self.physics_timestep_s)
                or self.physics_timestep_s <= 0 or type(self.steps_per_action) is not int
                or self.steps_per_action < 1 or type(self.maximum_physics_steps) is not int
                or self.maximum_physics_steps < self.steps_per_action
                or self.maximum_physics_steps % self.steps_per_action):
            raise ValueError('Declare a positive finite clock and an integer number of complete action intervals')
        try:
            duration = self.physics_timestep_s*self.maximum_physics_steps
        except OverflowError as error:
            raise ValueError('Episode duration must be finite') from error
        if not np.isfinite(duration): raise ValueError('Episode duration must be finite')


@dataclass(frozen=True)
class Rewards:
    sugar_scale: float
    contact_offset: float
    timeout: float

    def validate(self):
        if any(type(value) not in (int,float) or not np.isfinite(value) for value in asdict(self).values()):
            raise ValueError('Finite explicit terminal reward parameters required')
        if (self.sugar_scale < 0 or not -1 <= self.contact_offset <= 1
                or not -1 <= self.contact_offset+self.sugar_scale <= 1 or not -1 <= self.timeout <= 1):
            raise ValueError('Terminal rewards must remain in [-1,1] for every contact intensity')


def sampled_drive(action):
    if type(action) is not int or action not in (0,1,2):
        raise ValueError('Expected sampled right, straight or left action')
    turn = MAX_TURN*(action-1)
    return np.array([BASE_DRIVE-turn, BASE_DRIVE+turn], dtype=np.float64)


class FoodEpisode:
    def __init__(self, weights, initial_readout, clock, rewards, *, seed, binding, learning):
        if type(clock) is not Clock or type(rewards) is not Rewards:
            raise ValueError('Explicit Clock and Rewards required')
        clock.validate(); rewards.validate()
        if (type(seed) is not int or not 0 <= seed < 2**64 or type(learning) is not bool
                or not isinstance(binding,dict) or not binding):
            raise ValueError('Explicit sampling seed, nonempty source binding and learning flag required')
        self.core = FoodCoreRuntime(weights)  # Owns independent, read-only copies.
        self.learner = EpisodicReadout.restore(plain(initial_readout))
        if (self.learner.active or self.learner.weight.shape != self.core.weights['action_weight'].shape
                or self.learner.rule.maximum_decisions != clock.maximum_physics_steps//clock.steps_per_action):
            raise ValueError('Inactive readout dimensions and decision budget must match the episode')
        self.initial_readout = plain(self.learner.state())
        self.clock = clock; self.rewards = rewards; self.learning = learning
        self.sampler = np.random.Generator(np.random.PCG64(seed))
        source_paths = [Path(__file__), Path(__file__).with_name('food_readout_learning.py'),
            Path(__file__).with_name('food_motor.py'), Path(__file__).resolve().parents[1]/'experiments/embodiment/food_core_runtime.py']
        self.identity = plain(dict(format='flm-food-episode-v1', clock=asdict(clock), rewards=asdict(rewards),
            seed=seed, learning=learning, binding=binding, core_weights_sha256=array_digest(self.core.weights),
            sources={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            numpy=str(np.__version__), action_order=['turn_right','straight','turn_left'],
            motor=dict(base_drive=BASE_DRIVE,maximum_turn=MAX_TURN),
            terminal='Observe contact/horizon before sampling; terminal states receive no action credit'))
        self.events = []; self.status = 'active'; self.last_step = None; self.outcome = None
        self.learner.begin()

    def observe(self, sensory, physics_step):
        if self.status != 'active': raise ValueError('Episode has already terminated')
        expected = 0 if self.last_step is None else self.last_step+self.clock.steps_per_action
        if type(physics_step) is not int or physics_step != expected or physics_step > self.clock.maximum_physics_steps:
            raise ValueError('Observation does not follow the declared complete action clock')
        # Preserve the environment's contact intensity for terminal reward.
        # The frozen core performs its own declared float32 sensory conversion.
        x = np.array(sensory,dtype=np.float64,copy=True)
        if (x.shape != (6,) or not np.isfinite(x).all() or np.any((x < 0)|(x > 1))
                or x[5] not in (0.,1.) or (x[5] == 0 and x[4] != 0)):
            raise ValueError('Six normalized channels with binary source contact and consistent sugar required')
        old_fast = self.core.fast.copy(); old_slow = self.core.slow.copy()
        old_readout = self.learner.state(); old_rng = plain(self.sampler.bit_generator.state)
        status = 'active'; outcome = None; action = None; uniform = None; update = None
        try:
            _, _, features = self.core.step(x)
            probabilities = self.learner.probabilities(features)
            # The preceding interval has already acted. This observation can
            # terminate it, but cannot add an action that will never be applied.
            if bool(x[5]) or physics_step == self.clock.maximum_physics_steps:
                status = 'contact' if bool(x[5]) else 'timeout'
                reward = self.rewards.contact_offset+self.rewards.sugar_scale*float(x[4]) if bool(x[5]) else self.rewards.timeout
                decisions = self.learner.decisions
                if decisions == 0:
                    status = 'invalid_start'; reward = None
                    if self.learning: self.learner.abort()
                elif self.learning:
                    update = self.learner.finish(reward)
                if not self.learning:
                    self.learner = EpisodicReadout.restore(self.initial_readout)
                outcome = dict(status=status, reward=reward, decisions=decisions,
                    contact_sugar=float(x[4]) if bool(x[5]) else None,
                    terminal_physics_step=physics_step, time_s=physics_step*self.clock.physics_timestep_s,
                    update=update, updated=update is not None)
                drive = np.zeros(2,dtype=np.float64)
            else:
                uniform = float(self.sampler.random())
                result = self.learner.act(features,uniform)
                action = result['action']; probabilities = result['probabilities']
                drive = sampled_drive(action)
            event = plain(dict(kind='observation', physics_step=physics_step, sensory=x.tolist(),
                probabilities=probabilities.tolist(), action=action, uniform=uniform,
                descending_signal=drive.tolist(), status=status, outcome=outcome))
        except Exception:
            self.core.fast = old_fast; self.core.slow = old_slow
            self.learner = EpisodicReadout.restore(old_readout); self.sampler.bit_generator.state = old_rng
            raise
        self.last_step = physics_step; self.status = status; self.outcome = outcome; self.events.append(event)
        return dict(event, time_s=physics_step*self.clock.physics_timestep_s,
            fast=self.core.fast.copy(), slow=self.core.slow.copy(), features=features.copy())

    def abort(self, error_type, message):
        if self.status != 'active' or not isinstance(error_type,str) or not error_type or not isinstance(message,str):
            raise ValueError('An active episode and explicit failure description are required')
        decisions = self.learner.decisions
        if self.learning: self.learner.abort()
        else: self.learner = EpisodicReadout.restore(self.initial_readout)
        self.status = 'failed'
        self.outcome = dict(status='failed',reward=None,decisions=decisions,updated=False,update=None,
                            last_observed_physics_step=self.last_step,error_type=error_type,message=message)
        self.events.append(dict(kind='abort',error_type=error_type,message=message))
        return plain(self.outcome)

    def state(self):
        if array_digest(self.core.weights) != self.identity['core_weights_sha256']:
            raise ValueError('Frozen food core or sensory interface changed')
        return plain(dict(identity=self.identity,initial_readout=self.initial_readout,events=self.events,
            readout=self.learner.state(),sampler=self.sampler.bit_generator.state,
            fast=self.core.fast.tolist(),slow=self.core.slow.tolist(),status=self.status,
            last_step=self.last_step,outcome=self.outcome))

    @classmethod
    def restore(cls, weights, state, *, binding):
        identity = state['identity']
        actor = cls(weights,state['initial_readout'],Clock(**identity['clock']),Rewards(**identity['rewards']),
                    seed=identity['seed'],binding=binding,learning=identity['learning'])
        if actor.identity != identity: raise ValueError('Food episode source, environment binding or software changed')
        # Rebuild every sensory transition, sampled action and eligibility update.
        # This verifies controller state; it cannot reconstruct physical dynamics.
        for event in state['events']:
            if event['kind'] == 'observation': actor.observe(event['sensory'],event['physics_step'])
            elif event['kind'] == 'abort': actor.abort(event['error_type'],event['message'])
            else: raise ValueError('Unknown food episode event')
            if actor.events[-1] != event: raise ValueError('Food episode event does not replay')
        if actor.state() != state: raise ValueError('Food episode state differs from full event replay')
        return actor


def run_episode(actor, environment):
    """Run a fresh episode through a caller-owned environment, retaining failures.

    reset()/advance(drive, steps) return {sensory, record}; record is finite JSON
    diagnostic data, never a policy input. physics_step is the elapsed integer
    step after reset/warmup. The caller owns environment construction/cleanup.
    Full simulator and gait checkpoint/resumption are outside this function.
    """
    if not isinstance(actor,FoodEpisode) or actor.events or actor.status != 'active':
        raise ValueError('The rollout runner requires a fresh controller episode')
    samples = []
    try:
        observation = environment.reset()
        while True:
            if not isinstance(observation,dict) or set(observation) != {'sensory','record'}:
                raise ValueError('Environment must return sensory channels and a separate physical record')
            physical = plain(observation['record'])
            control = actor.observe(observation['sensory'],environment.physics_step)
            samples.append(dict(control=control,environment=physical))
            if actor.status != 'active': break
            observation = environment.advance(np.array(control['descending_signal']),actor.clock.steps_per_action)
    except Exception as error:
        actor.abort(type(error).__name__,str(error))
    return dict(outcome=plain(actor.outcome),samples=samples,controller=actor.state(),
        scope='A controller/environment rollout; physical validity and official study membership belong to the caller')
