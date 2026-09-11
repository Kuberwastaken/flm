"""Closed-loop/controller fixtures, not physical food-learning experiments."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from flm.food_episode import Clock, FoodEpisode, Rewards, array_digest, run_episode, sampled_drive
from flm.food_motor import FoodMotor
from flm.food_readout_learning import EpisodicReadout, Rule


CLOCK = Clock(.01,2,6)
REWARDS = Rewards(2.,-1.,-.25)
BINDING = {'model':'artificial-four-node-core','environment':'artificial-action-dependent-fixture'}


def weights():
    rng = np.random.default_rng(19)
    return dict(recurrent=np.array([[0,.5,0,0],[0,0,-.3,0],[0,0,0,.4],[.6,0,0,0]],dtype=np.float32),
        input_weight=rng.normal(0,.2,(4,3)).astype(np.float32),input_bias=np.zeros(4,dtype=np.float32),
        alpha=np.full(4,.3,dtype=np.float32),beta=np.full(4,.04,dtype=np.float32),gain=np.array(1.,dtype=np.float32),
        pool_index=np.array([0,1,0,1],dtype=np.int64),pool_sizes=np.array([2,2],dtype=np.float32),
        norm_weight=np.ones(4,dtype=np.float32),norm_bias=np.zeros(4,dtype=np.float32),norm_epsilon=np.array(1e-5,dtype=np.float32),
        sensor_weight=rng.normal(0,.2,(3,6)).astype(np.float32),sensor_bias=np.zeros(3,dtype=np.float32),
        action_weight=rng.normal(0,.1,(3,4)).astype(np.float32),action_bias=np.zeros(3,dtype=np.float32),
        body_ids=np.arange(4,dtype=np.int64))


def readout(w=None):
    w = weights() if w is None else w
    return EpisodicReadout(w['action_weight'],w['action_bias'],Rule(.03,1.,.1,3)).state()


def actor(learning=True):
    w = weights()
    return FoodEpisode(w,readout(w),CLOCK,REWARDS,seed=29,binding=BINDING,learning=learning)


class ToyEnvironment:
    """The second sensor observation depends on the actually supplied action."""
    def __init__(self, *, fail=False, contact=True, diagnostic='original'):
        self.physics_step = 0; self.calls = []; self.fail = fail
        self.contact = contact; self.diagnostic = diagnostic

    def observation(self, sugar=0., contact=0., odor=.3):
        return dict(sensory=[odor,.2,.1,.4,sugar,contact],
                    record=dict(physics_step=self.physics_step,diagnostic=self.diagnostic))

    def reset(self):
        self.physics_step = 0
        return self.observation()

    def advance(self, drive, steps):
        self.calls.append((drive.copy(),steps)); self.physics_step += steps
        if self.fail: raise RuntimeError('synthetic simulator failure')
        right = float(drive[0] > drive[1])
        return self.observation(sugar=right if self.contact else 0.,
                                contact=float(self.contact),odor=.2+.2*right)


class FoodEpisodeTests(unittest.TestCase):
    def test_discrete_drives_match_existing_motor_on_one_hot_actions(self):
        for action in range(3):
            probability = np.eye(3)[action]
            np.testing.assert_array_equal(sampled_drive(action),FoodMotor().step(np.zeros(6),probability))
        for invalid in (True,-1,3,.5):
            with self.assertRaises(ValueError): sampled_drive(invalid)

    def test_clock_and_readout_budget_are_explicit(self):
        for clock in (Clock(0,2,6),Clock(float('nan'),2,6),Clock(.01,2,5),Clock(.01,True,6),Clock(1e308,2,6)):
            with self.assertRaises(ValueError): FoodEpisode(weights(),readout(),clock,REWARDS,seed=29,binding=BINDING,learning=True)
        for rewards in (Rewards(3,-1,0),Rewards(1,-2,0),Rewards(2,-1,float('nan'))):
            with self.assertRaises(ValueError): rewards.validate()
        with self.assertRaisesRegex(ValueError,'decision budget'):
            FoodEpisode(weights(),readout(),Clock(.01,2,8),REWARDS,seed=29,binding=BINDING,learning=True)

    def test_terminal_contact_credits_only_preceding_applied_action(self):
        policy = actor(); initial = policy.learner.state()
        first = policy.observe([.2,.1,.3,.4,0,0],0)
        prior_trace = policy.learner.trace_weight.copy()
        sampler = copy.deepcopy(policy.sampler.bit_generator.state)
        terminal = policy.observe([.9,.8,.7,.6,1,1],2)
        self.assertIsNone(terminal['action']); self.assertIsNone(terminal['uniform'])
        np.testing.assert_array_equal(terminal['descending_signal'],[0,0])
        self.assertEqual(terminal['outcome']['decisions'],1)
        self.assertEqual(terminal['outcome']['reward'],1)
        np.testing.assert_allclose(policy.learner.weight,np.array(initial['weight'])+.03*prior_trace,atol=0,rtol=0)
        self.assertEqual(policy.sampler.bit_generator.state,sampler)
        self.assertFalse(np.array_equal(first['fast'],terminal['fast']))
        with self.assertRaisesRegex(ValueError,'terminated'): policy.observe([0]*6,4)

    def test_timeout_has_exactly_three_applied_actions_and_no_final_draw(self):
        policy = actor(); original = policy.learner.weight.copy()
        for step in (0,2,4):
            result = policy.observe([.1,.2,.3,.4,0,0],step)
            self.assertIsNotNone(result['action']); np.testing.assert_array_equal(policy.learner.weight,original)
        terminal = policy.observe([.1,.2,.3,.4,0,0],6)
        self.assertEqual(terminal['outcome']['decisions'],3)
        self.assertEqual(terminal['outcome']['reward'],-.25)
        self.assertEqual(policy.status,'timeout')
        rng = np.random.Generator(np.random.PCG64(29)); rng.random(3)
        self.assertEqual(policy.sampler.bit_generator.state,rng.bit_generator.state)

    def test_contact_at_horizon_has_precedence_and_initial_contact_does_not_train(self):
        policy = actor()
        for step in (0,2,4): policy.observe([0]*6,step)
        result = policy.observe([0,0,0,0,1,1],6)
        self.assertEqual(result['outcome']['status'],'contact'); self.assertEqual(result['outcome']['reward'],1)
        policy = actor(); original = policy.learner.weight.copy(); rng = copy.deepcopy(policy.sampler.bit_generator.state)
        result = policy.observe([0,0,0,0,1,1],0)
        self.assertEqual(result['outcome']['status'],'invalid_start'); self.assertIsNone(result['outcome']['reward'])
        self.assertFalse(result['outcome']['updated']); self.assertEqual(policy.learner.aborted_episodes,1)
        np.testing.assert_array_equal(original,policy.learner.weight); self.assertEqual(rng,policy.sampler.bit_generator.state)

    def test_evaluation_preserves_parameters_baseline_and_training_counters(self):
        trained = actor(); evaluated = actor(False); before = evaluated.initial_readout
        for step,sensory in ((0,[.2,.1,.3,.4,0,0]),(2,[.2,.1,.3,.4,1,1])):
            a = trained.observe(sensory,step); b = evaluated.observe(sensory,step)
            for key in ('action','uniform','probabilities','descending_signal'):
                self.assertEqual(a[key],b[key])
            np.testing.assert_array_equal(a['fast'],b['fast'])
        self.assertEqual(evaluated.learner.state(),before)
        self.assertFalse(evaluated.outcome['updated']); self.assertTrue(trained.outcome['updated'])

    def test_json_resume_rebuilds_core_sampler_and_eligibility_exactly(self):
        original = actor(); original.observe([.8,.1,.2,.3,0,0],0)
        restored = FoodEpisode.restore(weights(),json.loads(json.dumps(original.state())),binding=BINDING)
        for step,sensory in ((2,[.1,.8,.3,.2,0,0]),(4,[.8,.1,.2,.3,0,0]),(6,[0,0,0,0,1,1])):
            a = original.observe(sensory,step); b = restored.observe(sensory,step)
            for key in ('fast','slow','features'): np.testing.assert_array_equal(a.pop(key),b.pop(key))
            self.assertEqual(a,b)
        self.assertEqual(original.state(),restored.state())
        self.assertEqual(FoodEpisode.restore(weights(),original.state(),binding=BINDING).state(),original.state())

    def test_changed_event_state_source_binding_and_rng_fail_restoration(self):
        original = actor(); original.observe([.2,.3,.1,.4,0,0],0); state = original.state()
        for mutate in (lambda s:s['events'][0].update(action=(s['events'][0]['action']+1)%3),
                       lambda s:s['events'][0]['probabilities'].__setitem__(0,.999),
                       lambda s:s['fast'].__setitem__(0,.9),
                       lambda s:s['readout']['trace_bias'].__setitem__(0,99.),
                       lambda s:s['sampler']['state'].update(state=1)):
            changed = copy.deepcopy(state); mutate(changed)
            with self.assertRaises(ValueError): FoodEpisode.restore(weights(),changed,binding=BINDING)
        changed = weights(); changed['sensor_bias'][0] = .1
        with self.assertRaisesRegex(ValueError,'source'): FoodEpisode.restore(changed,state,binding=BINDING)
        with self.assertRaisesRegex(ValueError,'binding'): FoodEpisode.restore(weights(),state,binding={'other':True})

    def test_bad_observations_and_inference_failures_are_atomic(self):
        policy = actor(); policy.observe([.2,.1,.3,.4,0,0],0); before = policy.state()
        for sensory,step in (([0]*6,3),([0]*6,0),([0,0,0,0,1,0],2),([0,0,0,0,0,.5],2),([float('nan')]*6,2)):
            with self.assertRaises(ValueError): policy.observe(sensory,step)
            self.assertEqual(policy.state(),before)
        with patch.object(policy.learner,'act',side_effect=FloatingPointError('synthetic failure after core update')):
            with self.assertRaises(FloatingPointError): policy.observe([.1,.2,.3,.4,0,0],2)
        self.assertEqual(policy.state(),before)

    def test_failed_episode_retains_failure_without_reward_or_parameter_update(self):
        policy = actor(); initial = policy.learner.state(); result = run_episode(policy,ToyEnvironment(fail=True))
        self.assertEqual(result['outcome']['status'],'failed'); self.assertIsNone(result['outcome']['reward'])
        self.assertEqual(result['outcome']['decisions'],1); self.assertFalse(result['outcome']['updated'])
        for key in ('weight','bias','baseline','completed_episodes'):
            self.assertEqual(policy.learner.state()[key],initial[key])
        self.assertEqual(FoodEpisode.restore(weights(),result['controller'],binding=BINDING).state(),result['controller'])

    def test_closed_loop_uses_applied_drives_and_ignores_diagnostic_labels(self):
        first_env = ToyEnvironment(); second_env = ToyEnvironment(diagnostic={'rewarded_side':'changed','source_coordinates':[99,88,77]})
        first = run_episode(actor(),first_env); second = run_episode(actor(),second_env)
        self.assertEqual(len(first_env.calls),1); self.assertEqual(first_env.calls[0][1],2)
        self.assertEqual(first['controller'],second['controller'])
        sugar = float(first_env.calls[0][0][0]>first_env.calls[0][0][1])
        self.assertEqual(first['outcome']['reward'],2*sugar-1)
        self.assertEqual(len(first['samples']),2)
        self.assertEqual(first['samples'][-1]['control']['action'],None)

    def test_timeout_runs_full_action_intervals_and_invalid_start_runs_none(self):
        environment = ToyEnvironment(contact=False); result = run_episode(actor(),environment)
        self.assertEqual(len(environment.calls),3); self.assertEqual(len(result['samples']),4)
        self.assertEqual(result['outcome']['terminal_physics_step'],6)
        self.assertEqual(result['outcome']['status'],'timeout')
        environment = ToyEnvironment()
        environment.reset = lambda: environment.observation(sugar=1.,contact=1.)
        result = run_episode(actor(),environment)
        self.assertEqual(environment.calls,[]); self.assertEqual(result['outcome']['status'],'invalid_start')

    def test_next_episode_carries_learning_but_resets_core_and_action_rng(self):
        first = actor(); run_episode(first,ToyEnvironment())
        second = FoodEpisode(weights(),first.learner.state(),CLOCK,REWARDS,seed=29,binding=BINDING,learning=True)
        self.assertEqual(second.learner.completed_episodes,1)
        self.assertEqual(second.learner.baseline,first.learner.baseline)
        np.testing.assert_array_equal(second.core.fast,np.zeros(4,dtype=np.float32))
        np.testing.assert_array_equal(second.learner.weight,first.learner.weight)
        self.assertEqual(second.sampler.bit_generator.state,np.random.Generator(np.random.PCG64(29)).bit_generator.state)

    def test_source_arrays_and_numpy_global_rng_are_unchanged(self):
        source = weights(); original = array_digest(source); global_rng = np.random.get_state()
        policy = FoodEpisode(source,readout(source),CLOCK,REWARDS,seed=29,binding=BINDING,learning=True)
        run_episode(policy,ToyEnvironment())
        self.assertEqual(array_digest(source),original); self.assertEqual(array_digest(policy.core.weights),original)
        after = np.random.get_state()
        self.assertEqual(global_rng[0],after[0]); np.testing.assert_array_equal(global_rng[1],after[1])
        self.assertEqual(global_rng[2:],after[2:])

    def test_reward_retains_environment_precision_and_tiny_invalid_inputs_fail(self):
        policy = actor(); policy.observe([.1,.2,.3,.4,0,0],0)
        sugar = .123456789012345
        result = policy.observe([.1,.2,.3,.4,sugar,1],2)
        self.assertEqual(result['outcome']['reward'],2*sugar-1)
        self.assertEqual(result['sensory'][4],sugar)
        policy = actor()
        with self.assertRaises(ValueError): policy.observe([-1e-50,0,0,0,0,0],0)


if __name__ == '__main__': unittest.main()
