"""Mathematical and lifecycle checks; these fixtures are not food-training results."""
import copy
import json
import unittest
import numpy as np
from flm.food_readout_learning import EpisodicReadout, Rule


WEIGHT = np.array([[.2,-.3],[.1,.4],[-.5,.2]],dtype=np.float64)
BIAS = np.array([.03,-.02,.01])


def policy(decay=1., maximum=5):
    return EpisodicReadout(WEIGHT, BIAS, Rule(.07,decay,.1,maximum))


def probabilities(weight,bias,x):
    logits = weight @ x+bias; values = np.exp(logits-max(logits))
    return values/values.sum()


def draw_for(probability,action):
    return float(probability[:action].sum()+probability[action]/2)


def numerical_gradient(function,weight=WEIGHT,bias=BIAS):
    result=[]; epsilon=1e-6
    for value,other,is_weight in ((weight,bias,True),(bias,weight,False)):
        gradient=np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus=value.copy(); minus=value.copy(); plus[index]+=epsilon; minus[index]-=epsilon
            gradient[index]=((function(plus,other)-function(minus,other)) if is_weight else
                             (function(other,plus)-function(other,minus)))/(2*epsilon)
        result.append(gradient)
    return result


class ReadoutLearningTests(unittest.TestCase):
    def test_full_history_trace_matches_trajectory_log_likelihood_gradient(self):
        learner=policy(); learner.begin()
        inputs=[np.array([.4,-.7]),np.array([.8,.1]),np.array([-.2,.3])]; actions=[0,2,1]
        for x,a in zip(inputs,actions):
            result=learner.act(x,draw_for(learner.probabilities(x),a))
            self.assertEqual(result['action'],a)
        def objective(w,b): return sum(np.log(probabilities(w,b,x)[a]) for x,a in zip(inputs,actions))
        gw,gb=numerical_gradient(objective)
        np.testing.assert_allclose(learner.trace_weight,gw,atol=5e-10,rtol=0)
        np.testing.assert_allclose(learner.trace_bias,gb,atol=5e-10,rtol=0)
        # No update occurs before reward; input arrays are not aliased.
        np.testing.assert_array_equal(learner.weight,WEIGHT)
        np.testing.assert_array_equal(learner.bias,BIAS)
        learner.finish(.6)
        np.testing.assert_allclose(learner.weight,WEIGHT+.07*.6*gw,atol=5e-11,rtol=0)

    def test_expected_reward_gradient_in_action_dependent_two_step_environment(self):
        first=np.array([.2,-.4]); next_features=[np.array([.9,.1]),np.array([-.5,.7]),np.array([.3,-.8])]
        rewards=np.array([[1.,-.4,.2],[-.2,.8,-.7],[.3,-.5,.9]])
        def objective(w,b):
            p=probabilities(w,b,first)
            return sum(p[a]*probabilities(w,b,next_features[a])[b_action]*rewards[a,b_action]
                       for a in range(3) for b_action in range(3))
        expected_w,expected_b=numerical_gradient(objective)
        for baseline in (0.,.37,-.6):
            total_w=np.zeros_like(WEIGHT); total_b=np.zeros_like(BIAS)
            for a in range(3):
                for b in range(3):
                    learner=policy(); learner.begin()
                    p=learner.probabilities(first); learner.act(first,draw_for(p,a))
                    q=learner.probabilities(next_features[a]); learner.act(next_features[a],draw_for(q,b))
                    mass=p[a]*q[b]; advantage=rewards[a,b]-baseline
                    total_w+=mass*advantage*learner.trace_weight; total_b+=mass*advantage*learner.trace_bias
            np.testing.assert_allclose(total_w,expected_w,atol=2e-10,rtol=0)
            np.testing.assert_allclose(total_b,expected_b,atol=2e-10,rtol=0)

    def test_history_controls_keep_only_the_declared_credit(self):
        first=np.array([.8,-.3]); second=np.array([-.1,.9]); traces=[]
        for decay in (0.,.5,1.):
            learner=policy(decay); learner.begin(); learner.act(first,.1)
            before=(learner.trace_weight.copy(),learner.trace_bias.copy()); learner.act(second,.9)
            traces.append((learner.trace_weight.copy(),learner.trace_bias.copy()))
            if decay==1.: full_first=before
        for i in range(2):
            np.testing.assert_allclose(traces[2][i]-traces[0][i],full_first[i],atol=2e-16)
            np.testing.assert_allclose(traces[1][i],(traces[0][i]+traces[2][i])/2,atol=2e-16)

    def test_baseline_uses_only_previous_completed_rewards_and_abort_does_not_update(self):
        learner=policy(); learner.begin(); learner.act([.1,.4],.4)
        result=learner.finish(1.)
        self.assertEqual(result['baseline_before'],0.); self.assertEqual(result['baseline_after'],.1)
        learner.begin(); learner.act([.3,.6],.3); weight=learner.weight.copy(); baseline=learner.baseline
        learner.abort(); np.testing.assert_array_equal(learner.weight,weight); self.assertEqual(learner.baseline,baseline)
        learner.begin(); learner.act([.1,.4],.4); result=learner.finish(-1.)
        self.assertEqual(result['baseline_before'],.1); self.assertEqual(result['advantage'],-1.1)
        self.assertEqual(learner.completed_episodes,2); self.assertEqual(learner.aborted_episodes,1)
        self.assertFalse(np.any(learner.trace_weight)); self.assertFalse(learner.active)

    def test_json_resume_mid_episode_is_exact(self):
        learner=policy(); learner.begin(); learner.act([.4,.2],.63)
        resumed=EpisodicReadout.restore(json.loads(json.dumps(learner.state())))
        for actor in (learner,resumed):
            actor.act([-.2,.1],.2); actor.act([.9,.1],.9); actor.finish(.7)
        self.assertEqual(learner.state(),resumed.state())
        resumed.begin(); resumed.abort(); state=resumed.state()
        self.assertEqual(EpisodicReadout.restore(state).state(),state)

    def test_bad_inputs_and_budget_overrun_leave_state_unchanged(self):
        learner=policy(maximum=1)
        with self.assertRaises(ValueError): learner.act([.2,.1],.1)
        learner.begin(); before=copy.deepcopy(learner.state())
        for x,u in (([.2,.1],1.),([.2,.1],-.1),([.2,.1],True),([np.nan,.1],.2),([1,2,3],.2)):
            with self.assertRaises(ValueError): learner.act(x,u)
            self.assertEqual(before,learner.state())
        with self.assertRaises(ValueError): learner.finish(1.)
        learner.act([.2,.1],.1); before=copy.deepcopy(learner.state())
        with self.assertRaises(ValueError): learner.act([.2,.1],.2)
        with self.assertRaises(ValueError): learner.begin()
        for reward in (np.nan,np.inf,1.1,True):
            with self.assertRaises(ValueError): learner.finish(reward)
            self.assertEqual(before,learner.state())
        learner.abort()
        with self.assertRaises(ValueError): learner.finish(1.)

    def test_invalid_serialized_lifecycle_is_rejected(self):
        state=policy().state()
        for change in (lambda s:s.update(active=1),lambda s:s.update(baseline=2.),
                       lambda s:s.update(decisions=1),lambda s:s.update(completed_episodes=-1),
                       lambda s:s['trace_weight'][0].__setitem__(0,.1),
                       lambda s:s['rule'].update(trace_decay=1.1)):
            changed=copy.deepcopy(state); change(changed)
            with self.assertRaises(ValueError): EpisodicReadout.restore(changed)

    def test_extreme_finite_logits_skip_underflowed_actions(self):
        learner=EpisodicReadout(np.array([[-1000.,0.],[0.,0.],[1000.,0.]]),np.zeros(3),Rule(.01,1.,.1,1))
        learner.begin(); result=learner.act([1.,0.],0.)
        self.assertEqual(result['action'],2); np.testing.assert_array_equal(result['probabilities'],[0,0,1])


if __name__=='__main__': unittest.main()
