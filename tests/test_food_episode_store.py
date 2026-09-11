"""Durable-boundary tests with synthetic dynamics; no FlyGym or food training."""
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flm.food_episode import FoodEpisode, run_episode
from flm.food_episode_store import canonical, episode_lease, stored_episode
from test_food_episode import CLOCK, REWARDS, ToyEnvironment, readout, weights


ENVIRONMENT = {'fixture':'action-dependent-toy-v1','seed':17}


def actor(*, learning=True, seed=29):
    w = weights()
    return FoodEpisode(w, readout(w), CLOCK, REWARDS, seed=seed,
                       binding={'environment':ENVIRONMENT,'core':'four-node-fixture'}, learning=learning)


class Environment(ToyEnvironment):
    def __init__(self, *, fail=False, contact=True, cleanup_failure=False):
        super().__init__(fail=fail,contact=contact)
        self.closed = False; self.cleanup_failure = cleanup_failure

    def identity(self): return copy.deepcopy(ENVIRONMENT)

    def close(self):
        self.closed = True
        if self.cleanup_failure: raise RuntimeError('synthetic cleanup failure')


class FoodEpisodeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)/'episode'

    def test_terminal_update_matches_unstored_run_and_retry_does_no_work(self):
        template = actor(); before = template.state(); environment = Environment()
        record = stored_episode(self.directory,template,lambda:environment)
        reference = actor(); result = run_episode(reference,Environment())
        self.assertEqual(record['readout'],reference.learner.state())
        self.assertEqual(record['rollout'],json.loads(canonical(result)))
        self.assertEqual(record['readout']['completed_episodes'],1)
        self.assertNotEqual(record['readout']['weight'],before['initial_readout']['weight'])
        self.assertEqual(template.state(),before); self.assertTrue(environment.closed)
        with patch('flm.food_episode_store.run_episode',side_effect=AssertionError('must not rerun')):
            cached = stored_episode(self.directory,actor(),lambda:self.fail('must not construct'))
        self.assertEqual(cached,record)

    def test_evaluation_keeps_readout_and_timeout_is_a_terminal_update(self):
        template = actor(learning=False)
        result = stored_episode(self.directory,template,Environment)
        self.assertEqual(result['readout'],template.initial_readout)
        result = stored_episode(self.directory.parent/'timeout',actor(),lambda:Environment(contact=False))
        self.assertEqual(result['rollout']['outcome']['status'],'timeout')
        self.assertEqual(result['rollout']['outcome']['decisions'],3)
        self.assertEqual(result['readout']['completed_episodes'],1)

    def test_simulator_failure_is_retained_without_retry_or_readout_update(self):
        template = actor(); environment = Environment(fail=True)
        result = stored_episode(self.directory,template,lambda:environment)
        self.assertEqual(result['rollout']['outcome']['status'],'failed')
        self.assertEqual(result['readout'],template.initial_readout)
        self.assertEqual(len(result['rollout']['samples']),1)
        self.assertEqual(result['rollout']['controller']['events'][-1]['kind'],'abort')
        self.assertTrue(environment.closed)
        self.assertEqual(stored_episode(self.directory,actor(),lambda:self.fail('failure must not retry')),result)

    def test_initial_contact_and_cleanup_failure_do_not_commit_learning(self):
        class Invalid(Environment):
            def reset(self): return self.observation(sugar=1.,contact=1.)
        initial = actor().initial_readout
        invalid = stored_episode(self.directory,actor(),Invalid)
        self.assertEqual(invalid['rollout']['outcome']['status'],'invalid_start')
        self.assertEqual(invalid['readout'],initial)
        failed = stored_episode(self.directory.parent/'cleanup',actor(),lambda:Environment(cleanup_failure=True))
        self.assertTrue(failed['rollout']['outcome']['updated'])
        self.assertEqual(failed['readout'],initial)
        self.assertEqual(failed['cleanup_error']['type'],'RuntimeError')

    def test_constructor_failure_and_wrong_environment_prevent_rollout(self):
        def broken(): raise RuntimeError('synthetic constructor failure')
        record = stored_episode(self.directory,actor(),broken)
        self.assertEqual(record['rollout']['samples'],[])
        self.assertEqual(record['rollout']['outcome']['error_type'],'RuntimeError')
        class Wrong(Environment):
            def identity(self): return {'fixture':'wrong'}
            def reset(self): raise AssertionError('wrong environment must not reset')
        environment = Wrong()
        record = stored_episode(self.directory.parent/'wrong',actor(),lambda:environment)
        self.assertIn('differs from the declared identity',record['rollout']['outcome']['message'])
        self.assertTrue(environment.closed)

    def test_cancellation_preserves_attempt_and_restarts_original_state(self):
        class Interrupted(Environment):
            def advance(self,drive,steps): raise KeyboardInterrupt('synthetic cancellation')
        template = actor(); before = template.state(); environment = Interrupted()
        with self.assertRaises(KeyboardInterrupt): stored_episode(self.directory,template,lambda:environment)
        self.assertTrue(environment.closed); self.assertEqual(template.state(),before)
        first = self.directory/'attempt-000001'; self.assertTrue((first/'start.json').exists())
        self.assertFalse((first/'result.json.gz').exists())
        result = stored_episode(self.directory,actor(),Environment)
        self.assertEqual(result['attempt'],2)
        self.assertEqual(result['readout']['completed_episodes'],1)
        self.assertEqual(result['readout']['aborted_episodes'],0)
        self.assertTrue((first/'interrupted.json').exists())
        reference = stored_episode(self.directory.parent/'reference',actor(),Environment)
        self.assertEqual(result['rollout'],reference['rollout'])
        self.assertEqual(result['readout'],reference['readout'])

    def test_actual_process_exit_releases_lease_and_restarts_original_inputs(self):
        source = """
import os,sys
from test_food_episode_store import actor,Environment
from flm.food_episode_store import stored_episode
class Exit(Environment):
    def advance(self,drive,steps): os._exit(29)
stored_episode(sys.argv[1],actor(),Exit)
"""
        env = dict(os.environ)
        root = Path(__file__).resolve().parents[1]
        env['PYTHONPATH'] = os.pathsep.join([str(root/'tests'),str(root)])
        child = subprocess.run([sys.executable,'-c',source,str(self.directory)],env=env,
                               capture_output=True,text=True,timeout=30)
        self.assertEqual(child.returncode,29,child.stderr)
        result = stored_episode(self.directory,actor(),Environment)
        self.assertEqual(result['attempt'],2)
        self.assertEqual(result['readout']['completed_episodes'],1)
        self.assertTrue((self.directory/'attempt-000001/interrupted.json').exists())

    def test_process_cancellation_after_atomic_result_does_not_repeat_update(self):
        import flm.food_episode_store as module
        original = module.write_new
        def after_commit(path,value,**kwargs):
            original(path,value,**kwargs)
            if path.name == 'result.json.gz': raise KeyboardInterrupt('after durable result')
        with patch.object(module,'write_new',side_effect=after_commit):
            with self.assertRaises(KeyboardInterrupt): stored_episode(self.directory,actor(),Environment)
        result = stored_episode(self.directory,actor(),lambda:self.fail('committed result must be reused'))
        self.assertEqual(result['attempt'],1)
        self.assertEqual(result['readout']['completed_episodes'],1)

    def test_interrupted_staging_file_is_preserved_and_never_accepted(self):
        import flm.food_episode_store as module
        original = module.os.replace
        def before_commit(source,target):
            if target.name == 'result.json.gz': raise KeyboardInterrupt('before atomic result')
            original(source,target)
        with patch.object(module.os,'replace',side_effect=before_commit):
            with self.assertRaises(KeyboardInterrupt): stored_episode(self.directory,actor(),Environment)
        pending = list((self.directory/'attempt-000001').glob('result.json.gz.pending-*'))
        self.assertEqual(len(pending),1); original_bytes = pending[0].read_bytes()
        result = stored_episode(self.directory,actor(),Environment)
        self.assertEqual(result['attempt'],2); self.assertEqual(result['readout']['completed_episodes'],1)
        self.assertEqual(pending[0].read_bytes(),original_bytes)

    def test_held_lock_refuses_writer_but_stale_lock_file_is_harmless(self):
        with episode_lease(self.directory):
            with self.assertRaisesRegex(RuntimeError,'writer holds'):
                stored_episode(self.directory,actor(),Environment)
        result = stored_episode(self.directory,actor(),Environment)
        self.assertEqual(result['attempt'],1)

    def test_changed_seed_core_readout_or_source_identity_cannot_resume(self):
        stored_episode(self.directory,actor(),Environment)
        changed = actor(); changed.initial_readout['baseline'] = .25
        for template in (actor(seed=30),actor(learning=False),changed):
            with self.assertRaises(ValueError): stored_episode(self.directory,template,Environment)
        path = self.directory/'identity.json'; identity = json.loads(path.read_bytes())
        identity['store_source_sha256'] = '0'*64; path.write_bytes(canonical(identity))
        with self.assertRaisesRegex(ValueError,'identity changed'):
            stored_episode(self.directory,actor(),Environment)

    def test_damaged_neural_samples_missing_observations_and_readout_are_rejected(self):
        original = stored_episode(self.directory,actor(),Environment)
        path = self.directory/'attempt-000001/result.json.gz'
        damaged = copy.deepcopy(original); damaged['rollout']['samples'][0]['control']['fast'][0] += .1
        omitted = copy.deepcopy(original); omitted['rollout']['samples'].pop()
        wrong = copy.deepcopy(original); wrong['readout']['completed_episodes'] += 1
        for record in (damaged,omitted,wrong):
            path.write_bytes(gzip.compress(canonical(record),mtime=0))
            with self.assertRaises(ValueError): stored_episode(self.directory,actor(),Environment)
        path.write_bytes(b'not a gzip record')
        with self.assertRaises(gzip.BadGzipFile): stored_episode(self.directory,actor(),Environment)

    def test_extra_attempt_after_terminal_and_inventory_gaps_are_rejected(self):
        stored_episode(self.directory,actor(),Environment)
        extra = self.directory/'attempt-000002'; extra.mkdir()
        with self.assertRaisesRegex(ValueError,'inconsistent inventory'):
            stored_episode(self.directory,actor(),Environment)
        extra.rename(self.directory/'attempt-000003')
        with self.assertRaisesRegex(ValueError,'inventory has a gap'):
            stored_episode(self.directory,actor(),Environment)

    def test_next_episode_carries_readout_once_and_resets_neural_state(self):
        first = stored_episode(self.directory,actor(),Environment)
        w = weights()
        next_actor = FoodEpisode(w,first['readout'],CLOCK,REWARDS,seed=30,
            binding={'environment':ENVIRONMENT,'core':'four-node-fixture'},learning=True)
        self.assertTrue(np.all(next_actor.core.fast == 0)); self.assertTrue(np.all(next_actor.core.slow == 0))
        second = stored_episode(self.directory.parent/'second',next_actor,Environment)
        self.assertEqual(second['readout']['completed_episodes'],2)
        self.assertEqual(second['rollout']['controller']['initial_readout'],first['readout'])


if __name__ == '__main__': unittest.main()
