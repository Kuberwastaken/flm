"""Complete schedule fixtures with toy transitions, not physical study results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flm.food_episode_store import read
from flm.food_schedule import declaration, evaluate, initialize, train
from test_food_episode import CLOCK, REWARDS, weights
from test_food_episode_store import Environment


def fixture():
    initial = weights(); trained = {name:value.copy() for name,value in initial.items()}
    trained['alpha'] += np.float32(.02)
    cores = {'initial':initial,'language':trained}
    environments = {name:dict(fixture='two-action-reward-v1',scene=name)
                    for name in ('train-a','train-b','probe-a','probe-b')}
    streams = []
    for seed in (29,30):
        streams.append(dict(id=f'seed-{seed}',
            training=[dict(id='acquire',phase='acquisition',environment='train-a',action_seed=seed),
                      dict(id='reverse',phase='reversal',environment='train-b',action_seed=seed+100)],
            probes=[dict(id='original',phase='original-association',environment='probe-a',action_seed=seed+200),
                    dict(id='reversed',phase='reversed-association',environment='probe-b',action_seed=seed+300)],
            checkpoints=[0,1,2]))
    request = dict(purpose='Synthetic schedule execution checks; no physical training',
        protocol=dict(fixture=True,phase_order=['acquisition','reversal'],model_size=4),
        clock=vars(CLOCK),rewards=vars(REWARDS),
        models=[dict(id=name,pair='seed-19',role=name,source={'fixture':'four-node-core','origin':name}) for name in cores],
        methods=[dict(id=name,learning=True,rule=dict(learning_rate=.03,trace_decay=decay,baseline_rate=.1,maximum_decisions=3))
                 for name,decay in (('full',1.),('no-history',0.))],
        environments=environments,streams=streams)
    return request,cores


class ScheduledEnvironment(Environment):
    def __init__(self,identity,*,failure=None,on_close=None):
        super().__init__(); self.declared = identity; self.failure = failure; self.on_close = on_close

    def identity(self): return copy.deepcopy(self.declared)

    def advance(self,drive,steps):
        self.calls.append((drive.copy(),steps)); self.physics_step += steps
        if self.failure is not None: raise self.failure
        right = float(drive[0]>drive[1]); contact = self.physics_step >= 4
        sugar = right if self.declared['scene'].endswith('a') else 1-right
        return self.observation(sugar=sugar if contact else 0.,contact=float(contact),odor=.2+.2*right)

    def close(self):
        super().close()
        if self.on_close is not None: self.on_close()


class FoodScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'schedule'; self.request,self.weights = fixture(); self.calls = []

    def factory(self,name):
        self.calls.append(name)
        return ScheduledEnvironment(self.request['environments'][name])

    def frozen(self): return initialize(self.path,self.request,self.weights)

    def trained(self):
        self.frozen(); return train(self.path,self.request,self.weights,self.factory)

    def forbidden(self,name): self.fail('Unexpected environment construction: '+name)

    def test_full_cartesian_inventory_requires_complete_paired_interfaces(self):
        identity = declaration(self.request,self.weights)
        self.assertEqual(len(identity['conditions']),8)
        bad = copy.deepcopy(self.request); bad['models'].pop()
        with self.assertRaisesRegex(ValueError,'paired language core'):
            declaration(bad,{'initial':self.weights['initial']})
        changed = copy.deepcopy(self.weights); changed['language']['sensor_weight'][0,0] += .1
        with self.assertRaisesRegex(ValueError,'share the sensory/action interface'):
            declaration(self.request,changed)
        bad = copy.deepcopy(self.request); bad['models'][1]['role']='initial'
        with self.assertRaisesRegex(ValueError,'repeated roles'): declaration(bad,self.weights)

    def test_explicit_phase_order_checkpoint_and_inventory_validation(self):
        requests = []
        bad = copy.deepcopy(self.request); bad['streams'][1]['training'][0]['phase']='other'; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['streams'][0]['checkpoints']=[1,2]; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['streams'][0]['checkpoints']=[0,2,1]; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['streams'][0]['checkpoints']=[0,True,2]; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['methods'][0]['rule']['maximum_decisions']=4; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['environments']['unused']={'scene':'unused'}; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['streams'][0]['training'][0]['id']='../escape'; requests.append(bad)
        bad = copy.deepcopy(self.request); bad['streams'][0]['training'][0]['action_seed']=True; requests.append(bad)
        for request in requests:
            with self.assertRaises(ValueError): declaration(request,self.weights)

    def test_frozen_request_or_core_cannot_change(self):
        identity = self.frozen(); self.assertEqual(initialize(self.path,self.request,self.weights),identity)
        changed = copy.deepcopy(self.request); changed['methods'][0]['rule']['learning_rate']=.1
        with self.assertRaisesRegex(ValueError,'different food schedule'): initialize(self.path,changed,self.weights)
        with self.assertRaisesRegex(ValueError,'Frozen food schedule'): train(self.path,changed,self.weights,self.forbidden)
        changed_weights = copy.deepcopy(self.weights); changed_weights['language']['alpha'] += .01
        with self.assertRaisesRegex(ValueError,'Frozen food schedule'): train(self.path,self.request,changed_weights,self.forbidden)

    def test_bounded_training_and_whole_inventory_gate_before_any_probe(self):
        self.frozen()
        with self.assertRaisesRegex(ValueError,'Complete all declared'): evaluate(self.path,self.request,self.weights,self.forbidden)
        first = train(self.path,self.request,self.weights,self.factory,max_new=3)
        self.assertEqual(first,dict(status='partial',new_episodes=3))
        self.assertEqual(len(self.calls),3); self.assertFalse((self.path/'training-complete.json').exists())
        with self.assertRaisesRegex(ValueError,'Complete all declared'): evaluate(self.path,self.request,self.weights,self.forbidden)
        self.assertFalse((self.path/'evaluation').exists())
        second = train(self.path,self.request,self.weights,self.factory)
        self.assertEqual(second,dict(status='complete',new_episodes=13,conditions=8))
        self.assertEqual(self.calls,['train-a','train-b']*8)

    def test_all_probes_use_fixed_checkpoints_and_never_change_training(self):
        self.trained(); original = (self.path/'training-complete.json').read_bytes()
        result = evaluate(self.path,self.request,self.weights,self.factory)
        self.assertEqual(result,dict(status='complete',new_episodes=48,evaluated_episodes=48))
        self.assertEqual(original,(self.path/'training-complete.json').read_bytes())
        self.assertEqual(self.calls,['train-a','train-b']*8+['probe-a','probe-b']*24)
        trained = json.loads(original)['records']
        for row in read(self.path/'evaluation-complete.json')['records']['episodes']:
            point = trained['checkpoints'][row['condition']['id']][str(row['checkpoint'])]
            self.assertFalse(row['outcome']['updated'])
            self.assertIsNone(row['cleanup_error'])
            from flm.food_episode_store import digest
            self.assertEqual(row['readout_sha256'],digest(point))

    def test_completed_schedules_resume_without_new_physics(self):
        self.trained(); evaluate(self.path,self.request,self.weights,self.factory)
        before = {name:(self.path/name).read_bytes() for name in ('training-complete.json','evaluation-complete.json')}
        self.assertEqual(train(self.path,self.request,self.weights,self.forbidden)['new_episodes'],0)
        self.assertEqual(evaluate(self.path,self.request,self.weights,self.forbidden)['new_episodes'],0)
        self.assertEqual(before,{name:(self.path/name).read_bytes() for name in before})

    def test_interrupted_episode_restarts_without_skipping_or_doubled_update(self):
        self.frozen()
        def interrupted(name): return ScheduledEnvironment(self.request['environments'][name],failure=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt): train(self.path,self.request,self.weights,interrupted)
        self.assertFalse((self.path/'training-complete.json').exists())
        train(self.path,self.request,self.weights,self.factory)
        record = read(self.path/'training-complete.json')['records']
        self.assertEqual(record['conditions'][0]['episodes'][0]['attempt'],2)
        self.assertEqual(record['checkpoints']['condition-000001']['1']['completed_episodes'],1)
        self.assertEqual(record['checkpoints']['condition-000001']['2']['completed_episodes'],2)
        self.assertTrue((self.path/'training/condition-000001/episode-000001/attempt-000001/interrupted.json').exists())

    def test_failed_training_slot_is_retained_without_retry_or_readout_update(self):
        self.frozen(); count = 0
        def fail_once(name):
            nonlocal count
            count += 1
            return ScheduledEnvironment(self.request['environments'][name],failure=RuntimeError('failed slot') if count==1 else None)
        train(self.path,self.request,self.weights,fail_once)
        training = read(self.path/'training-complete.json')['records']
        self.assertEqual(count,16)
        self.assertEqual(training['conditions'][0]['episodes'][0]['outcome']['status'],'failed')
        points = training['checkpoints']['condition-000001']
        self.assertEqual(points['0'],points['1']); self.assertEqual(points['2']['completed_episodes'],1)
        train(self.path,self.request,self.weights,self.forbidden)
        evaluate(self.path,self.request,self.weights,self.factory)
        self.assertEqual(read(self.path/'training-complete.json')['records'],training)

    def test_missing_training_terminal_refuses_evaluation_before_probe_construction(self):
        self.trained()
        path = next((self.path/'training/condition-000008/episode-000002').glob('attempt-*/result.json.gz'))
        path.rename(path.with_name('removed-result'))
        with self.assertRaisesRegex(ValueError,'Complete every declared training episode'):
            evaluate(self.path,self.request,self.weights,self.forbidden)
        self.assertFalse((self.path/'evaluation').exists())

    def test_changed_completion_checkpoint_is_rejected_before_evaluation(self):
        self.trained(); path=self.path/'training-complete.json'; record=read(path)
        record['records']['checkpoints']['condition-000001']['1']['weight'][0][0] += .1
        path.write_text(json.dumps(record),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'completion changed'):
            evaluate(self.path,self.request,self.weights,self.forbidden)
        self.assertFalse((self.path/'evaluation').exists())

    def test_unknown_training_episode_is_rejected(self):
        self.trained(); (self.path/'training/condition-000001/episode-999999').mkdir()
        with self.assertRaisesRegex(ValueError,'Unexpected food episode artifact'):
            evaluate(self.path,self.request,self.weights,self.forbidden)

    def test_bounded_evaluation_resumes_fixed_order_without_training_changes(self):
        self.trained(); before=(self.path/'training-complete.json').read_bytes()
        first=evaluate(self.path,self.request,self.weights,self.factory,max_new=3)
        self.assertEqual(first,dict(status='partial',new_episodes=3))
        self.assertFalse((self.path/'evaluation-complete.json').exists())
        second=evaluate(self.path,self.request,self.weights,self.factory)
        self.assertEqual(second,dict(status='complete',new_episodes=45,evaluated_episodes=48))
        self.assertEqual(before,(self.path/'training-complete.json').read_bytes())

    def test_extra_episode_added_during_last_probe_blocks_completion(self):
        self.trained(); count=0
        def late_extra(name):
            nonlocal count
            count+=1
            def on_close():
                if count==48: (self.path/'evaluation/condition-000008/episode-999999').mkdir()
            return ScheduledEnvironment(self.request['environments'][name],on_close=on_close)
        with self.assertRaisesRegex(ValueError,'Unexpected food episode artifact'):
            evaluate(self.path,self.request,self.weights,late_extra)
        self.assertFalse((self.path/'evaluation-complete.json').exists())

    def test_frozen_method_retains_readout_across_all_slots_and_checkpoints(self):
        self.request['methods'][0]['learning']=False
        self.trained(); result=read(self.path/'training-complete.json')['records']
        for condition in result['conditions']:
            if condition['condition']['method']=='full':
                points=result['checkpoints'][condition['condition']['id']]
                self.assertEqual(points['0'],points['1']); self.assertEqual(points['0'],points['2'])

    def test_late_core_change_prevents_training_and_evaluation_completion(self):
        self.frozen(); count=0
        def mutate_last_training(name):
            nonlocal count
            count+=1
            def on_close():
                if count==16: self.weights['language']['alpha'][0] += np.float32(.001)
            return ScheduledEnvironment(self.request['environments'][name],on_close=on_close)
        with self.assertRaisesRegex(ValueError,'Frozen food schedule'):
            train(self.path,self.request,self.weights,mutate_last_training)
        self.assertFalse((self.path/'training-complete.json').exists())
        self.request,self.weights=fixture()
        train(self.path,self.request,self.weights,self.factory)
        count=0
        def mutate_last_probe(name):
            nonlocal count
            count+=1
            def on_close():
                if count==48: self.weights['language']['alpha'][0] += np.float32(.001)
            return ScheduledEnvironment(self.request['environments'][name],on_close=on_close)
        with self.assertRaisesRegex(ValueError,'Frozen food schedule'):
            evaluate(self.path,self.request,self.weights,mutate_last_probe)
        self.assertFalse((self.path/'evaluation-complete.json').exists())

    def test_bounded_call_refuses_state_changed_during_its_last_episode(self):
        self.frozen()
        def mutated(name):
            def on_close(): self.weights['language']['alpha'][0] += np.float32(.001)
            return ScheduledEnvironment(self.request['environments'][name],on_close=on_close)
        with self.assertRaisesRegex(ValueError,'Frozen food schedule'):
            train(self.path,self.request,self.weights,mutated,max_new=1)
        self.assertFalse((self.path/'training-complete.json').exists())

    def test_earlier_artifact_changed_in_last_episode_blocks_completion(self):
        self.frozen(); count=0; original=None
        target=self.path/'training/condition-000001/episode-000001/attempt-000001/start.json'
        def mutate_last_training(name):
            nonlocal count
            count+=1
            def on_close():
                nonlocal original
                if count==16:
                    original=target.read_bytes(); target.write_bytes(original+b' ')
            return ScheduledEnvironment(self.request['environments'][name],on_close=on_close)
        with self.assertRaisesRegex(ValueError,'artifact changed during execution'):
            train(self.path,self.request,self.weights,mutate_last_training)
        self.assertFalse((self.path/'training-complete.json').exists())
        target.write_bytes(original)
        train(self.path,self.request,self.weights,self.factory)
        count=0
        def mutate_last_probe(name):
            nonlocal count
            count+=1
            def on_close():
                if count==48: target.write_bytes(original+b' ')
            return ScheduledEnvironment(self.request['environments'][name],on_close=on_close)
        with self.assertRaisesRegex(ValueError,'artifact changed during execution'):
            evaluate(self.path,self.request,self.weights,mutate_last_probe)
        self.assertFalse((self.path/'evaluation-complete.json').exists())


if __name__=='__main__': unittest.main()
