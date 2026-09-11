"""Short physical restart checks, with every food-learning update disabled.

Four cores each run a direct 20-ms episode, an interrupted 10-ms attempt, and
a fresh 20-ms restart. Exact agreement tests restarting initial conditions;
this does not checkpoint/resume a running simulator or demonstrate navigation.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.embodiment.food_approach import lower_own_priority
from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from experiments.embodiment.food_learning_environment import FoodLearningEnvironment
from flm.food_approach import cases
from flm.food_episode import Clock, FoodEpisode, Rewards
from flm.food_episode_store import stored_episode
from flm.food_readout_learning import EpisodicReadout, Rule


MODELS = ('initial-s42','language-s42','initial-s43','language-s43')
SOURCES = ('flm/food_episode_store.py','tests/test_food_episode_store.py',
    'experiments/embodiment/verify_food_episode_store.py','flm/food_episode.py',
    'experiments/embodiment/food_learning_environment.py')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path,value):
    with path.open('x',encoding='utf8',newline='\n') as handle:
        json.dump(value,handle,indent=2,allow_nan=False); handle.write('\n')


def run(bundle,output):
    if json.loads((bundle/'manifest.json').read_bytes()) != json.loads((ROOT/'reports/food-core/preparation.json').read_bytes()):
        raise ValueError('Use the released four-core interface package')
    lower_own_priority(); output.mkdir(parents=True,exist_ok=False)
    clock = Clock(.0001,100,200); rewards = Rewards(2.,-1.,0.)
    identity = dict(scope=__doc__,source_sha256={name:sha(ROOT/name) for name in SOURCES},
        bundle_manifest_sha256=sha(bundle/'manifest.json'),model_ids=list(MODELS),
        clock=vars(clock),rewards=vars(rewards),action_seed=29,gait_seed=17,learning=False,
        complete_episodes_expected=8,interrupted_attempts_expected=4,task_training_updates=0)
    write_new(output/'identity.json',identity)
    reports = []
    for model_id in MODELS:
        record = dict(model_id=model_id,status='failed')
        try:
            core = FoodCoreRuntime.load(bundle,model_id)
            initial = EpisodicReadout(core.weights['action_weight'],core.weights['action_bias'],Rule(.03,1.,.1,2)).state()
            def factory(): return FoodLearningEnvironment(cases()[0].field(),clock,gait_seed=17,missing_odor=False)
            # identity() is available before constructing or resetting a simulator.
            prototype = factory(); environment_identity = prototype.identity(); prototype.close()
            binding = dict(core_record=core.record,environment=environment_identity)
            def template(): return FoodEpisode(core.weights,initial,clock,rewards,seed=29,binding=binding,learning=False)
            direct_path = output/model_id/'direct'; resumed_path = output/model_id/'restarted'
            direct = stored_episode(direct_path,template(),factory)
            interrupted_environment = factory()
            class Interrupted:
                @property
                def physics_step(self): return interrupted_environment.physics_step
                def identity(self): return interrupted_environment.identity()
                def reset(self): return interrupted_environment.reset()
                def advance(self,drive,steps):
                    interrupted_environment.advance(drive,steps)
                    raise KeyboardInterrupt('Declared interruption after the first physical interval')
                def close(self): interrupted_environment.close()
            try: stored_episode(resumed_path,template(),Interrupted)
            except KeyboardInterrupt: pass
            else: raise ValueError('Declared physical interruption did not happen')
            if interrupted_environment.physics_step != 100:
                raise ValueError('Physical interruption happened at an unexpected boundary')
            if (resumed_path/'attempt-000001/result.json.gz').exists():
                raise ValueError('Interrupted attempt incorrectly committed a terminal result')
            resumed = stored_episode(resumed_path,template(),factory)
            if (direct['rollout'] != resumed['rollout'] or direct['readout'] != resumed['readout']
                    or direct['attempt'] != 1 or resumed['attempt'] != 2):
                raise ValueError('Fresh physical restart differs from direct execution')
            if resumed['readout'] != initial or resumed['rollout']['outcome']['status'] != 'timeout':
                raise ValueError('Unexpected learning or terminal outcome in short restart check')
            if len(resumed['rollout']['samples']) != 3:
                raise ValueError('Short restart check omitted a sample')
            def forbidden(): raise AssertionError('A committed physical episode was rerun')
            if stored_episode(direct_path,template(),forbidden) != direct or stored_episode(resumed_path,template(),forbidden) != resumed:
                raise ValueError('Cached physical terminal results changed')
            record.update(status='complete',direct_attempt=1,restarted_attempt=2,complete_episodes=2,
                interrupted_attempts=1,post_warmup_physics_steps=500,exact_full_rollout_equal=True,
                unchanged_readout=True,cached_results_no_physics=True,
                observations_per_completed_episode=3,outcome=resumed['rollout']['outcome'],
                environment_identity=environment_identity,
                result_sha256={label:sha(path/'result.json.gz') for label,path in (
                    ('direct',direct_path/'attempt-000001'),('restarted',resumed_path/'attempt-000002'))})
        except Exception as error:
            record.update(error_type=type(error).__name__,error=str(error))
        write_new(output/(model_id+'.json'),record); reports.append(record)
        print(json.dumps(dict(model_id=model_id,status=record['status'])),flush=True)
    result = dict(finished_utc=datetime.now(timezone.utc).isoformat(),identity_sha256=sha(output/'identity.json'),
        complete=sum(row['status']=='complete' for row in reports),failed=sum(row['status']!='complete' for row in reports),
        checks=reports,task_training_updates=0,
        interpretation='Short integration checks only. Interrupted attempts restart original conditions. No simulator checkpoint resume, navigation, language advantage, timing benchmark or learned food adaptation is established.')
    write_new(output/'summary.json',result)
    if result['failed']: raise ValueError('Physical restart checks failed; preserve all attempts')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); run(args.bundle,args.output)
