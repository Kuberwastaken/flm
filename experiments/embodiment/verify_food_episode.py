"""Short no-learning physical integration checks, not a food-adaptation study.

Eight 20-ms rollouts (four exported cores, each repeated) follow the unchanged
factory's warmup. Only two descending commands are applied per rollout. This
checks executable wiring and reproducibility, not navigation or task learning.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.food_approach import lower_own_priority
from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from experiments.embodiment.food_learning_environment import FoodLearningEnvironment
from flm.food_approach import cases
from flm.food_episode import Clock,FoodEpisode,Rewards,array_digest,run_episode
from flm.food_readout_learning import EpisodicReadout,Rule

MODEL_IDS = ('initial-s42','language-s42','initial-s43','language-s43')
SOURCES = ('flm/food_episode.py','flm/food_readout_learning.py',
    'experiments/embodiment/food_learning_environment.py','experiments/embodiment/verify_food_episode.py',
    'experiments/embodiment/food_core_runtime.py','tests/test_food_episode.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with path.open('x',encoding='utf8',newline='\n') as handle:
        json.dump(value,handle,indent=2,allow_nan=False); handle.write('\n')


def run(bundle,output):
    bundle = Path(bundle); output = Path(output)
    if json.loads((bundle/'manifest.json').read_bytes()) != json.loads((ROOT/'reports/food-core/preparation.json').read_bytes()):
        raise ValueError('Use the released four-core interface package')
    lower_own_priority(); output.mkdir(parents=True,exist_ok=False)
    clock = Clock(.0001,100,200); rewards = Rewards(2.,-1.,0.)
    identity = dict(scope=__doc__,source_sha256={name:sha(ROOT/name) for name in SOURCES},
        bundle_manifest_sha256=sha(bundle/'manifest.json'),model_ids=list(MODEL_IDS),python=platform.python_version(),
        numpy=str(np.__version__),clock=vars(clock),rewards=vars(rewards),sampling_seed=29,gait_seed=17,
        learning=False,expected_trials=8,steps_per_trial=200,task_training_updates=0)
    write_new(output/'identity.json',identity)
    reports = []; pairs = []
    for model_id in MODEL_IDS:
        paired = []
        for repetition in (0,1):
            label = model_id+f'-repeat{repetition}'; started = time.perf_counter()
            environment = None; report = dict(label=label,model_id=model_id,status='failed')
            try:
                core = FoodCoreRuntime.load(bundle,model_id); original = array_digest(core.weights)
                initial = EpisodicReadout(core.weights['action_weight'],core.weights['action_bias'],Rule(.03,1.,.1,2)).state()
                environment = FoodLearningEnvironment(cases()[0].field(),clock,gait_seed=17,missing_odor=False)
                binding = dict(core_record=core.record,environment=environment.identity())
                actor = FoodEpisode(core.weights,initial,clock,rewards,seed=29,binding=binding,learning=False)
                result = run_episode(actor,environment)
                # Preserve the controller record even when the simulator failed.
                write_new(output/(label+'-controller.json'),result['controller'])
                if result['outcome']['status'] != 'timeout' or environment.physics_step != 200 or len(result['samples']) != 3:
                    raise ValueError('Incomplete no-contact physical smoke rollout: '+str(result['outcome']))
                if actor.learner.state() != initial or array_digest(actor.core.weights) != original or array_digest(core.weights) != original:
                    raise ValueError('Disabled-learning rollout changed core or readout')
                restored = FoodEpisode.restore(core.weights,result['controller'],binding=binding)
                if restored.state() != actor.state(): raise ValueError('Physical sensor/controller replay changed')
                rows = result['samples']
                arrays = {name:np.asarray([row['control'][name] for row in rows])
                          for name in ('time_s','sensory','fast','slow','features','probabilities','descending_signal')}
                arrays.update({name:np.asarray([row['environment'][name] for row in rows])
                               for name in ('qpos','body_positions_mm','thorax_quaternion_wxyz','raw_odor','contact_mask','simulation_time_s')})
                arrays['action'] = np.array([row['control']['action'] if row['control']['action'] is not None else -1 for row in rows])
                arrays['uniform'] = np.array([row['control']['uniform'] if row['control']['uniform'] is not None else -1. for row in rows])
                if any(not np.isfinite(value).all() for value in arrays.values()): raise ValueError('Nonfinite smoke records')
                if (arrays['action'][-1] != -1 or arrays['uniform'][-1] != -1
                        or not np.array_equal(arrays['descending_signal'][-1],[0.,0.])):
                    raise ValueError('Unused terminal action entered physical smoke records')
                with (output/(label+'.npz')).open('xb') as handle: np.savez_compressed(handle,**arrays)
                report.update(status='complete',physics_steps=environment.physics_step,observations=len(rows),
                    outcome=result['outcome'],core_unchanged=True,readout_unchanged=True,controller_replay_exact=True,
                    arrays_sha256=array_digest(arrays),trajectory_sha256=sha(output/(label+'.npz')),
                    controller_sha256=sha(output/(label+'-controller.json')),body_names=rows[0]['environment']['body_names'])
                paired.append(arrays)
            except Exception as error:
                report.update(error_type=type(error).__name__,error=str(error))
            finally:
                if environment is not None: environment.close()
            report['wall_seconds'] = time.perf_counter()-started
            write_new(output/(label+'.json'),report); reports.append(report)
            print(json.dumps(dict(label=label,status=report['status'])),flush=True)
        equal = {name:bool(np.array_equal(paired[0][name],paired[1][name])) for name in paired[0]} if len(paired)==2 else None
        pairs.append(dict(model_id=model_id,exact_array_equal=equal))
    result = dict(finished_utc=datetime.now(timezone.utc).isoformat(),identity=identity,
        identity_sha256=sha(output/'identity.json'),complete=sum(row['status']=='complete' for row in reports),
        failed=sum(row['status']!='complete' for row in reports),trials=reports,paired_repeats=pairs,
        task_training_updates=0,physical_steps_excluding_warmup=sum(row.get('physics_steps',0) for row in reports),
        interpretation='Short integration/replay checks only. No food-task learning, navigation success, language benefit or physical checkpoint resume is demonstrated. Terminal action/uniform -1 are explicit no-action sentinels. Warmup precedes the 20-ms recorded horizon.')
    write_new(output/'summary.json',result)
    if result['failed'] or any(pair['exact_array_equal'] is None or not all(pair['exact_array_equal'].values()) for pair in pairs):
        raise ValueError('Physical episode integration or repeat check failed; preserve the complete records')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); run(args.bundle,args.output)
