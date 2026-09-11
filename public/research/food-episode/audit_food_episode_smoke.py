"""Verify unpacked food-episode records with NumPy; do not run physics or fit.

Uses the same declared controller and sensory routines. This is numerical
replay and packaging verification, not an independent simulator implementation.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from flm.food_episode import Clock,FoodEpisode,Rewards,array_digest
from flm.food_sensors import FoodField,Source,body_sensor_positions

MODEL_IDS = ('initial-s42','language-s42','initial-s43','language-s43')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(path,bundle):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('record-manifest.json'))
        if set(archive.namelist()) != set(manifest)|{'record-manifest.json'} or archive.testzip() is not None:
            raise ValueError('Incomplete smoke archive or CRC failure')
        for name,record in manifest.items():
            value = archive.read(name)
            if len(value) != record['bytes'] or hashlib.sha256(value).hexdigest() != record['sha256']:
                raise ValueError('Smoke archive checksum changed: '+name)
        summary = json.loads(archive.read('summary.json'))
        expected = [model+f'-repeat{repeat}' for model in MODEL_IDS for repeat in (0,1)]
        if ([row['label'] for row in summary['trials']] != expected or summary['complete'] != 8 or summary['failed'] != 0
                or summary['task_training_updates'] != 0 or summary['physical_steps_excluding_warmup'] != 1600
                or hashlib.sha256(archive.read('identity.json')).hexdigest() != summary['identity_sha256']
                or json.loads(archive.read('identity.json')) != summary['identity']):
            raise ValueError('Whole smoke inventory or declaration changed')
        for name,value in summary['identity']['source_sha256'].items():
            if sha(ROOT/name) != value: raise ValueError('Use the declared controller/sensor sources')
        if sha(bundle/'manifest.json') != summary['identity']['bundle_manifest_sha256']:
            raise ValueError('Core package identity changed')
        checks = []; paired = {}
        for report in summary['trials']:
            label = report['label']
            if (json.loads(archive.read(label+'.json')) != report
                    or hashlib.sha256(archive.read(label+'.npz')).hexdigest() != report['trajectory_sha256']
                    or hashlib.sha256(archive.read(label+'-controller.json')).hexdigest() != report['controller_sha256']):
                raise ValueError('Trial records differ from their completion declaration')
            saved = json.loads(archive.read(label+'-controller.json')); identity = saved['identity']
            core = FoodCoreRuntime.load(bundle,report['model_id'])
            FoodEpisode.restore(core.weights,saved,binding=identity['binding'])
            actor = FoodEpisode(core.weights,saved['initial_readout'],Clock(**identity['clock']),Rewards(**identity['rewards']),
                                seed=identity['seed'],binding=identity['binding'],learning=identity['learning'])
            if actor.learning or len(saved['events']) != 3 or saved['status'] != 'timeout':
                raise ValueError('Require the complete three-observation no-learning smoke episode')
            source_rows = identity['binding']['environment']['field']['sources']
            sources = [Source(**{key:tuple(value) if isinstance(value,list) else value for key,value in row.items()}) for row in source_rows]
            field = FoodField(sources)
            with np.load(io.BytesIO(archive.read(label+'.npz')),allow_pickle=False) as payload:
                arrays = {name:payload[name].copy() for name in payload.files}
            if array_digest(arrays) != report['arrays_sha256']: raise ValueError('Recorded array inventory or bytes changed')
            for index,event in enumerate(saved['events']):
                antennae,feet = body_sensor_positions(report['body_names'],arrays['body_positions_mm'][index])
                observed = field.observe(antennae,feet,missing_odor=identity['binding']['environment']['missing_odor'])
                for name in ('sensory','raw_odor','contact_mask'):
                    if not np.array_equal(observed[name],arrays[name][index]): raise ValueError('Recorded sensor geometry changed')
                response = actor.observe(observed['sensory'],event['physics_step'])
                for name in ('time_s','sensory','fast','slow','features','probabilities','descending_signal'):
                    if not np.array_equal(np.asarray(response[name]),arrays[name][index]):
                        raise ValueError('Recorded neural/control replay changed: '+name)
                for name in ('action','uniform'):
                    if (response[name] if response[name] is not None else -1) != arrays[name][index]:
                        raise ValueError('Recorded action/draw or terminal sentinel changed')
            if actor.state() != saved or actor.learner.state() != saved['initial_readout']:
                raise ValueError('Controller replay or disabled learning changed')
            if report['model_id'] in paired:
                if any(not np.array_equal(value,paired[report['model_id']][name]) for name,value in arrays.items()):
                    raise ValueError('Paired physical/neural arrays do not repeat exactly')
            else: paired[report['model_id']] = arrays
            checks.append(dict(label=label,observations=3,sensor_replay_exact=True,controller_replay_exact=True,readout_unchanged=True))
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha(path),
        bundle_manifest_sha256=sha(bundle/'manifest.json'),auditor_sha256=sha(Path(__file__)),numpy=str(np.__version__),
        archive_members=len(manifest)+1,trials=checks,observations=24,paired_repeats_exact=4,
        new_physics_steps=0,task_training_updates=0,
        scope='Exact archive bytes, recorded sensor geometry, core/readout event replay and paired arrays. Same declared numerical routines, no new physical simulation or independent implementation.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True); parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); result = audit(args.archive,args.bundle)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8',newline='\n') as handle:
        json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(trials=len(result['trials']),observations=result['observations'],new_physics_steps=0)))
