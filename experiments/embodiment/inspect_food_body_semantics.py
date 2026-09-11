"""Measure stored-body lookup and integration-stage semantics on six complete trials."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.calibrate import make_simulation
from experiments.embodiment.food_approach import lower_own_priority


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(output):
    if output.exists(): raise ValueError('Preserve earlier body-semantics records')
    lower_own_priority(); run=ROOT/'runs/embodiment/food-core-physical-v1'
    identity=json.loads((run/'identity.json').read_bytes())
    assets_path=ROOT/'work/food-replay-assets-v1/model.json'; assets=json.loads(assets_path.read_bytes())
    if assets['physical_identity_sha256']!=sha(run/'identity.json'): raise ValueError('Body assets refer to another physical inventory')
    for name,digest in identity['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Physical source changed: '+name)
    for name,digest in identity['installed_sources'].items():
        if sha(__import__(name,fromlist=['__file__']).__file__)!=digest: raise ValueError('Installed source changed: '+name)
    if any(importlib.metadata.version(name)!=version for name,version in identity['environment']['versions'].items()):
        raise ValueError('Physical environment differs')
    fly,sim,_=make_simulation(); names=[b.name for b in fly.get_bodysegs_order()]
    ids=[mujoco.mj_name2id(sim.mj_model,mujoco.mjtObj.mjOBJ_BODY,element.name) for element in fly.bodyseg_to_mjcfbody.values()]
    if ids!=[r['compiled_body_id'] for r in assets['body_lookup']]: raise ValueError('Compiled body mapping changed')
    valid=np.array(ids)>=0; head=names.index('c_head'); alias=names.index('rh_tarsus5'); thorax=names.index('c_thorax')
    results=[]
    for case in identity['cases']:
        label='initial-s42--'+case['label']; record=json.loads((run/(label+'.json')).read_bytes())
        path=run/(label+'.npz')
        if (record['status']!='complete' or record['observations']!=201 or record['physics_steps']!=20000
                or record['identity_sha256']!=sha(run/'identity.json') or sha(path)!=record['trajectory_sha256']):
            raise ValueError('Require the six completed original initial-core trials')
        with np.load(path,allow_pickle=False) as archive:
            arrays={key:archive[key] for key in ('qpos','body_positions_mm','thorax_quaternion_wxyz')}
        if not np.array_equal(arrays['body_positions_mm'][:,head],arrays['body_positions_mm'][:,alias]):
            raise ValueError('Stored head alias differs from the inspected API behavior')
        maxima=dict(valid_body_coordinate_mm=0.,valid_body_distance_mm=0.,thorax_quaternion_component=0.)
        largest=None
        for frame,qpos in enumerate(arrays['qpos']):
            sim.mj_data.qpos[:]=qpos
            mujoco.mj_kinematics(sim.mj_model,sim.mj_data)
            positions=sim.get_body_positions(fly.name); rotations=sim.get_body_rotations(fly.name)
            error=positions[valid]-arrays['body_positions_mm'][frame,valid]
            distance=np.linalg.norm(error,axis=1); index=int(distance.argmax())
            if float(distance[index])>maxima['valid_body_distance_mm']:
                largest=dict(frame=frame,segment=np.array(names)[valid][index].item(),distance_mm=float(distance[index]))
            maxima['valid_body_coordinate_mm']=max(maxima['valid_body_coordinate_mm'],float(np.abs(error).max()))
            maxima['valid_body_distance_mm']=max(maxima['valid_body_distance_mm'],float(distance.max()))
            maxima['thorax_quaternion_component']=max(maxima['thorax_quaternion_component'],float(np.abs(rotations[thorax]-arrays['thorax_quaternion_wxyz'][frame]).max()))
        results.append(dict(label=label,trajectory_sha256=record['trajectory_sha256'],observations=201,
            head_row_exactly_equals_rh_tarsus5=True,maximum_recomputed_minus_cached=maxima,largest_distance=largest))
    report=dict(inspected_utc=datetime.now(timezone.utc).isoformat(),physical_identity_sha256=sha(run/'identity.json'),
        body_assets_manifest_sha256=sha(assets_path),inspector_sha256=sha(__file__),body_lookup=assets['body_lookup'],
        unresolved_body_segments=assets['unresolved_body_segments'],unresolved_numpy_alias=assets['unresolved_numpy_alias'],
        all_sensor_and_outcome_body_ids_resolve=True,checked_trials=results,observations=1206,
        simulation_integrator='Euler',physics_timestep_s=float(sim.timestep),physics_steps_run=0,training_updates=0,
        documentation='https://mujoco.readthedocs.io/en/stable/computation/index.html#consistency-in-mjdata',
        scope='Post-hoc getter and kinematic inspection of all six initial-s42 physical cases. Original cached sensors, commands, poses and outcomes are preserved. Recomputed qpos geometry belongs to a different integration stage; the head body row aliases the last body and is not a valid head position.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8',newline='\n') as handle: json.dump(report,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(trials=len(results),observations=report['observations'],
        maximum_valid_body_distance_mm=max(r['maximum_recomputed_minus_cached']['valid_body_distance_mm'] for r in results),
        unresolved_body_segments=report['unresolved_body_segments'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    inspect(parser.parse_args().output)
