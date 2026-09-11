"""Calibrate engineered food sensors against actual FlyGym body positions.

No language model, trainable controller, optimizer, food policy or renderer runs.
Two short prescribed-motion repetitions check geometry and sensor reproducibility.
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import argparse
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.calibrate import make_simulation
from flm.food_sensors import FoodField,Source,CHANNELS,ODOR_BODY_ORIGINS,CONTACT_BODY_ORIGINS,body_sensor_positions
from flygym_demo.complex_terrain import (HybridTurningController,HybridControllerObservation,
    LocomotionAction,PreprogrammedSteps,apply_locomotion_action)
import flygym.simulation
import flygym.anatomy
import flygym.compose.fly.base_fly


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(field,antennae,contacts,missing=False):
    values=field.observe(antennae,contacts,missing_odor=missing)
    return {name:value.tolist() if isinstance(value,np.ndarray) else value for name,value in values.items()}


def motion_probe():
    fly,sim,_=make_simulation(); steps=PreprogrammedSteps(); order=fly.get_actuated_jointdofs_order('position')
    if sim.timestep!=.0001 or len(order)!=42: raise ValueError('Calibrated physical interface changed')
    gait=HybridTurningController(timestep=sim.timestep,preprogrammed_steps=steps,output_dof_order=order)
    sim.reset(); gait.reset(seed=17)
    apply_locomotion_action(sim,fly.name,LocomotionAction(joint_angles=steps.default_pose_by_dof_order(order),adhesion_onoff=np.ones(6,dtype=bool)))
    sim.warmup(); names=[body.name for body in fly.get_bodysegs_order()]
    field=FoodField([Source('a',(8.,3.,0.),(5.,4.,2.),(1.,0.),1.,.5,.2),
                     Source('b',(8.,-3.,0.),(5.,4.,2.),(0.,1.),0.,.5,.2)])
    frames=[]
    def record(step):
        positions=sim.get_body_positions(fly.name).copy(); antennae,contacts=body_sensor_positions(names,positions)
        if not np.isfinite(sim.mj_data.qpos).all(): raise ValueError('Nonfinite physical state')
        frames.append(dict(time_s=step*sim.timestep,body_positions_mm=positions.tolist(),qpos=sim.mj_data.qpos.tolist(),
            antennae_mm=antennae.tolist(),contact_points_mm=contacts.tolist(),observation=observation(field,antennae,contacts)))
    record(0)
    for step in range(1,201):
        sensory=HybridControllerObservation.from_sim(sim,fly.name)
        apply_locomotion_action(sim,fly.name,gait.step(np.array([1.,1.]),sensory)); sim.step()
        if step%100==0: record(step)
    return names,field,frames


def run(root,output):
    if output.exists(): raise ValueError('Preserve previous sensor calibration records')
    if importlib.metadata.version('flygym')!='2.1.0': raise ValueError('Use the pinned isolated FlyGym 2.1.0 environment')
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.SetPriorityClass.argtypes=(wintypes.HANDLE,wintypes.DWORD)
        kernel.SetPriorityClass.restype=wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    names,field,frames=motion_probe(); repeated_names,repeated_field,repeated=motion_probe()
    if names!=repeated_names or field.card()!=repeated_field.card() or frames!=repeated:
        raise ValueError('Repeated physical positions or sensor observations differ')
    antennae=np.asarray(frames[0]['antennae_mm']); contacts=np.asarray(frames[0]['contact_points_mm'])
    anchor=contacts[0].copy(); anchor[2]-=.01
    touch=Source('contact-probe',tuple(map(float,anchor)),(2.,2.,2.),(.4,.2),1.,.1,.02)
    far=replace(touch,position_mm=(float(anchor[0]+100.),float(anchor[1]),float(anchor[2])))
    cases=[]
    for label,source,missing in (('distant-sweet',far,False),('distant-neutral',replace(far,sugar=0.),False),
            ('touched-sweet',touch,False),('touched-neutral',replace(touch,sugar=0.),False),
            ('touched-missing-odor',touch,True),('touched-odorless-sugar',replace(touch,odor=(0.,0.)),False)):
        stimulus=FoodField([source])
        cases.append(dict(label=label,field=stimulus.card(),missing_odor=missing,
            observation=observation(stimulus,antennae,contacts,missing)))
    lookup={row['label']:row['observation']['sensory'] for row in cases}
    if lookup['distant-sweet']!=lookup['distant-neutral']: raise ValueError('Sugar identity leaked at a distance')
    if lookup['touched-sweet'][:4]!=lookup['touched-neutral'][:4] or lookup['touched-sweet'][4:]!=[1.,1.] or lookup['touched-neutral'][4:]!=[0.,1.]:
        raise ValueError('Taste/contact separation failed')
    if lookup['touched-missing-odor']!=[0.,0.,0.,0.,1.,1.] or lookup['touched-odorless-sugar']!=[0.,0.,0.,0.,1.,1.]:
        raise ValueError('Missing odor incorrectly removed contact')
    source_files=('flm/food_sensors.py','tests/test_food_sensors.py','experiments/embodiment/food_sensor_probe.py',
                  'experiments/embodiment/calibrate.py','requirements-embodied-lock.txt')
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=platform.python_version(),
        versions={name:importlib.metadata.version(name) for name in ('flygym','mujoco','numpy','scipy','numba')},
        source_sha256={name:sha(root/name) for name in source_files},
        installed_source_sha256={module.__name__:sha(module.__file__) for module in
            (flygym.simulation,flygym.anatomy,flygym.compose.fly.base_fly)},
        body_names=names,odor_body_origins=list(ODOR_BODY_ORIGINS),contact_body_origins=list(CONTACT_BODY_ORIGINS),
        channels=list(CHANNELS),field=field.card(),physics_seed=17,physics_timestep_s=.0001,
        simulation_seconds=.02,physics_steps_per_repeat=200,repetitions=2,numerical_threads=1,
        descending_signal=[1.,1.],low_level_controller='Existing upstream hybrid walking controller',
        frames=frames,exact_repeated_geometry_and_sensor_records=True,
        geometry_counterfactuals=dict(antennae_mm=antennae.tolist(),contact_points_mm=contacts.tolist(),cases=cases),
        language_model_loaded=False,neural_policy_used=False,training_updates=0,food_approach_or_feeding_measured=False,
        scope='Actual body-position readout and prescribed-motion sensor calibration only. Counterfactual virtual contact patches are positioned under a measured foot for geometry testing; no learned food behavior, language transfer or ingestion is tested.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(verified=True,frames=len(frames),repetitions=2,counterfactuals=len(cases),scope=result['scope'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    run(ROOT,args.output)
