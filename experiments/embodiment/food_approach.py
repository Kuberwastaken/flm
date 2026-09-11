"""Run the declared seven-case scripted food approach in actual FlyGym physics."""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.embodiment.calibrate import make_simulation
from flm.food_sensors import CHANNELS, ODOR_BODY_ORIGINS, CONTACT_BODY_ORIGINS, body_sensor_positions
from flm.food_approach import cases, OdorApproach, policy_card, CONTROL_STEP_S, DURATION_S, GAIT_SEED
from flygym_demo.complex_terrain import (HybridTurningController, HybridControllerObservation,
    LocomotionAction, PreprogrammedSteps, apply_locomotion_action)
import flygym.simulation
import flygym.anatomy
import flygym.compose.fly.base_fly

SOURCES = ('flm/food_sensors.py', 'flm/food_approach.py', 'tests/test_food_approach.py',
    'experiments/embodiment/food_approach.py', 'experiments/embodiment/calibrate.py',
    'requirements-embodied-lock.txt', 'docs/FOOD-APPROACH-REFERENCE.md')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')


def identity():
    return dict(study='scripted-food-approach-v1', cases=[asdict(c) for c in cases()],
        fields={c.label: c.field().card() for c in cases()}, policy=policy_card(),
        sources={name: sha(ROOT/name) for name in SOURCES},
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            versions={name: importlib.metadata.version(name) for name in ('flygym', 'mujoco', 'numpy', 'scipy', 'numba')},
            numerical_threads=1, windows_priority='BELOW_NORMAL_PRIORITY_CLASS' if sys.platform == 'win32' else None),
        installed_sources={m.__name__: sha(m.__file__) for m in (flygym.simulation, flygym.anatomy, flygym.compose.fly.base_fly)},
        channels=list(CHANNELS), odor_body_origins=list(ODOR_BODY_ORIGINS), contact_body_origins=list(CONTACT_BODY_ORIGINS),
        physics_seed=GAIT_SEED, physics_timestep_s=.0001, duration_s=DURATION_S,
        control_step_s=CONTROL_STEP_S, expected_physics_steps=20000, expected_observations=201,
        language_model_loaded=False, neural_policy_used=False, training_updates=0,
        scope='Scripted sensor-to-action physical reference. No neural learning, language transfer, feeding or ingestion.')


def lower_own_priority():
    if sys.platform != 'win32': return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.SetPriorityClass.restype = wintypes.BOOL
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())


def trial(case, output):
    report_path = output/(case.label+'.json'); archive_path = output/(case.label+'.npz')
    if report_path.exists():
        saved = json.loads(report_path.read_bytes())
        if saved['case'] != asdict(case) or saved['identity_sha256'] != sha(output/'identity.json'):
            raise ValueError('Saved trial identity differs')
        if sha(archive_path) != saved['trajectory_sha256']: raise ValueError('Saved trajectory differs')
        return saved
    if archive_path.exists(): raise ValueError('Unfinished trial archive exists; preserve it for investigation')
    field = case.field(); policy = OdorApproach(case.mode)
    rows = []; names = []; status = 'complete'; failure = None; step = 0; started = time.perf_counter()
    try:
        fly, sim, _ = make_simulation()
        order = fly.get_actuated_jointdofs_order('position')
        if sim.timestep != .0001 or len(order) != 42: raise ValueError('Calibrated physical interface changed')
        steps = PreprogrammedSteps()
        gait = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=order)
        sim.reset(); gait.reset(seed=GAIT_SEED)
        apply_locomotion_action(sim, fly.name, LocomotionAction(
            joint_angles=steps.default_pose_by_dof_order(order), adhesion_onoff=np.ones(6, dtype=bool)))
        sim.warmup(); names = [body.name for body in fly.get_bodysegs_order()]; thorax = names.index('c_thorax')
        def sample(current_step):
            positions = sim.get_body_positions(fly.name).copy()
            quaternion = sim.get_body_rotations(fly.name)[thorax].copy()
            qpos = sim.mj_data.qpos.copy()
            if not np.isfinite(qpos).all() or not np.isfinite(quaternion).all(): raise FloatingPointError('Nonfinite physical state')
            antennae, feet = body_sensor_positions(names, positions)
            observation = field.observe(antennae, feet, missing_odor=case.missing_odor)
            # This is the entire high-level policy call: no pose or diagnostic input.
            signal = policy.step(observation['sensory'])
            rows.append(dict(time_s=current_step*sim.timestep, body_positions_mm=positions,
                thorax_quaternion_wxyz=quaternion, qpos=qpos, sensory=observation['sensory'],
                raw_odor=observation['raw_odor'], contact_mask=observation['contact_mask'], descending_signal=signal))
            return signal
        signal = sample(0)
        for step in range(1, 20001):
            mechanical = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, gait.step(signal, mechanical)); sim.step()
            if not np.isfinite(sim.mj_data.qpos).all(): raise FloatingPointError('Nonfinite physical state')
            if step % 100 == 0: signal = sample(step)
        if len(rows) != 201: raise ValueError('Incomplete observation budget')
    except Exception as exc:
        status = 'failed'; failure = dict(type=type(exc).__name__, message=str(exc), physics_step=step)
    keys = ('time_s', 'body_positions_mm', 'thorax_quaternion_wxyz', 'qpos', 'sensory', 'raw_odor', 'contact_mask', 'descending_signal')
    arrays = {key: np.asarray([r[key] for r in rows]) for key in keys}
    with archive_path.open('xb') as handle: np.savez_compressed(handle, **arrays)
    metrics = None
    if rows:
        times = arrays['time_s']; contacts = arrays['contact_mask'].any(axis=1); hits = contacts.any(axis=1)
        first = int(np.flatnonzero(hits)[0]) if hits.any() else None
        indices = np.flatnonzero(contacts[first]).tolist() if first is not None else []
        thorax_positions = arrays['body_positions_mm'][:, names.index('c_thorax')]
        quaternions = arrays['thorax_quaternion_wxyz']
        metrics = dict(first_contact_s=float(times[first]) if first is not None else None,
            first_contact_sources=[field.sources[i].name for i in indices],
            first_contact_sugar=max((field.sources[i].sugar for i in indices), default=None),
            contact_latency_censored=first is None, censor_time_s=float(times[-1]) if first is None else None,
            all_contacted_sources=[s.name for i, s in enumerate(field.sources) if contacts[:, i].any()],
            minimum_thorax_planar_distance_mm={s.name: float(np.linalg.norm(thorax_positions[:, :2]-np.array(s.position_mm[:2]), axis=1).min()) for s in field.sources},
            final_thorax_position_mm=thorax_positions[-1].tolist(),
            minimum_thorax_height_mm=float(thorax_positions[:, 2].min()),
            minimum_thorax_up_z=float((1-2*(quaternions[:, 1]**2+quaternions[:, 2]**2)).min()))
    result = dict(case=asdict(case), status=status, failure=failure, identity_sha256=sha(output/'identity.json'),
        body_names=names, observations=len(rows), physics_steps=step, wall_seconds=time.perf_counter()-started,
        trajectory_sha256=sha(archive_path), metrics=metrics)
    write_new(report_path, result)
    print(json.dumps(dict(case=case.label, status=status, metrics=metrics, wall_seconds=result['wall_seconds'])), flush=True)
    return result


def run(output):
    lower_own_priority()
    if importlib.metadata.version('flygym') != '2.1.0': raise ValueError('Use pinned FlyGym 2.1.0 environment')
    output.mkdir(parents=True, exist_ok=True)
    declared = identity(); identity_path = output/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_bytes()) != declared: raise ValueError('Frozen physical identity changed')
    else:
        if any(output.iterdir()): raise ValueError('A new physical run requires an empty output directory')
        write_new(identity_path, declared)
    if (output/'summary.json').exists(): raise ValueError('Completed physical inventory already exists')
    reports = []
    for case in cases():
        if identity() != declared: raise ValueError('Source or environment changed during physical queue')
        reports.append(trial(case, output))
    comparisons = []
    physical = ('time_s', 'body_positions_mm', 'thorax_quaternion_wxyz', 'qpos', 'descending_signal')
    for first, second, all_arrays in (
        ('odor-a-left', 'odor-a-left-repeat', True),
        ('odor-a-left', 'odor-a-left-reversed', False),
        ('odor-a-left', 'odor-a-left-neutral', False),
        ('odor-a-left-missing', 'straight-a-left', False)):
        pair_complete = all(r['status'] == 'complete' for r in reports if r['case']['label'] in (first, second))
        with np.load(output/(first+'.npz'), allow_pickle=False) as a, np.load(output/(second+'.npz'), allow_pickle=False) as b:
            keys = a.files if all_arrays else physical
            equal = {k: bool(np.array_equal(a[k], b[k])) for k in keys}
        comparisons.append(dict(first=first, second=second, both_complete=pair_complete, exact_array_equal=equal))
    summary = dict(finished_utc=datetime.now(timezone.utc).isoformat(), identity_sha256=sha(identity_path),
        complete_cases=sum(r['status']=='complete' for r in reports), failed_cases=sum(r['status']!='complete' for r in reports),
        trials=reports, paired_replays=comparisons, scope=declared['scope'])
    write_new(output/'summary.json', summary)
    print(json.dumps(dict(complete_cases=summary['complete_cases'], failed_cases=summary['failed_cases'], paired_replays=comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
