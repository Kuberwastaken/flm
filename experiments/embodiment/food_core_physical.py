"""Execute the declared 24 unadapted FLM-controlled physical food trials."""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.embodiment.food_approach import (
    identity as reference_identity, lower_own_priority, make_simulation, write_new,
    HybridTurningController, HybridControllerObservation, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action)
from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from flm.food_approach import cases
from flm.food_sensors import body_sensor_positions
from flm.food_motor import FoodMotor, motor_card

MODEL_IDS = ('initial-s42', 'language-s42', 'initial-s43', 'language-s43')
KEYS = ('time_s', 'body_positions_mm', 'thorax_quaternion_wxyz', 'qpos', 'sensory', 'raw_odor',
        'contact_mask', 'descending_signal', 'fast', 'slow', 'features', 'logits', 'probabilities')
SOURCES = ('experiments/embodiment/food_core_physical.py', 'experiments/embodiment/food_core_runtime.py',
           'flm/food_motor.py', 'tests/test_food_motor.py', 'docs/FOOD-CORE-PHYSICAL-REFERENCE.md')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions(): return tuple(case for case in cases() if case.mode != 'straight')


def identity(bundle):
    inherited = reference_identity()
    manifest_path = bundle/'manifest.json'; manifest = json.loads(manifest_path.read_bytes())
    released = json.loads((ROOT/'reports/food-core/preparation.json').read_bytes())
    if manifest != released or tuple(r['id'] for r in manifest['models']) != MODEL_IDS:
        raise ValueError('Use the complete released four-core preparation')
    if manifest['adaptation_updates'] != 0: raise ValueError('Expected zero-adaptation interfaces')
    for model_id in MODEL_IDS: FoodCoreRuntime.load(bundle, model_id)
    if tuple(manifest['action_names']) != tuple(motor_card()['action_order']):
        raise ValueError('Motor and model action orders differ')
    inherited.update(study='unadapted-food-core-physical-v1', cases=[asdict(c) for c in conditions()],
        fields={c.label: c.field().card() for c in conditions()}, model_ids=list(MODEL_IDS),
        policy=motor_card(), language_model_loaded=True, neural_policy_used=True, training_updates=0,
        expected_trials=24, bundle_manifest_sha256=sha(manifest_path), models=manifest['models'],
        physical_reference_identity_sha256=sha(ROOT/'runs/embodiment/food-approach-v1/identity.json'),
        scope='Actual closed-loop physics with initial and language-trained cores behind identical unadapted adapters. No food-task learning, feeding or demonstrated transfer benefit.')
    inherited['sources'].update({name: sha(ROOT/name) for name in SOURCES})
    return inherited


def metrics(arrays, names, field):
    if not len(arrays['time_s']): return None
    contacts = arrays['contact_mask'].any(axis=1); hits = contacts.any(axis=1)
    first = int(np.flatnonzero(hits)[0]) if hits.any() else None
    indices = np.flatnonzero(contacts[first]).tolist() if first is not None else []
    body = arrays['body_positions_mm'][:, names.index('c_thorax')]
    quat = arrays['thorax_quaternion_wxyz']; times = arrays['time_s']
    return dict(first_contact_s=float(times[first]) if first is not None else None,
        first_contact_sources=[field.sources[i].name for i in indices],
        first_contact_sugar=max((field.sources[i].sugar for i in indices), default=None),
        contact_latency_censored=first is None, censor_time_s=float(times[-1]) if first is None else None,
        all_contacted_sources=[s.name for i, s in enumerate(field.sources) if contacts[:, i].any()],
        minimum_thorax_planar_distance_mm={s.name: float(np.linalg.norm(body[:, :2]-np.array(s.position_mm[:2]), axis=1).min()) for s in field.sources},
        final_thorax_position_mm=body[-1].tolist(), minimum_thorax_height_mm=float(body[:, 2].min()),
        minimum_thorax_up_z=float((1-2*(quat[:, 1]**2+quat[:, 2]**2)).min()))


def trial(model_id, case, bundle, output):
    label = model_id+'--'+case.label
    report_path = output/(label+'.json'); archive_path = output/(label+'.npz')
    if report_path.exists():
        saved = json.loads(report_path.read_bytes())
        if (saved['case'] != asdict(case) or saved['model_id'] != model_id
                or saved['identity_sha256'] != sha(output/'identity.json')
                or sha(archive_path) != saved['trajectory_sha256']):
            raise ValueError('Saved trial identity or payload differs')
        return saved
    if archive_path.exists(): raise ValueError('Preserve and investigate the unfinished trial archive')
    field = case.field(); core = FoodCoreRuntime.load(bundle, model_id); motor = FoodMotor()
    rows = []; names = []; status = 'complete'; failure = None; step = 0; started = time.perf_counter()
    try:
        fly, sim, _ = make_simulation(); order = fly.get_actuated_jointdofs_order('position')
        if sim.timestep != .0001 or len(order) != 42: raise ValueError('Physical interface changed')
        steps = PreprogrammedSteps()
        gait = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=order)
        sim.reset(); gait.reset(seed=17)
        apply_locomotion_action(sim, fly.name, LocomotionAction(
            joint_angles=steps.default_pose_by_dof_order(order), adhesion_onoff=np.ones(6, dtype=bool)))
        sim.warmup(); names = [body.name for body in fly.get_bodysegs_order()]; thorax = names.index('c_thorax')
        def sample(current_step):
            positions = sim.get_body_positions(fly.name).copy()
            quaternion = sim.get_body_rotations(fly.name)[thorax].copy(); qpos = sim.mj_data.qpos.copy()
            if not np.isfinite(qpos).all() or not np.isfinite(quaternion).all(): raise FloatingPointError('Nonfinite physical state')
            antennae, feet = body_sensor_positions(names, positions)
            observed = field.observe(antennae, feet, missing_odor=case.missing_odor)
            sensory = observed['sensory'].astype(np.float32)
            logits, probabilities, features = core.step(sensory)
            signal = motor.step(sensory, probabilities)
            rows.append(dict(time_s=current_step*sim.timestep, body_positions_mm=positions,
                thorax_quaternion_wxyz=quaternion, qpos=qpos, sensory=sensory,
                raw_odor=observed['raw_odor'], contact_mask=observed['contact_mask'], descending_signal=signal,
                fast=core.fast.copy(), slow=core.slow.copy(), features=features.copy(),
                logits=logits.copy(), probabilities=probabilities.copy()))
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
    arrays = {key: np.asarray([r[key] for r in rows]) for key in KEYS}
    with archive_path.open('xb') as handle: np.savez_compressed(handle, **arrays)
    result = dict(label=label, model_id=model_id, case=asdict(case), status=status, failure=failure,
        identity_sha256=sha(output/'identity.json'), model_payload_sha256=core.record['sha256'],
        body_names=names, observations=len(rows), physics_steps=step,
        wall_seconds=time.perf_counter()-started, trajectory_sha256=sha(archive_path),
        metrics=metrics(arrays, names, field))
    write_new(report_path, result)
    print(json.dumps({key: result[key] for key in ('label', 'status', 'metrics', 'wall_seconds')}), flush=True)
    return result


def run(bundle, output):
    lower_own_priority(); output.mkdir(parents=True, exist_ok=True)
    declared = identity(bundle); identity_path = output/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_bytes()) != declared: raise ValueError('Frozen identity changed')
    else:
        if any(output.iterdir()): raise ValueError('New physical inventory requires an empty directory')
        write_new(identity_path, declared)
    if (output/'summary.json').exists(): raise ValueError('Completed physical inventory already exists')
    reports = []
    for model_id in MODEL_IDS:
        for case in conditions():
            if identity(bundle) != declared: raise ValueError('Source or environment changed during queue')
            reports.append(trial(model_id, case, bundle, output))
    pairs = []
    for model_id in MODEL_IDS:
        for counterpart in ('odor-a-left-repeat', 'odor-a-left-reversed', 'odor-a-left-neutral'):
            first = model_id+'--odor-a-left'; second = model_id+'--'+counterpart
            with np.load(output/(first+'.npz'), allow_pickle=False) as a, np.load(output/(second+'.npz'), allow_pickle=False) as b:
                equal = {name: bool(np.array_equal(a[name], b[name])) for name in KEYS}
            pairs.append(dict(first=first, second=second,
                both_complete=all(r['status']=='complete' for r in reports if r['label'] in (first, second)),
                exact_array_equal=equal))
    summary = dict(finished_utc=datetime.now(timezone.utc).isoformat(), identity_sha256=sha(identity_path),
        complete_cases=sum(r['status']=='complete' for r in reports), failed_cases=sum(r['status']!='complete' for r in reports),
        trials=reports, paired_replays=pairs, scope=declared['scope'])
    write_new(output/'summary.json', summary)
    print(json.dumps(dict(complete_cases=summary['complete_cases'], failed_cases=summary['failed_cases'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.bundle, args.output)
