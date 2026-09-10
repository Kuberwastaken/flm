"""Run the declared live/frozen pose assay in actual FlyGym/MuJoCo physics."""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
from contextlib import contextmanager
import csv
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
from experiments.embodiment.choice_runtime import ChoiceRuntime, verify_parity
from experiments.embodiment.feedback import FeedbackController, NEURAL_STEP_S, target_at, bearing_error, cases
from flygym.anatomy import BodySegment
from flygym_demo.complex_terrain import (HybridTurningController, HybridControllerObservation,
    LocomotionAction, PreprogrammedSteps, apply_locomotion_action)

OUTPUT = ROOT / 'runs/embodiment/closed-loop-v1'
PACKAGE = ROOT / 'data/controllers/choice-v1'
PROTOCOL = 'docs/CLOSED-LOOP-PROTOCOL.md'
VIDEO_CASE = 'eligibility-live-switch'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8'); temporary.replace(path)


@contextmanager
def assay_lease():
    """Only one physical writer may own this registered output directory."""
    path=OUTPUT/'queue.lock'; handle=path.open('a+b')
    if path.stat().st_size==0: handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if sys.platform=='win32':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close(); raise RuntimeError('Another process already owns the physical assay')
    try:
        yield
    finally:
        handle.close()


def yaw_of(quaternion):
    w,x,y,z = quaternion
    return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def study_identity():
    sources = ['experiments/embodiment/' + name for name in
               ('closed_loop.py','choice_runtime.py','feedback.py','calibrate.py')]
    inputs = [ROOT / PROTOCOL, ROOT / 'requirements-embodied-lock.txt', *sorted(PACKAGE.glob('*'))]
    return dict(study='Online pose feedback through the learned choice network v1',
        sources={name:sha(ROOT/name) for name in sources},
        inputs={path.relative_to(ROOT).as_posix():sha(path) for path in inputs if path.is_file()},
        environment=dict(python=platform.python_version(),platform=platform.platform(),
            versions={name:importlib.metadata.version(name) for name in ('flygym','mujoco','numpy','scipy','numba')},
            numerical_threads=1),
        simulation_seconds=2., physics_seed=17, cases=cases(), repeat_case=VIDEO_CASE,
        video_case=VIDEO_CASE, neural_step_seconds=NEURAL_STEP_S,
        statement='Inference-only engineering assay. No language weights, new learning or learned gait.')


def run_trial(case, identity, repeat=False):
    label = case['label'] + ('-repeat' if repeat else '')
    report_path = OUTPUT / (label + '.json'); archive_path = OUTPUT / (label + '.npz')
    if report_path.exists():
        saved=read(report_path)
        if saved['study_identity_sha256'] != sha(OUTPUT/'identity.json') or saved['case'] != case or saved['repeat'] != repeat:
            raise ValueError('Saved trial identity changed')
        if saved.get('status') != 'complete':
            raise RuntimeError('This trial has a recorded failure; investigate before a new study identity')
        if saved['recorded_samples']!=2001 or saved['neural_frames']!=400 or len(saved['decisions'])!=36:
            raise ValueError('Saved trial is missing the declared observation/decision budget')
        if sha(archive_path) != saved['trajectory_sha256'] or sha(OUTPUT/(label+'.csv')) != saved['csv_sha256']:
            raise ValueError('Saved physical trajectory changed')
        if saved.get('video') and sha(OUTPUT/(label+'.mp4')) != saved['video']['sha256']:
            raise ValueError('Saved physical video changed')
        return saved
    fly, sim, camera=make_simulation()
    if sim.timestep != .0001: raise ValueError('Physical timestep changed')
    steps=PreprogrammedSteps(); order=fly.get_actuated_jointdofs_order('position')
    if len(order)!=42: raise ValueError('Body actuator count changed')
    gait=HybridTurningController(timestep=sim.timestep,preprogrammed_steps=steps,output_dof_order=order)
    sim.reset(); gait.reset(seed=17)
    apply_locomotion_action(sim,fly.name,LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(order),adhesion_onoff=np.ones(6,dtype=bool)))
    sim.warmup()
    thorax=fly.get_bodysegs_order().index(BodySegment('c_thorax'))
    initial_position=sim.get_body_positions(fly.name)[thorax].copy()
    initial_rotation=sim.get_body_rotations(fly.name)[thorax].copy()
    neural=None if case['model']=='scripted' else ChoiceRuntime(PACKAGE,case['model'])
    feedback=FeedbackController(neural,case['scenario'],case['pose_mode'],initial_position,yaw_of(initial_rotation))
    video=case['label']==VIDEO_CASE and not repeat
    renderer=sim.set_renderer([camera],camera_res=(360,480),playback_speed=.25,output_fps=25) if video else None
    records=[]; neural_records=[]; decisions=[]
    def measure(t):
        position=sim.get_body_positions(fly.name)[thorax].copy()
        rotation=sim.get_body_rotations(fly.name)[thorax].copy(); yaw=yaw_of(rotation)
        target=target_at(case['scenario'],t)
        records.append(dict(time=t,position=position,rotation=rotation,yaw=yaw,
            error=bearing_error(position,yaw,target),target=target,
            distance=float(np.linalg.norm(target-position[:2])),
            up=float(1-2*(rotation[1]**2+rotation[2]**2)),contacts=int(sim.mj_data.ncon),
            qpos=sim.mj_data.qpos.copy(),signal=feedback.signal.copy()))
    measure(0.); started=time.perf_counter(); stride=round(NEURAL_STEP_S/sim.timestep)
    count=round(identity['simulation_seconds']/sim.timestep)
    status='complete'; failure=None
    try:
        for step in range(count):
            if step % stride == 0:
                position=sim.get_body_positions(fly.name)[thorax].copy()
                rotation=sim.get_body_rotations(fly.name)[thorax].copy()
                record,decision=feedback.frame(step*sim.timestep,position,yaw_of(rotation))
                neural_records.append(record)
                if decision is not None: decisions.append(decision)
            observation=HybridControllerObservation.from_sim(sim,fly.name)
            action=gait.step(feedback.signal,observation)
            apply_locomotion_action(sim,fly.name,action); sim.step()
            if not np.isfinite(sim.mj_data.qpos).all(): raise FloatingPointError('Nonfinite physical state')
            if (step+1)%10==0: measure((step+1)*sim.timestep)
            if renderer: sim.render_as_needed()
    except Exception as exc:
        status='failed'; failure=dict(type=type(exc).__name__,message=str(exc),physics_step=step)
    if status=='complete' and (len(records)!=2001 or len(neural_records)!=400 or len(decisions)!=36):
        status='failed'; failure=dict(type='IncompleteBudget',message='Missing declared observations or decisions')
    elapsed=time.perf_counter()-started
    arrays={name:np.asarray([row[key] for row in records]) for name,key in
        [('time_s','time'),('thorax_position_mm','position'),('thorax_quaternion_wxyz','rotation'),
         ('yaw_rad','yaw'),('true_bearing_error_rad','error'),('target_mm','target'),
         ('target_distance_mm','distance'),('thorax_up_z','up'),('contact_count','contacts'),
         ('qpos','qpos'),('descending_signal','signal')]}
    arrays.update({name:np.asarray([row[key] for row in neural_records]) for name,key in
        [('neural_time_s','time_s'),('neural_phase','phase'),('sensory','sensory'),('query','query')]})
    if neural is not None:
        arrays.update({name:np.asarray([row[key] for row in neural_records]) for name,key in
            [('fast_state','fast'),('slow_state','slow'),('logits','logits')]})
    observed_seconds=float(arrays['time_s'][-1])
    phase_progress=[]
    if status=='complete':
        boundaries=(0.,1.,2.) if case['scenario']=='switch' else (0.,2.)
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            target=target_at(case['scenario'],left)
            before=float(np.linalg.norm(target-arrays['thorax_position_mm'][round(left*1000),:2]))
            after=float(np.linalg.norm(target-arrays['thorax_position_mm'][round(right*1000),:2]))
            phase_progress.append(dict(start_s=left,end_s=right,target_mm=target.tolist(),
                start_distance_mm=before,end_distance_mm=after,progress_toward_target_mm=before-after))
    with archive_path.with_suffix('.tmp').open('wb') as handle: np.savez_compressed(handle,**arrays)
    archive_path.with_suffix('.tmp').replace(archive_path)
    csv_path=OUTPUT/(label+'.csv')
    with csv_path.open('w',newline='',encoding='utf8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['time_s','x_mm','y_mm','z_mm','yaw_rad','true_bearing_error_rad',
            'target_x_mm','target_y_mm','distance_mm','up_z','contacts','drive_0','drive_1'])
        writer.writerows((r['time'],*r['position'],r['yaw'],r['error'],*r['target'],r['distance'],
                          r['up'],r['contacts'],*r['signal']) for r in records)
    result=dict(case=case,repeat=repeat,status=status,failure=failure,
        study_identity_sha256=sha(OUTPUT/'identity.json'),trajectory_sha256=sha(archive_path),
        csv_sha256=sha(csv_path),recorded_samples=len(records),neural_frames=len(neural_records),
        decisions=decisions,wall_seconds=elapsed,includes_rendering=video,
        model_checkpoint=None if neural is None else neural.record,
        metrics=dict(mean_absolute_bearing_error_rad=float(np.trapezoid(np.abs(arrays['true_bearing_error_rad']),arrays['time_s'])/observed_seconds) if observed_seconds else None,
            observed_seconds=observed_seconds,phase_progress=phase_progress,
            heading_change_rad=float(np.unwrap(arrays['yaw_rad'])[-1]-arrays['yaw_rad'][0]),
            final_absolute_bearing_error_rad=float(abs(arrays['true_bearing_error_rad'][-1])),
            final_target_distance_mm=float(arrays['target_distance_mm'][-1]),
            minimum_thorax_height_mm=float(arrays['thorax_position_mm'][:,2].min()),
            minimum_thorax_up_z=float(arrays['thorax_up_z'].min()),
            mean_contact_count=float(arrays['contact_count'].mean()),
            descending_signal_switches=int(np.any(np.diff(arrays['descending_signal'],axis=0)!=0,axis=1).sum())))
    if renderer:
        renderer.save_video(OUTPUT/(label+'.mp4'))
        result['video']=dict(sha256=sha(OUTPUT/(label+'.mp4')),playback_speed=.25)
    write(report_path,result)
    if status!='complete': raise RuntimeError(f'Recorded failed physical trial: {label}: {failure}')
    print(json.dumps(dict(label=label,status=status,metrics=result['metrics'],wall_seconds=elapsed)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=[r['label'] for r in cases()])
    args=parser.parse_args(); OUTPUT.mkdir(parents=True,exist_ok=True)
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.SetPriorityClass.argtypes=(wintypes.HANDLE,wintypes.DWORD)
        kernel.SetPriorityClass.restype=wintypes.BOOL
        kernel.GetPriorityClass.argtypes=(wintypes.HANDLE,)
        kernel.GetPriorityClass.restype=wintypes.DWORD
        process=kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(process,0x4000): raise ctypes.WinError(ctypes.get_last_error())
        if kernel.GetPriorityClass(process)!=0x4000: raise OSError('Physical worker priority was not lowered')
    with assay_lease(): run_assay(args.case)


def run_assay(selected_case=None):
    identity=study_identity(); identity_path=OUTPUT/'identity.json'
    if identity_path.exists() and read(identity_path)!=identity:
        raise ValueError('The frozen closed-loop assay identity changed')
    parity=verify_parity(PACKAGE)
    if not identity_path.exists(): write(identity_path,identity)
    write(OUTPUT/'runtime-parity.json',dict(environment=identity['environment'],models=parity))
    selected=[r for r in cases() if not selected_case or r['label']==selected_case]
    results=[run_trial(case,identity) for case in selected]
    if not selected_case:
        repeat_case=next(row for row in cases() if row['label']==VIDEO_CASE)
        repeated=run_trial(repeat_case,identity,repeat=True)
        with np.load(OUTPUT/(VIDEO_CASE+'.npz'),allow_pickle=False) as first, np.load(OUTPUT/(VIDEO_CASE+'-repeat.npz'),allow_pickle=False) as second:
            if set(first.files)!=set(second.files) or any(not np.array_equal(first[name],second[name]) for name in first.files):
                raise ValueError('The repeated physical/neural trajectory did not reproduce exactly')
        if next(row for row in results if row['case']['label']==VIDEO_CASE)['decisions']!=repeated['decisions']:
            raise ValueError('Repeated control decisions changed')
        write(ROOT/'reports/embodiment/closed-loop.json',dict(identity=identity,
            identity_sha256=sha(identity_path),trials=results,repeat=dict(case=VIDEO_CASE,all_recorded_arrays_exact=True),
            runtime_parity=parity,scope=identity['statement']))


if __name__=='__main__': main()
