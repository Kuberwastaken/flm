"""Calibrate actual NeuroMechFly locomotion before attaching an FLM controller.

Run with the isolated Python 3.12 environment in requirements-embodied-lock.txt.
The upstream hybrid controller is a designed feedback policy, not a learned FLM.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (HybridTurningController, HybridControllerObservation,
    LocomotionAction, PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly)


def make_simulation():
    fly = make_locomotion_fly(name='flm_body', add_adhesion=True, colorize=True)
    camera = fly.add_tracking_camera(name='side', pos_offset=(-.5, -7.5, 0.),
        rotation=Rotation3D('euler', (1.57, 0., 0.)), fovy=35.)
    world = FlatGroundWorld()
    world.add_fly(fly, [0., 0., .8], Rotation3D('quat', [1., 0., 0., 0.]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False)
    simulation = Simulation(world)
    return fly, simulation, camera


def trial(signal, seed, seconds, label, output, video=False):
    fly, sim, camera = make_simulation()
    steps = PreprogrammedSteps(); order = fly.get_actuated_jointdofs_order('position')
    controller = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=order)
    sim.reset(); controller.reset(seed=seed)
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(order), adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    renderer = sim.set_renderer([camera], camera_res=(360, 480), playback_speed=.25, output_fps=25) if video else None
    thorax = fly.get_bodysegs_order().index(BodySegment('c_thorax'))
    positions, rotations, times, contacts, qpos = [], [], [], [], []
    initial_position = sim.get_body_positions(fly.name)[thorax].copy()
    count = round(seconds / sim.timestep); started = time.perf_counter()
    for step in range(count):
        observation = HybridControllerObservation.from_sim(sim, fly.name)
        action = controller.step(np.asarray(signal), observation)
        apply_locomotion_action(sim, fly.name, action); sim.step()
        if not np.isfinite(sim.mj_data.qpos).all(): raise FloatingPointError('Nonfinite physics state')
        if step % 10 == 0 or step == count - 1:
            times.append((step + 1) * sim.timestep)
            positions.append(sim.get_body_positions(fly.name)[thorax].copy())
            rotations.append(sim.get_body_rotations(fly.name)[thorax].copy())
            contacts.append(sim.mj_data.ncon); qpos.append(sim.mj_data.qpos.copy())
        if renderer: sim.render_as_needed()
    elapsed = time.perf_counter() - started
    positions = np.asarray(positions); rotations = np.asarray(rotations)
    w, x, y, z = rotations.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    up_z = 1 - 2 * (x * x + y * y)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'{label}.npz'
    np.savez_compressed(archive, time_s=times, thorax_position_mm=positions,
        thorax_quaternion_wxyz=rotations, yaw_rad=yaw, thorax_up_z=up_z, contact_count=contacts, qpos=qpos)
    with (output / f'{label}.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream); writer.writerow(['time_s','x_mm','y_mm','z_mm','yaw_rad','up_z','contacts'])
        writer.writerows((t, *position, angle, up, contact) for t, position, angle, up, contact in zip(times, positions, yaw, up_z, contacts))
    if renderer: renderer.save_video(output / f'{label}.mp4')
    result = dict(label=label, seed=seed, descending_signal=signal, simulation_seconds=seconds,
        physics_steps=count, timestep_seconds=sim.timestep, recorded_samples=len(times), actuated_joint_dofs=len(order),
        displacement_mm=(positions[-1] - initial_position).tolist(), yaw_change_rad=float(yaw[-1] - yaw[0]),
        minimum_thorax_height_mm=float(positions[:, 2].min()), minimum_thorax_up_z=float(up_z.min()),
        mean_contact_count=float(np.mean(contacts)), wall_seconds=elapsed, includes_rendering=video,
        trajectory_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print(json.dumps(result), flush=True); return result, positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/embodiment/calibration'))
    p.add_argument('--seconds', type=float, default=1.)
    p.add_argument('--video', action='store_true')
    a = p.parse_args()
    if not 0 < a.seconds <= 30: p.error('Use a positive calibration duration of at most 30 seconds')
    results = []; trajectories = []
    for signal, name in [([1., 1.], 'symmetric'), ([1.2, .4], 'left_drive'), ([.4, 1.2], 'right_drive'), ([1., 1.], 'symmetric_repeat')]:
        result, trajectory = trial(signal, 17, a.seconds, name, a.output, a.video and name == 'left_drive')
        results.append(result); trajectories.append(trajectory)
    repeat_error = float(np.max(np.abs(trajectories[0] - trajectories[-1])))
    if repeat_error > 1e-10: raise ValueError(f'Identical-seed physical replay diverged: {repeat_error}')
    report = dict(platform=platform.platform(), python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in ('flygym','mujoco','numpy','scipy','numba')},
        source='https://neuromechfly.org/tutorials/4d_turning_controller/',
        controller='Upstream designed hybrid turning controller with preprogrammed steps and contact feedback',
        flm_connected=False, learning_used=False, repeat_max_position_error_mm=repeat_error, trials=results,
        caution='Physics and controller calibration only. Descending commands are prescribed; these trials do not demonstrate learned behavior or language transfer.')
    destination = Path('reports/embodiment/calibration.json'); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__': main()
