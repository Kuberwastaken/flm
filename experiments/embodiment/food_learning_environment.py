"""FlyGym adapter for sampled-action food episodes; no scheduler or training budget.

Use the pinned embodiment environment. The existing reference factory, sensory
geometry and gait are reused unchanged. The caller owns each episode's learner,
study membership, checkpoints, resource lifetime and complete result inventory.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from flm.food_episode import Clock, sampled_drive
from flm.food_sensors import FoodField, body_sensor_positions


class FoodLearningEnvironment:
    def __init__(self, field, clock, *, gait_seed, missing_odor):
        if (type(field) is not FoodField or type(clock) is not Clock or type(gait_seed) is not int
                or not 0 <= gait_seed < 2**32 or type(missing_odor) is not bool):
            raise ValueError('Explicit food field, clock, gait seed and odor intervention required')
        clock.validate()
        if importlib.metadata.version('flygym') != '2.1.0': raise ValueError('Use pinned FlyGym 2.1.0')
        from experiments.embodiment.food_approach import (make_simulation, HybridTurningController,
            HybridControllerObservation, LocomotionAction, PreprogrammedSteps, apply_locomotion_action)
        self.factory = make_simulation; self.controller_type = HybridTurningController
        self.observation_type = HybridControllerObservation; self.action_type = LocomotionAction
        self.steps_type = PreprogrammedSteps; self.apply = apply_locomotion_action
        self.field = FoodField(field.sources); self.clock = clock; self.gait_seed = gait_seed
        self.missing_odor = missing_odor; self.physics_step = 0; self.sim = None; self.ready = False

    def identity(self):
        root = Path(__file__).resolve().parents[2]
        files = ['experiments/embodiment/food_learning_environment.py','experiments/embodiment/calibrate.py',
                 'experiments/embodiment/food_approach.py','flm/food_sensors.py','flm/food_episode.py',
                 'requirements-embodied-lock.txt']
        return json.loads(json.dumps(dict(format='flm-sampled-food-environment-v1',field=self.field.card(),
            gait_seed=self.gait_seed,missing_odor=self.missing_odor,clock=vars(self.clock),
            versions={name:importlib.metadata.version(name) for name in ('flygym','mujoco','numpy','scipy','numba')},
            sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
            installed_sources={implementation.__module__:hashlib.sha256(Path(sys.modules[implementation.__module__].__file__).read_bytes()).hexdigest()
                               for implementation in (self.controller_type,self.steps_type,self.apply)},
            body_record='Unmodified cached body origins plus qpos and thorax quaternion. The known invalid c_head row aliases a hind foot; the six feet and two antennae used for sensors are the calibrated body origins.',
            scope='Sampled descending commands through the existing designed gait; no learned walking or feeding'),allow_nan=False))

    def reset(self):
        if self.sim is not None: raise ValueError('Create a fresh physical environment for each episode')
        self.fly,self.sim,_ = self.factory()
        order = self.fly.get_actuated_jointdofs_order('position')
        if self.sim.timestep != self.clock.physics_timestep_s or len(order) != 42:
            raise ValueError('Calibrated timestep or physical action interface changed')
        steps = self.steps_type()
        self.gait = self.controller_type(timestep=self.sim.timestep,preprogrammed_steps=steps,output_dof_order=order)
        self.sim.reset(); self.gait.reset(seed=self.gait_seed)
        self.apply(self.sim,self.fly.name,self.action_type(
            joint_angles=steps.default_pose_by_dof_order(order),adhesion_onoff=np.ones(6,dtype=bool)))
        self.sim.warmup()
        self.names = [body.name for body in self.fly.get_bodysegs_order()]
        self.thorax = self.names.index('c_thorax'); self.physics_step = 0; self.ready = True
        return self.observe()

    def observe(self):
        if not self.ready: raise ValueError('Reset the physical environment first')
        positions = self.sim.get_body_positions(self.fly.name).copy()
        quaternion = self.sim.get_body_rotations(self.fly.name)[self.thorax].copy()
        qpos = self.sim.mj_data.qpos.copy()
        if any(not np.isfinite(value).all() for value in (positions,quaternion,qpos)):
            raise FloatingPointError('Nonfinite physical observation')
        antennae,feet = body_sensor_positions(self.names,positions)
        sensed = self.field.observe(antennae,feet,missing_odor=self.missing_odor)
        return dict(sensory=sensed['sensory'],record=dict(physics_step=self.physics_step,
            time_s=self.physics_step*self.clock.physics_timestep_s,simulation_time_s=float(self.sim.mj_data.time),
            body_names=self.names,body_positions_mm=positions.tolist(),thorax_quaternion_wxyz=quaternion.tolist(),
            qpos=qpos.tolist(),raw_odor=sensed['raw_odor'].tolist(),contact_mask=sensed['contact_mask'].tolist(),
            diagnostic_contacted_sources=sensed['diagnostic_contacted_sources']))

    def advance(self, descending_signal, physics_steps):
        if not self.ready: raise ValueError('Reset the physical environment first')
        signal = np.array(descending_signal,dtype=np.float64,copy=True)
        if (type(physics_steps) is not int or physics_steps != self.clock.steps_per_action
                or self.physics_step+physics_steps > self.clock.maximum_physics_steps
                or signal.shape != (2,) or not any(np.array_equal(signal,sampled_drive(action)) for action in range(3))):
            raise ValueError('Apply one declared complete sampled-action interval within the horizon')
        for _ in range(physics_steps):
            mechanical = self.observation_type.from_sim(self.sim,self.fly.name)
            self.apply(self.sim,self.fly.name,self.gait.step(signal,mechanical)); self.sim.step()
            self.physics_step += 1
            if not np.isfinite(self.sim.mj_data.qpos).all(): raise FloatingPointError('Nonfinite physics state')
        return self.observe()

    def close(self):
        if self.sim is not None: self.sim.close()
        self.ready = False
