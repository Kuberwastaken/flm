"""Declared pose-to-cue wrapper; keeps the sampled observation causal."""
import math
import numpy as np

NEURAL_STEP_S = .005
CYCLE_FRAMES = 11
DEADBAND_RAD = .10
SCENARIOS = ('positive', 'negative', 'switch')
MODELS = ('initial', 'bptt', 'reservoir', 'eligibility')


def target_at(scenario, time_s):
    if scenario not in SCENARIOS or not math.isfinite(time_s) or time_s < 0:
        raise ValueError('Invalid scenario or time')
    negative = scenario == 'negative' or (scenario == 'switch' and time_s >= 1.)
    return np.array([40., -20. if negative else 20.])


def bearing_error(position, yaw, target):
    position = np.asarray(position, dtype=np.float64); target = np.asarray(target, dtype=np.float64)
    if position.shape != (3,) or target.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(target).all() or not math.isfinite(yaw):
        raise ValueError('Expected finite world pose and target')
    delta = target - position[:2]
    if np.linalg.norm(delta) < 1e-8:
        raise ValueError('Target coincides with the sensed position')
    return float((math.atan2(delta[1], delta[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)


def cases():
    rows = [dict(label=f'{model}-{mode}-{scenario}', model=model, pose_mode=mode, scenario=scenario)
            for model in MODELS for mode in ('live', 'frozen') for scenario in SCENARIOS]
    rows += [dict(label=f'scripted-live-{scenario}', model='scripted', pose_mode='live', scenario=scenario)
             for scenario in SCENARIOS]
    return rows


class FeedbackController:
    def __init__(self, model, scenario, pose_mode, initial_position, initial_yaw):
        if scenario not in SCENARIOS or pose_mode not in ('live', 'frozen'):
            raise ValueError('Unknown feedback condition')
        self.model = model; self.scenario = scenario; self.pose_mode = pose_mode
        self.initial_position = np.asarray(initial_position, dtype=np.float64).copy()
        self.initial_yaw = float(initial_yaw)
        self.frames = 0; self.signal = np.ones(2); self.sample = None

    def frame(self, time_s, position, yaw):
        expected_time = self.frames * NEURAL_STEP_S
        if not math.isclose(time_s, expected_time, abs_tol=1e-10, rel_tol=0):
            raise ValueError('Neural frames must follow the declared physical clock')
        phase = self.frames % CYCLE_FRAMES
        if phase == 0:
            sensed_position = position if self.pose_mode == 'live' else self.initial_position
            sensed_yaw = yaw if self.pose_mode == 'live' else self.initial_yaw
            target = target_at(self.scenario, time_s)
            error = bearing_error(sensed_position, sensed_yaw, target)
            self.sample = dict(time_s=time_s, position_mm=np.asarray(sensed_position).tolist(),
                               yaw_rad=float(sensed_yaw), target_mm=target.tolist(), bearing_error_rad=error,
                               cue=0 if error >= 0 else 1)
            if self.model is not None: self.model.reset()
        sensory = np.zeros(4, dtype=np.float32)
        if phase < 2: sensory[self.sample['cue']] = 1.
        if phase == 10: sensory[3] = 1.
        if self.model is not None:
            logits, probabilities = self.model.step(sensory)
            fast = self.model.fast.copy(); slow = self.model.slow.copy()
        else:
            logits = probabilities = None; fast = slow = None
        record = dict(time_s=time_s, frame=self.frames, phase=phase, sensory=sensory,
                      fast=fast, slow=slow, logits=logits, query=phase == 10)
        decision = None
        if phase == 10:
            action = int(np.argmax(logits)) if logits is not None else self.sample['cue']
            straight = abs(self.sample['bearing_error_rad']) <= DEADBAND_RAD
            self.signal = np.array([1.,1.] if straight else ([.4,1.2] if action == 0 else [1.2,.4]))
            decision = dict(time_s=time_s, sampled=self.sample.copy(), action=action,
                probabilities=None if probabilities is None else probabilities.tolist(),
                straight_deadband=straight, descending_signal=self.signal.tolist())
        self.frames += 1
        return record, decision
