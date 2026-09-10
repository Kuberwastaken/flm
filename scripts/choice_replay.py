"""Derive a compact browser replay from all 40 completed physical-choice trials.

No fitting, physical reruns, interpolation, checkpoint selection or new scores.
The two distinct paths are shared only after exact full-array equality checks.
"""
from pathlib import Path
import json
import numpy as np
from flm.provenance import sha256, write_json

METHODS = ('bptt', 'reservoir', 'eligibility', 'instantaneous', 'reward')
STEPS = (0, 300, 600, 900)


def build(root):
    source = root / 'reports/embodiment/learned-choice.json'
    report = json.loads(source.read_text(encoding='utf8'))
    summary = json.loads((root / 'public/research/learned-choice.json').read_text(encoding='utf8'))
    if sha256(source) != summary['report_sha256']:
        raise ValueError('The published physical-choice identity changed')
    if (root / 'public/research/learned-choice-records.json').read_bytes() != source.read_bytes():
        raise ValueError('Public and original physical-choice records differ')
    expected = {(method, step, cue) for method in METHODS for step in STEPS for cue in (0, 1)}
    if len(report['trials']) != len(expected):
        raise ValueError('All 40 original conditions are required')
    raw_paths, paths, cases, initial = {}, {}, [], {}
    for trial in report['trials']:
        case, physical = trial['identity']['case'], trial['physical']
        episode = case['neural_episode']
        method, step, cue = case['method'], episode['step'], episode['cue']
        key = (method, step, cue)
        if key not in expected:
            raise ValueError('Unexpected or duplicate physical condition')
        expected.remove(key)
        label = f'{method}-{step:06d}-cue{cue}'
        probabilities = np.asarray(episode['action_probabilities'])
        action = episode['chosen_action']
        if (case['label'] != label or physical['label'] != label or
                case['training_seed'] != 17 or physical['seed'] != 17 or
                episode['delay'] != 8 or episode['distractor'] != 'zero' or
                probabilities.shape != (2,) or not np.isfinite(probabilities).all() or
                np.any(probabilities < 0) or np.any(probabilities > 1) or
                not np.isclose(probabilities.sum(), 1, atol=1e-6) or
                action != int(probabilities.argmax()) or
                episode['expected_action'] != (1 - cue if step == 600 else cue)):
            raise ValueError('Neural condition identity or recorded decision changed')
        if step == 0:
            stimulus = [episode[name] for name in ('fast_state', 'slow_state', 'action_probabilities')]
            if cue in initial and initial[cue] != stimulus:
                raise ValueError('Initial responses differ across learning methods')
            initial[cue] = stimulus
        signal = [0.4, 1.2] if action == 0 else [1.2, 0.4]
        if episode['descending_signal'] != signal or physical['descending_signal'] != signal:
            raise ValueError('Recorded action and physical command differ')
        original = root / 'runs/embodiment/learned-choice-v1' / f'{label}.npz'
        if sha256(original) != physical['trajectory_sha256']:
            raise ValueError('Recorded trajectory checksum changed')
        with np.load(original, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in ('time_s', 'thorax_position_mm', 'yaw_rad', 'qpos')}
        t, position, yaw = (arrays[name] for name in ('time_s', 'thorax_position_mm', 'yaw_rad'))
        if (t.shape != (1001,) or position.shape != (1001, 3) or yaw.shape != (1001,) or
                arrays['qpos'].shape != (1001, 73) or
                any(not np.isfinite(value).all() for value in arrays.values()) or
                not np.all(np.diff(t) > 0) or not np.isclose(t[0], .0001, atol=1e-12, rtol=0) or
                not np.isclose(t[-1], 1, atol=1e-12, rtol=0) or
                not np.isclose(position[:, 2].min(), physical['minimum_thorax_height_mm'], atol=1e-10, rtol=0) or
                not np.isclose(yaw[-1] - yaw[0], physical['yaw_change_rad'], atol=1e-10, rtol=0)):
            raise ValueError('Physical coverage or summary disagrees with arrays')
        # Summary displacement uses the pre-step reset pose, which is not stored
        # in the NPZ. Keep the actual first sampled time; never invent a t=0 pose.
        if action in raw_paths and any(not np.array_equal(arrays[name], raw_paths[action][name]) for name in arrays):
            raise ValueError('Equal commands do not have exactly equal recorded paths')
        raw_paths[action] = arrays
        path_id = f'action-{action}'
        indices = np.arange(0, 1001, 10)
        paths[path_id] = dict(time_s=t[indices].tolist(), x_mm=position[indices, 0].tolist(),
                              y_mm=position[indices, 1].tolist(), yaw_rad=yaw[indices].tolist())
        cases.append(dict(method=method, step=step, cue=cue, path=path_id,
            chosen_action=action, expected_action=episode['expected_action'],
            action_probabilities=probabilities.tolist(), checkpoint_sha256=case['checkpoint_sha256'],
            trajectory_sha256=physical['trajectory_sha256']))
    if expected or set(paths) != {'action-0', 'action-1'}:
        raise ValueError('Incomplete physical path inventory')
    for case in cases:
        published = next(row for row in summary['cases'] if all(row[k] == case[k] for k in ('method', 'step', 'cue')))
        for key in ('chosen_action', 'expected_action', 'trajectory_sha256'):
            if case[key] != published[key]:
                raise ValueError('Published case mapping changed')
        if published['action_probability_0'] != case['action_probabilities'][0]:
            raise ValueError('Published neural probability changed')
    points = np.concatenate([np.column_stack((path['x_mm'], path['y_mm'])) for path in paths.values()])
    low, high = points.min(0), points.max(0)
    center, side = (low + high) / 2, float(max(high - low) + 4)
    return dict(schema_version=1, complete=True, training_seed=17, physics_seed=17,
        source_report_sha256=sha256(source), source_summary_sha256=sha256(root / 'public/research/learned-choice.json'),
        generator_sha256=sha256(Path(__file__)), original_samples_per_trial=1001,
        retained_indices=list(range(0, 1001, 10)),
        full_recorded_arrays_equal_within_each_action=True,
        bounds_mm=dict(x_min=float(center[0] - side/2), x_max=float(center[0] + side/2),
                       y_min=float(center[1] - side/2), y_max=float(center[1] + side/2)),
        paths=paths, cases=cases,
        scope='Recorded sensory-choice behavior; separate from language training and food sensing. '
              'All 40 trials reduce to two designed motor commands and two exactly repeated paths. '
              'Every tenth recorded frame is retained without interpolation; first frame is at 0.0001 s.')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    replay = build(root)
    write_json(root / 'public/research/choice-replay.json', replay)
    print(f'Verified {len(replay["cases"])} original trials; exported two exact paths at 101 recorded frames.')
