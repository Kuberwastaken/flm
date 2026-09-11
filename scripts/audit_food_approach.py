"""Replay supplied physical food-approach sensors, policy commands and outcomes.

Requires NumPy for the stored arrays, but imports no FLM or FlyGym implementation.
This is a record audit, not independent physics execution or learned behavior.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .audit_food_sensor_probe import replay, close
except ImportError:
    from audit_food_sensor_probe import replay, close

CASES = [
    ('odor-a-left', 1, 1., 0., False, 'odor', None),
    ('odor-a-right', -1, 1., 0., False, 'odor', None),
    ('odor-a-left-reversed', 1, 0., 1., False, 'odor', None),
    ('odor-a-left-neutral', 1, 0., 0., False, 'odor', None),
    ('odor-a-left-missing', 1, 1., 0., True, 'odor', None),
    ('straight-a-left', 1, 1., 0., False, 'straight', None),
    ('odor-a-left-repeat', 1, 1., 0., False, 'odor', 'odor-a-left')]
KEYS = ('time_s', 'body_positions_mm', 'thorax_quaternion_wxyz', 'qpos', 'sensory', 'raw_odor', 'contact_mask', 'descending_signal')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError('Metric fields changed')
        for name, value in expected.items(): same(actual[name], value)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError('Metric inventory changed')
        for a, b in zip(actual, expected): same(a, b)
    elif expected is None or type(expected) in (str, bool):
        if type(actual) is not type(expected) or actual != expected: raise ValueError('Metric value changed')
    else: close(actual, expected)


def audit_trial(declared, record, arrays):
    case = record['case']; field = declared['fields'][case['label']]
    n = record['observations']; names = record['body_names']
    if record['status'] not in ('complete', 'failed'): raise ValueError('Unknown trial status')
    if record['status'] == 'complete' and (n != 201 or record['physics_steps'] != 20000 or record['failure'] is not None):
        raise ValueError('Incomplete physical budget')
    if record['status'] == 'failed' and not isinstance(record['failure'], dict): raise ValueError('Missing failure record')
    if set(arrays) != set(KEYS) or not 0 <= n <= 201: raise ValueError('Incomplete physical arrays')
    if n == 0:
        if record['status'] != 'failed' or record['metrics'] is not None or any(a.size for a in arrays.values()):
            raise ValueError('Invalid empty failed record')
        return 0
    if len(names) != 69 or len(set(names)) != 69: raise ValueError('Body identity inventory changed')
    shapes = dict(time_s=(n,), body_positions_mm=(n, 69, 3), thorax_quaternion_wxyz=(n, 4),
        sensory=(n, 6), raw_odor=(n, 2, 2), contact_mask=(n, 6, 2), descending_signal=(n, 2))
    for key, shape in shapes.items():
        if arrays[key].shape != shape: raise ValueError('Physical array shape changed: '+key)
    if arrays['qpos'].ndim != 2 or arrays['qpos'].shape[0] != n or arrays['qpos'].shape[1] < 42:
        raise ValueError('Invalid generalized position shape')
    if arrays['contact_mask'].dtype != np.bool_: raise ValueError('Contact masks must be boolean')
    if any(not np.isfinite(a).all() for a in arrays.values()): raise ValueError('Nonfinite physical record')
    if not np.allclose(arrays['time_s'], np.arange(n)*.01, atol=1e-14, rtol=0): raise ValueError('Physical clock changed')
    stop = False
    antenna_indices = [names.index(x) for x in declared['odor_body_origins']]
    foot_indices = [names.index(x) for x in declared['contact_body_origins']]
    for i in range(n):
        points = arrays['body_positions_mm'][i]
        expected = replay(field, points[antenna_indices].tolist(), points[foot_indices].tolist(), case['missing_odor'])
        for key in ('sensory', 'raw_odor', 'contact_mask'): close(arrays[key][i].tolist(), expected[key])
        values = arrays['sensory'][i].tolist(); stop = stop or values[5] >= .5
        turn = max(-.4, min(.4, 24*(values[0]-values[1])/(values[0]+values[1]+1e-12))) if case['mode'] == 'odor' else 0.
        signal = [0., 0.] if stop else [.8-turn, .8+turn]
        close(arrays['descending_signal'][i].tolist(), signal)
    hits_by_source = arrays['contact_mask'].any(axis=1); hits = hits_by_source.any(axis=1)
    first = int(np.flatnonzero(hits)[0]) if hits.any() else None
    source_indices = np.flatnonzero(hits_by_source[first]).tolist() if first is not None else []
    body = arrays['body_positions_mm'][:, names.index('c_thorax')]
    quat = arrays['thorax_quaternion_wxyz']; sources = field['sources']; times = arrays['time_s']
    metrics = dict(first_contact_s=float(times[first]) if first is not None else None,
        first_contact_sources=[sources[i]['name'] for i in source_indices],
        first_contact_sugar=max((sources[i]['sugar'] for i in source_indices), default=None),
        contact_latency_censored=first is None, censor_time_s=float(times[-1]) if first is None else None,
        all_contacted_sources=[s['name'] for i, s in enumerate(sources) if hits_by_source[:, i].any()],
        minimum_thorax_planar_distance_mm={s['name']: float(np.linalg.norm(body[:, :2]-np.array(s['position_mm'][:2]), axis=1).min()) for s in sources},
        final_thorax_position_mm=body[-1].tolist(), minimum_thorax_height_mm=float(body[:, 2].min()),
        minimum_thorax_up_z=float((1-2*(quat[:, 1]**2+quat[:, 2]**2)).min()))
    same(record['metrics'], metrics)
    return n


def audit(folder):
    folder = Path(folder); declared = json.loads((folder/'identity.json').read_bytes())
    summary = json.loads((folder/'summary.json').read_bytes()); identity_hash = sha(folder/'identity.json')
    expected_cases = [dict(zip(('label', 'a_side', 'a_sugar', 'b_sugar', 'missing_odor', 'mode', 'repeat_of'), row)) for row in CASES]
    if declared['cases'] != expected_cases or [r['case'] for r in summary['trials']] != expected_cases:
        raise ValueError('Declared seven-case inventory changed')
    if (declared['language_model_loaded'] is not False or declared['neural_policy_used'] is not False
            or declared['training_updates'] != 0 or declared['physics_seed'] != 17
            or declared['duration_s'] != 2. or declared['physics_timestep_s'] != .0001
            or declared['control_step_s'] != .01 or declared['expected_observations'] != 201
            or declared['expected_physics_steps'] != 20000): raise ValueError('Physical reference scope changed')
    if declared['channels'] != ['odor_a_left', 'odor_a_right', 'odor_b_left', 'odor_b_right', 'sugar_contact', 'source_contact']:
        raise ValueError('Channel order changed')
    policy = declared['policy']
    if (policy['control_step_s'] != .01 or policy['base_drive'] != .8 or policy['steering_gain'] != 24.
            or policy['maximum_turn'] != .4 or policy['contrast_epsilon'] != 1e-12
            or policy['coordinates_or_reward_identity_input'] is not False or policy['training_updates'] != 0
            or policy['used_channels'] != ['odor_a_left', 'odor_a_right', 'source_contact']
            or policy['unused_channels'] != ['odor_b_left', 'odor_b_right', 'sugar_contact']):
        raise ValueError('Declared policy differs from reference')
    if declared['odor_body_origins'] != ['l_funiculus', 'r_funiculus'] or declared['contact_body_origins'] != [f'{leg}_tarsus5' for leg in ('lf','lm','lh','rf','rm','rh')]:
        raise ValueError('Sensor attachment changed')
    for case in expected_cases:
        expected_sources = [dict(name=name, position_mm=[8., y, 0.], spread_mm=[5., 4., 2.],
            odor=odor, sugar=sugar, contact_radius_mm=.75, contact_height_mm=.25)
            for name, y, odor, sugar in (('a', 3.*case['a_side'], [1., 0.], case['a_sugar']), ('b', -3.*case['a_side'], [0., 1.], case['b_sugar']))]
        same(declared['fields'][case['label']]['sources'], expected_sources)
    if summary['identity_sha256'] != identity_hash: raise ValueError('Summary identity differs')
    saved_arrays = {}; total = 0
    for record in summary['trials']:
        label = record['case']['label']; path = folder/(label+'.npz')
        if json.loads((folder/(label+'.json')).read_bytes()) != record: raise ValueError('Standalone trial differs from summary')
        if record['identity_sha256'] != identity_hash or sha(path) != record['trajectory_sha256']:
            raise ValueError('Trial identity or trajectory checksum differs')
        with np.load(path, allow_pickle=False) as archive: arrays = {k: archive[k] for k in archive.files}
        total += audit_trial(declared, record, arrays); saved_arrays[label] = arrays
    complete = sum(r['status'] == 'complete' for r in summary['trials'])
    if summary['complete_cases'] != complete or summary['failed_cases'] != 7-complete: raise ValueError('Outcome counts changed')
    comparisons = []
    for first, second, all_arrays in (
        ('odor-a-left', 'odor-a-left-repeat', True), ('odor-a-left', 'odor-a-left-reversed', False),
        ('odor-a-left', 'odor-a-left-neutral', False), ('odor-a-left-missing', 'straight-a-left', False)):
        keys = KEYS if all_arrays else ('time_s', 'body_positions_mm', 'thorax_quaternion_wxyz', 'qpos', 'descending_signal')
        comparisons.append(dict(first=first, second=second,
            both_complete=all(r['status'] == 'complete' for r in summary['trials'] if r['case']['label'] in (first, second)),
            exact_array_equal={k: bool(np.array_equal(saved_arrays[first][k], saved_arrays[second][k])) for k in keys}))
    same(summary['paired_replays'], comparisons)
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), identity_sha256=identity_hash,
        summary_sha256=sha(folder/'summary.json'), auditor_sha256=sha(__file__), sensor_auditor_sha256=sha(Path(__file__).with_name('audit_food_sensor_probe.py')),
        trials=7, complete_cases=complete, failed_cases=7-complete, observations_replayed=total,
        paired_replays=comparisons, physical_simulation_rerun=False, language_or_learning_result=False,
        scope='Independent scalar sensor and policy replay plus outcome arithmetic from supplied physical arrays. No independent physics rerun or learned behavior validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    result = audit(args.folder); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as handle: json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))
