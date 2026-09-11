"""Audit stored food-core physics using scalar sensors and complete neural replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .audit_food_approach import same, CASES
    from .audit_food_sensor_probe import replay, close
except ImportError:
    from audit_food_approach import same, CASES
    from audit_food_sensor_probe import replay, close
try:
    from experiments.embodiment.food_core_runtime import FoodCoreRuntime
except ModuleNotFoundError:
    from food_core_runtime import FoodCoreRuntime

MODEL_IDS = ['initial-s42', 'language-s42', 'initial-s43', 'language-s43']
KEYS = ('time_s', 'body_positions_mm', 'thorax_quaternion_wxyz', 'qpos', 'sensory', 'raw_odor',
        'contact_mask', 'descending_signal', 'fast', 'slow', 'features', 'logits', 'probabilities')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_trial(declared, record, arrays, core):
    case = record['case']; field = declared['fields'][case['label']]
    n = record['observations']; names = record['body_names']; core.reset()
    if record['status'] not in ('complete', 'failed'): raise ValueError('Unknown physical status')
    if record['status'] == 'complete' and (n != 201 or record['physics_steps'] != 20000 or record['failure'] is not None):
        raise ValueError('Incomplete physical budget')
    if record['status'] == 'failed' and not isinstance(record['failure'], dict): raise ValueError('Missing failure record')
    if set(arrays) != set(KEYS) or type(n) is not int or not 0 <= n <= 201: raise ValueError('Physical inventory changed')
    if n == 0:
        if record['status'] != 'failed' or record['metrics'] is not None or any(a.size for a in arrays.values()):
            raise ValueError('Invalid empty failure')
        return dict(observations=0, maximum_core_error={})
    if len(names) != 69 or len(set(names)) != 69: raise ValueError('Body inventory changed')
    shapes = dict(time_s=(n,), body_positions_mm=(n, 69, 3), thorax_quaternion_wxyz=(n, 4),
        sensory=(n, 6), raw_odor=(n, 2, 2), contact_mask=(n, 6, 2), descending_signal=(n, 2),
        fast=(n, len(core.fast)), slow=(n, len(core.slow)), features=(n, len(core.weights['norm_weight'])),
        logits=(n, 3), probabilities=(n, 3))
    if any(arrays[k].shape != shape for k, shape in shapes.items()): raise ValueError('Recorded shape changed')
    if arrays['qpos'].ndim != 2 or arrays['qpos'].shape[0] != n or arrays['qpos'].shape[1] < 42:
        raise ValueError('Invalid generalized coordinates')
    if arrays['contact_mask'].dtype != np.bool_ or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite record or nonboolean contacts')
    for name in ('sensory', 'fast', 'slow', 'features', 'logits', 'probabilities'):
        if arrays[name].dtype != np.float32: raise ValueError('Recorded neural dtype differs')
    if not np.allclose(arrays['time_s'], np.arange(n)*.01, atol=1e-14, rtol=0): raise ValueError('Control clock differs')
    antennae = [names.index(x) for x in declared['odor_body_origins']]
    feet = [names.index(x) for x in declared['contact_body_origins']]
    stop = False; maxima = {name: 0. for name in ('fast', 'slow', 'features', 'logits', 'probabilities')}
    for i in range(n):
        points = arrays['body_positions_mm'][i]
        expected = replay(field, points[antennae].tolist(), points[feet].tolist(), case['missing_odor'])
        close(arrays['raw_odor'][i].tolist(), expected['raw_odor'])
        close(arrays['contact_mask'][i].tolist(), expected['contact_mask'])
        np.testing.assert_allclose(arrays['sensory'][i], np.asarray(expected['sensory'], dtype=np.float32), atol=2e-8, rtol=2e-6)
        logits, probabilities, features = core.step(arrays['sensory'][i])
        for name, actual in (('fast', core.fast), ('slow', core.slow), ('features', features), ('logits', logits), ('probabilities', probabilities)):
            np.testing.assert_allclose(arrays[name][i], actual, atol=5e-6, rtol=5e-5)
            maxima[name] = max(maxima[name], float(np.max(np.abs(arrays[name][i]-actual))))
        recorded = arrays['probabilities'][i].astype(np.float64)
        if np.any((recorded < 0) | (recorded > 1)) or not np.isclose(recorded.sum(), 1., atol=1e-6, rtol=0):
            raise ValueError('Action probability normalization differs')
        stop = stop or bool(arrays['sensory'][i, 5] >= .5)
        turn = .4*float(recorded[2]-recorded[0])
        close(arrays['descending_signal'][i].tolist(), [0., 0.] if stop else [.8-turn, .8+turn])
    contacts = arrays['contact_mask'].any(axis=1); hit = contacts.any(axis=1)
    first = int(np.flatnonzero(hit)[0]) if hit.any() else None
    indices = np.flatnonzero(contacts[first]).tolist() if first is not None else []
    sources = field['sources']; body = arrays['body_positions_mm'][:, names.index('c_thorax')]
    quat = arrays['thorax_quaternion_wxyz']; times = arrays['time_s']
    expected_metrics = dict(first_contact_s=float(times[first]) if first is not None else None,
        first_contact_sources=[sources[i]['name'] for i in indices],
        first_contact_sugar=max((sources[i]['sugar'] for i in indices), default=None),
        contact_latency_censored=first is None, censor_time_s=float(times[-1]) if first is None else None,
        all_contacted_sources=[s['name'] for i, s in enumerate(sources) if contacts[:, i].any()],
        minimum_thorax_planar_distance_mm={s['name']: float(np.linalg.norm(body[:, :2]-np.array(s['position_mm'][:2]), axis=1).min()) for s in sources},
        final_thorax_position_mm=body[-1].tolist(), minimum_thorax_height_mm=float(body[:, 2].min()),
        minimum_thorax_up_z=float((1-2*(quat[:, 1]**2+quat[:, 2]**2)).min()))
    same(record['metrics'], expected_metrics)
    return dict(observations=n, maximum_core_error=maxima)


def audit(folder, bundle):
    folder = Path(folder); bundle = Path(bundle)
    declared = json.loads((folder/'identity.json').read_bytes()); summary = json.loads((folder/'summary.json').read_bytes())
    identity_hash = sha(folder/'identity.json')
    case_fields = ('label', 'a_side', 'a_sugar', 'b_sugar', 'missing_odor', 'mode', 'repeat_of')
    expected_cases = [dict(zip(case_fields, row)) for row in CASES if row[5] != 'straight']
    if (declared['model_ids'] != MODEL_IDS or declared['cases'] != expected_cases
            or declared['expected_trials'] != 24 or declared['training_updates'] != 0
            or declared['language_model_loaded'] is not True or declared['neural_policy_used'] is not True
            or declared['physics_seed'] != 17 or declared['expected_physics_steps'] != 20000
            or declared['expected_observations'] != 201 or declared['physics_timestep_s'] != .0001
            or declared['control_step_s'] != .01 or declared['duration_s'] != 2.):
        raise ValueError('Declared physical inventory or budget differs')
    if declared['channels'] != ['odor_a_left', 'odor_a_right', 'odor_b_left', 'odor_b_right', 'sugar_contact', 'source_contact']:
        raise ValueError('Sensory channel order changed')
    if (declared['odor_body_origins'] != ['l_funiculus', 'r_funiculus']
            or declared['contact_body_origins'] != [leg+'_tarsus5' for leg in ('lf','lm','lh','rf','rm','rh')]):
        raise ValueError('Physical sensor attachment differs')
    expected_policy = dict(action_order=['turn_right', 'straight', 'turn_left'], base_drive=.8, maximum_turn=.4,
        formula='turn=0.4*(p_left-p_right); drive=[0.8-turn,0.8+turn]',
        stop='Latch source_contact >= 0.5; thereafter both drives are zero',
        core_after_contact='Continue one recurrent transition per sample, even after the motor stop latches',
        probability_sampling=False, location_or_reward_identity_input=False,
        scope='Engineered continuous motor mapping and contact stop, not learned walking or feeding.')
    if declared['policy'] != expected_policy: raise ValueError('Motor interface changed')
    for case in expected_cases:
        expected_sources = [dict(name=name, position_mm=[8., y, 0.], spread_mm=[5., 4., 2.],
            odor=odor, sugar=sugar, contact_radius_mm=.75, contact_height_mm=.25)
            for name, y, odor, sugar in (('a', 3.*case['a_side'], [1., 0.], case['a_sugar']), ('b', -3.*case['a_side'], [0., 1.], case['b_sugar']))]
        same(declared['fields'][case['label']]['sources'], expected_sources)
    manifest = json.loads((bundle/'manifest.json').read_bytes())
    if (sha(bundle/'manifest.json') != declared['bundle_manifest_sha256'] or manifest['models'] != declared['models']
            or manifest['adaptation_updates'] != 0 or summary['identity_sha256'] != identity_hash):
        raise ValueError('Bundle or summary identity changed')
    expected_labels = [(model_id, case) for model_id in MODEL_IDS for case in expected_cases]
    if [(r['model_id'], r['case']) for r in summary['trials']] != expected_labels: raise ValueError('Incomplete result inventory')
    arrays_by_label = {}; checks = []
    for record in summary['trials']:
        label = record['model_id']+'--'+record['case']['label']
        path = folder/(label+'.npz'); core = FoodCoreRuntime.load(bundle, record['model_id'])
        if (record['label'] != label or record['identity_sha256'] != identity_hash
                or record['model_payload_sha256'] != core.record['sha256'] or sha(path) != record['trajectory_sha256']
                or json.loads((folder/(label+'.json')).read_bytes()) != record):
            raise ValueError('Trial, model or trajectory identity differs')
        with np.load(path, allow_pickle=False) as archive: arrays = {name: archive[name] for name in archive.files}
        checks.append(dict(label=label, **audit_trial(declared, record, arrays, core)))
        arrays_by_label[label] = arrays
    complete = sum(r['status']=='complete' for r in summary['trials'])
    if summary['complete_cases'] != complete or summary['failed_cases'] != 24-complete: raise ValueError('Outcome totals differ')
    pairs = []
    for model_id in MODEL_IDS:
        for counterpart in ('odor-a-left-repeat', 'odor-a-left-reversed', 'odor-a-left-neutral'):
            first = model_id+'--odor-a-left'; second = model_id+'--'+counterpart
            pairs.append(dict(first=first, second=second,
                both_complete=all(r['status']=='complete' for r in summary['trials'] if r['label'] in (first, second)),
                exact_array_equal={name: bool(np.array_equal(arrays_by_label[first][name], arrays_by_label[second][name])) for name in KEYS}))
    same(summary['paired_replays'], pairs)
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), identity_sha256=identity_hash,
        summary_sha256=sha(folder/'summary.json'), auditor_sha256=sha(__file__),
        complete_cases=complete, failed_cases=24-complete, observations_replayed=sum(r['observations'] for r in checks),
        trials=checks, paired_replays=pairs, physics_rerun=False, training_updates=0,
        scope='Scalar geometry and motor-command audit plus full NumPy neural-state replay from stored physical observations; no independent physics rerun or food-transfer conclusion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path); parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior audit records')
    result = audit(args.folder, args.bundle); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps({key: result[key] for key in ('complete_cases', 'failed_cases', 'observations_replayed')}))
