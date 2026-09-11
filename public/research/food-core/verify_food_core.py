"""Verify every packaged food-core sensory frame with NumPy only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
try:
    from .food_core_runtime import FoodCoreRuntime
except ImportError:
    from food_core_runtime import FoodCoreRuntime


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder):
    folder = Path(folder); manifest = json.loads((folder/'manifest.json').read_bytes())
    expected_ids = ['initial-s42', 'language-s42', 'initial-s43', 'language-s43']
    if [r['id'] for r in manifest['models']] != expected_ids: raise ValueError('Complete paired core inventory required')
    if manifest['format'] != 'flm-food-core-interface-v1' or manifest['adaptation_updates'] != 0:
        raise ValueError('Expected an unadapted core-interface preparation')
    fixture = folder/'fixtures.npz'
    if sha(fixture) != manifest['fixture']['sha256']: raise ValueError('Sensory parity fixture changed')
    if manifest['parity_tolerance'] != dict(atol=5e-6, rtol=5e-5): raise ValueError('Parity threshold changed')
    with np.load(fixture, allow_pickle=False) as data:
        sensory = data['sensory']; reset = data['reset']
        if sensory.shape != (466, 6) or reset.shape != (466,) or reset.dtype != np.bool_:
            raise ValueError('Fixture observation inventory changed')
        if np.flatnonzero(reset).tolist() != [0, 201, 402]: raise ValueError('Episode reset policy changed')
        results = []; adapters = None; body_ids = None
        for identity in expected_ids:
            model = FoodCoreRuntime.load(folder, identity)
            current = {k: v for k, v in model.weights.items() if k.startswith(('sensor_', 'action_'))}
            if adapters is None: adapters = current; body_ids = model.weights['body_ids']
            elif (any(not np.array_equal(adapters[k], v) for k, v in current.items())
                    or not np.array_equal(body_ids, model.weights['body_ids'])):
                raise ValueError('Paired adapters or anatomical ordering differ')
            maxima = {name: 0. for name in ('fast', 'slow', 'features', 'logits', 'probabilities')}
            for i, x in enumerate(sensory):
                if reset[i]: model.reset()
                logits, probabilities, features = model.step(x)
                for name, actual in (('fast', model.fast), ('slow', model.slow), ('features', features),
                                     ('logits', logits), ('probabilities', probabilities)):
                    expected = data[identity+'_'+name][i]
                    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)
                    maxima[name] = max(maxima[name], float(np.max(np.abs(actual-expected))))
                if int(logits.argmax()) != int(data[identity+'_logits'][i].argmax()):
                    raise ValueError('A greedy interface action differs')
            results.append(dict(id=identity, observations=466, maximum_absolute_error=maxima, greedy_action_exact=True))
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), manifest_sha256=sha(folder/'manifest.json'),
        runtime_sha256=sha(Path(__file__).with_name('food_core_runtime.py')), verifier_sha256=sha(__file__),
        models=results, same_adapters_and_body_order=True, torch_imported=False,
        scope='Numerical sensory replay of four unadapted interfaces. No new training, closed-loop physics, reward success or language-to-behavior benefit is evaluated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    result = verify(args.folder); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as handle: json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))
