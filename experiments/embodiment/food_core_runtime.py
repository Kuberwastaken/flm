"""NumPy-only sensory inference for the exported frozen FLM food-core bridge.

No source coordinates, language tokenizer, lexical head, learning or physics
are used here. Action probabilities are an interface output, not a food skill.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


class FoodCoreRuntime:
    def __init__(self, weights):
        self.weights = {key: np.array(value, copy=True) for key, value in weights.items()}
        w = self.weights
        try: n = len(w['alpha']); d = len(w['sensor_bias']); p = len(w['pool_sizes'])
        except (KeyError, TypeError) as exc: raise ValueError('Missing core dimensions') from exc
        if min(n, d, p) < 1 or n > 2048: raise ValueError('Unsupported dense runtime dimensions')
        shapes = dict(recurrent=(n,n), input_weight=(n,d), input_bias=(n,), alpha=(n,), beta=(n,), gain=(),
            pool_index=(n,), pool_sizes=(p,), norm_weight=(2*p,), norm_bias=(2*p,), norm_epsilon=(),
            sensor_weight=(d,6), sensor_bias=(d,), action_weight=(3,2*p), action_bias=(3,), body_ids=(n,))
        if set(w) != set(shapes) or any(w[k].shape != shape for k, shape in shapes.items()):
            raise ValueError('Exported tensor inventory or shape differs')
        for name, value in w.items():
            expected = np.dtype('int64') if name in ('pool_index', 'body_ids') else np.dtype('float32')
            if value.dtype != expected or not np.isfinite(value).all(): raise ValueError('Invalid tensor dtype or value: '+name)
        if (len(np.unique(w['body_ids'])) != n or np.any(w['pool_index'] < 0) or np.any(w['pool_index'] >= p)
                or not np.array_equal(w['pool_sizes'], np.maximum(np.bincount(w['pool_index'], minlength=p), 1))):
            raise ValueError('Invalid neuron identities or pooling')
        if (np.any((w['alpha'] < .05) | (w['alpha'] > .95)) or np.any((w['beta'] < .002) | (w['beta'] > .1))
                or not .05 <= w['gain'] <= 3. or not w['norm_epsilon'] > 0
                or np.any(np.abs(w['recurrent']).sum(axis=1) > 1.00001)):
            raise ValueError('Invalid bounded dynamics or normalized recurrent matrix')
        for value in w.values(): value.setflags(write=False)
        self.reset()

    @classmethod
    def load(cls, folder, identity):
        folder = Path(folder).resolve(); manifest = json.loads((folder/'manifest.json').read_bytes())
        if manifest['format'] != 'flm-food-core-interface-v1': raise ValueError('Unsupported food-core package')
        records = [record for record in manifest['models'] if record['id'] == identity]
        if len(records) != 1: raise ValueError('Unknown or repeated food-core identity')
        record = records[0]; name = record['file']
        if not isinstance(name, str) or Path(name).name != name or '\\' in name or ':' in name:
            raise ValueError('Core payload must be a package-local filename')
        path = folder/name
        if not path.resolve().is_relative_to(folder): raise ValueError('Core payload leaves package')
        payload = path.read_bytes()
        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Core payload checksum differs')
        with np.load(path, allow_pickle=False) as archive:
            runtime = cls({name: archive[name] for name in archive.files})
        runtime.record = record
        return runtime

    def reset(self):
        self.fast = np.zeros_like(self.weights['alpha']); self.slow = np.zeros_like(self.fast)

    def step(self, sensory):
        x = np.asarray(sensory, dtype=np.float32)
        if x.shape != (6,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError('Six finite normalized sensory channels required')
        w = self.weights
        projected = w['sensor_weight'] @ x + w['sensor_bias']
        drive = w['input_weight'] @ projected + w['input_bias']
        fast = (1-w['alpha'])*self.fast + w['alpha']*np.tanh(drive+w['gain']*(w['recurrent'] @ self.fast))
        slow = (1-w['beta'])*self.slow+w['beta']*fast
        def pool(state):
            out = np.zeros_like(w['pool_sizes']); np.add.at(out, w['pool_index'], state)
            return out/w['pool_sizes']
        joined = np.concatenate((pool(fast), pool(slow)))
        centered = joined-joined.mean()
        features = centered/np.sqrt(np.mean(centered**2)+w['norm_epsilon'])*w['norm_weight']+w['norm_bias']
        logits = w['action_weight'] @ features+w['action_bias']
        if any(not np.isfinite(v).all() for v in (fast, slow, features, logits)):
            raise FloatingPointError('Nonfinite sensory recurrent inference')
        self.fast = fast; self.slow = slow
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        return logits, probabilities, features
