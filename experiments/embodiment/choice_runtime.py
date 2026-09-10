"""NumPy inference for the four-input FLM choice model; no fitted behavior here."""
from pathlib import Path
import hashlib
import json
import numpy as np


class ChoiceRuntime:
    def __init__(self, folder, identity):
        folder = Path(folder)
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
        if manifest['format'] != 'flm-choice-inference-v1':
            raise ValueError('Unsupported choice runtime package')
        self.record = next((row for row in manifest['models'] if row['id'] == identity), None)
        if self.record is None:
            raise ValueError('Unknown choice model')
        path = folder / self.record['file']
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Model path leaves package')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.record['sha256']:
            raise ValueError('Choice model package changed')
        with np.load(path, allow_pickle=False) as archive:
            self.weights = {name: archive[name].copy() for name in archive.files}
        w = self.weights; n = len(w['alpha']); pools = len(w['pool_sizes'])
        shapes = dict(recurrent=(n,n), input_weight=(n,4), input_bias=(n,), alpha=(n,),
                      beta=(n,), gain=(), pool_index=(n,), pool_sizes=(pools,),
                      norm_weight=(2*pools,), norm_bias=(2*pools,),
                      readout_weight=(2,2*pools), readout_bias=(2,), norm_epsilon=())
        if set(w) != set(shapes) or any(w[k].shape != shape for k,shape in shapes.items()):
            raise ValueError('Invalid choice tensor dimensions')
        if (any(not np.isfinite(value).all() for value in w.values()) or
                w['pool_index'].dtype != np.int64 or np.any(w['pool_index'] < 0) or
                np.any(w['pool_index'] >= pools) or np.any(w['pool_sizes'] <= 0)):
            raise ValueError('Invalid choice tensor values')
        self.reset()

    def reset(self):
        self.fast = np.zeros_like(self.weights['alpha'])
        self.slow = np.zeros_like(self.fast)

    def step(self, sensory):
        x = np.asarray(sensory, dtype=np.float32)
        if x.shape != (4,) or not np.isfinite(x).all():
            raise ValueError('Expected four finite sensory inputs')
        w = self.weights
        drive = w['input_weight'] @ x + w['input_bias']
        z = np.tanh(drive + w['gain'] * (w['recurrent'] @ self.fast))
        self.fast = (1 - w['alpha']) * self.fast + w['alpha'] * z
        self.slow = (1 - w['beta']) * self.slow + w['beta'] * self.fast
        def pool(state):
            result = np.zeros_like(w['pool_sizes'])
            np.add.at(result, w['pool_index'], state)
            return result / w['pool_sizes']
        features = np.concatenate((pool(self.fast), pool(self.slow)))
        centered = features - features.mean()
        features = centered / np.sqrt(np.mean(centered**2) + w['norm_epsilon'])
        logits = w['readout_weight'] @ (features * w['norm_weight'] + w['norm_bias']) + w['readout_bias']
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite neural state')
        probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
        return logits, probabilities


def verify_parity(folder):
    folder = Path(folder); manifest = json.loads((folder / 'manifest.json').read_text())
    path = folder / manifest['fixture']['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['fixture']['sha256']:
        raise ValueError('Parity fixtures changed')
    results = []
    with np.load(path, allow_pickle=False) as fixtures:
        for record in manifest['models']:
            model = ChoiceRuntime(folder, record['id']); maxima = dict(fast=0., slow=0., logits=0.)
            for index, sensory in enumerate(fixtures['sensory']):
                if fixtures['reset'][index]: model.reset()
                logits, _ = model.step(sensory)
                for name, actual in [('fast', model.fast), ('slow', model.slow), ('logits', logits)]:
                    expected = fixtures[record['id'] + '_' + name][index]
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                    maxima[name] = max(maxima[name], float(np.max(np.abs(actual - expected))))
                if int(logits.argmax()) != int(fixtures[record['id'] + '_logits'][index].argmax()):
                    raise ValueError('Neural argmax changed')
            results.append(dict(model=record['id'], frames=len(fixtures['sensory']),
                                maximum_absolute_error=maxima, argmax_exact=True))
    return results
