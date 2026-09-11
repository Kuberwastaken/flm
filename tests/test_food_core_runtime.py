"""Portable sensory inference and malformed-export rejection on a tiny graph."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from flm.model import FLM, Config
from flm.food_core import FoodCore
from flm.food_core_export import arrays
from experiments.embodiment.food_core_runtime import FoodCoreRuntime


def bridge():
    graph = dict(body_ids=np.arange(4), row=np.array([0,1,2,3,0]), col=np.array([1,2,3,0,3]),
        weight=np.array([.3,-.4,.6,.2,-.1]), pool=np.array([0,1,0,1]))
    return FoodCore(FLM(graph, Config(neurons=4, pools=2, embedding=3, vocabulary=9, tied_readout=True)), 711)


class RuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(42)

    def test_all_states_features_logits_and_probabilities_match_native_steps(self):
        model = bridge(); runtime = FoodCoreRuntime(arrays(model, np.arange(4)))
        stream = np.random.default_rng(8).uniform(size=(37,6)).astype(np.float32); state = None
        with torch.no_grad():
            for i, x in enumerate(stream):
                if i in (0, 13, 27): state = None; runtime.reset()
                expected, state, features = model(torch.from_numpy(x)[None,None], state)
                logits, probabilities, actual_features = runtime.step(x)
                for actual, reference in ((runtime.fast, state[0][0]), (runtime.slow, state[1][0]),
                    (actual_features, features[0,0]), (logits, expected[0,0]), (probabilities, expected[0,0].softmax(-1))):
                    np.testing.assert_allclose(actual, reference.numpy(), atol=5e-6, rtol=5e-5)
                self.assertEqual(int(logits.argmax()), int(expected[0,0].argmax()))

    def test_export_is_independent_and_pooling_or_dtype_damage_is_rejected(self):
        model = bridge(); exported = arrays(model, np.arange(4)); runtime = FoodCoreRuntime(exported)
        original = runtime.weights['input_weight'].copy(); exported['input_weight'][:] = 9
        np.testing.assert_array_equal(runtime.weights['input_weight'], original)
        self.assertFalse(runtime.weights['input_weight'].flags.writeable)
        for key, value in (('pool_sizes', np.array([9,9], dtype=np.float32)),
            ('body_ids', np.zeros(4, dtype=np.int64)), ('alpha', np.full(4, 2., dtype=np.float32)),
            ('sensor_weight', np.zeros((3,7), dtype=np.float32)), ('action_bias', np.zeros(3, dtype=np.float64))):
            changed = arrays(model, np.arange(4)); changed[key] = value
            with self.assertRaises(ValueError): FoodCoreRuntime(changed)

    def test_payload_checksum_and_identity_are_checked_before_load(self):
        model = bridge()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'core.npz'
            np.savez_compressed(path, **arrays(model, np.arange(4)))
            manifest = dict(format='flm-food-core-interface-v1', models=[dict(id='fixture', file='core.npz',
                bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
            (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
            runtime = FoodCoreRuntime.load(root, 'fixture'); self.assertEqual(runtime.record['id'], 'fixture')
            with self.assertRaises(ValueError): FoodCoreRuntime.load(root, 'absent')
            path.write_bytes(path.read_bytes()+b'damaged')
            with self.assertRaises(ValueError): FoodCoreRuntime.load(root, 'fixture')

    def test_bad_sensory_input_does_not_advance_state(self):
        runtime = FoodCoreRuntime(arrays(bridge(), np.arange(4)))
        for values in ([0.]*5, [float('nan')]*6, [-.1]*6, [1.1]*6):
            with self.assertRaises(ValueError): runtime.step(values)
            np.testing.assert_array_equal(runtime.fast, np.zeros(4, dtype=np.float32))
            np.testing.assert_array_equal(runtime.slow, np.zeros(4, dtype=np.float32))


if __name__ == '__main__': unittest.main()
