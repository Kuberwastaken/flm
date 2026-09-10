import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from flm.inference import RUNTIME_FILES, bundle_path, generate, load_model, state_hash, verify_bundle
from flm.provenance import sha256


class ToyLexicon:
    pieces = [b'', b'', b'\x00', b'a', b'b', b'\n']
    vocabulary = len(pieces)

    def encode(self, prompt):
        return [3 if char == 'a' else 4 for char in prompt]


class ToyModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer('scores', torch.tensor(scores, dtype=torch.float32))
        self.calls = []

    def forward(self, tokens, state=None):
        previous = 0 if state is None else state
        self.calls.append((tokens.clone(), previous))
        return self.scores.repeat(1, tokens.shape[1], 1), previous + tokens.shape[1]


class InferenceTests(unittest.TestCase):
    def test_blocks_escaping_or_ambiguous_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(bundle_path(root, 'checkpoints/flm.pt'), root / 'checkpoints/flm.pt')
            for value in ('../secret', '/secret', 'a/../../secret', 'C:/secret',
                          'C:secret', 'a\\b', '', '.', 'a:stream'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    bundle_path(root, value)

    def make_bundle(self, root):
        source = Path(__file__).resolve().parents[1]
        for name in RUNTIME_FILES:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
        models = [dict(id=f'{variant}-s{seed}', file=f'{variant}-{seed}.pt')
                  for seed in (42, 43) for variant in ('flm', 'gru', 'transformer')]
        for name in ('graph.npz', 'tokenizer.json', *(r['file'] for r in models)):
            (root / name).write_bytes(b'fixture')
        files = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256(p))
                 for p in root.rglob('*') if p.is_file()}
        manifest = dict(format='flm-wikitext-inference-v1', runtime_sources=list(RUNTIME_FILES),
                        files=files, models=models, graph='graph.npz', tokenizer='tokenizer.json')
        (root / 'bundle.json').write_text(json.dumps(manifest), encoding='utf8')
        return manifest

    def test_corrupt_model_is_rejected_before_checkpoint_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_bundle(root)
            self.assertEqual(len(verify_bundle(root)['models']), 6)
            (root / 'flm-42.pt').write_bytes(b'changed')
            with patch('flm.inference.restore') as restore:
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    load_model(root, 'flm-s42')
                restore.assert_not_called()

    def test_rejects_incomplete_comparison_and_mismatched_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = self.make_bundle(root)
            missing = dict(manifest, models=manifest['models'][:-1])
            (root / 'bundle.json').write_text(json.dumps(missing), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'six-model'):
                verify_bundle(root)
            runtime = root / 'flm/inference.py'; runtime.write_bytes(b'changed runtime')
            manifest['files']['flm/inference.py'] = dict(bytes=runtime.stat().st_size, sha256=sha256(runtime))
            (root / 'bundle.json').write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'matching bundled runtime'):
                verify_bundle(root)

    def test_sampling_masks_control_tokens_and_greedy_ignores_seed(self):
        model = ToyModel([100., -100., 90., 2., 1., 0.])
        first = generate(model, ToyLexicon(), '', temperature=0, top_k=6, maximum_tokens=4)
        second = generate(model, ToyLexicon(), '', seed=99, temperature=0, top_k=1, maximum_tokens=4)
        self.assertEqual(first, second)
        self.assertEqual(first['tokens'], [3, 3, 3, 3])
        self.assertEqual(first['continuation'], 'aaaa')
        self.assertEqual(first['bytes'], 4)

    def test_eos_stops_without_being_decoded(self):
        model = ToyModel([100., 50., 90., 2., 1., 0.])
        result = generate(model, ToyLexicon(), 'ab', temperature=0, top_k=1)
        self.assertEqual(result, dict(prompt='ab', continuation='', tokens=[], bytes=0))
        self.assertEqual(len(model.calls), 1)

    def test_long_prompt_preserves_all_tokens_and_carried_state(self):
        model = ToyModel([0., 0., 0., 2., 1., 0.]); prompt = 'a' * 210
        generate(model, ToyLexicon(), prompt, temperature=0, top_k=1, maximum_tokens=1)
        self.assertEqual([state for _, state in model.calls], [0, 96, 192, 211])
        self.assertTrue(torch.equal(torch.cat([x for x, _ in model.calls[:3]], 1),
                                    torch.tensor([[0] + [3] * 210])))

    def test_sampling_uses_an_independent_reproducible_generator(self):
        model = ToyModel([0., -100., 0., 2., 1., 0.]); lexicon = ToyLexicon()
        torch.manual_seed(37); before = torch.get_rng_state().clone()
        first = generate(model, lexicon, 'ab', seed=17, top_k=3, maximum_tokens=40)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = generate(model, lexicon, 'ab', seed=17, top_k=3, maximum_tokens=40)
        self.assertEqual(first, second)
        self.assertNotEqual(first['tokens'], generate(model, lexicon, 'ab', seed=29,
                                                      top_k=3, maximum_tokens=40)['tokens'])

    def test_invalid_settings_fail_before_running_model(self):
        model = ToyModel([0.] * 6)
        for settings in (dict(temperature=float('nan')), dict(temperature=-1),
                         dict(top_k=7), dict(top_k=0), dict(top_k=1.5),
                         dict(maximum_tokens=0), dict(maximum_tokens=1025),
                         dict(maximum_tokens=True), dict(seed=-1), dict(seed=2**63)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generate(model, ToyLexicon(), '', **settings)
        self.assertEqual(model.calls, [])

    def test_state_hash_covers_fixed_buffers_and_learned_parameters(self):
        model = torch.nn.Linear(2, 2); model.register_buffer('anatomy', torch.arange(3))
        first = state_hash(model)
        model.anatomy[0] = 3
        second = state_hash(model); self.assertNotEqual(first, second)
        with torch.no_grad(): model.weight[0, 0] += 1
        self.assertNotEqual(second, state_hash(model))


if __name__ == '__main__':
    unittest.main()
