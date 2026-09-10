import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm.core_inference import (BUNDLE_FORMAT, CHECKPOINT_FORMAT, RUNTIME_FILES,
    condition_specs, load_model, parameter_counts, verify_bundle)
from flm.inference import generate, state_hash
from flm.language_core_controls import FROZEN_DYNAMICS, construct as construct_control
from flm.language_train import construct as construct_original
from flm.provenance import sha256, write_json
from flm.tokenizer import Lexicon
from test_model import fixture


def fixture_bundle(root):
    """Eight three-update 16-node fixtures with artificial selection metadata.

    These are runtime tests, not completed language fits. No corpus is read.
    """
    source = Path(__file__).resolve().parents[1]
    for name in RUNTIME_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
    tokenizer = 'tokenizer.json'
    shutil.copyfile(source / 'data/tokenizers/wikitext2-4096/tokenizer.json', root / tokenizer)
    lexicon = Lexicon(root / tokenizer)
    graph = 'graph.npz'
    np.savez_compressed(root / graph, **fixture())
    models = {}
    declared = []
    selected = []
    records = []
    tokens = torch.tensor([[0, 2, 3, 7], [0, 5, 6, 7]])
    for name, spec in condition_specs().items():
        if spec['reference']:
            model = construct_original('flm', root / graph, lexicon.vocabulary, spec['seed'])
        else:
            model = construct_control(spec['control'], root / graph, lexicon.vocabulary, spec['seed'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, lexicon.vocabulary), tokens[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        models[name] = model
        source_hash = hashlib.sha256(('synthetic-source/' + name).encode()).hexdigest()
        payload = dict(format=CHECKPOINT_FORMAT, control=spec['control'], model=model.state_dict(), config=asdict(model.config),
            step=500, run=dict(seed=spec['seed'], graph_sha256=sha256(root / graph), tokenizer_sha256=lexicon.sha256),
            source_checkpoint_sha256=source_hash, state_sha256=state_hash(model))
        checkpoint = 'checkpoints/' + name + '.pt'
        (root / checkpoint).parent.mkdir(exist_ok=True)
        torch.save(payload, root / checkpoint)
        condition = dict(label=name, graph=graph, **spec)
        declared.append(condition)
        selected.append(dict(**condition, checkpoint='synthetic-source/' + name + '.pt', checkpoint_step=500,
                             checkpoint_sha256=source_hash, **parameter_counts(model)))
        records.append(dict(id=name, graph=graph, file=checkpoint, training_seed=spec['seed'], control=spec['control'],
            reference=spec['reference'], graph_sha256=sha256(root / graph), source_checkpoint_sha256=source_hash,
            checkpoint_step=500, state_sha256=state_hash(model), **parameter_counts(model)))
    identity = dict(conditions=declared, sources={name: sha256(root / name) for name in RUNTIME_FILES},
        inputs={name: sha256(root / name) for name in (tokenizer, graph)},
        reference_checkpoints={r['id']: r['source_checkpoint_sha256'] for r in records if r['reference']})
    write_json(root / 'identity.json', identity)
    selection = dict(runs=selected, study_identity_sha256=sha256(root / 'identity.json'), tokenizer_sha256=lexicon.sha256)
    write_json(root / 'selection.json', selection)
    manifest = dict(format=BUNDLE_FORMAT, models=records, runtime_sources=list(RUNTIME_FILES), tokenizer=tokenizer,
        study_identity='identity.json', selection='selection.json', study_identity_sha256=sha256(root / 'identity.json'),
        selection_sha256=sha256(root / 'selection.json'),
        files={p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256(p))
               for p in root.rglob('*') if p.is_file()})
    write_json(root / 'bundle.json', manifest)
    return models, lexicon


class CoreInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-inference-fixture-')
        cls.addClassCleanup(temporary.cleanup)
        cls.base = Path(temporary.name).resolve()
        assert cls.base.is_relative_to(Path(tempfile.gettempdir()).resolve())
        cls.models, cls.lexicon = fixture_bundle(cls.base)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-inference-check-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)

    def manifest(self):
        return json.loads((self.root / 'bundle.json').read_text(encoding='utf8'))

    def rehash_files(self, manifest):
        for name, record in manifest['files'].items():
            path = self.root / name
            record.update(bytes=path.stat().st_size, sha256=sha256(path))
        write_json(self.root / 'bundle.json', manifest)

    def mutate_payload(self, label, mutation, *, update_state_hash=False):
        manifest = self.manifest()
        record = next(row for row in manifest['models'] if row['id'] == label)
        path = self.root / record['file']
        saved = torch.load(path, weights_only=True)
        mutation(saved)
        if update_state_hash:
            changed = copy.deepcopy(self.models[label])
            changed.load_state_dict(saved['model'])
            record['state_sha256'] = saved['state_sha256'] = state_hash(changed)
        torch.save(saved, path)
        self.rehash_files(manifest)

    def test_all_eight_conditions_preserve_fitted_tensors_logits_states_and_counts(self):
        tokens = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]])
        with torch.no_grad():
            for label, original in self.models.items():
                restored, lexicon, record = load_model(self.root, label)
                self.assertEqual(lexicon.sha256, self.lexicon.sha256)
                self.assertEqual(asdict(restored.config), asdict(original.config))
                for name, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[name]), (label, name))
                expected, states = original(tokens)
                actual, end = restored(tokens)
                self.assertTrue(torch.equal(expected, actual), label)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, end)), label)
                self.assertEqual(parameter_counts(restored), parameter_counts(original))
                if record['control'] == 'fixed_dynamics':
                    restored.verify_frozen_dynamics()

    def test_memoryless_runtime_ignores_prefix_and_hostile_supplied_state(self):
        restored, _, _ = load_model(self.root, 'no_temporal_state-s42')
        tokens = torch.tensor([[0, 2, 3, 7], [0, 5, 6, 7]])
        changed = tokens.clone()
        changed[:, :-1] = tokens.flip(0)[:, :-1]
        hostile = (torch.full((2, 16), float('nan')), torch.full((2, 16), float('inf')))
        with torch.no_grad():
            expected, states = restored(tokens)
            actual, end = restored(tokens, hostile)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(expected[:, -1], restored(changed)[0][:, -1]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, end)))
            temporal, _, _ = load_model(self.root, 'no_lateral-s42')
            self.assertFalse(torch.equal(temporal(tokens)[0][:, -1], temporal(changed)[0][:, -1]))

    def test_all_frozen_groups_are_checked_even_if_release_hashes_are_updated(self):
        label = 'fixed_dynamics-s42'
        for name in FROZEN_DYNAMICS:
            with self.subTest(group=name):
                shutil.copyfile(self.base / 'checkpoints' / (label + '.pt'), self.root / 'checkpoints' / (label + '.pt'))
                self.mutate_payload(label, lambda p: p['model'][name].add_(.1), update_state_hash=True)
                with self.assertRaisesRegex(ValueError, 'differs from initialization'):
                    load_model(self.root, label)

    def test_graph_pooling_and_nonfinite_parameters_fail_after_file_rehashing(self):
        for name in ('row', 'pool_sizes', 'embedding.weight'):
            with self.subTest(tensor=name):
                shutil.copyfile(self.base / 'checkpoints/full-s42.pt', self.root / 'checkpoints/full-s42.pt')
                def mutate(saved):
                    value = saved['model'][name]
                    value.reshape(-1)[0] = float('nan') if name == 'embedding.weight' else value.reshape(-1)[0] + 1
                self.mutate_payload('full-s42', mutate, update_state_hash=True)
                with self.assertRaisesRegex(ValueError, 'graph/pooling buffer|Invalid inference tensor'):
                    load_model(self.root, 'full-s42')

    def test_a_memoryless_model_cannot_silently_be_restored_as_temporal(self):
        self.mutate_payload('no_temporal_state-s42', lambda saved: saved['config'].update(control='no_lateral'))
        with self.assertRaisesRegex(ValueError, 'selected mechanism'):
            load_model(self.root, 'no_temporal_state-s42')

    def test_payload_mechanism_and_model_only_schema_are_checked(self):
        for mutate, message in [(lambda saved: saved.update(control='fixed_dynamics'), 'mechanism'),
                                (lambda saved: saved.update(optimizer={'fixture': True}), 'model-only')]:
            shutil.copyfile(self.base / 'checkpoints/full-s42.pt', self.root / 'checkpoints/full-s42.pt')
            self.mutate_payload('full-s42', mutate)
            with self.assertRaisesRegex(ValueError, message):
                load_model(self.root, 'full-s42')

    def test_partial_inventory_and_corrupt_bytes_fail_before_deserialization(self):
        manifest = self.manifest()
        manifest['models'].pop()
        write_json(self.root / 'bundle.json', manifest)
        with patch('flm.core_inference.torch.load') as loader:
            with self.assertRaisesRegex(ValueError, 'eight-condition'):
                load_model(self.root, 'full-s42')
            loader.assert_not_called()
        shutil.copyfile(self.base / 'bundle.json', self.root / 'bundle.json')
        (self.root / 'checkpoints/full-s42.pt').write_bytes(b'changed')
        with patch('flm.core_inference.torch.load') as loader:
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_model(self.root, 'full-s42')
            loader.assert_not_called()

    def test_graph_replacement_and_runtime_drift_are_rejected(self):
        graph = fixture()
        graph['col'] = np.roll(graph['col'], 1)
        np.savez_compressed(self.root / 'graph.npz', **graph)
        self.rehash_files(self.manifest())
        with self.assertRaisesRegex(ValueError, 'measured graph changed'):
            verify_bundle(self.root)
        shutil.copyfile(self.base / 'graph.npz', self.root / 'graph.npz')
        (self.root / 'flm/core_inference.py').write_bytes(b'changed runtime')
        self.rehash_files(self.manifest())
        with self.assertRaisesRegex(ValueError, 'matching bundled'):
            verify_bundle(self.root)

    def test_fresh_external_cli_reproduces_each_distinct_mechanism(self):
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        runner = ("import pathlib,runpy,flm; assert pathlib.Path(flm.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
                  "runpy.run_module('flm.core_inference',run_name='__main__')")
        for label in ('full-s43', 'fixed_dynamics-s42', 'no_lateral-s43', 'no_temporal_state-s42'):
            expected = generate(self.models[label], self.lexicon, 'A small animal', temperature=0, maximum_tokens=4)
            result = subprocess.run([sys.executable, '-E', '-X', 'utf8', '-c', runner, '--condition', label,
                '--prompt', 'A small animal', '--temperature', '0', '--tokens', '4'], cwd=self.root, env=environment,
                capture_output=True, text=True, encoding='utf8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(result.stdout)
            self.assertEqual(actual['model']['id'], label)
            self.assertTrue(all(actual[key] == value for key, value in expected.items()))


if __name__ == '__main__':
    unittest.main()
