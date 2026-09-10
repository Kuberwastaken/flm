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

from flm.inference import generate, state_hash
from flm.language_train import construct
from flm.provenance import sha256
from flm.tokenizer import Lexicon
from flm.topology_inference import RUNTIME_FILES, condition_specs, load_model, verify_bundle
from test_model import fixture


class TopologyInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = tempfile.TemporaryDirectory(prefix='flm-topology-runtime-fixture-')
        cls.root = Path(cls.base.name).resolve()
        assert cls.root.is_relative_to(Path(tempfile.gettempdir()).resolve())
        cls.addClassCleanup(cls.base.cleanup)
        source = Path(__file__).resolve().parents[1]
        for name in RUNTIME_FILES:
            path = cls.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, path)
        tokenizer = 'tokenizer.json'
        shutil.copyfile(source/'data/tokenizers/wikitext2-4096/tokenizer.json', cls.root/tokenizer)
        cls.lexicon = Lexicon(cls.root/tokenizer)
        graph_paths = {}
        for index, graph_seed in enumerate((None, 101, 103, 107)):
            graph = fixture()
            graph['col'] = np.roll(graph['col'], index*2)
            graph['weight'] = graph['source_sign'][graph['col']].astype(np.float32)/3
            path = f'graph-{graph_seed}.npz'; graph_paths[graph_seed] = path
            np.savez_compressed(cls.root/path, **graph)
        cls.models = {}; records = []; declared = []; selected = []
        for label, spec in condition_specs().items():
            graph = graph_paths[spec['graph_seed']]
            model = construct(spec['variant'], cls.root/graph, cls.lexicon.vocabulary, spec['seed'])
            cls.models[label] = model
            source_hash = hashlib.sha256(('synthetic-source/'+label).encode()).hexdigest()
            payload = dict(format='flm-inference-checkpoint-v1', model=model.state_dict(), config=asdict(model.config),
                step=500, run=dict(seed=spec['seed'], graph_sha256=sha256(cls.root/graph), tokenizer_sha256=cls.lexicon.sha256),
                source_checkpoint_sha256=source_hash, state_sha256=state_hash(model))
            name = 'checkpoints/'+label+'.pt'; (cls.root/name).parent.mkdir(exist_ok=True)
            torch.save(payload, cls.root/name)
            condition = dict(label=label, graph=graph, **spec)
            declared.append(condition)
            selected.append(dict(**condition, checkpoint='synthetic-source/'+label+'.pt', checkpoint_step=500,
                                 checkpoint_sha256=source_hash, parameters=model.parameter_card()['trainable_parameters']))
            records.append(dict(id=label, graph=graph, file=name, training_seed=spec['seed'], variant=spec['variant'],
                graph_seed=spec['graph_seed'], topology=spec['topology'], reference=spec['reference'],
                graph_sha256=sha256(cls.root/graph), source_checkpoint_sha256=source_hash, checkpoint_step=500,
                parameters=model.parameter_card()['trainable_parameters'], state_sha256=state_hash(model)))
        identity = dict(conditions=declared, sources={name:sha256(cls.root/name) for name in RUNTIME_FILES},
            inputs={name:sha256(cls.root/name) for name in [tokenizer, *graph_paths.values()]},
            reference_checkpoints={r['id']:r['source_checkpoint_sha256'] for r in records if r['reference']})
        (cls.root/'identity.json').write_text(json.dumps(identity), encoding='utf8')
        selection = dict(runs=selected, study_identity_sha256=sha256(cls.root/'identity.json'), tokenizer_sha256=cls.lexicon.sha256)
        (cls.root/'selection.json').write_text(json.dumps(selection), encoding='utf8')
        manifest = dict(format='flm-language-topology-inference-v1', models=records, runtime_sources=list(RUNTIME_FILES),
            tokenizer=tokenizer, study_identity='identity.json', selection='selection.json',
            study_identity_sha256=sha256(cls.root/'identity.json'), selection_sha256=sha256(cls.root/'selection.json'),
            files={p.relative_to(cls.root).as_posix():dict(bytes=p.stat().st_size, sha256=sha256(p))
                   for p in cls.root.rglob('*') if p.is_file()})
        (cls.root/'bundle.json').write_text(json.dumps(manifest), encoding='utf8')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='flm-topology-runtime-check-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        shutil.copytree(self.root, self.folder, dirs_exist_ok=True)

    def manifest(self):
        return json.loads((self.folder/'bundle.json').read_text())

    def save_manifest(self, manifest):
        for name, record in manifest['files'].items():
            path = self.folder/name
            record.update(bytes=path.stat().st_size, sha256=sha256(path))
        (self.folder/'bundle.json').write_text(json.dumps(manifest), encoding='utf8')

    def test_all_ten_conditions_reproduce_original_tensors_logits_and_state(self):
        tokens = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]])
        with torch.no_grad():
            for label, original in self.models.items():
                restored, lexicon, record = load_model(self.folder, label)
                self.assertEqual(lexicon.sha256, self.lexicon.sha256)
                for name, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[name]), (label, name))
                expected, states = original(tokens); actual, end = restored(tokens)
                self.assertTrue(torch.equal(expected, actual), label)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, end)), label)
                if record['variant'] == 'no_slow':
                    self.assertEqual(int(torch.count_nonzero(end[1])), 0)

    def test_wrong_graph_binding_and_incomplete_inventory_fail_before_loading(self):
        for change in (lambda m:m['models'].pop(), lambda m:m['models'][1].update(graph='graph-103.npz')):
            manifest = self.manifest(); change(manifest)
            with patch('flm.topology_inference.restore') as reader:
                (self.folder/'bundle.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): load_model(self.folder, 'null101-s42')
                reader.assert_not_called()
            # Restore the pristine manifest between mutations.
            shutil.copyfile(self.root/'bundle.json', self.folder/'bundle.json')

    def test_checkpoint_corruption_is_rejected_before_deserialization(self):
        (self.folder/'checkpoints/measured-s42.pt').write_bytes(b'changed')
        with patch('flm.topology_inference.restore') as reader:
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_model(self.folder, 'measured-s42')
            reader.assert_not_called()

    def test_changed_graph_buffer_is_rejected_even_with_updated_export_hashes(self):
        manifest = self.manifest(); record = manifest['models'][1]
        path = self.folder/record['file']; payload = torch.load(path, weights_only=True)
        payload['model']['col'] = payload['model']['col'].roll(1)
        changed = construct(record['variant'], self.folder/record['graph'], self.lexicon.vocabulary, record['training_seed'])
        changed.load_state_dict(payload['model'])
        record['state_sha256'] = payload['state_sha256'] = state_hash(changed)
        torch.save(payload, path); self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'graph buffer'):
            load_model(self.folder, record['id'])

    def test_mechanism_and_model_only_schema_survive_file_rehashing(self):
        for change, message in [(lambda p:p['config'].update(variant='no_slow'), 'mechanism'),
                                 (lambda p:p.update(optimizer={'fixture':True}), 'model-only')]:
            path = self.folder/'checkpoints/measured-s42.pt'
            shutil.copyfile(self.root/'checkpoints/measured-s42.pt', path)
            payload = torch.load(path, weights_only=True); change(payload); torch.save(payload, path)
            self.save_manifest(self.manifest())
            with self.assertRaisesRegex(ValueError, message):
                load_model(self.folder, 'measured-s42')

    def test_selected_source_graph_cannot_be_replaced_by_a_different_hashed_file(self):
        shutil.copyfile(self.folder/'graph-103.npz', self.folder/'graph-101.npz')
        self.save_manifest(self.manifest())
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            verify_bundle(self.folder)

    def test_matching_runtime_is_required(self):
        (self.folder/'flm/model.py').write_bytes(b'changed code')
        self.save_manifest(self.manifest())
        with self.assertRaisesRegex(ValueError, 'matching bundled'):
            verify_bundle(self.folder)

    def test_standalone_cli_reproduces_a_rewired_and_a_no_slow_fixture(self):
        environment = dict(os.environ); environment.pop('PYTHONPATH', None)
        for name in ('null101-s43', 'no-slow-s43'):
            expected = generate(self.models[name], self.lexicon, 'A small animal', temperature=0, maximum_tokens=4)
            runner = ("import pathlib,runpy,flm; assert pathlib.Path(flm.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
                      "runpy.run_module('flm.topology_inference',run_name='__main__')")
            completed = subprocess.run([sys.executable, '-E', '-X', 'utf8', '-c', runner,
                '--condition', name, '--prompt', 'A small animal', '--temperature', '0', '--tokens', '4'],
                cwd=self.folder, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            actual = json.loads(completed.stdout)
            self.assertEqual(actual['model']['id'], name)
            self.assertTrue(all(actual[key] == value for key, value in expected.items()))


if __name__ == '__main__':
    unittest.main()
