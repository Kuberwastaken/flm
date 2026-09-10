from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import torch

from flm.core_inference import CHECKPOINT_FORMAT, RUNTIME_FILES, parameter_counts
from flm.language_core_study import GRAPH, LEXICON, REPORTS, conditions, run_binding
from flm.language_core_train import optimizer_names
from flm.provenance import sha256, write_json
from scripts.package_core_inference import ARCHIVE, ASSETS, RELEASE, PUBLIC_RELEASE, package
from scripts.verify_core_inference_release import read_archive, verify
from test_core_inference import fixture_bundle


def export_fixture(root):
    """Three-update 16-node models with artificial 500-step selection metadata.

    No corpus or real training selection is read. Only the real-study gate is
    mocked by the caller; checkpoint restore/export and all samples are real.
    """
    source = Path(__file__).resolve().parents[1]
    models, lexicon = fixture_bundle(root)
    for name in ASSETS:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        if name == GRAPH.as_posix():
            shutil.copyfile(root / 'graph.npz', path)
        elif name == GRAPH.with_name('graph-card.json').as_posix():
            write_json(path, dict(scope='Synthetic 16-node inference-release test'))
        else:
            shutil.copyfile(source / name, path)
    declared = conditions(); selected = []
    payloads = {}
    references = {}
    for condition in declared:
        name = condition['label']; model = models[name]
        checkpoint = 'synthetic-source/' + name + '.pt'
        payload = dict(model=model.state_dict(), config=asdict(model.config), step=500,
            run=dict(seed=condition['seed'], graph_sha256=sha256(root / GRAPH), tokenizer_sha256=lexicon.sha256),
            optimizer={'fixture_only': torch.ones(1)}, torch_rng=torch.get_rng_state(),
            source_only_fixture='Do not export optimizer/RNG/private payloads')
        if condition['reference']:
            path = root / checkpoint; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path); references[name] = sha256(path)
        payloads[name] = payload
        selected.append(dict(**condition, checkpoint=checkpoint, checkpoint_step=500,
                             **parameter_counts(model)))
    identity = dict(scope='Synthetic release fixture, not a completed computation study',
        conditions=declared, sources={name: sha256(root / name) for name in RUNTIME_FILES},
        inputs={name: sha256(root / name) for name in (GRAPH, LEXICON)},
        reference_checkpoints=references, training_protocol=dict(tokenizer_sha256=lexicon.sha256),
        initial_parameter_sha256={str(seed): hashlib.sha256(f'synthetic-init-{seed}'.encode()).hexdigest()
                                  for seed in (42, 43)})
    # JSON object keys use archive-relative POSIX paths.
    identity['inputs'] = {name.as_posix(): value for name, value in identity['inputs'].items()}
    write_json(root / REPORTS / 'identity.json', identity)
    for selected_row in selected:
        name = selected_row['label']; model = models[name]; payload = payloads[name]
        path = root / selected_row['checkpoint']
        if not selected_row['reference']:
            payload['run'] = dict(run_binding(root, selected_row, identity),
                optimizer_parameter_names=optimizer_names(model), parameter_card=model.parameter_card())
            torch.save(payload, path)
        selected_row['checkpoint_sha256'] = sha256(path)
    selection = dict(runs=selected, study_identity_sha256=sha256(root / REPORTS / 'identity.json'),
                     tokenizer_sha256=lexicon.sha256)
    write_json(root / REPORTS / 'selection.json', selection)
    report = dict(study_identity_sha256=selection['study_identity_sha256'],
        selection_sha256=sha256(root / REPORTS / 'selection.json'),
        runs=[dict(**row, score=dict(bits_per_byte=2. + index / 100)) for index, row in enumerate(selected)],
        scope='Artificial scores, no held-out inference performed')
    write_json(root / REPORTS / 'summary.json', report)
    return report


class CoreInferenceReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-export-fixture-')
        cls.addClassCleanup(temporary.cleanup)
        cls.base = Path(temporary.name).resolve()
        assert cls.base.is_relative_to(Path(tempfile.gettempdir()).resolve())
        cls.report = export_fixture(cls.base)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-export-check-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)

    def test_all_eight_exports_and_32_samples_survive_fresh_external_cli_replay(self):
        with patch('scripts.package_core_inference.verified_report', return_value=self.report) as gate:
            release = package(self.root)
            gate.assert_called_once_with(self.root)
        self.assertFalse(release['fresh_archive_cli_verified'])
        self.assertEqual(len(release['models']), 8)
        self.assertTrue(all(row['fixed_prompt_continuations_exact'] == 4 for row in release['parity']))
        archive_path = self.root / ARCHIVE
        self.assertEqual(sha256(archive_path), release['archive_sha256'])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertFalse(any(name.startswith(('synthetic-source/', 'runs/', 'data/processed/', 'data/raw/'))
                                 for name in archive.namelist()))
            samples = json.loads(archive.read('samples.json'))
            self.assertEqual(sum(len(row['passages']) for row in samples['models']), 32)
            for model in release['models']:
                payload = torch.load(io.BytesIO(archive.read(model['file'])), weights_only=True)
                self.assertEqual(set(payload), {'format', 'control', 'model', 'config', 'step', 'run',
                                                'source_checkpoint_sha256', 'state_sha256'})
                self.assertEqual(payload['format'], CHECKPOINT_FORMAT)
                self.assertEqual(payload['control'], model['control'])
                self.assertEqual(set(payload['run']), {'seed', 'graph_sha256', 'tokenizer_sha256'})
                if model['reference']:
                    self.assertNotIn('control', payload['config'])
                else:
                    self.assertEqual(payload['config']['control'], model['control'])
        checked = verify(self.root)
        self.assertTrue(checked['fresh_archive_cli_verified'])
        self.assertTrue(checked['standalone_cli_verification']['extraction_outside_repository'])
        self.assertEqual(len(checked['standalone_cli_verification']['cases']), 8)
        self.assertEqual(sha256(archive_path), release['archive_sha256'])
        self.assertEqual(json.loads((self.root / PUBLIC_RELEASE).read_bytes()), checked)

    def test_incomplete_real_gate_rejects_before_loading_sampling_or_writing(self):
        (self.root / REPORTS / 'summary.json').unlink()
        with patch('scripts.package_core_inference.restore_original') as original, \
             patch('scripts.package_core_inference.restore_control') as control, \
             patch('scripts.package_core_inference.generate') as generate:
            with self.assertRaisesRegex(ValueError, 'not available'):
                package(self.root)
            original.assert_not_called(); control.assert_not_called(); generate.assert_not_called()
        self.assertFalse((self.root / 'public').exists())

    def test_source_checkpoint_change_after_gate_rejects_without_release(self):
        row = self.report['runs'][0]
        checkpoint = self.root / row['checkpoint']
        payload = torch.load(checkpoint, weights_only=True)
        payload['model']['alpha_logit'].add_(.1)
        torch.save(payload, checkpoint)
        with patch('scripts.package_core_inference.verified_report', return_value=self.report):
            with self.assertRaisesRegex(ValueError, 'source checkpoint changed'):
                package(self.root)
        self.assertFalse((self.root / 'public').exists())

    def test_failed_continuation_parity_cannot_publish_archive(self):
        with patch('scripts.package_core_inference.verified_report', return_value=self.report), \
             patch('scripts.package_core_inference.generate', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'continuation changed'):
                package(self.root)
        self.assertFalse((self.root / 'public').exists())

    def test_different_selection_cannot_overwrite_existing_release(self):
        write_json(self.root / RELEASE, dict(selection_sha256='0' * 64))
        with patch('scripts.package_core_inference.verified_report', return_value=self.report), \
             patch('scripts.package_core_inference.restore_original') as restore:
            with self.assertRaisesRegex(ValueError, 'different selected study'):
                package(self.root)
            restore.assert_not_called()
        self.assertFalse((self.root / 'public').exists())

    def test_corrupt_archive_cannot_execute_or_upgrade_release_record(self):
        path = self.root / ARCHIVE; path.parent.mkdir(parents=True)
        path.write_bytes(b'changed')
        record = dict(archive=ARCHIVE, archive_sha256='0' * 64, bytes=7, fresh_archive_cli_verified=False)
        write_json(self.root / RELEASE, record)
        before = (self.root / RELEASE).read_bytes()
        with patch('scripts.verify_core_inference_release.subprocess.run') as command:
            with self.assertRaisesRegex(ValueError, 'archive changed'):
                verify(self.root)
            command.assert_not_called()
        self.assertEqual((self.root / RELEASE).read_bytes(), before)

    def test_unsafe_duplicate_and_symlink_zip_entries_reject_before_payload_loading(self):
        for name, duplicate, symlink in (('../escape', False, False), ('C:/escape', False, False),
                ('a\\escape', False, False), ('/escape', False, False),
                ('a/./alias', False, False), ('duplicate', True, False), ('link', False, True)):
            with self.subTest(name=name):
                path = self.root / 'unsafe.zip'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    with zipfile.ZipFile(path, 'w') as archive:
                        item = zipfile.ZipInfo(name)
                        # On Windows the constructor normalizes backslashes;
                        # retain the malformed raw name this rejection tests.
                        item.filename = name
                        if symlink:
                            item.external_attr = (stat.S_IFLNK | 0o777) << 16
                        archive.writestr(item, b'payload')
                        if duplicate:
                            archive.writestr(name, b'other')
                record = dict(archive_sha256=sha256(path), bytes=path.stat().st_size,
                              payloads=2 if duplicate else 1)
                with self.assertRaisesRegex(ValueError, 'archive inventory|archive path'):
                    read_archive(path, record)


if __name__ == '__main__':
    unittest.main()
