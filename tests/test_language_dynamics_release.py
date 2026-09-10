"""Integrity and intervention checks against the real exported core archive."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from flm.language_dynamics import CORE_GROUPS, core_hash
from flm.language_dynamics_study import FOLDER, GRAPH, LEXICON
from flm.language_train import construct
from flm.tokenizer import Lexicon
from scripts.replay_language_dynamics import verify_bundle, restore_cores


class DynamicsReleaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='flm-dynamics-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        archive = Path(__file__).resolve().parents[1] / 'public/research/language-dynamics-replay.zip'
        with zipfile.ZipFile(archive) as source:
            source.extractall(self.root)

    def test_export_restores_all_seven_cores_without_full_checkpoints(self):
        identity, cases = verify_bundle(self.root)
        models = restore_cores(self.root, identity, cases)
        self.assertEqual(len(models), 7)
        self.assertFalse(list(self.root.rglob('*.pt')))
        self.assertFalse((self.root / 'runs').exists())

    def test_changed_parameter_archive_is_rejected_before_measurement(self):
        path = self.root / FOLDER / 'case-trained-s42.npz'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Archive file changed'):
            verify_bundle(self.root)

    def test_manifest_cannot_omit_a_required_numerical_source(self):
        path = self.root / 'manifest.json'
        manifest = json.loads(path.read_text())
        del manifest['files']['flm/language_dynamics.py']
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'missing required files'):
            verify_bundle(self.root)

    def test_rehashed_intervention_cannot_change_undeclared_time_constants(self):
        identity, cases = verify_bundle(self.root)
        cases = copy.deepcopy(cases)
        case = next(row for row in cases if row['condition']['label'] == 'edges_gain_only-s42')
        path = self.root / FOLDER / case['arrays']
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        arrays['parameter_alpha_logit'] += np.float32(.1)
        np.savez_compressed(path, **arrays)
        model = construct('flm', self.root / GRAPH, Lexicon(self.root / LEXICON).vocabulary, 42)
        with torch.no_grad():
            for name in CORE_GROUPS:
                model.get_parameter(name).copy_(torch.from_numpy(arrays['parameter_' + name]))
        case['metrics']['core_sha256'] = core_hash(model)
        with self.assertRaisesRegex(ValueError, 'Acute swap differs'):
            restore_cores(self.root, identity, cases)


if __name__ == '__main__':
    unittest.main()
