from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from flm.language_train import construct
from flm.provenance import sha256
from flm.tokenizer import Lexicon
from flm.topology_inference import RUNTIME_FILES
from scripts.package_topology_inference import package
from scripts.verify_topology_inference_release import verify
from test_language_topology_audit import report_fixture
from test_model import fixture


def export_fixture(root):
    """Ten untrained 16-node models and artificial scores; no study data read."""
    source = Path(__file__).resolve().parents[1]
    tokenizer = 'data/tokenizers/wikitext2-4096/tokenizer.json'
    names = set(RUNTIME_FILES) | {tokenizer, 'data/tokenizers/wikitext2-4096/tokenizer-card.json',
        'data/cards/wikitext2.json', 'docs/LANGUAGE-TOPOLOGY-PROTOCOL.md', 'docs/WIKITEXT-PROTOCOL.md',
        'LICENSE', 'licenses/CC-BY-4.0.txt', 'licenses/DATA-ATTRIBUTION.md',
        'scripts/package_topology_inference.py', 'flm/language_report.py'}
    for name in names:
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
    report = report_fixture(); lexicon = Lexicon(root/tokenizer)
    graphs = sorted({run['graph'] for run in report['runs']})
    for index, name in enumerate(graphs):
        graph = fixture(); graph['col'] = np.roll(graph['col'], index*2)
        graph['weight'] = graph['source_sign'][graph['col']].astype(np.float32)/3
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **graph)
        path.with_name('graph-card.json').write_text(json.dumps(dict(scope='Synthetic 16-node release test')))
    declared = []; selected = []
    for run in report['runs']:
        condition = {key:run[key] for key in ('label', 'seed', 'graph_seed', 'graph', 'variant', 'topology', 'reference', 'output')}
        declared.append(condition)
        model = construct(run['variant'], root/run['graph'], lexicon.vocabulary, run['seed'])
        checkpoint = f'runs/synthetic/{run["label"]}/best.pt'
        path = root/checkpoint; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), config=asdict(model.config), step=run['checkpoint_step'],
            run=dict(seed=run['seed'], graph_sha256=sha256(root/run['graph']), tokenizer_sha256=lexicon.sha256),
            optimizer={'synthetic_only':torch.ones(1)}, rng=torch.get_rng_state(),
            source_only_fixture='Must not appear in model-only payloads'), path)
        run.update(checkpoint=checkpoint, checkpoint_sha256=sha256(path), parameters=model.parameter_card()['trainable_parameters'])
        run['score']['tokenizer_sha256'] = lexicon.sha256
        selected.append(dict(**condition, **{key:run[key] for key in ('checkpoint', 'checkpoint_sha256', 'checkpoint_step', 'parameters')}))
    folder = root/'reports/language-topology'; folder.mkdir(parents=True)
    identity = dict(conditions=declared, sources={name:sha256(root/name) for name in RUNTIME_FILES},
        inputs={name:sha256(root/name) for name in [tokenizer, *graphs]},
        reference_checkpoints={run['label']:run['checkpoint_sha256'] for run in report['runs'] if run['reference']})
    (folder/'identity.json').write_text(json.dumps(identity), encoding='utf8')
    report['study_identity_sha256'] = sha256(folder/'identity.json')
    selection = dict(runs=selected, study_identity_sha256=report['study_identity_sha256'], tokenizer_sha256=lexicon.sha256)
    (folder/'selection.json').write_text(json.dumps(selection), encoding='utf8')
    report['selection_sha256'] = sha256(folder/'selection.json')
    for run in report['runs']:
        run['selection_sha256'] = report['selection_sha256']
    published = root/'public/research/language-topology-results.json'; published.parent.mkdir(parents=True)
    published.write_text(json.dumps(report), encoding='utf8')
    return report


class TopologyInferenceReleaseTests(unittest.TestCase):
    def test_complete_synthetic_export_survives_fresh_replay_and_browser_release_checks(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='flm-complete-export-fixture-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            report = export_fixture(root)
            # Only the real-study completion gate is mocked. Serialization, all
            # ten restorations, forty samples, ZIP and fresh CLI replay are real.
            with patch('scripts.package_topology_inference.verified_report', return_value=report) as gate:
                release = package(root)
                gate.assert_called_once_with(root)
            self.assertFalse(release['fresh_archive_cli_verified'])
            self.assertEqual(len(release['models']), 10)
            self.assertTrue(all(row['fixed_prompt_continuations_exact'] == 4 for row in release['parity']))
            archive_path = root/release['archive']
            self.assertEqual(sha256(archive_path), release['archive_sha256'])
            with zipfile.ZipFile(archive_path) as archive:
                self.assertFalse(any(name.startswith(('runs/', 'data/processed/', 'data/raw/')) for name in archive.namelist()))
                samples = json.loads(archive.read('samples.json'))
                self.assertEqual(sum(len(row['passages']) for row in samples['models']), 40)
                for model in release['models']:
                    payload = torch.load(io.BytesIO(archive.read(model['file'])), weights_only=True)
                    self.assertEqual(set(payload), {'format', 'model', 'config', 'step', 'run', 'source_checkpoint_sha256', 'state_sha256'})
                    self.assertEqual(payload['source_checkpoint_sha256'], model['source_checkpoint_sha256'])
                    self.assertEqual(set(payload['run']), {'seed', 'graph_sha256', 'tokenizer_sha256'})
            source = Path(__file__).resolve().parents[1]
            node = shutil.which('node')
            self.assertIsNotNone(node, 'The browser release integration test requires Node.js')
            checker = ("import assert from 'node:assert/strict'; import fs from 'node:fs'; "
                f"import {{ validateReleaseDownload }} from {json.dumps((source/'web/language-topology.js').as_uri())}; "
                "const r=JSON.parse(fs.readFileSync(process.argv[1])); const m=JSON.parse(fs.readFileSync(process.argv[2])); "
                "if(process.argv[3]==='pending') assert.throws(()=>validateReleaseDownload(r,m,'inference'),/standalone replay/); "
                "else assert.equal(validateReleaseDownload(r,m,'inference').path,'research/language-topology-inference.zip');")
            command = [node, '--input-type=module', '-e', checker,
                str(root/'public/research/language-topology-results.json'),
                str(root/'public/research/language-topology-inference-release.json')]
            subprocess.run([*command, 'pending'], cwd=root, check=True, capture_output=True, text=True, timeout=30)
            verified = verify(root)
            self.assertTrue(verified['fresh_archive_cli_verified'])
            self.assertEqual(len(verified['standalone_cli_verification']['cases']), 10)
            self.assertEqual(sha256(archive_path), release['archive_sha256'])
            self.assertEqual(json.loads((root/'public/research/language-topology-inference-release.json').read_text()), verified)
            subprocess.run([*command, 'verified'], cwd=root, check=True, capture_output=True, text=True, timeout=30)

    def test_incomplete_study_cannot_export_or_sample_models(self):
        with tempfile.TemporaryDirectory(prefix='flm-export-gate-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            with patch('scripts.package_topology_inference.restore') as restore, \
                 patch('scripts.package_topology_inference.generate') as generate:
                with self.assertRaisesRegex(ValueError, 'not available'):
                    package(root)
                restore.assert_not_called(); generate.assert_not_called()
            self.assertFalse((root/'public').exists())
            self.assertFalse((root/'output').exists())

    def test_corrupt_archive_fails_before_extraction_or_any_command(self):
        with tempfile.TemporaryDirectory(prefix='flm-export-integrity-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            report = root/'reports/language-topology/inference-release.json'
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps(dict(archive='fixture.zip', archive_sha256='0'*64)))
            (root/'fixture.zip').write_bytes(b'changed')
            with patch('scripts.verify_topology_inference_release.subprocess.run') as command:
                with self.assertRaisesRegex(ValueError, 'archive changed'):
                    verify(root)
                command.assert_not_called()
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
