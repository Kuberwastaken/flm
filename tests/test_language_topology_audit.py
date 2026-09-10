import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from flm.language_topology import conditions
from flm.language_topology_test import summarize
from scripts.audit_language_topology_report import audit


def report_fixture():
    """Sixty artificial article records, generated without reading any corpus."""
    runs = []
    for c in conditions():
        rows = []
        for i in range(60):
            size = (i+1)*3
            measured = 1.6 + i/100 + (c['seed']-42)*.15
            effect = ((i % 7)-3)/50 + ((c['graph_seed'] or 100)-103)/500
            if c['seed'] == 43:
                effect *= -.5
            if c['variant'] == 'no_slow':
                effect = -.03 + (i % 3)/100
            if c['reference']:
                effect = 0
            loss = measured - effect
            rows.append(dict(document=f'article-{i:02d}', bytes=size, tokens=i+1,
                             nll=loss*size*math.log(2), bits_per_byte=loss))
        if c['seed'] == 43:
            rows.reverse()
        nll = sum(r['nll'] for r in rows); size = sum(r['bytes'] for r in rows); tokens = sum(r['tokens'] for r in rows)
        score = dict(documents=rows, nll=nll, bytes=size, tokens=tokens,
                     bits_per_byte=nll/size/math.log(2), token_perplexity=math.exp(nll/tokens), tokenizer_sha256='b'*64)
        runs.append(dict(**c, score=score, checkpoint_step=6000 if c['reference'] else 5500, parameters=600003,
                         checkpoint_sha256=hashlib.sha256(c['label'].encode()).hexdigest(),
                         selection_sha256='c'*64, test_cache_sha256='d'*64))
    return dict(study_identity_sha256='a'*64, selection_sha256='c'*64, runs=list(reversed(runs)), **summarize(runs))


class LanguageTopologyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = report_fixture()

    def test_independent_arithmetic_matches_all_registered_contrasts(self):
        result = audit(self.report)
        self.assertEqual(result['verified_runs'], 10)
        self.assertEqual(result['unique_test_articles'], 60)
        self.assertEqual(result['scored_bytes_per_run'], 5490)
        self.assertEqual(result['scored_tokens_per_run'], 1830)
        signs = {row['difference_bpb'] > 0 for row in result['topology_contrasts']}
        self.assertEqual(signs, {True, False})
        self.assertEqual(len(result['slow_state_contrasts']), 2)

    def test_partial_wrong_identity_or_inconsistent_article_evidence_is_rejected(self):
        mutations = [lambda r: r['runs'].pop(), lambda r: r['runs'].__setitem__(1, r['runs'][0]),
            lambda r: r['runs'][0].update(seed=42), lambda r: r['runs'][0].update(selection_sha256='f'*64),
            lambda r: r['runs'][0].update(checkpoint_step=6001),
            lambda r: r['runs'][0]['score']['documents'].pop(),
            lambda r: r['runs'][0]['score']['documents'].__setitem__(1, r['runs'][0]['score']['documents'][0]),
            lambda r: r['runs'][0]['score']['documents'][0].update(nll=float('nan')),
            lambda r: r['runs'][0]['score']['documents'][0].update(tokens=999),
            lambda r: r['runs'][0]['score'].update(bits_per_byte=0),
            lambda r: r['runs'][0]['score'].update(token_perplexity=float('inf')),
            lambda r: r['runs'][0]['score'].update(tokenizer_sha256='e'*64)]
        for index, change in enumerate(mutations):
            with self.subTest(mutation=index):
                report = copy.deepcopy(self.report); change(report)
                with self.assertRaises(ValueError):
                    audit(report)

    def test_changing_a_published_interval_pair_or_mean_is_detected(self):
        mutations = [lambda r: r['topology_contrasts'][0].update(lower_95=-999),
            lambda r: r['topology_contrasts'][0].update(seed=1),
            lambda r: r['topology_contrasts'][0].update(unit='independent tokens'),
            lambda r: r.update(sign='Control minus measured'),
            lambda r: r['slow_state_contrasts'].__setitem__(1, r['slow_state_contrasts'][0]),
            lambda r: r['topology_by_graph'][0].update(mean_difference_bpb=99),
            lambda r: r.update(topology_mean_difference_bpb=99)]
        for index, change in enumerate(mutations):
            with self.subTest(mutation=index):
                report = copy.deepcopy(self.report); change(report)
                with self.assertRaises(ValueError):
                    audit(report)

    def test_script_runs_without_repository_imports(self):
        with tempfile.TemporaryDirectory(prefix='flm-arithmetic-') as directory:
            folder = Path(directory).resolve()
            self.assertTrue(folder.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            source = Path(__file__).resolve().parents[1]/'scripts/audit_language_topology_report.py'
            shutil.copyfile(source, folder/'audit.py')
            (folder/'fixture.json').write_text(json.dumps(self.report), encoding='utf8')
            environment = dict(os.environ); environment.pop('PYTHONPATH', None)
            # Keep installed NumPy available, ignore PYTHON* environment options,
            # and make accidental repository/PyTorch imports fail explicitly.
            runner = ("import runpy,sys; sys.modules['flm']=None; sys.modules['torch']=None; "
                      "sys.argv=['audit.py','fixture.json']; runpy.run_path('audit.py',run_name='__main__')")
            process = subprocess.run([sys.executable, '-E', '-c', runner],
                cwd=folder, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)['verified_runs'], 10)
            self.assertFalse((folder/'flm').exists())

    def test_record_package_cannot_create_outputs_for_an_incomplete_study(self):
        from scripts.package_language_topology import package
        with tempfile.TemporaryDirectory(prefix='flm-record-gate-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            with patch('scripts.package_language_topology.draw') as plot:
                with self.assertRaisesRegex(ValueError, 'not available'):
                    package(root)
                plot.assert_not_called()
            self.assertFalse((root/'public').exists())

    def test_packaged_fixture_preserves_every_score_and_can_be_audited_after_extraction(self):
        from scripts.package_language_topology import package
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='flm-record-fixture-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            def write(name, payload):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            # This exercises serialization after a mocked checkpoint gate. All
            # scores are synthetic; it does not assert checkpoint verification.
            report = copy.deepcopy(self.report)
            write('reports/language-topology/identity.json', b'{"sources":{}}')
            write('reports/language-topology/selection.json', b'{"scoring_sources":{}}')
            for name in ('reports/language-topology/summary.json', 'public/research/language-topology-results.json'):
                write(name, json.dumps(report).encode())
            for run in report['runs']:
                write(f'reports/language-topology/test-{run["label"]}.json', json.dumps(run).encode())
            for name in ('docs/LANGUAGE-TOPOLOGY-PROTOCOL.md', 'docs/WIKITEXT-PROTOCOL.md',
                         'data/cards/wikitext2.json', 'LICENSE', 'scripts/audit_language_topology_report.py',
                         'scripts/language_topology_report.py', 'scripts/package_language_topology.py'):
                write(name, (repository/name).read_bytes())
            def fixture_images(rows, path):
                self.assertEqual(len(rows), 8)
                path.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ('.png', '.svg'):
                    path.with_suffix(suffix).write_bytes(b'SYNTHETIC TEST PLACEHOLDER - not a result figure')
            with patch('scripts.package_language_topology.verified_report', return_value=report), \
                 patch('scripts.package_language_topology.draw', side_effect=fixture_images):
                release = package(root)
            archive_path = root/release['archive']
            self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), release['archive_sha256'])
            extraction = root/'extracted'; extraction.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read('manifest.json'))
                self.assertEqual(len([n for n in archive.namelist() if '/test-' in n]), 10)
                for name, record in manifest['files'].items():
                    payload = archive.read(name)
                    self.assertEqual(len(payload), record['bytes'])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), record['sha256'])
                    self.assertTrue((extraction/name).resolve().is_relative_to(extraction.resolve()))
                archive.extractall(extraction)
            environment = dict(os.environ); environment.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-E', 'scripts/audit_language_topology_report.py',
                                     'public/research/language-topology-results.json'],
                cwd=extraction, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), release['arithmetic_audit'])


if __name__ == '__main__':
    unittest.main()
