import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from flm.provenance import sha256, write_json
from scripts.package_language_core import (ARCHIVE, ASSETS, FOLDER, PUBLIC_RELEASE,
    PUBLIC_RESULT, RELEASE, SUMMARY, extract_checked, package)
from test_language_core_audit import report_fixture


def records_fixture(root):
    """Synthetic article scores and selection; checkpoint gate mocked by caller."""
    repository = Path(__file__).resolve().parents[1]
    source_identity = json.loads((repository / FOLDER / 'identity.json').read_bytes())
    source_names = list(source_identity['sources'])
    for name in set(ASSETS) | set(source_names):
        if name.startswith('reports/wikitext2/test-'):
            continue
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, path)
    report = report_fixture()
    identity = dict(study=report['study'], scope='Synthetic records-release fixture',
                    sources={name: sha256(root / name) for name in source_names})
    write_json(root / FOLDER / 'identity.json', identity)
    identity_hash = sha256(root / FOLDER / 'identity.json')
    selected = []
    for run in report['runs']:
        selected.append({key: value for key, value in run.items()
                         if key not in ('score', 'test_cache_sha256', 'selection_sha256', 'reused_score_sha256')})
        if run['reference']:
            path = root / f'reports/wikitext2/test-flm-s{run["seed"]}.json'
            write_json(path, dict(score=run['score'], scope='Synthetic reference fixture'))
            run['reused_score_sha256'] = sha256(path)
    selection = dict(runs=selected, study_identity_sha256=identity_hash, tokenizer_sha256='b' * 64,
                     test_cache_sha256='d' * 64)
    write_json(root / FOLDER / 'selection.json', selection)
    report['study_identity_sha256'] = identity_hash
    report['selection_sha256'] = sha256(root / FOLDER / 'selection.json')
    for run in report['runs']:
        run['selection_sha256'] = report['selection_sha256']
        write_json(root / FOLDER / ('test-' + run['label'] + '.json'), run)
    write_json(root / SUMMARY, report)
    return report


class CoreRecordsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-records-fixture-')
        cls.addClassCleanup(temporary.cleanup)
        cls.base = Path(temporary.name).resolve()
        assert cls.base.is_relative_to(Path(tempfile.gettempdir()).resolve())
        cls.report = records_fixture(cls.base)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-records-check-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        shutil.copytree(self.base, self.root, dirs_exist_ok=True)

    def test_all_records_and_csvs_pass_fresh_external_archive_audit(self):
        # Only the real training/checkpoint gate is mocked. Independent arithmetic,
        # reference binding, serialization, CSVs, ZIP and fresh CLI audit run normally.
        with patch('scripts.package_language_core.verified_report', return_value=self.report) as gate:
            release = package(self.root)
            gate.assert_called_once_with(self.root)
        self.assertTrue(release['fresh_archive_arithmetic_verified'])
        self.assertTrue(release['extraction_outside_repository'])
        self.assertTrue(release['flm_and_torch_imports_disabled'])
        self.assertEqual(release['arithmetic_audit']['verified_runs'], 8)
        self.assertEqual(sha256(self.root / ARCHIVE), release['archive_sha256'])
        self.assertEqual((self.root / PUBLIC_RESULT).read_bytes(), (self.root / SUMMARY).read_bytes())
        self.assertEqual(json.loads((self.root / PUBLIC_RELEASE).read_bytes()), release)
        with zipfile.ZipFile(self.root / ARCHIVE) as archive:
            names = archive.namelist()
            self.assertEqual(len([name for name in names if name.startswith(FOLDER + '/test-')]), 8)
            self.assertFalse(any(name.startswith(('runs/', 'data/raw/', 'data/processed/')) or name.endswith(('.pt', '.npz'))
                                 for name in names))
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(set(names), set(manifest['files']) | {'manifest.json'})
            for name, record in manifest['files'].items():
                self.assertEqual(len(archive.read(name)), record['bytes'])
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), record['sha256'])
            for name, count in [('model-scores.csv', 8), ('article-scores.csv', 480), ('paired-contrasts.csv', 8)]:
                data = archive.read('tables/' + name)
                rows = list(csv.DictReader(io.StringIO(data.decode('utf8'))))
                self.assertEqual(len(rows), count)
                self.assertEqual((self.root / ('public/research/language-core-' + name)).read_bytes(), data)
            articles = list(csv.DictReader(io.StringIO(archive.read('tables/article-scores.csv').decode('utf8'))))
            actual = {(row['condition'], row['article']): row for row in articles}
            for run in self.report['runs']:
                for article in run['score']['documents']:
                    row = actual[(run['label'], article['document'])]
                    self.assertEqual(float(row['nll']), article['nll'])
                    self.assertEqual(float(row['bits_per_byte']), article['bits_per_byte'])
                    self.assertEqual(int(row['bytes']), article['bytes'])
                    self.assertEqual(int(row['tokens']), article['tokens'])

    def test_incomplete_study_cannot_audit_decode_or_write_release(self):
        (self.root / SUMMARY).unlink()
        with patch('scripts.package_language_core.audit') as checker, \
             patch('scripts.package_language_core.fresh_audit') as fresh, \
             patch('scripts.language_core_report.read_cache') as decoder:
            with self.assertRaisesRegex(ValueError, 'not available'):
                package(self.root)
            checker.assert_not_called(); fresh.assert_not_called(); decoder.assert_not_called()
        self.assertFalse((self.root / 'public').exists())

    def test_individual_score_or_reused_reference_changed_after_gate_rejects(self):
        for relative in (FOLDER + '/test-' + self.report['runs'][0]['label'] + '.json',
                         'reports/wikitext2/test-flm-s42.json'):
            with self.subTest(path=relative):
                path = self.root / relative; original = path.read_bytes()
                changed = json.loads(original); changed['score']['bits_per_byte'] += 1
                write_json(path, changed)
                with patch('scripts.package_language_core.verified_report', return_value=self.report):
                    with self.assertRaisesRegex(ValueError, 'individual score|reference score'):
                        package(self.root)
                self.assertFalse((self.root / 'public').exists())
                path.write_bytes(original)

    def test_fresh_audit_failure_cannot_publish_a_record(self):
        with patch('scripts.package_language_core.verified_report', return_value=self.report), \
             patch('scripts.package_language_core.fresh_audit', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'Fresh computation score arithmetic differs'):
                package(self.root)
        self.assertFalse((self.root / 'public').exists())
        self.assertFalse((self.root / RELEASE).exists())

    def test_different_selection_cannot_overwrite_owned_records(self):
        write_json(self.root / RELEASE, dict(selection_sha256='0' * 64))
        before = (self.root / RELEASE).read_bytes()
        with patch('scripts.package_language_core.verified_report', return_value=self.report):
            with self.assertRaisesRegex(ValueError, 'different selected study'):
                package(self.root)
        self.assertEqual((self.root / RELEASE).read_bytes(), before)
        self.assertFalse((self.root / 'public').exists())

    def test_archive_byte_changes_and_unsafe_paths_fail_before_audit(self):
        path = self.root / 'fixture.zip'; extraction = self.root / 'extracted'
        for name, payload, expected in [('report.json', b'changed', {'report.json': b'original'}),
                                        ('../outside', b'data', {'../outside': b'data'})]:
            with self.subTest(name=name):
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(name, payload)
                with self.assertRaisesRegex(ValueError, 'payload changed|archive path'):
                    extract_checked(path, extraction, expected)
                self.assertFalse(extraction.exists())


if __name__ == '__main__':
    unittest.main()
