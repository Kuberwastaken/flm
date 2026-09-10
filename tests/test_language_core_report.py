import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from flm.language_core_study import conditions
from flm.language_core_test import summarize
from flm.provenance import sha256, write_json
from scripts.language_core_report import verified_report


def report_fixture(root):
    """Artificial article scores and selections; no trained models or corpus."""
    lexicon = SimpleNamespace(vocabulary=32, sha256='2' * 64, lengths=np.array([0, 0] + [1] * 30))
    documents = []
    for index in range(60):
        tokens = np.full(4 + (index * 7) % 17, 2, dtype=np.int64)
        tokens[0] = 0
        documents.append((f'synthetic-article-{index}', tokens))
    test_relative = 'data/processed/wikitext2-bpe/test.npz'
    cache = root / test_relative
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'Artificial fixture; test token decoding is mocked.')
    folder = root / 'reports/language-core'
    identity = dict(study='Synthetic computation report fixture', inputs={test_relative: sha256(cache)})
    write_json(folder / 'identity.json', identity)
    selected = []
    results = []
    shifts = dict(full=0., fixed_dynamics=.03, no_lateral=.05, no_temporal_state=.2)
    for condition in conditions():
        control = condition['control']
        row = dict(condition, checkpoint=condition['output'] + '/selected.pt', checkpoint_step=500,
            checkpoint_sha256=hashlib.sha256(('fixture/' + condition['label']).encode()).hexdigest(),
            allocated_parameters=600003, trainable_parameters=521824 if control == 'fixed_dynamics' else 600003,
            frozen_parameters=78179 if control == 'fixed_dynamics' else 0, selection_validation_bpb=2.)
        selected.append(row)
        write_json(root / condition['output'] / 'complete.json', {'scope': 'Mocked completion verification'})
        articles = []
        for index, (name, tokens) in enumerate(documents):
            count = len(tokens) - 1
            bpb = 2. + shifts[control] * (1 + math.sin(index + condition['seed']) * .1)
            nll = bpb * count * math.log(2)
            articles.append(dict(document=name, nll=nll, tokens=count, bytes=count, bits_per_byte=bpb))
        nll = math.fsum(article['nll'] for article in articles)
        count = sum(article['bytes'] for article in articles)
        score = dict(documents=articles, nll=nll, bytes=count, tokens=count,
                     bits_per_byte=nll / count / math.log(2), token_perplexity=math.exp(nll / count),
                     tokenizer_sha256=lexicon.sha256)
        result = dict(row, score=score, test_cache_sha256=sha256(cache))
        if condition['reference']:
            original = root / f'reports/wikitext2/test-flm-s{condition["seed"]}.json'
            write_json(original, dict(score=score))
            result['reused_score_sha256'] = sha256(original)
        else:
            score['mechanism'] = ('Reset both states for every token, including within chunks' if control == 'no_temporal_state'
                                  else 'Carry native fast/slow state within each article; reset per article')
        results.append(result)
    selection = dict(runs=selected, study_identity_sha256=sha256(folder / 'identity.json'),
                     tokenizer_sha256=lexicon.sha256, test_cache_sha256=sha256(cache))
    write_json(folder / 'selection.json', selection)
    selection_hash = sha256(folder / 'selection.json')
    for result in results:
        result['selection_sha256'] = selection_hash
        write_json(folder / ('test-' + result['label'] + '.json'), result)
    report = dict(study=identity['study'], study_identity_sha256=selection['study_identity_sha256'],
                  selection_sha256=selection_hash, runs=results, **summarize(results))
    write_json(folder / 'summary.json', report)
    return report, selected, lexicon, documents


class CoreReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-report-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report, self.selected, self.lexicon, self.documents = report_fixture(self.root)
        self.folder = self.root / 'reports/language-core'

    def context(self):
        # Real checkpoint completion is exercised by the frozen training harness;
        # this fixture isolates report assembly and its no-test-before-gate rule.
        return (
            patch('scripts.language_core_report.verify_identity'),
            patch('scripts.language_core_report.verify_complete', side_effect=copy.deepcopy(self.selected)),
            patch('scripts.language_core_report.Lexicon', return_value=self.lexicon),
            patch('scripts.language_core_report.read_cache', return_value=self.documents),
        )

    def test_missing_final_report_or_any_run_fails_before_test_decoding(self):
        for missing in (self.folder / 'summary.json', self.root / self.selected[-1]['output'] / 'complete.json'):
            original = missing.read_bytes()
            missing.unlink()
            with patch('scripts.language_core_report.verify_identity'), \
                 patch('scripts.language_core_report.Lexicon') as lexicon, \
                 patch('scripts.language_core_report.read_cache') as decoder:
                with self.assertRaisesRegex(ValueError, 'not available|incomplete'):
                    verified_report(self.root)
                decoder.assert_not_called()
                lexicon.assert_not_called()
            missing.write_bytes(original)

    def test_failed_checkpoint_verification_or_changed_selection_precedes_test_decoding(self):
        a, b, c, d = self.context()
        with a, b as checker, c, d as decoder:
            checker.side_effect = ValueError('Invalid saved checkpoint')
            with self.assertRaisesRegex(ValueError, 'Invalid saved checkpoint'):
                verified_report(self.root)
            decoder.assert_not_called()
        selected = json.loads((self.folder / 'selection.json').read_text(encoding='utf8'))
        selected['runs'][-1]['checkpoint_step'] = 1000
        write_json(self.folder / 'selection.json', selected)
        a, b, c, d = self.context()
        with a, b, c, d as decoder:
            with self.assertRaisesRegex(ValueError, 'checkpoints changed'):
                verified_report(self.root)
            decoder.assert_not_called()

    def test_complete_synthetic_report_passes_without_mutating_artifacts(self):
        before = {p.relative_to(self.root).as_posix(): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        a, b, c, d = self.context()
        with a, b as checker, c, d as decoder:
            actual = verified_report(self.root)
            self.assertEqual(actual, self.report)
            self.assertEqual(checker.call_count, 8)
            decoder.assert_called_once_with(self.root / 'data/processed/wikitext2-bpe/test.npz')
        after = {p.relative_to(self.root).as_posix(): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_changed_article_denominator_reference_or_mechanism_is_rejected(self):
        for fault in ('article', 'reference', 'mechanism'):
            with self.subTest(fault=fault):
                report = copy.deepcopy(self.report)
                run = report['runs'][-1]
                if fault == 'article':
                    run['score']['documents'][0]['bytes'] += 1
                elif fault == 'mechanism':
                    run['score']['mechanism'] = 'Carry temporal state'
                else:
                    run = report['runs'][0]
                    run['reused_score_sha256'] = '0' * 64
                write_json(self.folder / ('test-' + run['label'] + '.json'), run)
                write_json(self.folder / 'summary.json', report)
                a, b, c, d = self.context()
                with a, b, c, d:
                    with self.assertRaisesRegex(ValueError, 'denominator|reference score|mechanism declaration'):
                        verified_report(self.root)
                original = next(row for row in self.report['runs'] if row['label'] == run['label'])
                write_json(self.folder / ('test-' + run['label'] + '.json'), original)

    def test_report_recomputes_primary_intervals_and_secondary_mean(self):
        for fault in ('interval', 'mean'):
            report = copy.deepcopy(self.report)
            if fault == 'interval':
                report['primary_contrasts'][0]['lower_95'] += .01
            else:
                report['independent_unit_memory_mean_difference_bpb'] += .01
            write_json(self.folder / 'summary.json', report)
            a, b, c, d = self.context()
            with a, b, c, d:
                with self.assertRaisesRegex(ValueError, 'Recomputed computation contrasts'):
                    verified_report(self.root)


if __name__ == '__main__':
    unittest.main()
