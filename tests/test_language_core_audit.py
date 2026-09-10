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

from flm.language_core_study import conditions
from flm.language_core_test import summarize
from scripts.audit_language_core_report import audit


def report_fixture():
    """Unequal synthetic article sizes, mixed effect signs, and shuffled order.

    The frozen scorer builds the comparison oracle. The independent auditor
    never imports it; no corpus, checkpoint, or trained prediction is read.
    """
    runs = []
    for condition in conditions():
        control = condition['control']; seed = condition['seed']
        effect = dict(full=0., fixed_dynamics=.03, no_lateral=.06, no_temporal_state=.1)[control]
        if seed == 43:
            effect *= -.5
        articles = []
        for index in range(60):
            tokens = index + 1; size = tokens * (2 + index % 4)
            bpb = 1.8 + index / 200 + effect * (1 + math.sin(index + seed) / 3)
            articles.append(dict(document=f'fixture-article-{index:02d}', nll=bpb * size * math.log(2),
                                 bytes=size, tokens=tokens, bits_per_byte=bpb))
        if seed == 43:
            articles.reverse()
        nll = sum(row['nll'] for row in articles)
        size = sum(row['bytes'] for row in articles); tokens = sum(row['tokens'] for row in articles)
        score = dict(documents=articles, nll=nll, bytes=size, tokens=tokens,
            bits_per_byte=nll / size / math.log(2), token_perplexity=math.exp(nll / tokens), tokenizer_sha256='b' * 64)
        if not condition['reference']:
            score['mechanism'] = ('Reset both states for every token, including within chunks' if control == 'no_temporal_state'
                                  else 'Carry native fast/slow state within each article; reset per article')
        run = dict(**condition, checkpoint_step=6000 if condition['reference'] else 5500,
            checkpoint=condition['output'] + '/synthetic-selected.pt',
            checkpoint_sha256=hashlib.sha256(condition['label'].encode()).hexdigest(),
            allocated_parameters=600003, trainable_parameters=521824 if control == 'fixed_dynamics' else 600003,
            frozen_parameters=78179 if control == 'fixed_dynamics' else 0,
            selection_sha256='c' * 64, test_cache_sha256='d' * 64, score=score)
        if condition['reference']:
            run['reused_score_sha256'] = hashlib.sha256(('reference/' + condition['label']).encode()).hexdigest()
        runs.append(run)
    return dict(study='Synthetic computation arithmetic fixture', study_identity_sha256='a' * 64,
                selection_sha256='c' * 64, runs=list(reversed(runs)), **summarize(runs))


class CoreArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = report_fixture()

    def test_independent_byte_weighted_arithmetic_matches_all_declared_comparisons(self):
        result = audit(self.report)
        self.assertEqual(result['verified_runs'], 8)
        self.assertEqual(result['unique_test_articles'], 60)
        self.assertEqual(result['scored_tokens_per_run'], 1830)
        self.assertEqual(result['scored_bytes_per_run'], 6480)
        self.assertEqual(len(result['primary_contrasts']), 6)
        self.assertEqual(len(result['independent_unit_memory_contrasts']), 2)
        self.assertEqual(len(result['primary_means']), 3)
        self.assertEqual({row['difference_bpb'] > 0 for row in result['primary_contrasts']}, {True, False})
        score = self.report['runs'][0]['score']
        self.assertNotAlmostEqual(score['bits_per_byte'],
                                 sum(row['bits_per_byte'] for row in score['documents']) / 60)

    def test_incomplete_wrong_identity_counts_or_mechanism_reject(self):
        mutations = [lambda r: r['runs'].pop(), lambda r: r['runs'].__setitem__(1, r['runs'][0]),
            lambda r: r['runs'][0].update(seed=42), lambda r: r['runs'][0].update(selection_sha256='f' * 64),
            lambda r: r['runs'][0].update(checkpoint_step=6001),
            lambda r: r['runs'][0].update(graph='a-different-graph.npz'),
            lambda r: r['runs'][0].update(allocated_parameters=600003.),
            lambda r: r['runs'][0]['score'].update(mechanism='Reset only at chunk boundaries'),
            lambda r: next(x for x in r['runs'] if x['control'] == 'fixed_dynamics').update(trainable_parameters=600003),
            lambda r: next(x for x in r['runs'] if x['reference']).pop('reused_score_sha256')]
        for index, change in enumerate(mutations):
            with self.subTest(mutation=index):
                report = copy.deepcopy(self.report); change(report)
                with self.assertRaises(ValueError):
                    audit(report)

    def test_missing_duplicated_nonfinite_or_mismatched_article_evidence_rejects(self):
        mutations = [lambda r: r['runs'][0]['score']['documents'].pop(),
            lambda r: r['runs'][0]['score']['documents'].__setitem__(1, r['runs'][0]['score']['documents'][0]),
            lambda r: r['runs'][0]['score']['documents'][0].update(nll=float('nan')),
            lambda r: r['runs'][0]['score']['documents'][0].update(tokens=999),
            lambda r: r['runs'][0]['score']['documents'][0].update(bytes=0),
            lambda r: r['runs'][0]['score'].update(bits_per_byte=0),
            lambda r: r['runs'][0]['score'].update(token_perplexity=float('inf')),
            lambda r: r['runs'][0]['score'].update(tokenizer_sha256='e' * 64)]
        for index, change in enumerate(mutations):
            with self.subTest(mutation=index):
                report = copy.deepcopy(self.report); change(report)
                with self.assertRaises(ValueError):
                    audit(report)

    def test_wrong_paired_direction_bootstrap_interval_or_mean_rejects(self):
        mutations = [lambda r: r['primary_contrasts'][0].update(lower_95=-999),
            lambda r: r['primary_contrasts'][0].update(first='no_lateral'),
            lambda r: r['primary_contrasts'][0].update(seed=1),
            lambda r: r['primary_contrasts'][0].update(unit='independent tokens'),
            lambda r: r.update(sign='Second minus first'),
            lambda r: r['independent_unit_memory_contrasts'].__setitem__(1, r['independent_unit_memory_contrasts'][0]),
            lambda r: r['primary_means'][0].update(mean_difference_bpb=99),
            lambda r: r['primary_means'][0].update(lower_95=-99),
            lambda r: r.update(independent_unit_memory_mean_difference_bpb=99)]
        for index, change in enumerate(mutations):
            with self.subTest(mutation=index):
                report = copy.deepcopy(self.report); change(report)
                with self.assertRaises(ValueError):
                    audit(report)

    def test_report_and_article_order_do_not_change_audited_results(self):
        report = copy.deepcopy(self.report)
        report['runs'].reverse()
        for run in report['runs']:
            run['score']['documents'].reverse()
        report['primary_contrasts'].reverse(); report['primary_means'].reverse()
        report['independent_unit_memory_contrasts'].reverse()
        self.assertEqual(audit(report), audit(self.report))

    def test_standalone_script_runs_with_torch_and_flm_imports_disabled(self):
        with tempfile.TemporaryDirectory(prefix='flm-core-arithmetic-') as directory:
            folder = Path(directory).resolve()
            self.assertTrue(folder.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            source = Path(__file__).resolve().parents[1] / 'scripts/audit_language_core_report.py'
            shutil.copyfile(source, folder / 'audit.py')
            (folder / 'fixture.json').write_text(json.dumps(self.report), encoding='utf8')
            environment = dict(os.environ); environment.pop('PYTHONPATH', None)
            runner = ("import runpy,sys; sys.modules['flm']=None; sys.modules['torch']=None; "
                      "sys.argv=['audit.py','fixture.json']; runpy.run_path('audit.py',run_name='__main__')")
            process = subprocess.run([sys.executable, '-E', '-c', runner], cwd=folder,
                env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout), audit(self.report))
            self.assertFalse((folder / 'flm').exists())


if __name__ == '__main__':
    unittest.main()
