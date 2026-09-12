"""Fast synthetic decision checks; no model, corpus, fitting or resampling."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import selection_decision_report as decision


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value)+'\n').encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def contrasts(rule, delta=-.006):
    rows = []
    for group in rule['groups']:
        for seed in rule['training_seeds']:
            for subset in rule['analysis_subsets']:
                for kind, level, name, first, second in decision.comparisons(group):
                    value = delta if kind == 'topology' else 0.
                    rows.append(dict(group=group, training_seed=seed, subset=subset,
                        kind=kind, level=level, name=name, first=first, second=second, reason=None,
                        estimate=dict(difference_bpb=value, lower_95=value-.001,
                                      upper_95=value+.001, replicates=10000, seed=31415)))
    return rows


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.rule = json.loads((decision.ROOT / decision.RULE).read_text())

    def test_margin_inclusive_and_no_rounding(self):
        self.assertEqual(decision.apply_rule(self.rule, contrasts(self.rule, -.005))['decision'], 'pass')
        self.assertEqual(decision.apply_rule(self.rule, contrasts(self.rule, -.004999999))['decision'], 'fail')

    def test_mac_lineage_cannot_change_budget_or_graph(self):
        parent = json.loads((decision.ROOT/decision.IDENTITY).read_text())
        native = copy.deepcopy(parent)
        native['torch'] = '2.8.0'
        native['execution_amendment'] = {'parent_study_identity_sha256': self.rule['original_identity_sha256']}
        for row in native['conditions']:
            for key in ('initial_state_sha256', 'initial_parameters_sha256', 'initial_nonedge_parameters_sha256'):
                row['base_binding'][key] = '1'*64
        decision.validate_mac_identity(parent, native)
        changed = copy.deepcopy(native)
        changed['conditions'][0]['settings']['steps'] = 2000
        with self.assertRaises(ValueError):
            decision.validate_mac_identity(parent, changed)
        changed = copy.deepcopy(native)
        changed['conditions'][0]['condition']['graph'] = 'another-graph'
        with self.assertRaises(ValueError):
            decision.validate_mac_identity(parent, changed)

    def test_filtered_interval_touching_zero_fails(self):
        rows = contrasts(self.rule)
        row = next(r for r in rows if r['name'] == self.rule['primary_contrast'] and r['subset'] == 'overlap_filtered')
        row['estimate']['upper_95'] = 0
        result = decision.apply_rule(self.rule, rows)
        self.assertEqual(result['decision'], 'fail')
        self.assertTrue(all(result['analyses'][0]['checks'].values()))

    def test_one_nonnegative_rewire_cannot_hide_in_mean(self):
        rows = contrasts(self.rule)
        cell = [r for r in rows if r['group'] == self.rule['groups'][0] and r['training_seed'] == 42
                and r['subset'] == 'official' and r['name'].startswith('candidate-measured-minus-')]
        for row in cell:
            if row['level'] == 'individual':
                row['estimate']['difference_bpb'] = 0. if row['name'].endswith('101') else -.009
        self.assertEqual(decision.apply_rule(self.rule, rows)['decision'], 'fail')

    def test_undefined_nonfinite_duplicate_and_missing_are_invalid(self):
        base = contrasts(self.rule)
        variants = [base[:-1], base+[base[0]]]
        for value in (None, float('nan'), float('inf')):
            changed = copy.deepcopy(base)
            changed[0]['estimate']['difference_bpb'] = value
            variants.append(changed)
        for rows in variants:
            with self.subTest(), self.assertRaises(ValueError):
                decision.apply_rule(self.rule, rows)

    def test_record_consumer_complete_pending_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (decision.RULE, decision.IDENTITY, decision.DOCUMENT):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(decision.ROOT/name, target)
            self.assertEqual(decision.report(root)['decision'], 'pending')
            # Partial held-out data is deliberately malformed and must not be opened.
            write(root, decision.DIRECTORY+'test/partial/result.json', {'broken': True})
            self.assertEqual(decision.report(root)['decision'], 'pending')
            identity = json.loads((root / decision.IDENTITY).read_text())
            for name in identity['sources']:
                target = root / 'flm' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(decision.ROOT/'flm'/name, target)
            selected = [dict(condition=r['condition'], checkpoint_step=3000) for r in identity['conditions']]
            selection_hash = write(root, decision.SELECTION,
                dict(study_identity_sha256=self.rule['original_identity_sha256'], conditions=selected))
            eval_hash = write(root, decision.EVALUATION,
                dict(study_identity_sha256=self.rule['original_identity_sha256'], selection_sha256=selection_hash,
                     policy=identity['evaluation_policy'], source_sha256=identity['sources'], test_metadata=identity['test_metadata']))
            runs, hashes, aggregates = [], {}, []
            for choice in selected:
                c = choice['condition']
                bpb = 2. if c['graph'].endswith('/measured') else 2.006
                components = {'all': {s: dict(bits_per_byte=bpb) for s in self.rule['analysis_subsets']}}
                row = dict(condition=c, checkpoint_step=3000, components=components,
                           identity=dict(evaluation_identity_sha256=eval_hash, selected=choice))
                runs.append(row)
                hashes[c['label']] = write(root, decision.DIRECTORY+'test/'+c['label']+'/result.json', dict(row, blocks=[]))
                if c['seed'] == 42:
                    for subset in self.rule['analysis_subsets']:
                        aggregates.append(dict(graph=c['graph'], component='all', subset=subset,
                                               seeds=[42, 43], bits_per_byte=[bpb, bpb]))
            rows = contrasts(self.rule)
            across = [dict(**{k: r[k] for k in ('group', 'subset', 'kind', 'level', 'name')},
                           seeds=[42, 43], difference_bpb=[r['estimate']['difference_bpb']]*2)
                      for r in rows if r['training_seed'] == 42]
            summary = dict(evaluation_identity_sha256=eval_hash, runs=runs, result_sha256=hashes,
                           paired_comparisons=rows, aggregates=aggregates, paired_seed_summary=across)
            write(root, decision.SUMMARY, summary)
            self.assertEqual(decision.report(root)['decision'], 'pass')
            write(root, decision.DIRECTORY+'test/'+selected[0]['condition']['label']+'/result.json', {})
            self.assertEqual(decision.report(root)['decision'], 'invalid')


if __name__ == '__main__':
    unittest.main()
