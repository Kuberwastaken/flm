"""Synthetic publication fixtures; these are not language experiment results."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from flm.provenance import sha256
from scripts.language_core_figures import PANELS, contrast_rows, draw, generate, paper_inputs
from test_language_core_report import report_fixture


class CoreFigureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='flm-core-figure-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_incomplete_real_gate_writes_no_artifacts_or_decodes_test_tokens(self):
        with patch('scripts.language_core_report.read_cache') as decoder:
            with self.assertRaisesRegex(ValueError, 'not available'):
                generate(self.root)
        decoder.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_all_pairs_retain_their_own_selection_steps_and_order(self):
        report, *_ = report_fixture(self.root)
        for run in report['runs']:
            run['checkpoint_step'] = 6000 if run['reference'] else 5500
        rows = contrast_rows(report)
        self.assertEqual([(r['family'], r['first'], r['second'], r['training_seed']) for r in rows],
                         [(*panel, seed) for panel in PANELS for seed in (42, 43)])
        for row in rows:
            self.assertEqual(row['first_selected_update'], 6000 if row['family'] == 'primary' else 5500)
            self.assertEqual(row['second_selected_update'], 5500)
            self.assertAlmostEqual(row['difference_bpb'], row['first_bpb'] - row['second_bpb'])

    def test_omitted_condition_or_reversed_effect_cannot_be_plotted(self):
        report, *_ = report_fixture(self.root)
        missing = copy.deepcopy(report); missing['runs'].pop()
        reversed_effect = copy.deepcopy(report)
        reversed_effect['primary_contrasts'][0]['difference_bpb'] *= -1
        for invalid in (missing, reversed_effect):
            with self.assertRaises(ValueError):
                contrast_rows(invalid)

    def test_each_panel_contains_zero_and_every_interval_and_point(self):
        report, *_ = report_fixture(self.root)
        rows = contrast_rows(report)
        # Deliberately use mixed signs, wide and narrow effects for layout only.
        for index, row in enumerate(rows):
            value = (-1 if index % 2 else 1) * (0.0001, .002, .4, 1.2)[index // 2]
            row.update(difference_bpb=value, lower_95=value-abs(value)/2, upper_95=value+abs(value)/2)
        original = plt.subplots
        captured = []
        def capture(*args, **kwargs):
            result = original(*args, **kwargs); captured.append(result); return result
        with patch('scripts.language_core_figures.plt.subplots', side_effect=capture):
            draw(rows, self.root / 'fixture', title='TEST FIXTURE · mixed signs and unequal scales')
        figure, axes = captured[0]
        self.assertIn('own horizontal scale', figure.texts[1].get_text())
        for axis, panel in zip(axes.flat, PANELS):
            lower, upper = axis.get_xlim()
            self.assertLess(lower, 0); self.assertGreater(upper, 0)
            pair = [r for r in rows if (r['family'], r['first'], r['second']) == panel]
            points = [line for line in axis.lines if line.get_marker() in ('o', 's')]
            self.assertEqual([float(line.get_xdata()[0]) for line in points], [r['difference_bpb'] for r in pair])
            self.assertEqual(len(axis.collections), 2)
            for row, interval in zip(pair, axis.collections):
                ends = interval.get_segments()[0][:, 0]
                self.assertEqual(list(ends), [row['lower_95'], row['upper_95']])
                self.assertLess(lower, min(row['lower_95'], row['difference_bpb']))
                self.assertGreater(upper, max(row['upper_95'], row['difference_bpb']))

    def test_fixture_pipeline_binds_all_outputs_and_preserves_eight_csv_rows(self):
        report, *_ = report_fixture(self.root)
        def fixture_draw(rows, output):
            return draw(rows, output, title='TEST FIXTURE · not measured language results')
        with patch('scripts.language_core_figures.verified_report', return_value=report), \
                patch('scripts.language_core_figures.draw', side_effect=fixture_draw):
            record = generate(self.root)
        for name, expected in record['files'].items():
            self.assertEqual(sha256(self.root / name), expected)
        with (self.root / 'public/research/figures/language-core-test.csv').open(newline='') as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 8)
        self.assertEqual([r['family'] for r in rows], ['primary'] * 6 + ['secondary'] * 2)
        tex = paper_inputs(report)
        self.assertEqual(tex.count(r' \\'), 12)  # Four model rows and all eight fitted-pair rows.
        note = (self.root / 'docs/LANGUAGE-CORE-RESULTS.md').read_text(encoding='utf8')
        self.assertIn('do not include training or graph uncertainty', note)
        self.assertIn('78,179', note)
        self.assertIn('76,131', note)


if __name__ == '__main__':
    unittest.main()
