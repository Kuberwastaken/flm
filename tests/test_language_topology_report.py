import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flm.language_topology import conditions
from scripts.language_topology_report import contrast_rows, verified_report


def layout_fixture():
    """Artificial scores for plotting tests only; never a published result."""
    runs = []; topology = []; slow = []
    for condition in conditions():
        difference = -.014 if condition['variant'] == 'no_slow' else {
            (101, 42): .002, (101, 43): -.001,
            (103, 42): .001, (103, 43): .003,
            (107, 42): -.002, (107, 43): .0015}.get((condition['graph_seed'], condition['seed']), 0)
        measured = 2 if condition['seed'] == 42 else 2.01
        runs.append(dict(**condition, checkpoint_step=6000 if condition['reference'] else 5500,
                         score={'bits_per_byte': measured-difference}))
        if not condition['reference']:
            point = dict(training_seed=condition['seed'], difference_bpb=difference,
                         lower_95=difference-.001, upper_95=difference+.001)
            if condition['graph_seed']:
                topology.append(dict(**point, graph_seed=condition['graph_seed']))
            else:
                slow.append(point)
    return dict(runs=runs, topology_contrasts=topology, slow_state_contrasts=slow)


class LanguageTopologyReportTests(unittest.TestCase):
    def test_partial_study_has_no_test_reads_or_plot_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('scripts.language_topology_report.read_cache') as reader:
                with self.assertRaisesRegex(ValueError, 'not available'):
                    verified_report(root)
                reader.assert_not_called()
            self.assertFalse((root/'public').exists())

    def test_all_eight_rows_keep_their_own_selected_checkpoints_and_pair(self):
        rows = contrast_rows(layout_fixture())
        self.assertEqual([row['control'] for row in rows], [
            'null101-s42', 'null101-s43', 'null103-s42', 'null103-s43',
            'null107-s42', 'null107-s43', 'no-slow-s42', 'no-slow-s43'])
        for row in rows:
            self.assertEqual(row['measured_selected_update'], 6000)
            self.assertEqual(row['control_selected_update'], 5500)
            self.assertAlmostEqual(row['measured_bpb']-row['control_bpb'], row['difference_bpb'])
        self.assertEqual(rows[0]['lower_95'], .001)
        self.assertEqual(rows[1]['upper_95'], 0)
        self.assertEqual([row['graph_seed'] for row in rows[-2:]], ['', ''])

    def test_wrong_effect_direction_cannot_be_plotted_as_the_measured_pair(self):
        report = copy.deepcopy(layout_fixture())
        report['topology_contrasts'][0]['difference_bpb'] *= -1
        with self.assertRaisesRegex(ValueError, 'differs from its paired scores'):
            contrast_rows(report)


if __name__ == '__main__':
    unittest.main()
