import tempfile
from pathlib import Path
import unittest
from flm.language_test import freeze_selection, paired_interval


class ReportTests(unittest.TestCase):
    def test_test_selection_fails_closed_when_a_registered_run_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'incomplete'): freeze_selection(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_paired_bootstrap_is_exact_for_a_constant_byte_normalized_gap(self):
        a = [dict(document=str(i), bytes=n, nll=n * 2.) for i, n in enumerate([10, 100, 1000])]
        b = [dict(document=str(i), bytes=n, nll=n) for i, n in enumerate([10, 100, 1000])]
        import math
        result = paired_interval(a, b, 1000)
        for key in ('difference_bpb', 'lower_95', 'upper_95'): self.assertAlmostEqual(result[key], 1 / math.log(2))
        self.assertEqual(paired_interval(a, a, 1000)['upper_95'], 0)
        with self.assertRaises(ValueError): paired_interval(a, b[:-1])
        with self.assertRaises(ValueError): paired_interval(a, a + [a[0]])


if __name__ == '__main__': unittest.main()
