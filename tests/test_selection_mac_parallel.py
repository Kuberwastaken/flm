"""Small scheduling-policy/failure checks; no training or background processes."""
import unittest
from scripts import selection_mac_parallel as parallel


def row(workers, speed, exact=True):
    return dict(workers=workers, updates_per_second=speed, exact_final_state_and_optimizer=exact)


class ParallelTests(unittest.TestCase):
    def test_rejects_fast_configuration_with_different_tensors(self):
        self.assertEqual(parallel.choose_workers([row(1, 10), row(2, 17), row(3, 30, False), row(4, 40, False)]), 2)

    def test_near_tie_prefers_fewer_workers_and_small_gain_stays_serial(self):
        self.assertEqual(parallel.choose_workers([row(1, 10), row(2, 20), row(3, 21), row(4, 22)]), 2)
        self.assertEqual(parallel.choose_workers([row(1, 10), row(2, 11.4)]), 1)

    def test_initializer_failure_becomes_task_failure(self):
        previous = parallel.CTX
        try:
            parallel.CTX = {'initialization_error': 'input integrity failure'}
            with self.assertRaisesRegex(RuntimeError, 'input integrity failure'):
                parallel.fit_worker(0)
        finally:
            parallel.CTX = previous


if __name__ == '__main__':
    unittest.main()
