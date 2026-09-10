from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flm.language_topology_queue import queue_lease, schedule, training_command


class FakeChild:
    def __init__(self, ticks=2, code=0):
        self.ticks, self.code, self.terminated = ticks, code, False

    def poll(self):
        if self.terminated:
            return -1
        self.ticks -= 1
        return self.code if self.ticks <= 0 else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return -1 if self.terminated else self.code


class LanguageQueueTests(unittest.TestCase):
    def test_limits_concurrency_and_runs_every_condition_once(self):
        started, finished, children = [], [], []
        def start(job):
            self.assertLess(sum(child.ticks > 0 for child in children), 2)
            started.append(job['label']); child = FakeChild(3)
            children.append(child); return child
        jobs = [dict(label=str(i)) for i in range(5)]
        schedule(jobs, 2, start, lambda job: finished.append(job['label']), pause=lambda _: None)
        self.assertEqual(started, [str(i) for i in range(5)])
        self.assertEqual(finished, started)
        self.assertFalse(any(child.terminated for child in children))

    def test_failure_finishes_live_sibling_but_never_launches_more_work(self):
        started, finished = [], []
        children = [FakeChild(1, 1), FakeChild(3)]
        def start(job):
            started.append(job['label']); return children[len(started) - 1]
        with self.assertRaisesRegex(RuntimeError, 'remaining jobs were not launched'):
            schedule([dict(label=str(i)) for i in range(3)], 2, start,
                     lambda job: finished.append(job['label']), pause=lambda _: None)
        self.assertEqual(started, ['0', '1']); self.assertEqual(finished, ['1'])
        self.assertFalse(children[1].terminated)

    def test_verification_failure_stops_only_live_owned_children(self):
        children = [FakeChild(1), FakeChild(10)]
        def reject(_):
            raise ValueError('Checkpoint verification rejected')
        with self.assertRaisesRegex(ValueError, 'verification rejected'):
            schedule([dict(label='0'), dict(label='1')], 2,
                     lambda job: children[int(job['label'])], reject, pause=lambda _: None)
        self.assertFalse(children[0].terminated); self.assertTrue(children[1].terminated)

    def test_low_memory_keeps_one_worker_without_changing_jobs(self):
        active, finished = [], []
        def start(job):
            self.assertFalse(any(child.ticks > 0 for child in active))
            child = FakeChild(2); active.append(child); return child
        schedule([dict(label='a'), dict(label='b')], 2, start,
                 lambda job: finished.append(job['label']), can_add=lambda: False, pause=lambda _: None)
        self.assertEqual(finished, ['a', 'b'])

    def test_os_lease_blocks_another_process_and_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'queue.lock'
            code = ('from pathlib import Path; import sys; '
                    'from flm.language_topology_queue import queue_lease; '
                    'lease=queue_lease(Path(sys.argv[1])); lease.__enter__(); lease.__exit__(None,None,None)')
            with queue_lease(path):
                result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('holds the queue lease', result.stderr)
            result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    @patch('flm.language_topology_queue.verify_source_identity')
    def test_commands_keep_exact_training_settings_and_require_verified_resume(self, _):
        condition = dict(label='null101-s42', seed=42, variant='flm',
                         graph='data/graphs/central-1024-null101/graph.npz', output='runs/control')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = training_command(root, condition, {}, None)
            for key, value in [('--variant', 'flm'), ('--threads', '4'), ('--steps', '6000'),
                               ('--seed', '42'), ('--graph', condition['graph']), ('--output', condition['output'])]:
                self.assertEqual(command[command.index(key) + 1], value)
            directory = root / condition['output']; directory.mkdir(parents=True)
            (directory / 'run.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'without a saved checkpoint'):
                training_command(root, condition, {}, None)
            (directory / 'last.pt').write_bytes(b'fixture')
            with patch('flm.language_topology_queue.restore', return_value=(None, {'step': 4500})), \
                 patch('flm.language_topology_queue.verify_saved') as verify:
                command = training_command(root, condition, {}, None)
                verify.assert_called_once()
                self.assertEqual(command[-2:], ['--resume', str(directory / 'last.pt')])
                verify.side_effect = ValueError('RNG/exposure drift')
                with self.assertRaisesRegex(ValueError, 'RNG/exposure drift'):
                    training_command(root, condition, {}, None)


if __name__ == '__main__':
    unittest.main()
