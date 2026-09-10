import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.package_topology_inference import package
from scripts.verify_topology_inference_release import verify


class TopologyInferenceReleaseTests(unittest.TestCase):
    def test_incomplete_study_cannot_export_or_sample_models(self):
        with tempfile.TemporaryDirectory(prefix='flm-export-gate-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            with patch('scripts.package_topology_inference.restore') as restore, \
                 patch('scripts.package_topology_inference.generate') as generate:
                with self.assertRaisesRegex(ValueError, 'not available'):
                    package(root)
                restore.assert_not_called(); generate.assert_not_called()
            self.assertFalse((root/'public').exists())
            self.assertFalse((root/'output').exists())

    def test_corrupt_archive_fails_before_extraction_or_any_command(self):
        with tempfile.TemporaryDirectory(prefix='flm-export-integrity-') as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            report = root/'reports/language-topology/inference-release.json'
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps(dict(archive='fixture.zip', archive_sha256='0'*64)))
            (root/'fixture.zip').write_bytes(b'changed')
            with patch('scripts.verify_topology_inference_release.subprocess.run') as command:
                with self.assertRaisesRegex(ValueError, 'archive changed'):
                    verify(root)
                command.assert_not_called()
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
