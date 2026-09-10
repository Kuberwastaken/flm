"""Verify the candidate archive, then run its audit from a fresh extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable, help='Interpreter for the extracted NumPy-only audit')
    args = parser.parse_args()
    release_path = ROOT / 'reports/embodiment/closed-loop/release-candidate.json'
    release = json.loads(release_path.read_text())
    folder = ROOT / 'output/closed-loop-release'
    for name, digest in release['files'].items():
        if sha(folder / name) != digest:
            raise ValueError('Feedback candidate changed: ' + name)
    archive_path = folder / 'closed-loop-records.zip'
    if sha(archive_path) != release['archive_sha256']:
        raise ValueError('Feedback archive changed')
    parent = (ROOT / 'output').resolve()
    with tempfile.TemporaryDirectory(prefix='closed-loop-audit-', dir=parent) as temporary:
        extracted = Path(temporary).resolve()
        # Verify the exact absolute cleanup target before any archive writes.
        if extracted.parent != parent or not extracted.name.startswith('closed-loop-audit-'):
            raise ValueError('Temporary audit directory is outside the intended output folder')
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or archive.testzip() is not None:
                raise ValueError('Duplicate or damaged archive entry')
            manifest = json.loads(archive.read('manifest.json'))
            if set(names) != set(manifest) | {'manifest.json'}:
                raise ValueError('Unexpected or missing manifest payload')
            for name in names:
                relative = PurePosixPath(name)
                target = (extracted / name).resolve()
                if relative.is_absolute() or '\\' in name or '..' in relative.parts or not target.is_relative_to(extracted):
                    raise ValueError('Unsafe archive path')
                payload = archive.read(name)
                if name != 'manifest.json' and hashlib.sha256(payload).hexdigest() != manifest[name]:
                    raise ValueError('Archive payload changed: ' + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        result = subprocess.run([args.python, '-X', 'utf8', 'scripts/audit_feedback_release.py'],
                                cwd=extracted, capture_output=True, text=True, encoding='utf8')
        if result.returncode:
            raise RuntimeError('Fresh extraction audit failed:\n' + result.stdout + result.stderr)
        audit = json.loads(result.stdout)
        if audit['conditions'] != 27 or audit['control_frames'] != 10800 or audit['delayed_decisions'] != 972 or not audit['repeat_arrays_exact']:
            raise ValueError('Fresh audit did not verify the complete cohort')
    report = dict(archive_sha256=sha(archive_path), candidate_sha256=sha(release_path),
        all_payloads_verified=True, fresh_extraction=True, manifest_entries=len(manifest),
        interpreter=args.python, audit=audit, verifier_sha256=sha(Path(__file__)))
    destination = ROOT / 'reports/embodiment/closed-loop/fresh-extraction.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
