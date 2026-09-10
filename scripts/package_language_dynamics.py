"""Publish a core-only dynamics archive after a fresh, outside-repo replay."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flm.language_dynamics_study import FOLDER, INPUTS, SOURCES, conditions, verified_results
from flm.provenance import sha256, write_json

README = '''# FLM recurrent dynamics replay

This archive reproduces the seven synthetic-pulse conditions in the findings.
It contains the four recurrent parameter groups, the graph subset, the original
numerical sources, complete results and attribution. It does not contain lexical
language weights, training checkpoints, optimizer state or a corpus.

Use a fresh Python environment (the recorded run used Python 3.10.11):

    python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install numpy==2.2.6 scipy==1.13.1 tokenizers==0.22.2
    python -E scripts/replay_language_dynamics.py --output fresh-replay.json

The command reconstitutes each recurrent core, validates the acute swaps, and
recomputes all metrics and saved arrays. Exact reproduction is checked against
the recorded one-thread CPU environment; other platforms may differ numerically.
No training repository or original checkpoints are accessed. A freshly initialized
lexical interface exists in memory but is bypassed by the direct-drive diagnostic.
For text generation use the separate topology inference release on the website.

Read docs/LANGUAGE-DYNAMICS-FINDINGS.md and the frozen protocol first. Curves are
model updates, not biological time or a measure of useful memory. These results
do not establish anatomical advantage, better language prediction or behavior
transfer. Historical full-checkpoint hashes document provenance; the archive
does not independently establish the training history of those checkpoints.

manifest.json checks file consistency, not publisher authenticity. The script
language_dynamics_study.py is the historical repository harness; use the replay
wrapper above because the historical harness expects the full local run files.

Source code: MIT, Kuber Mehta. Graph data: MaleCNS, CC BY 4.0; see licenses/.
'''


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode('utf8')


def file_record(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def package(root):
    report = verified_results(root)
    names = set(SOURCES) | set(INPUTS) | {
        'scripts/replay_language_dynamics.py', 'scripts/package_language_dynamics.py',
        'scripts/language_dynamics_report.py', 'docs/LANGUAGE-DYNAMICS-FINDINGS.md',
        'docs/LANGUAGE-CORE-PROTOCOL.md', 'LICENSE',
        'public/research/figures/language-dynamics-pulses.png',
        'public/research/figures/language-dynamics-pulses.svg',
        'public/research/language-dynamics-curves.csv',
        'public/research/language-dynamics-summary.csv'}
    names |= {f'{FOLDER.as_posix()}/{name}.json' for name in ('identity', 'summary', 'replay', 'figure')}
    names.add(f'{FOLDER.as_posix()}/pulses.npz')
    names |= {f'{FOLDER.as_posix()}/case-{c["label"]}.{s}' for c in conditions() for s in ('json', 'npz')}
    names |= {(Path('licenses') / name).as_posix() for name in ('CC-BY-4.0.txt', 'DATA-ATTRIBUTION.md')}
    contents = {name: (root / name).read_bytes() for name in sorted(names)}
    contents['README.md'] = README.encode('utf8')
    manifest = dict(format='flm-language-dynamics-v1',
                    study_identity_sha256=report['study_identity_sha256'],
                    files={name: file_record(data) for name, data in contents.items()})
    contents['manifest.json'] = encoded(manifest)
    # Stage outside the repository, and replay from only the bytes in the ZIP.
    with tempfile.TemporaryDirectory(prefix='flm-dynamics-') as temporary:
        stage = Path(temporary)
        archive = stage / 'language-dynamics-replay.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for name, data in sorted(contents.items()):
                item = zipfile.ZipInfo(name, (2026, 9, 10, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                output.writestr(item, data)
        extracted = stage / 'extracted'
        with zipfile.ZipFile(archive) as source:
            source.extractall(extracted)
        environment = {key: value for key, value in os.environ.items() if key.upper() != 'PYTHONPATH'}
        result_path = stage / 'replayed.json'
        command = [sys.executable, '-E', '-u', '-X', 'utf8',
                   str(extracted / 'scripts/replay_language_dynamics.py'), '--output', str(result_path)]
        completed = subprocess.run(command, cwd=extracted, env=environment, capture_output=True, text=True)
        print(completed.stdout, end='')
        if completed.returncode:
            raise RuntimeError('Fresh dynamics archive replay failed: ' + completed.stderr)
        replay = json.loads(result_path.read_text(encoding='utf8'))
        expected_source = (extracted / 'flm/language_dynamics_study.py').resolve()
        if (Path(replay['imported_measure_source']) != expected_source or replay['conditions'] != 7 or
                replay['exact_metrics_and_arrays'] is not True or
                replay['study_identity_sha256'] != report['study_identity_sha256']):
            raise ValueError('Fresh replay did not use the extracted sources and complete study')
        payload = archive.read_bytes()
        replay.pop('imported_measure_source')
    record = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                  study_identity_sha256=report['study_identity_sha256'],
                  archive='language-dynamics-replay.zip', **file_record(payload),
                  included_files=len(contents), fresh_outside_repository_replay=replay,
                  imported_sources_verified_inside_extracted_archive=True,
                  source_sha256={name: sha256(root / name) for name in
                      ('scripts/replay_language_dynamics.py', 'scripts/package_language_dynamics.py')})
    destination = root / 'public/research'
    (destination / record['archive']).write_bytes(payload)
    write_json(destination / 'language-dynamics-release.json', record)
    write_json(root / FOLDER / 'standalone-replay.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    package(ROOT)
