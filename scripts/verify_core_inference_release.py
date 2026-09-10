"""Replay every computation condition from a fresh archive outside the checkout."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'reports/language-core/inference-release.json'
PUBLIC_RELEASE = 'public/research/language-core-inference-release.json'
ARCHIVE = 'public/research/language-core-inference.zip'
CONDITIONS = {f'{control}-s{seed}' for seed in (42, 43)
              for control in ('full', 'fixed_dynamics', 'no_lateral', 'no_temporal_state')}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read_archive(path, report):
    """Check exact ZIP/manifest inventory and paths before executing any source."""
    payload = path.read_bytes()
    if digest(payload) != report['archive_sha256'] or len(payload) != report['bytes']:
        raise ValueError('The computation inference archive changed')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != report['payloads']:
            raise ValueError('Duplicate or changed computation archive inventory')
        for item in archive.infolist():
            # ZipInfo normalizes backslashes on Windows and truncates at NUL.
            # Validate the stored name before trusting that normalized form.
            name = item.orig_filename; relative = PurePosixPath(name)
            if (name != item.filename or not name or '\\' in name or ':' in name or relative.is_absolute() or
                    '..' in relative.parts or relative.as_posix() != name or item.is_dir() or
                    stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Invalid computation inference archive path')
        if archive.testzip() is not None:
            raise ValueError('Computation archive CRC failed')
        contents = {name: archive.read(name) for name in names}
    if 'bundle.json' not in contents or digest(contents['bundle.json']) != report['manifest_sha256']:
        raise ValueError('The computation inference manifest changed')
    manifest = json.loads(contents['bundle.json'])
    if set(contents) != set(manifest['files']) | {'bundle.json'}:
        raise ValueError('Computation archive and manifest inventories differ')
    for name, record in manifest['files'].items():
        if len(contents[name]) != record['bytes'] or digest(contents[name]) != record['sha256']:
            raise ValueError('Computation archive payload changed: ' + name)
    if manifest['format'] != 'flm-language-core-inference-v1' or manifest['models'] != report['models']:
        raise ValueError('Computation manifest/model binding changed')
    for key in ('selection_sha256', 'study_identity_sha256'):
        if manifest[key] != report[key]:
            raise ValueError('Computation study binding changed')
    samples = json.loads(contents['samples.json'])
    if samples['selection_sha256'] != report['selection_sha256']:
        raise ValueError('Computation samples changed selection')
    for rows, field in ((report['models'], 'id'), (samples['models'], 'condition')):
        if len(rows) != 8 or {row[field] for row in rows} != CONDITIONS:
            raise ValueError('All eight unique computation conditions are required')
    for record in report['models']:
        row = next(item for item in samples['models'] if item['condition'] == record['id'])
        if row['checkpoint_sha256'] != record['source_checkpoint_sha256'] or len(row['passages']) != 4:
            raise ValueError('Computation samples changed checkpoint or prompt count')
    return contents, samples


def verify(root):
    root = Path(root).resolve()
    report_path = root / RELEASE
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    if report['archive'] != ARCHIVE:
        raise ValueError('Unexpected computation archive path')
    archive_path = root / ARCHIVE
    contents, samples = read_archive(archive_path, report)
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    runner = ("import pathlib,runpy,flm; assert pathlib.Path(flm.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
              "runpy.run_module('flm.core_inference',run_name='__main__')")
    cases = []
    with tempfile.TemporaryDirectory(prefix='flm-core-standalone-') as temporary:
        folder = Path(temporary).resolve()
        if folder.is_relative_to(root):
            raise ValueError('Fresh computation archive verification must run outside the repository')
        for name, data in contents.items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder):
                raise ValueError('Computation archive path leaves its extraction directory')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        for record in report['models']:
            example = next(row for row in samples['models'] if row['condition'] == record['id'])['passages'][0]
            settings = samples['settings']
            result = subprocess.run([sys.executable, '-E', '-X', 'utf8', '-c', runner,
                '--condition', record['id'], '--prompt', example['prompt'],
                '--sampling-seed', str(settings['seed']), '--temperature', str(settings['temperature']),
                '--top-k', str(settings['top_k']), '--tokens', str(settings['maximum_tokens'])],
                cwd=folder, env=environment, capture_output=True, text=True, encoding='utf8',
                timeout=180, check=True)
            actual = json.loads(result.stdout)
            if actual['model'] != record or any(actual.get(key) != value for key, value in example.items()):
                raise ValueError('Standalone computation continuation differs: ' + record['id'])
            if actual['sampling']['threads'] != 1 or any(actual['sampling'][key] != settings[key]
                    for key in ('seed', 'temperature', 'top_k', 'maximum_tokens')):
                raise ValueError('Standalone computation sampling settings changed')
            cases.append(dict(condition=record['id'], source_prompt_reproduced=True,
                              output_tokens=len(actual['tokens']), cli_exit_code=result.returncode))
            print('Standalone computation condition verified: ' + record['id'], flush=True)
    if report_path.read_bytes() != original_report or digest(archive_path.read_bytes()) != report['archive_sha256']:
        raise ValueError('Computation release changed during standalone verification')
    report['fresh_archive_cli_verified'] = True
    report['standalone_cli_verification'] = dict(fresh_archive_extraction=True,
        extraction_outside_repository=True, repository_pythonpath_removed=True,
        extracted_runtime_import_path_verified=True, cases=cases,
        verified_utc=datetime.now(timezone.utc).isoformat(), verifier_sha256=digest(Path(__file__).read_bytes()),
        scope='One fixed-prompt continuation per condition; no independent held-out likelihood evaluation')
    payload = (json.dumps(report, indent=2) + '\n').encode('utf8')
    report_path.write_bytes(payload)
    (root / PUBLIC_RELEASE).write_bytes(payload)
    return report


if __name__ == '__main__':
    result = verify(ROOT)
    print(json.dumps(dict(verified_conditions=8, archive_sha256=result['archive_sha256']), indent=2))
