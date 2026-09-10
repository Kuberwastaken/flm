"""Exercise every released topology condition from a fresh archive extraction."""
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


def verify(root):
    report_path = root/'reports/language-topology/inference-release.json'
    report = json.loads(report_path.read_text(encoding='utf8'))
    archive_path = root/report['archive']
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != report['archive_sha256']:
        raise ValueError('The topology inference archive changed')
    destination_parent = root/'output/inference'
    destination_parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='standalone-topology-', dir=destination_parent)).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (folder/name).resolve().is_relative_to(folder) or '\\' in name or ':' in name:
                raise ValueError('Invalid topology inference archive path')
        archive.extractall(folder)
    samples = json.loads((folder/'samples.json').read_text(encoding='utf8'))
    cases = []
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    # Verify that Python actually imports the extracted package, not an installed
    # editable checkout which happens to contain matching source bytes.
    runner = ("import pathlib,runpy,flm; assert pathlib.Path(flm.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
              "runpy.run_module('flm.topology_inference',run_name='__main__')")
    for record in report['models']:
        example = next(row for row in samples['models'] if row['condition'] == record['id'])['passages'][0]
        result = subprocess.run([sys.executable, '-E', '-X', 'utf8', '-c', runner,
            '--condition', record['id'], '--prompt', example['prompt']], cwd=folder, env=environment,
            capture_output=True, text=True, encoding='utf8', timeout=180, check=True)
        actual = json.loads(result.stdout)
        if actual['model'] != record or any(actual.get(key) != value for key, value in example.items()):
            raise ValueError('Standalone topology continuation differs: ' + record['id'])
        for key in ('seed', 'temperature', 'top_k', 'maximum_tokens'):
            if actual['sampling'][key] != samples['settings'][key]:
                raise ValueError('Standalone sampling settings changed')
        cases.append(dict(condition=record['id'], source_prompt_reproduced=True,
                          output_tokens=len(actual['tokens']), cli_exit_code=result.returncode))
        print('Standalone topology condition verified: ' + record['id'], flush=True)
    expected = {f'{condition}-s{seed}' for condition in ('measured', 'null101', 'null103', 'null107', 'no-slow')
                for seed in (42, 43)}
    if len(cases) != 10 or {case['condition'] for case in cases} != expected:
        raise ValueError('All ten standalone conditions are required')
    report['fresh_archive_cli_verified'] = True
    report['standalone_cli_verification'] = dict(fresh_archive_extraction=True,
        repository_pythonpath_removed=True, extracted_runtime_import_path_verified=True, cases=cases,
        verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload = (json.dumps(report, indent=2)+'\n').encode('utf8')
    report_path.write_bytes(payload)
    (root/'public/research/language-topology-inference-release.json').write_bytes(payload)
    return report


if __name__ == '__main__':
    result = verify(ROOT)
    print(json.dumps(dict(verified_conditions=10, archive_sha256=result['archive_sha256']), indent=2))
