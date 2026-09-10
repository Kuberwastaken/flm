"""Exercise the released CLI from a fresh extraction without repository imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report_path = ROOT / 'reports/wikitext2/inference-release.json'
    report = json.loads(report_path.read_text(encoding='utf8'))
    archive_path = ROOT / report['archive']
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != report['archive_sha256']:
        raise ValueError('The exported archive changed')
    directory = Path(tempfile.mkdtemp(prefix='standalone-', dir=ROOT / 'output/inference'))
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or '\\' in name or ':' in name:
                raise ValueError('Invalid archive entry')
        archive.extractall(directory)
    environment = os.environ.copy(); environment.pop('PYTHONPATH', None)
    checks = []
    for record in report['models']:
        command = [sys.executable, '-X', 'utf8', '-m', 'flm.inference', '--model', record['variant'],
                   '--training-seed', str(record['training_seed']), '--prompt', 'The history of science']
        result = subprocess.run(command, cwd=directory, env=environment, check=True,
                                capture_output=True, text=True, encoding='utf8', timeout=180)
        output = json.loads(result.stdout)
        published = json.loads((ROOT / f'public/research/samples-006000-s{record["training_seed"]}.json').read_text())
        expected = next(row for row in published['models'] if row['variant'] == record['variant'])['passages'][0]
        if output['model'] != record or any(output[key] != value for key, value in expected.items()):
            raise ValueError(f'The standalone CLI changed {record["id"]}')
        checks.append(dict(model=record['id'], published_prompt_reproduced=True,
                           cli_exit_code=result.returncode, output_tokens=len(output['tokens'])))
    report['standalone_cli_verification'] = dict(
        fresh_archive_extraction=True, repository_pythonpath_removed=True,
        runtime='Matching extracted sources verified by each CLI invocation',
        python=sys.version, cases=checks)
    report['standalone_verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = json.dumps(report, indent=2) + '\n'
    report_path.write_text(content, encoding='utf8')
    (ROOT / 'public/research/inference-release.json').write_text(content, encoding='utf8')
    print(json.dumps(dict(standalone_models_verified=len(checks), archive_sha256=report['archive_sha256']), indent=2))


if __name__ == '__main__':
    main()
