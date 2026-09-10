"""Verify the successful Pages run and bytes served by the public custom domain."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://flm.kuber.studio/'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    workflow = json.loads(subprocess.check_output(['gh', 'run', 'view', args.run_id,
        '--json', 'status,conclusion,headSha,url,workflowName'], cwd=ROOT, text=True))
    if workflow['status'] != 'completed' or workflow['conclusion'] != 'success' or workflow['headSha'] != commit:
        raise ValueError('Pages workflow must have succeeded for the current release commit')
    paths = ['index.html', 'research/closed-loop.json', 'research/closed-loop.csv',
        'research/closed-loop-records.zip', 'research/closed-loop.mp4',
        'research/closed-loop-poster.png', 'research/closed-loop.pdf',
        'research/closed-loop-protocol.md', 'research/closed-loop-reproduction.md',
        'research/paper-source.zip', 'research/language-topology-progress.json']
    paths.extend('research/figures/closed-loop-' + name + suffix for name in ('positive', 'negative', 'switch', 'neural') for suffix in ('.svg', '.png'))
    artifacts = []
    for path in paths:
        local = ROOT / 'dist' / path
        digest = hashlib.sha256(); size = 0
        with urlopen(Request(BASE + path, headers={'User-Agent': 'FLM-release-verification', 'Cache-Control': 'no-cache'}), timeout=30) as response:
            status = response.status
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                digest.update(chunk); size += len(chunk)
        if digest.hexdigest() != file_sha(local) or size != local.stat().st_size:
            raise ValueError('The public artifact differs from the reviewed build: ' + path)
        artifacts.append(dict(path=path, status=status, bytes=size, sha256=digest.hexdigest()))
        print('Verified ' + path, flush=True)
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), release_commit=commit,
        site=BASE, workflow=workflow, artifacts=artifacts)
    (ROOT / 'reports/embodiment/closed-loop/deployment.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('All public feedback artifacts match the reviewed local build.', flush=True)


if __name__ == '__main__':
    main()
