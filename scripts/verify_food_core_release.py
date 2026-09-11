"""Check the food-core archive, source snapshot, documentation links and HTTP bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(base_url):
    release = json.loads((ROOT / 'reports/food-core/release.json').read_bytes())
    payload = (ROOT / release['archive']).read_bytes()
    if len(payload) != release['bytes'] or sha(payload) != release['sha256']:
        raise ValueError('Interface archive differs from its release record')
    for name, key in [('preparation.json', 'preparation_sha256'), ('isolated-parity.json', 'isolated_parity_sha256')]:
        if sha((ROOT / 'reports/food-core' / name).read_bytes()) != release[key]:
            raise ValueError('Preserved preparation or parity record changed')
    if sha((ROOT / 'scripts/package_food_core.py').read_bytes()) != release['packaging_source_sha256']:
        raise ValueError('Packaging source changed')
    with zipfile.ZipFile(ROOT / release['archive']) as archive:
        inventory = json.loads(archive.read('archive-manifest.json'))
        if (inventory != release['files'] or len(inventory) != release['file_count']
                or archive.testzip() is not None
                or set(archive.namelist()) != {*inventory, 'archive-manifest.json'}):
            raise ValueError('Archive inventory or CRC differs')
        for name, record in inventory.items():
            data = archive.read(name)
            if sha(data) != record['sha256'] or len(data) != record['bytes']:
                raise ValueError('Archive member changed: ' + name)
            if name.startswith('sources/') and data != (ROOT / name.removeprefix('sources/')).read_bytes():
                raise ValueError('Published numerical source differs: ' + name)
        for member, source in [('manifest.json', 'preparation.json'), ('isolated-parity.json', 'isolated-parity.json')]:
            if archive.read(member) != (ROOT / 'reports/food-core' / source).read_bytes():
                raise ValueError('Archived proof differs: ' + member)
    links = []
    for name in ('food-core-interface.md', 'food-response-plan.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (ROOT / 'public/research' / name).read_text(encoding='utf8')):
            if urlparse(target).scheme:
                continue
            path = (ROOT / 'public/research' / target).resolve()
            if not path.is_relative_to((ROOT / 'public').resolve()) or not path.is_file():
                raise ValueError('Missing published note target: ' + target)
            links.append(target)
    with zipfile.ZipFile(ROOT / 'public/research/paper-source.zip') as archive:
        source_manifest = json.loads(archive.read('source-manifest.json'))
        if archive.testzip() is not None:
            raise ValueError('Paper source CRC failed')
        for name, record in source_manifest.items():
            if name.startswith(('data/raw/', 'data/processed/', 'runs/')):
                raise ValueError('Unexpected corpus or run payload in paper sources')
            data = archive.read(name)
            if (len(data) != record['bytes'] or sha(data) != record['sha256']
                    or data != (ROOT / name).read_bytes()):
                raise ValueError('Paper source differs: ' + name)
        if not {'README.md', 'docs/FOOD-CORE-INTERFACE.md', 'reports/food-core/release.json'}.issubset(source_manifest):
            raise ValueError('Updated overview or interface records missing from sources')
    files = ['index.html', 'research/food-core-interface.md', 'research/food-response-plan.md',
             'research/food-core-interface.zip', 'research/paper-source.zip']
    files += [path.relative_to(ROOT / 'public').as_posix() for path in sorted((ROOT / 'public/research/food-core').iterdir()) if path.is_file()]
    artifacts = []
    for name in files:
        with urlopen(base_url.rstrip('/') + '/' + name, timeout=45) as response:
            data = response.read()
            status = response.status
        if status != 200 or data != (ROOT / 'dist' / name).read_bytes():
            raise ValueError('HTTP response differs from build: ' + name)
        artifacts.append(dict(path=name, status=status, bytes=len(data), sha256=sha(data)))
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), base_url=base_url,
        artifacts=artifacts, http_bytes_match_build=True, archive_files_verified=len(inventory),
        paper_source_files_verified=len(source_manifest), local_markdown_links_verified=links,
        release_sha256=sha((ROOT / 'reports/food-core/release.json').read_bytes()),
        verification_source_sha256=sha(Path(__file__).read_bytes()),
        scope='Publication and exact archive/source/proof correspondence. This check performs no model inference, training or physics; numerical replay is preserved in the separately bound isolated proof.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new verification output path')
    result = verify(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('base_url', 'archive_files_verified', 'paper_source_files_verified')}))
