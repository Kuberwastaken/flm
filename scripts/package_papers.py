"""Package reviewed paper sources and published figure data with file hashes."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted(path for path in (ROOT / 'papers').rglob('*') if path.is_file())
    files += sorted((ROOT / 'public/research/figures').glob('*.csv'))
    files += [ROOT / 'scripts' / name for name in (
        'research_figures.py', 'continuing_figures.py', 'behavior_figures.py',
        'physical_choice_report.py', 'choice_paper_data.py', 'build_papers.py',
        'package_papers.py', 'wiring_figures.py', 'wiring_paper_data.py')]
    files += [ROOT / name for name in ('README.md', 'pyproject.toml',
        'docs/LOCAL-LEARNING-PROTOCOL.md', 'reports/local-learning/summary.json',
        'public/research/learned-choice.json', 'docs/WIRING-RESULTS.md',
        'docs/WIRING-LEARNING-PROTOCOL.md', 'docs/LANGUAGE-TOPOLOGY-PROTOCOL.md',
        'reports/wiring-learning/summary.json', 'reports/wiring-learning/paper-inputs.json')]
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                for name, data in sorted(contents.items())}
    contents['source-manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination = ROOT / 'public/research/paper-source.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Paper source archive failed CRC validation')
        for name, record in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != record['sha256']:
                raise RuntimeError(f'Paper source archive changed {name}')
    print(f'Packaged and verified {len(manifest)} source files: {destination}')


if __name__ == '__main__':
    main()
