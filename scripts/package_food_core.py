"""Package source-bound, zero-adaptation food-core interfaces and parity fixtures."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    folder = ROOT / 'work/food-core-interface-v1'
    preparation = (ROOT / 'reports/food-core/preparation.json').read_bytes()
    manifest = json.loads(preparation)
    parity_bytes = (ROOT / 'reports/food-core/isolated-parity.json').read_bytes()
    parity = json.loads(parity_bytes)
    if (folder / 'manifest.json').read_bytes() != preparation:
        raise ValueError('Prepared manifest differs from preserved record')
    if parity['manifest_sha256'] != sha(preparation):
        raise ValueError('Isolated parity proof belongs to a different interface')
    if ([r['id'] for r in parity['models']] != [r['id'] for r in manifest['models']]
            or any(r['observations'] != 466 or not r['greedy_action_exact'] for r in parity['models'])
            or not parity['same_adapters_and_body_order'] or parity['torch_imported']):
        raise ValueError('Incomplete isolated parity proof')
    contents = {'manifest.json': preparation, 'isolated-parity.json': parity_bytes}
    for record in [*manifest['models'], manifest['fixture']]:
        name = record['file']
        if Path(name).name != name or '\\' in name or ':' in name:
            raise ValueError('Package member must be a local filename')
        data = (folder / name).read_bytes()
        if sha(data) != record['sha256'] or len(data) != record['bytes']:
            raise ValueError('Prepared payload changed: ' + name)
        contents[name] = data
    for name, expected in manifest['sources'].items():
        data = (ROOT / name).read_bytes()
        if sha(data) != expected:
            raise ValueError('Bound source changed: ' + name)
        contents['sources/' + name] = data
    for name in ('food_core_runtime.py', 'verify_food_core.py'):
        data = (folder / name).read_bytes()
        if sha(data) != manifest['sources']['experiments/embodiment/' + name]:
            raise ValueError('Portable source differs: ' + name)
        proof_key = 'runtime_sha256' if name == 'food_core_runtime.py' else 'verifier_sha256'
        if sha(data) != parity[proof_key]:
            raise ValueError('Isolated source differs: ' + name)
        contents[name] = data
    for name in ('LICENSE', 'CC-BY-4.0.txt', 'DATA-ATTRIBUTION.md', 'BODY-PROVENANCE.md',
                 'Body-Apache-2.0.txt', 'Body-MIT.txt', 'graph-card.json'):
        contents[name] = (folder / name).read_bytes()
    contents['README.md'] = (
        '# FLM frozen sensory interfaces\n\n'
        'Four paired initial/language-trained cores, zero food-adaptation updates.\n'
        'Open-loop numerical replay only; no food skill or transfer result.\n\n'
        'With NumPy installed, run from this extracted directory:\n\n'
        '```console\npython verify_food_core.py . --output fresh-parity.json\n```\n\n'
        'Use a new output filename. Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and\n'
        'MKL_NUM_THREADS to 1 before launch. Repeated compressed-reference loading\n'
        'can take several minutes. No training corpus, PyTorch or FlyGym is needed.\n\n'
        'manifest.json binds model exports and fixtures; archive-manifest.json\n'
        'hashes every other member. sources/ preserves the native numerical code;\n'
        'it is not a full runnable training repository. See the project appendix:\n'
        'https://flm.kuber.studio/research/food-core-interface.md\n\n'
        'Code is MIT licensed. Graph and body provenance and component licenses\n'
        'are included; preserve their attribution when reusing the material.\n'
    ).encode('utf8')
    inventory = {name: dict(bytes=len(data), sha256=sha(data)) for name, data in sorted(contents.items())}
    contents['archive-manifest.json'] = (json.dumps(inventory, indent=2) + '\n').encode('utf8')
    destination = ROOT / 'public/research/food-core-interface.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(contents):
            raise ValueError('Archive integrity or inventory failure')
        for name, data in contents.items():
            if archive.read(name) != data:
                raise ValueError('Archive member differs: ' + name)
    release = dict(archive=destination.relative_to(ROOT).as_posix(), bytes=destination.stat().st_size,
        sha256=sha(destination.read_bytes()), files=inventory, file_count=len(inventory),
        preparation_sha256=sha(preparation), isolated_parity_sha256=sha(parity_bytes),
        packaging_source_sha256=sha(Path(__file__).read_bytes()),
        scope='Byte-verified packaging of four interfaces and their completed isolated numerical replay; no additional model training or physics.')
    (ROOT / 'reports/food-core/release.json').write_text(
        json.dumps(release, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({key: release[key] for key in ('archive', 'bytes', 'sha256', 'file_count')}))


if __name__ == '__main__':
    main()
