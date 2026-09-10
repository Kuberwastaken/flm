"""Acquire immutable BLiMP paradigms and freeze a hash-selected diagnostic panel."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
from .provenance import sha256, write_json

REVISION = '3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
RAW = Path('data/raw/blimp')


def pair_key(record):
    return hashlib.sha256(f"{record['UID']}/{record['pairID']}".encode()).hexdigest()


def main():
    manifest_path = Path('data/sources/blimp.json')
    if manifest_path.exists(): manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    else:
        response = requests.get(f'https://api.github.com/repos/alexwarstadt/blimp/git/trees/{REVISION}?recursive=1', timeout=60)
        response.raise_for_status(); tree = response.json()
        if tree.get('truncated'): raise ValueError('Incomplete source tree')
        files = [dict(path=r['path'], bytes=r['size'], git_blob_sha1=r['sha']) for r in tree['tree']
                 if r['type'] == 'blob' and r['path'].startswith('data/') and r['path'].endswith('.jsonl')]
        if len(files) != 67: raise ValueError('Unexpected BLiMP inventory')
        manifest = dict(dataset='BLiMP', revision=REVISION, files=sorted(files, key=lambda r:r['path']),
            source='https://github.com/alexwarstadt/blimp', license='CC BY 4.0',
            authors='Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang and Samuel R. Bowman')
        write_json(manifest_path, manifest)
    RAW.mkdir(parents=True, exist_ok=True)
    def acquire(record):
        path = RAW / Path(record['path']).name
        payload = path.read_bytes() if path.exists() else requests.get(
            f"https://raw.githubusercontent.com/alexwarstadt/blimp/{REVISION}/{record['path']}", timeout=60).content
        digest = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        if len(payload) != record['bytes'] or digest != record['git_blob_sha1']: raise ValueError('BLiMP source mismatch')
        if not path.exists(): path.write_bytes(payload)
        rows = [json.loads(line) for line in payload.decode('utf8').splitlines()]
        if len(rows) != 1000 or len({r['pairID'] for r in rows}) != 1000: raise ValueError('Invalid paradigm pair IDs')
        if {r['UID'] for r in rows} != {path.stem}: raise ValueError('Unexpected paradigm identity')
        for row in rows:
            if not row['sentence_good'] or not row['sentence_bad']: raise ValueError('Empty sentence')
        selected = sorted(rows, key=pair_key)[:100]
        return dict(path=str(path), sha256=sha256(path), paradigm=path.stem, pairs=len(rows),
            selected_pair_ids=[r['pairID'] for r in selected], category=rows[0]['linguistics_term'])
    with ThreadPoolExecutor(max_workers=3) as pool: files = list(pool.map(acquire, manifest['files']))
    card = dict(dataset='BLiMP', revision=REVISION, manifest_sha256=sha256(manifest_path), files=files,
        total_pairs=67000, selected_pairs=6700, selection='First 100 pairs per paradigm sorted by SHA-256 of UID/pairID.',
        protocol_sha256=sha256(Path('docs/SYNTAX-PROTOCOL.md')), model_scores_used_for_selection=False,
        training_usage='None. Expert-grammar-generated evaluation data only.')
    destination = Path('data/cards/blimp-panel.json')
    if destination.exists() and json.loads(destination.read_text(encoding='utf8')) != card: raise ValueError('Existing panel changed')
    write_json(destination, card); print('Verified 67,000 BLiMP pairs; froze 6,700 diagnostic pairs without model scores.')


if __name__ == '__main__': main()
