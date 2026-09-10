"""Pin, acquire and byte-audit the official BabyLM 2026 English corpora."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests
from .provenance import sha256, write_json

REPOSITORIES = {
    'train-10m': ('BabyLM-2026-Strict-Small', 'c92ab16b4f08858304b0815706065b3354d8fc0a', 'train.txt'),
    'train-100m': ('BabyLM-2026-Strict', '9e57baaaa91ac3c638746be14d1d5fa6c789f4cf', 'train.txt'),
    'validation': ('BabyLM-dev', '169f42e32d0aaf65ec6b91d55bafad27a3afc729', 'dev'),
    'test': ('BabyLM-Test', '2c47b98e2dc3707465aed81da69dc36cdca5d13b', 'test'),
}
COMPONENTS = ('bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard')
MANIFEST = Path('data/sources/babylm-2026.json')


def pin():
    records = []
    for partition, (repo, revision, suffix) in REPOSITORIES.items():
        endpoint = f'https://huggingface.co/api/datasets/BabyLM-community/{repo}/tree/{revision}'
        response = requests.get(endpoint, timeout=60); response.raise_for_status()
        files = {item['path']: item for item in response.json() if item['type'] == 'file'}
        for component in COMPONENTS:
            name = f'{component}.{suffix}'; entry = files[name]
            records.append(dict(partition=partition, component=component, repository='BabyLM-community/' + repo,
                revision=revision, filename=name, bytes=entry['size'], git_blob_sha1=entry['oid'],
                lfs_sha256=entry.get('lfs', {}).get('oid'),
                url=f'https://huggingface.co/datasets/BabyLM-community/{repo}/resolve/{revision}/{name}'))
    result = dict(dataset='BabyLM 2026 English', files=records,
        publisher='BabyLM-community', source='https://babylm.github.io/',
        licensing='Training repository metadata says MIT. Component provenance and rights remain distinct; no raw corpus text is redistributed by FLM.',
        interpretation='10M and 100M are publisher budget labels. Count actual words, lines and bytes locally; no model-specific token count is implied.')
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding='utf8')) != result:
        raise ValueError('Pinned manifest differs; do not overwrite an existing acquisition identity')
    write_json(MANIFEST, result); print(f'Pinned {len(records)} files; {sum(x["bytes"] for x in records):,} source bytes.')


def verify(path, record):
    if not path.exists() or path.stat().st_size != record['bytes']: return False
    if record['lfs_sha256']: return sha256(path) == record['lfs_sha256']
    digest = hashlib.sha1(f'blob {record["bytes"]}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest() == record['git_blob_sha1']


def acquire(record, root):
    path = root / record['partition'] / record['filename']; path.parent.mkdir(parents=True, exist_ok=True)
    if not verify(path, record):
        temporary = path.with_suffix(path.suffix + '.part')
        for attempt in range(3):
            try:
                with requests.get(record['url'], stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status(); written = 0
                    with temporary.open('wb') as stream:
                        for block in response.iter_content(1024 * 1024):
                            written += len(block)
                            if written > record['bytes']: raise ValueError('Download exceeds the pinned source size')
                            stream.write(block)
                if not verify(temporary, record): raise ValueError('Downloaded file does not match its pinned Git/LFS object')
                temporary.replace(path); break
            except (requests.RequestException, OSError):
                if attempt == 2: raise
                time.sleep(2 ** attempt)
    return path


def audit(path, record):
    lines = blank = words = characters = crlf = 0; longest = 0
    with path.open('rb') as stream:
        for line in stream:
            text = line.decode('utf8'); lines += 1; blank += not text.strip()
            words += len(text.split()); characters += len(text); crlf += line.endswith(b'\r\n')
            longest = max(longest, len(line))
    return dict(partition=record['partition'], component=record['component'], path=str(path),
        utf8_bytes=path.stat().st_size, sha256=sha256(path), source_verified=True,
        lines=lines, blank_lines=blank, whitespace_words=words, characters=characters,
        crlf_lines=crlf, longest_line_bytes=longest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('pin', 'acquire'))
    p.add_argument('--raw', type=Path, default=Path('data/raw/babylm-2026'))
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if a.operation == 'pin': pin(); return
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    def process(record):
        result = audit(acquire(record, a.raw), record)
        print(json.dumps({k: result[k] for k in ('partition','component','utf8_bytes','whitespace_words','lines')}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=a.workers) as pool: results = list(pool.map(process, manifest['files']))
    write_json(Path('data/cards/babylm-2026-acquisition.json'), dict(dataset=manifest['dataset'],
        manifest_sha256=sha256(MANIFEST), files=results, model_training_started=False,
        status='Raw acquisition and size/text audit only; document boundaries, leakage, tokenizer and experiment protocol are separate steps.'))


if __name__ == '__main__': main()
