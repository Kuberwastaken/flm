"""Acquire and validate the pinned full CSR graph independently of a local workspace."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import shutil
import numpy as np
import requests
from .provenance import GRAPH_REVISION, sha256, write_json


def safe_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or len(relative.parts) != 1 or '\\' in name or ':' in name or name in ('.', '..'):
        raise ValueError('Invalid source manifest filename')
    return root / name


def acquire(manifest, cache):
    pinned = json.loads(manifest.read_text(encoding='utf8'))
    if pinned['revision'] != GRAPH_REVISION: raise ValueError('Source manifest revision mismatch')
    cache.mkdir(parents=True, exist_ok=True)
    for record in pinned['files']:
        path = safe_file(cache, record['path'])
        if path.exists() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']: continue
        expected_url = f"https://huggingface.co/spaces/Xenova/fruit-fly-simulation/resolve/{GRAPH_REVISION}/public/data/{record['path']}"
        if record['url'] != expected_url: raise ValueError('Unexpected source URL')
        temporary = path.with_suffix(path.suffix + '.part'); size = 0
        with requests.get(expected_url, timeout=90, stream=True) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > record['bytes']: raise ValueError('Source exceeded pinned byte length')
                    handle.write(chunk)
        if size != record['bytes'] or sha256(temporary) != record['sha256']: raise ValueError('Source checksum mismatch')
        temporary.replace(path); print(f"Verified {record['path']}", flush=True)
    return pinned


def extract(cache, output, pinned):
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf8'))
    metadata = safe_file(cache, manifest['metadata'])
    rows = json.loads(gzip.decompress(metadata.read_bytes()))
    n = manifest['neurons']
    if len(rows) != n: raise ValueError('Metadata length mismatch')
    arrays = {}
    for specification in manifest['arrays']:
        chunks = []
        for part in specification['parts']:
            path = safe_file(cache, part['file'])
            if sha256(path) != part['sha256']: raise ValueError('CSR chunk checksum mismatch')
            raw = gzip.decompress(path.read_bytes())
            if len(raw) % 4: raise ValueError('Misaligned CSR chunk')
            chunks.append(np.frombuffer(raw, dtype='<u4'))
        array = np.concatenate(chunks)
        if len(array) != specification['length']: raise ValueError('CSR array length mismatch')
        arrays[specification['name']] = array
    offsets, sources, counts = [arrays[name] for name in ('offsets', 'sources', 'counts')]
    if len(offsets) != n + 1 or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]): raise ValueError('Invalid row offsets')
    if offsets[-1] != len(sources) or len(sources) != len(counts) or len(counts) != manifest['edges']: raise ValueError('Edge count mismatch')
    if sources.max() >= n or counts.min() == 0 or int(counts.sum(dtype=np.uint64)) != manifest['synapses']: raise ValueError('Invalid edges/contacts')
    ids = np.asarray([r[0] for r in rows], dtype=np.uint64)
    signs = np.asarray([r[5] for r in rows], dtype=np.int8)
    if len(np.unique(ids)) != n or not set(signs.tolist()) <= {-1, 0, 1}: raise ValueError('Invalid identities/signs')
    arrays.update(body_ids=ids, fast_sign=signs,
        soma_positions_8nm=np.asarray([r[6] if r[6] is not None else [np.nan] * 3 for r in rows], dtype=np.float32))
    output.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items(): np.save(output / f'{name}.npy', array, allow_pickle=False)
    shutil.copyfile(metadata, output / 'neurons.json.gz')
    write_json(output / 'acquisition.json', dict(revision=GRAPH_REVISION, neurons=n, directed_edges=len(counts),
        contacts=int(counts.sum(dtype=np.uint64)), orientation='W[postsynaptic, presynaptic]',
        signs='Runtime metadata sign field; zero-sign outputs remain zero, including modulatory/unclear labels',
        source_files=len(pinned['files']), source_manifest=pinned,
        outputs={name: sha256(output / f'{name}.npy') for name in arrays}))
    print(json.dumps(dict(neurons=n, directed_edges=len(counts), output=str(output)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/source-manifest.json'))
    parser.add_argument('--cache', type=Path, default=Path('data/raw/connectome'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/connectome'))
    args = parser.parse_args(); pinned = acquire(args.manifest, args.cache); extract(args.cache, args.output, pinned)


if __name__ == '__main__': main()
