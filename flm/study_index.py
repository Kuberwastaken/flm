"""Publish an index only of completed, present study artifacts."""
from pathlib import Path
from .provenance import sha256, write_json


def publish_index(folder=Path('public/research')):
    artifacts = {}
    for key, filename in [('test', 'test-results.json'), ('runtime', 'runtime.json')]:
        path = folder / filename
        artifacts[key] = dict(file=filename, sha256=sha256(path)) if path.exists() else None
    write_json(folder / 'study-index.json', artifacts)


if __name__ == '__main__': publish_index()
