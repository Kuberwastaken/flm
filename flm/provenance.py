"""Shared, explicit provenance and atomic artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

GRAPH_REVISION = "776d115ee5aa934578a87fd6d260d138084f59c1"
GRAPH_BASE = f"https://huggingface.co/spaces/Xenova/fruit-fly-simulation/resolve/{GRAPH_REVISION}/public/data/"
SPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
SPEECH_DATASET = "openslr/librispeech_asr"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    os.replace(temporary, path)
