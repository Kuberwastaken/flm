"""Acquire revision-checked, text-only LibriSpeech and build document splits."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import requests

from .provenance import SPEECH_DATASET, SPEECH_REVISION, sha256, write_json

SPLITS = {"train": "train.100", "validation": "validation", "test": "test"}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fetch_batch(cache: Path, split: str, offset: int) -> dict:
    path = cache / f"{split}-{offset:06d}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf8"))
        if saved.get("revision") != SPEECH_REVISION:
            raise ValueError(f"Cached revision mismatch: {path}")
        canonical = json.dumps(saved["rows"], sort_keys=True).encode()
        if hashlib.sha256(canonical).hexdigest() != saved["rows_sha256"]:
            raise ValueError(f"Cached row checksum mismatch: {path}")
        return saved
    parameters = dict(dataset=SPEECH_DATASET, config="clean", split=split, offset=offset, length=100)
    for attempt in range(6):
        try:
            response = requests.get("https://datasets-server.huggingface.co/rows", params=parameters, timeout=60)
            response.raise_for_status()
            revision = response.headers.get("x-revision")
            if revision != SPEECH_REVISION:
                raise ValueError(f"Dataset changed: expected {SPEECH_REVISION}, got {revision}; pin/audit a new release explicitly")
            payload = response.json()
            rows = []
            for item in payload["rows"]:
                if "text" in item.get("truncated_cells", []):
                    raise ValueError("Server truncated a transcript")
                row = item["row"]
                rows.append({k: row[k] for k in ("id", "speaker_id", "chapter_id", "text")})
            saved = dict(revision=revision, split=split, offset=offset,
                         total=payload["num_rows_total"], rows=rows,
                         response_sha256=hashlib.sha256(response.content).hexdigest(),
                         rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                         url=response.url)
            write_json(path, saved)
            return saved
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("Unreachable download state")


def acquire(raw: Path, workers: int = 4) -> dict[str, list[dict]]:
    result = {}
    for name, split in SPLITS.items():
        first = fetch_batch(raw, split, 0)
        batches = {0: first}
        offsets = list(range(100, first["total"], 100))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(fetch_batch, raw, split, offset): offset for offset in offsets}
            for index, future in enumerate(as_completed(jobs), 1):
                batches[jobs[future]] = future.result()
                if index % 25 == 0:
                    print(f"{name}: {index + 1}/{len(offsets) + 1} transcript batches", flush=True)
        rows = [row for offset in sorted(batches) for row in batches[offset]["rows"]]
        if len(rows) != first["total"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Incomplete or repeated rows in {split}")
        result[name] = rows
        print(f"{name}: {len(rows)} verified transcripts", flush=True)
    return result


def prepare(rows: dict[str, list[dict]], output: Path, card_path: Path) -> dict:
    speakers = {k: {r["speaker_id"] for r in v} for k, v in rows.items()}
    chapters = {k: {r["chapter_id"] for r in v} for k, v in rows.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if speakers[a] & speakers[b] or chapters[a] & chapters[b]:
            raise ValueError(f"Speaker/chapter leakage between {a} and {b}")
    seen = set()
    stats = {}
    output.mkdir(parents=True, exist_ok=True)
    # Evaluation has priority when identical longer passages recur across splits.
    for split in ("test", "validation", "train"):
        grouped = defaultdict(list)
        discarded = 0
        for row in sorted(rows[split], key=lambda r: tuple(int(x) for x in r["id"].split("-"))):
            text = normalized(row["text"])
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            if len(text) >= 80 and fingerprint in seen:
                discarded += 1
                continue
            if len(text) >= 80:
                seen.add(fingerprint)
            grouped[(row["speaker_id"], row["chapter_id"])].append((row["id"], text))
        documents = []
        for (speaker, chapter), utterances in sorted(grouped.items()):
            documents.append(dict(id=f"librispeech-{speaker}-{chapter}", source="LibriSpeech",
                speaker_ids=[str(speaker)], group_id=str(chapter), utterance_ids=[x[0] for x in utterances],
                text="\n".join(x[1] for x in utterances)))
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents), encoding="utf8")
        stats[split] = dict(documents=len(documents), input_utterances=len(rows[split]),
            words=sum(len(d["text"].split()) for d in documents),
            utf8_bytes=sum(len(d["text"].encode()) for d in documents),
            speakers=len(speakers[split]), discarded_exact_duplicates=discarded, sha256=sha256(path))
    card = dict(name="LibriSpeech clean 100h transcripts", dataset=SPEECH_DATASET,
        revision=SPEECH_REVISION, license="CC-BY-4.0", source="https://www.openslr.org/12",
        retrieval="Hugging Face official dataset viewer, text/IDs only, x-revision checked on every batch",
        normalization="Lowercase; collapse internal whitespace; retain utterance boundaries as newlines",
        partition="Official clean train.100/validation/test; documents grouped by speaker/chapter",
        leakage_checks="Disjoint speaker IDs and chapter IDs; exact passages >=80 characters deduplicated with test then validation priority",
        limitations=["Audiobook speech, not spontaneous conversation", "Different chapter IDs do not guarantee different books", "Near-duplicate/source-book overlap has not been exhaustively excluded", "Orthographic transcription removes many punctuation/case cues", "No audio was used"],
        statistics=stats)
    write_json(card_path, card)
    return card


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/librispeech"))
    p.add_argument("--output", type=Path, default=Path("data/processed/librispeech"))
    p.add_argument("--card", type=Path, default=Path("data/cards/librispeech.json"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(prepare(acquire(a.raw, a.workers), a.output, a.card), indent=2), flush=True)


if __name__ == "__main__":
    main()
