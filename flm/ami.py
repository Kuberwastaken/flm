"""Prepare licensed human meeting transcripts with participant-disjoint splits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import requests

from .provenance import sha256, write_json

URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
CHECKSUM = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"


def acquire(path: Path) -> None:
    if not path.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    if sha256(path) != CHECKSUM:
        raise ValueError("AMI archive checksum mismatch")


def detokenize(words: list[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def prepare(archive: Path, output: Path, card_path: Path) -> dict:
    acquire(archive)
    documents = {}
    with zipfile.ZipFile(archive) as z:
        meetings = ET.fromstring(z.read("corpusResources/meetings.xml"))
        members = set(z.namelist())
        for meeting in meetings.findall("meeting"):
            meeting_id = meeting.attrib["observation"]
            turns, participants = [], []
            for speaker in meeting.findall("speaker"):
                agent = speaker.attrib["nxt_agent"]
                person = speaker.attrib.get("global_name")
                if not person:
                    raise ValueError(f"No global participant identity for {meeting_id}/{agent}")
                filename = f"words/{meeting_id}.{agent}.words.xml"
                if filename not in members:
                    continue
                participants.append(person)
                words = ET.fromstring(z.read(filename))
                pending, start, last_end = [], None, None
                for w in words:
                    if w.tag != "w" or not w.text or "starttime" not in w.attrib:
                        continue
                    time = float(w.attrib["starttime"])
                    end = float(w.attrib.get("endtime", time))
                    if pending and last_end is not None and time - last_end > 1.0:
                        turns.append((start, agent, detokenize(pending))); pending, start = [], None
                    if start is None:
                        start = time
                    pending.append(w.text)
                    last_end = end
                    if w.text.strip() in (".", "?", "!"):
                        turns.append((start, agent, detokenize(pending))); pending, start = [], None
                if pending:
                    turns.append((start, agent, detokenize(pending)))
            if not turns:
                continue
            turns.sort(key=lambda t: (t[0], t[1]))
            lines = [f"{agent.lower()}: {text}" for _, agent, text in turns if text]
            documents[meeting_id] = dict(id=f"ami-{meeting_id}", source="AMI", group_id=meeting_id,
                speaker_ids=sorted(set(participants)), text="\n".join(lines))
    # Connected components include every session sharing even one participant.
    parent = {key: key for key in documents}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    owners = {}
    for key, doc in sorted(documents.items()):
        family = re.sub(r"(?<=\d)[a-z]$", "", key)
        for identity in ["family:" + family] + ["person:" + p for p in doc["speaker_ids"]]:
            if identity in owners:
                a, b = find(key), find(owners[identity]); parent[max(a, b)] = min(a, b)
            owners[identity] = key
    groups = defaultdict(list)
    for key in documents:
        groups[find(key)].append(key)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(("flm-ami-2026:" + key).encode()).hexdigest())
    if len(ordered) < 10:
        raise ValueError("Too few independent participant groups for this split recipe")
    assignments, splits = {}, {"train": [], "validation": [], "test": []}
    for i, group in enumerate(ordered):
        split = "test" if i % 10 == 0 else "validation" if i % 10 == 1 else "train"
        for key in sorted(groups[group]):
            assignments[key] = dict(split=split, participant_component=group)
            splits[split].append(documents[key])
    speaker_sets = {split: {p for d in docs for p in d["speaker_ids"]} for split, docs in splits.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if speaker_sets[a] & speaker_sets[b]:
            raise ValueError("Participant leakage")
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, docs in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf8")
        stats[split] = dict(documents=len(docs), speakers=len(speaker_sets[split]),
            words=sum(len(d["text"].split()) for d in docs), utf8_bytes=sum(len(d["text"].encode()) for d in docs),
            sha256=sha256(path))
    card = dict(name="AMI manual meeting transcripts", version="1.6.2", license="CC-BY-4.0",
        source="https://groups.inf.ed.ac.uk/ami/download/", archive_url=URL, archive_sha256=CHECKSUM,
        independent_participant_components=len(groups), statistics=stats, assignments=assignments,
        transformations=["Read word and punctuation annotations; omit nonlexical events", "Split on sentence punctuation or >1 second pause", "Sort speaker segments by onset; retain generic a/b/c/d speaker markers", "Lowercase and normalize spacing; retain spoken hesitations", "Partition participant/family connected components by fixed hash order, 8/10 train, 1/10 validation, 1/10 test"],
        limitations=["A custom participant-disjoint text split, not the official ASR benchmark", "Overlapping speech is linearized by segment onset", "Mostly design-team meetings; narrow conversational domain", "Word counts include speaker labels", "No audio or teacher-generated dialogue used"])
    write_json(card_path, card)
    return {k: v for k, v in card.items() if k != "assignments"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("data/raw/ami/ami_public_manual_1.6.2.zip"))
    p.add_argument("--output", type=Path, default=Path("data/processed/ami"))
    p.add_argument("--card", type=Path, default=Path("data/cards/ami.json"))
    a = p.parse_args()
    print(json.dumps(prepare(a.archive, a.output, a.card), indent=2))


if __name__ == "__main__":
    main()
