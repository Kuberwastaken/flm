"""Build a compact, traceable induced graph from the retained MaleCNS graph.

The default ranking depends only on anatomy. It is fixed before language data
is inspected. The compact graph must never be described as the whole CNS.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .provenance import GRAPH_REVISION, sha256, write_json


def build(source: Path, destination: Path, neurons: int = 1024, pools: int = 128) -> dict:
    metadata = json.loads(gzip.decompress((source / "neurons.json.gz").read_bytes()))
    offsets = np.load(source / "offsets.npy", allow_pickle=False)
    sources = np.load(source / "sources.npy", allow_pickle=False)
    counts = np.load(source / "counts.npy", allow_pickle=False)
    signs = np.load(source / "fast_sign.npy", allow_pickle=False)
    ids = np.load(source / "body_ids.npy", allow_pickle=False)
    n = len(metadata)
    if len(offsets) != n + 1 or offsets[-1] != len(sources) or len(sources) != len(counts):
        raise ValueError("Malformed source CSR")
    if len(np.unique(ids)) != n or not np.all(offsets[1:] >= offsets[:-1]) or sources.max() >= n:
        raise ValueError("Invalid source identities or offsets")
    eligible = np.array([i for i, row in enumerate(metadata) if row[2] == "cb_intrinsic"], dtype=np.int32)
    if not 16 <= neurons <= len(eligible) or not 1 <= pools <= neurons:
        raise ValueError("Invalid compact graph/pooling size")
    full = csr_matrix((counts.astype(np.float32), sources, offsets), shape=(n, n))
    central = full[eligible][:, eligible]
    strength = np.asarray(central.sum(0)).ravel() + np.asarray(central.sum(1)).ravel()
    order = np.lexsort((ids[eligible], -strength))
    selected = eligible[order[:neurons]]
    # Canonical ordering groups related cell types while retaining unique source IDs.
    selected = np.array(sorted(selected.tolist(), key=lambda i: (metadata[i][1], int(ids[i]))), dtype=np.int32)
    compact = full[selected][:, selected].tocsr()
    compact.sort_indices()
    destination.mkdir(parents=True, exist_ok=True)
    row = np.repeat(np.arange(neurons, dtype=np.int32), np.diff(compact.indptr))
    col = compact.indices.astype(np.int32)
    contact = compact.data.astype(np.uint32)
    sign = signs[selected].astype(np.int8)
    active = sign[col] != 0
    row, col, contact = row[active], col[active], contact[active]
    raw = np.log1p(contact.astype(np.float32))
    denominator = np.bincount(row, weights=raw, minlength=neurons).astype(np.float32)
    weight = raw / np.maximum(denominator[row], 1e-12) * sign[col]
    pool = np.minimum(np.arange(neurons) * pools // neurons, pools - 1).astype(np.int32)
    positions = np.asarray([metadata[i][6] or [float("nan")] * 3 for i in selected], dtype=np.float32)
    arrays = dict(row=row, col=col, weight=weight.astype(np.float32), contacts=contact,
                  source_sign=sign, body_ids=ids[selected], source_indices=selected,
                  positions=positions, pool=pool)
    np.savez_compressed(destination / "graph.npz", **arrays)
    source_files = ["offsets.npy", "sources.npy", "counts.npy", "fast_sign.npy", "body_ids.npy", "neurons.json.gz"]
    card = {
        "name": f"MaleCNS central {neurons}", "source_revision": GRAPH_REVISION,
        "source_neurons": n, "source_edges": int(len(sources)),
        "neurons": neurons, "edges": len(row), "pools": pools,
        "selection": "Induced cb_intrinsic subgraph; highest incoming+outgoing central contact strength; ID tie break; before language training",
        "ordering": "Cell type then source body ID; balanced contiguous readout pools",
        "sign_rule": "Runtime fast sign: acetylcholine +1, GABA/glutamate -1, other/unclear 0; zero-sign outgoing edges excluded from fast computation",
        "weight_rule": "log1p(contact_count), normalized by absolute incoming sum, then presynaptic sign",
        "orientation": "row postsynaptic, column presynaptic",
        "license": "CC-BY-4.0", "source": "https://male-cns.janelia.org/download/",
        "graph_sha256": sha256(destination / "graph.npz"),
        "source_file_sha256": {f: sha256(source / f) for f in source_files},
        "cell_types": [metadata[i][1] for i in selected],
        "cut_edges": "Only connections between selected nodes retained; boundary input is absent",
        "limitations": ["A selected central-brain subgraph, not the complete fly brain", "Contact strengths/signs are modeling assumptions", "Selection favors highly connected neurons and may bias memory/computation"]}
    write_json(destination / "graph-card.json", card)
    return {k: v for k, v in card.items() if k not in ("cell_types", "source_file_sha256")}


def rewire(row: np.ndarray, col: np.ndarray, signs: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Directed double-edge swaps preserving each degree and input sign counts.

    No duplicate edges are introduced. Self edges are allowed only when already
    present; proposed self edges are rejected. Rewiring actually changes topology.
    """
    rng = np.random.default_rng(seed)
    result = col.copy()
    edges = set(zip(row.tolist(), result.tolist()))
    swapped = 0
    for _ in range(max(1000, len(row) * 20)):
        a, b = rng.integers(0, len(row), size=2)
        ra, rb, ca, cb = int(row[a]), int(row[b]), int(result[a]), int(result[b])
        if a == b or ra == rb or ca == cb or signs[ca] != signs[cb] or ra == cb or rb == ca:
            continue
        if (ra, cb) in edges or (rb, ca) in edges:
            continue
        edges.remove((ra, ca)); edges.remove((rb, cb))
        edges.add((ra, cb)); edges.add((rb, ca))
        result[a], result[b] = cb, ca
        swapped += 1
    if swapped == 0:
        raise ValueError("No valid topology swaps; this graph cannot supply the requested control")
    return row.copy(), result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/graphs/central-1024"))
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--pools", type=int, default=128)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output, a.neurons, a.pools), indent=2))


if __name__ == "__main__":
    main()
