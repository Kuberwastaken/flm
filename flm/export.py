"""Export a trained FLM to a small, framework-free browser bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import Config, FLM, VOCAB, load_graph
from .provenance import sha256, write_json
from .train import restored


@torch.no_grad()
def export_model(checkpoint: Path, graph_path: Path, output: Path, anatomy_source: Path | None = None):
    torch.set_num_threads(2)
    model, saved = restored(checkpoint, graph_path)
    model.eval()
    graph = load_graph(graph_path)
    n, p = model.config.neurons, model.config.pools
    w, alpha, beta, gain = model.constants()
    weights = (w.to_dense() if w.is_sparse else w)[model.row, model.col] * gain
    drives = model.input(model.embedding(torch.arange(VOCAB)))
    row = model.row.cpu().numpy(); col = model.col.cpu().numpy()
    order = np.lexsort((col, row))
    offsets = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=n)))).astype("<u4")
    to_numpy = lambda tensor: tensor.detach().cpu().numpy().astype("<f4")
    arrays = {
        "drives": to_numpy(drives), "offsets": offsets, "sources": col[order].astype("<u4"),
        "weights": to_numpy(weights)[order], "alpha": to_numpy(alpha), "beta": to_numpy(beta),
        "pool": model.pool_index.cpu().numpy().astype("<u4"), "pool_sizes": to_numpy(model.pool_sizes),
        "norm_weight": to_numpy(model.norm.weight), "norm_bias": to_numpy(model.norm.bias),
        "readout_weight": to_numpy(model.readout.weight), "readout_bias": to_numpy(model.readout.bias),
    }
    output.mkdir(parents=True, exist_ok=True)
    binary = bytearray(); index = {}
    for name, array in arrays.items():
        raw = array.tobytes(order="C")
        index[name] = dict(offset=len(binary), length=array.size, shape=list(array.shape),
                           dtype="uint32" if array.dtype.kind == "u" else "float32")
        binary.extend(raw)
    (output / "weights.bin").write_bytes(binary)
    graph_card = json.loads(graph_path.with_name("graph-card.json").read_text(encoding="utf8"))
    positions = graph["positions"].copy()
    finite = np.isfinite(positions).all(axis=1)
    midpoint = np.nanmedian(positions, axis=0)
    # Missing somas remain absent from rendering but present in the model.
    anatomy = dict(positions=[pos.tolist() if valid else None for pos, valid in zip(positions, finite)],
                   body_ids=[str(x) for x in graph["body_ids"]], cell_types=graph_card["cell_types"],
                   source_sign=graph["source_sign"].tolist(), coordinate_units="8 nm voxels")
    if anatomy_source:
        raw_positions = np.load(anatomy_source, allow_pickle=False)
        available = raw_positions[np.isfinite(raw_positions).all(axis=1)]
        indices = np.linspace(0, len(available) - 1, min(24000, len(available)), dtype=int)
        anatomy["context_positions"] = available[indices].tolist()
        anatomy["context_note"] = "Downsampled anatomical reference only; these background points do not display simulated activity"
    write_json(output / "anatomy.json", anatomy)
    configuration = dict(format="flm-browser-v1", name="FLM 0.1", model_id=f"flm-central-{n}-ami-s{saved['run']['seed']}",
        neurons=n, pools=p, features=p * 2, vocabulary=VOCAB, arrays=index, variant=model.config.variant,
        checkpoint_step=saved["step"], checkpoint_sha256=saved["_file_sha256"],
        weights_sha256=sha256(output / "weights.bin"), weights_bytes=len(binary),
        anatomy_sha256=sha256(output / "anatomy.json"), source_graph_sha256=sha256(graph_path),
        trained_parameters=saved["run"]["parameter_card"]["trainable_parameters"],
        retained_edges=len(row), source_neurons=graph_card["source_neurons"],
        training=saved["run"], norm_epsilon=float(model.norm.eps),
        description="A compact fly-wired model trained from scratch on human meeting transcripts.",
        capability="Experimental dialogue/text continuation; not instruction tuned or factually reliable.",
        license="MIT for original implementation; anatomy and corpus attribution CC BY 4.0")
    write_json(output / "model.json", configuration)
    # Ground truth for independent JS implementation parity, not UI output samples.
    tokens = [256] + list("a: what should we make?\nb:".encode())
    logits, state, features = model(torch.tensor([tokens]), return_features=True)
    parity = dict(tokens=tokens, logits=to_numpy(logits[0, -1]).tolist(),
                  h=to_numpy(state[0][0]).tolist(), slow=to_numpy(state[1][0]).tolist(),
                  features=to_numpy(features[0, -1]).tolist())
    write_json(output / "parity.json", parity)
    print(json.dumps({"model_id": configuration["model_id"], "step": saved["step"],
                      "bytes": len(binary), "neurons": n, "edges": len(row)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--graph", type=Path, default=Path("data/graphs/central-1024/graph.npz"))
    p.add_argument("--output", type=Path, default=Path("public/models/flm-compact"))
    p.add_argument("--anatomy-source", type=Path)
    a = p.parse_args()
    export_model(a.checkpoint, a.graph, a.output, a.anatomy_source)


if __name__ == "__main__":
    main()
