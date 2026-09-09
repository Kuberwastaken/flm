"""Bounded, resumable training and document-isolated likelihood evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .graph import rewire
from .model import Config, FLM, encode
from .model import load_graph
from .provenance import sha256, write_json


def read_documents(path: Path) -> list[tuple[str, np.ndarray]]:
    documents = []
    for line in path.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        documents.append((row["id"], np.asarray(encode(row["text"], boundaries=True), dtype=np.int64)))
    if not documents:
        raise ValueError(f"Empty corpus: {path}")
    return documents


class Sampler:
    def __init__(self, documents, seed: int, length: int):
        self.documents = [d for _, d in documents if len(d) > length + 1]
        if not self.documents:
            raise ValueError("No documents longer than one training window")
        self.length = length
        self.rng = np.random.default_rng(seed)
        sizes = np.array([len(d) - length for d in self.documents], dtype=np.float64)
        self.probabilities = sizes / sizes.sum()

    def sample(self, batch: int, device: str):
        indices = self.rng.choice(len(self.documents), size=batch, p=self.probabilities)
        chunks = []
        for index in indices:
            document = self.documents[index]
            offset = int(self.rng.integers(0, len(document) - self.length))
            chunks.append(document[offset:offset + self.length + 1])
        tokens = torch.from_numpy(np.stack(chunks)).to(device)
        return tokens[:, :-1], tokens[:, 1:]


@torch.no_grad()
def evaluate(model: FLM, documents, sequence: int = 128, byte_limit: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    losses, total_nll, total_bytes = [], 0.0, 0
    per_document = max(sequence, byte_limit // len(documents)) if byte_limit else None
    started = time.perf_counter()
    for identity, document in documents:
        if per_document:
            document = document[:per_document + 1]
        state = None
        doc_nll, doc_bytes = 0.0, 0
        for start in range(0, len(document) - 1, sequence):
            chunk = torch.from_numpy(document[start:start + sequence + 1].copy()).to(device).unsqueeze(0)
            logits, state = model(chunk[:, :-1], state)
            targets = chunk[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1), reduction="none")
            mask = targets.reshape(-1) < 256
            doc_nll += float(loss[mask].sum()); doc_bytes += int(mask.sum())
        losses.append(dict(document=identity, nll=doc_nll, bytes=doc_bytes,
                           bits_per_byte=doc_nll / max(doc_bytes, 1) / math.log(2)))
        total_nll += doc_nll; total_bytes += doc_bytes
    model.train(was_training)
    return dict(bits_per_byte=total_nll / total_bytes / math.log(2), nll_nats_per_byte=total_nll / total_bytes,
                byte_perplexity=math.exp(total_nll / total_bytes), bytes=total_bytes,
                seconds=time.perf_counter() - started, documents=losses,
                protocol="Reset state per document, carry within document; byte targets only; BOS/EOS excluded from score",
                subset="fixed document prefixes" if byte_limit else "entire supplied split")


def save_checkpoint(path: Path, model: FLM, optimizer, sampler: Sampler, step: int, best: float, run: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(config=asdict(model.config), model=model.state_dict(), optimizer=optimizer.state_dict(),
        step=step, best=best, sampler_rng=sampler.rng.bit_generator.state,
        torch_rng=torch.get_rng_state(), run=run), temporary)
    temporary.replace(path)


def restored(path: Path, graph_path: Path, device: str = "cpu"):
    # PyTorch 2.8 exposes __version__ as a harmless str subclass; accept that
    # legacy metadata type while retaining the restricted tensor-only loader.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FLM(load_graph(graph_path), Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    if checkpoint["run"]["graph_sha256"] != sha256(graph_path):
        raise ValueError("Checkpoint source graph mismatch")
    return model, checkpoint


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); random.seed(args.seed)
    train_path = args.data / "train.jsonl"; validation_path = args.data / "validation.jsonl"
    documents = read_documents(train_path); validation = read_documents(validation_path)
    sampler = Sampler(documents, args.seed, args.sequence)
    graph = load_graph(args.graph)
    if args.variant == "rewired":
        graph["row"], graph["col"] = rewire(graph["row"], graph["col"], graph["source_sign"], args.seed + 1000)
    config = Config(neurons=len(graph["body_ids"]), pools=int(graph["pool"].max()) + 1,
                    embedding=args.embedding, variant=args.variant, backend=args.backend)
    model = FLM(graph, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(seed=args.seed, train_sha256=sha256(train_path), validation_sha256=sha256(validation_path),
        graph_sha256=sha256(args.graph), graph=str(args.graph), data=str(args.data),
        training=dict(batch=args.batch, sequence=args.sequence, steps=args.steps, learning_rate=args.learning_rate,
                      warmup_tokens=args.warmup, threads=args.threads, device=args.device),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python_torch=str(torch.__version__), parameter_card=model.parameter_card(),
        state_protocol="Random within-document windows; fresh state per window; warmup prefix excluded from training loss",
        test_set_used_for_training=False)
    start_step, best = 0, float("inf")
    if args.resume:
        model, checkpoint = restored(args.resume, args.graph, args.device)
        if checkpoint["run"]["train_sha256"] != run["train_sha256"]:
            raise ValueError("Resume training corpus mismatch")
        if checkpoint["config"] != asdict(config):
            raise ValueError("Resume configuration mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.rng.bit_generator.state = checkpoint["sampler_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        start_step, best = checkpoint["step"], checkpoint["best"]
        run["resumed_from"] = dict(path=str(args.resume), sha256=sha256(args.resume), step=start_step)
    write_json(args.output / "run.json", run)
    if not args.resume:
        initial = evaluate(model, validation, args.sequence, args.eval_bytes)
        write_json(args.output / "initial-validation.json", initial)
        print(json.dumps({"initial_bpb": initial["bits_per_byte"], **model.parameter_card()}), flush=True)
    started = time.perf_counter(); accumulated = 0.0; interval_start = started
    history = args.output / "history.jsonl"
    for step in range(start_step + 1, args.steps + 1):
        model.train(); x, y = sampler.sample(args.batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, args.warmup:].reshape(-1, 258), y[:, args.warmup:].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step(); accumulated += loss.item()
        if step % args.log_every == 0:
            elapsed = time.perf_counter() - interval_start
            record = dict(step=step, train_loss=accumulated / args.log_every,
                          presented_tokens=step * args.batch * args.sequence,
                          tokens_per_second=args.log_every * args.batch * args.sequence / elapsed,
                          gradient_norm=float(gradient), wall_seconds=time.perf_counter() - started)
            with history.open("a", encoding="utf8") as f: f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True); accumulated = 0; interval_start = time.perf_counter()
        if step % args.eval_every == 0 or step == args.steps:
            measured = evaluate(model, validation, args.sequence, args.eval_bytes)
            write_json(args.output / f"validation-{step:06d}.json", measured)
            improved = measured["bits_per_byte"] < best
            best = min(best, measured["bits_per_byte"])
            save_checkpoint(args.output / "last.pt", model, optimizer, sampler, step, best, run)
            if improved: save_checkpoint(args.output / "best.pt", model, optimizer, sampler, step, best, run)
            print(json.dumps({"step": step, "validation_bpb": measured["bits_per_byte"], "best": best}), flush=True)
            interval_start = time.perf_counter()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/ami"))
    p.add_argument("--graph", type=Path, default=Path("data/graphs/central-1024/graph.npz"))
    p.add_argument("--output", type=Path, default=Path("runs/flm-ami-42"))
    p.add_argument("--variant", choices=["flm", "rewired", "no_recurrence", "no_slow"], default="flm")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sequence", type=int, default=96)
    p.add_argument("--embedding", type=int, default=64)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", choices=["auto", "dense", "sparse"], default="auto")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-bytes", type=int, default=32768)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if not 0 <= args.warmup < args.sequence or min(args.steps, args.batch, args.log_every, args.eval_every) < 1:
        p.error("Invalid training dimensions/intervals")
    train(args)


if __name__ == "__main__":
    main()
