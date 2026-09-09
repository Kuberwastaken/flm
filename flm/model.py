"""A differentiable fly-wired recurrent core with two memory timescales."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

VOCAB = 258
BOS, EOS = 256, 257


@dataclass
class Config:
    neurons: int = 1024
    pools: int = 128
    embedding: int = 64
    variant: str = "flm"
    backend: str = "auto"
    vocabulary: int = VOCAB
    tied_readout: bool = False


class FLM(nn.Module):
    def __init__(self, graph: dict, config: Config | None = None):
        super().__init__()
        n = len(graph["body_ids"])
        self.config = config or Config(neurons=n, pools=int(np.max(graph["pool"])) + 1)
        c = self.config
        if c.neurons != n or c.variant not in ("flm", "rewired", "no_recurrence", "no_slow"):
            raise ValueError("Invalid graph/configuration")
        if c.backend not in ("auto", "dense", "sparse"):
            raise ValueError("Unknown matrix backend")
        if c.backend == "dense" and n > 4096:
            raise ValueError("Dense backend prohibited for large graphs")
        self.register_buffer("row", torch.as_tensor(graph["row"], dtype=torch.long))
        self.register_buffer("col", torch.as_tensor(graph["col"], dtype=torch.long))
        self.register_buffer("base_weight", torch.as_tensor(graph["weight"], dtype=torch.float32))
        self.register_buffer("pool_index", torch.as_tensor(graph["pool"], dtype=torch.long))
        sizes = np.bincount(graph["pool"], minlength=c.pools)
        self.register_buffer("pool_sizes", torch.tensor(np.maximum(sizes, 1), dtype=torch.float32))
        self.embedding = nn.Embedding(c.vocabulary, c.embedding)
        self.input = nn.Linear(c.embedding, n)
        self.edge_log_gain = nn.Parameter(torch.zeros(len(self.row)))
        self.alpha_logit = nn.Parameter(torch.linspace(-1.5, 1.5, n))
        self.beta_logit = nn.Parameter(torch.linspace(-2.5, 0.5, n))
        self.recurrent_logit = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(c.pools * 2)
        self.readout = nn.Linear(c.pools * 2, c.embedding if c.tied_readout else c.vocabulary)
        if c.tied_readout:
            self.output_bias = nn.Parameter(torch.zeros(c.vocabulary))
        nn.init.normal_(self.embedding.weight, std=0.2)
        nn.init.xavier_uniform_(self.input.weight, gain=0.7)
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.readout.weight, std=0.025)
        nn.init.zeros_(self.readout.bias)

    def initial_state(self, batch: int) -> tuple[Tensor, Tensor]:
        return (self.input.weight.new_zeros(batch, self.config.neurons),
                self.input.weight.new_zeros(batch, self.config.neurons))

    def constants(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = self.base_weight * self.edge_log_gain.clamp(-3, 3).exp()
        sums = values.new_zeros(self.config.neurons).scatter_add(0, self.row, values.abs())
        values = values / sums[self.row].clamp_min(1e-8)
        indices = torch.stack((self.row, self.col))
        w = torch.sparse_coo_tensor(indices, values, (self.config.neurons,) * 2, device=values.device).coalesce()
        dense = self.config.backend == "dense" or (self.config.backend == "auto" and self.config.neurons <= 2048)
        if dense:
            w = w.to_dense()
        alpha = 0.05 + 0.90 * self.alpha_logit.sigmoid()
        beta = 0.002 + 0.098 * self.beta_logit.sigmoid()
        gain = 0.05 + 2.95 * self.recurrent_logit.sigmoid()
        return w, alpha, beta, gain

    def pool(self, state: Tensor) -> Tensor:
        out = state.new_zeros(state.shape[0], self.config.pools)
        return out.scatter_add(1, self.pool_index.expand(state.shape[0], -1), state) / self.pool_sizes

    def transition(self, drive: Tensor, state: tuple[Tensor, Tensor], constants) -> tuple[Tensor, Tensor]:
        h, slow = state
        w, alpha, beta, gain = constants
        if self.config.variant != "no_recurrence":
            incoming = torch.sparse.mm(w, h.T).T if w.is_sparse else F.linear(h, w)
            drive = drive + gain * incoming
        h = (1 - alpha) * h + alpha * torch.tanh(drive)
        slow = torch.zeros_like(slow) if self.config.variant == "no_slow" else (1 - beta) * slow + beta * h
        return h, slow

    def forward(self, tokens: Tensor, state: tuple[Tensor, Tensor] | None = None,
                return_features: bool = False):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must be [batch, nonempty time]")
        state = state if state is not None else self.initial_state(tokens.shape[0])
        constants = self.constants()
        drives = self.input(self.embedding(tokens))
        features = []
        for t in range(tokens.shape[1]):
            state = self.transition(drives[:, t], state, constants)
            features.append(torch.cat((self.pool(state[0]), self.pool(state[1])), dim=-1))
        encoded = self.norm(torch.stack(features, dim=1))
        if self.config.tied_readout:
            encoded = self.readout(encoded)
            logits = F.linear(encoded, self.embedding.weight, self.output_bias)
        else:
            logits = self.readout(encoded)
        return (logits, state, encoded) if return_features else (logits, state)

    def parameter_card(self) -> dict:
        groups = {name: parameter.numel() for name, parameter in self.named_parameters()}
        return dict(config=asdict(self.config), trainable_parameters=sum(groups.values()),
                    parameter_groups=groups, edges=len(self.row), vocabulary=self.config.vocabulary)


def load_graph(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def encode(text: str, boundaries: bool = False) -> list[int]:
    result = list(text.encode("utf8"))
    return [BOS] + result + [EOS] if boundaries else result
