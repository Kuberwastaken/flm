"""Small parameter-matched recurrent and attention baselines, trained from scratch."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .model import VOCAB


@dataclass
class BaselineConfig:
    variant: str
    embedding: int = 64
    hidden: int = 200
    width: int = 104
    heads: int = 4
    layers: int = 2
    window: int = 96
    vocabulary: int = VOCAB
    tied_readout: bool = False


class Baseline(nn.Module):
    def parameter_card(self):
        groups = {name: p.numel() for name, p in self.named_parameters()}
        return dict(config=asdict(self.config), trainable_parameters=sum(groups.values()),
                    parameter_groups=groups, vocabulary=self.config.vocabulary)


class GRU(Baseline):
    def __init__(self, config=None):
        super().__init__(); self.config = config or BaselineConfig('gru'); c = self.config
        self.embedding = nn.Embedding(c.vocabulary, c.embedding)
        self.core = nn.GRU(c.embedding, c.hidden, batch_first=True)
        self.readout = nn.Linear(c.hidden, c.embedding if c.tied_readout else c.vocabulary)
        if c.tied_readout: self.output_bias = nn.Parameter(torch.zeros(c.vocabulary))

    def forward(self, tokens, state=None):
        values, state = self.core(self.embedding(tokens), state)
        features = self.readout(values)
        return (F.linear(features, self.embedding.weight, self.output_bias) if self.config.tied_readout else features), state


def rotary(values, positions):
    """Adjacent-pair RoPE; absolute offsets cancel in attention dot products."""
    dimension = values.shape[-1]
    frequencies = 10000.0 ** (-torch.arange(0, dimension, 2, device=values.device, dtype=values.dtype) / dimension)
    angles = positions.to(values.dtype)[:, None] * frequencies[None, :]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class AttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__(); self.config = c
        self.norm1 = nn.LayerNorm(c.width); self.norm2 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, c.width * 3); self.output = nn.Linear(c.width, c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, c.width * 2), nn.GELU(), nn.Linear(c.width * 2, c.width))

    def forward(self, values, offset, cache=None):
        c = self.config; batch, length, _ = values.shape
        qkv = self.qkv(self.norm1(values)).reshape(batch, length, 3, c.heads, c.width // c.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        positions = torch.arange(offset, offset + length, device=values.device)
        query, key = rotary(query, positions), rotary(key, positions)
        past = 0
        if cache is not None:
            past = cache[0].shape[2]; key = torch.cat((cache[0], key), dim=2); value = torch.cat((cache[1], value), dim=2)
        keys = torch.arange(offset - past, offset + length, device=values.device)
        allowed = (keys[None, :] <= positions[:, None]) & (keys[None, :] > positions[:, None] - c.window)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.0)
        values = values + self.output(attended.transpose(1, 2).reshape(batch, length, c.width))
        values = values + self.ff(self.norm2(values))
        # Future chunks need only the last window-1 keys. No state crosses documents.
        return values, (key[:, :, -(c.window - 1):], value[:, :, -(c.window - 1):])


class Transformer(Baseline):
    def __init__(self, config=None):
        super().__init__(); self.config = config or BaselineConfig('transformer'); c = self.config
        if c.width % (2 * c.heads) or c.window < 2: raise ValueError('Invalid transformer dimensions')
        self.embedding = nn.Embedding(c.vocabulary, c.embedding if c.tied_readout else c.width)
        if c.tied_readout:
            self.input_projection = nn.Linear(c.embedding, c.width)
            self.output_bias = nn.Parameter(torch.zeros(c.vocabulary))
        self.blocks = nn.ModuleList([AttentionBlock(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width); self.readout = nn.Linear(c.width, c.embedding if c.tied_readout else c.vocabulary)

    def forward(self, tokens, state=None):
        offset, cached = state if state is not None else (0, [None] * len(self.blocks))
        values = self.embedding(tokens); next_cache = []
        if self.config.tied_readout: values = self.input_projection(values)
        for block, cache in zip(self.blocks, cached):
            values, cache = block(values, offset, cache); next_cache.append(cache)
        features = self.readout(self.norm(values))
        logits = F.linear(features, self.embedding.weight, self.output_bias) if self.config.tied_readout else features
        return logits, (offset + tokens.shape[1], next_cache)
