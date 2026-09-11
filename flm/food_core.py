"""Fresh sensory/action interfaces around a copied, frozen language core.

This module defines a transfer interface, not a trained food policy. It uses
the existing FLM transition unchanged and does not assign biological time units.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from .model import FLM

ACTION_NAMES = ('turn_right', 'straight', 'turn_left')
USED_CORE_PARAMETERS = ('input.weight', 'input.bias', 'edge_log_gain', 'alpha_logit',
    'beta_logit', 'recurrent_logit', 'norm.weight', 'norm.bias')


class FoodCore(nn.Module):
    def __init__(self, language_model, adapter_seed):
        super().__init__()
        if type(language_model) is not FLM: raise ValueError('An FLM recurrent model is required')
        if language_model.config.variant not in ('flm', 'rewired'): raise ValueError('This bridge requires full fast/slow recurrent dynamics')
        if type(adapter_seed) is not int or not 0 <= adapter_seed < 2**63: raise ValueError('Invalid adapter seed')
        if language_model.input.weight.device.type != 'cpu': raise ValueError('This preparation supports CPU models only')
        self.core = copy.deepcopy(language_model)
        self.core.requires_grad_(False)
        self.adapter_seed = adapter_seed
        dtype = self.core.input.weight.dtype
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(adapter_seed)
            self.sensor = nn.Linear(6, self.core.config.embedding, dtype=dtype)
            self.action = nn.Linear(2*self.core.config.pools, len(ACTION_NAMES), dtype=dtype)
            nn.init.xavier_uniform_(self.sensor.weight, gain=.2); nn.init.zeros_(self.sensor.bias)
            nn.init.normal_(self.action.weight, std=.025); nn.init.zeros_(self.action.bias)

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1: raise ValueError('Positive batch size required')
        return self.core.initial_state(batch)

    def encode_projected(self, projected, state=None):
        """Input embeddings to normalized pooled states, using native FLM methods.

        A separate entry point permits lexical forward parity before replacing
        token embeddings with the fresh sensory adapter.
        """
        if (projected.ndim != 3 or min(projected.shape[:2]) < 1
                or projected.shape[2] != self.core.config.embedding
                or projected.dtype != self.core.input.weight.dtype
                or projected.device != self.core.input.weight.device or not torch.isfinite(projected).all()):
            raise ValueError('Finite [batch, time, embedding] input matching core dtype/device required')
        if state is None: state = self.initial_state(projected.shape[0])
        if not isinstance(state, tuple) or len(state) != 2 or any(
                x.shape != (projected.shape[0], self.core.config.neurons) or x.dtype != projected.dtype
                or x.device != projected.device or not torch.isfinite(x).all() for x in state):
            raise ValueError('State must contain two matching finite [batch, neuron] tensors')
        constants = self.core.constants()
        drives = self.core.input(projected); features = []
        for t in range(projected.shape[1]):
            state = self.core.transition(drives[:, t], state, constants)
            features.append(torch.cat((self.core.pool(state[0]), self.core.pool(state[1])), dim=-1))
        return self.core.norm(torch.stack(features, dim=1)), state

    def forward(self, sensory, state=None):
        if (sensory.ndim != 3 or min(sensory.shape[:2]) < 1 or sensory.shape[2] != 6
                or sensory.dtype != self.sensor.weight.dtype or sensory.device != self.sensor.weight.device
                or not torch.isfinite(sensory).all() or ((sensory < 0) | (sensory > 1)).any()):
            raise ValueError('Six normalized sensory inputs in [batch, time, 6] required')
        features, state = self.encode_projected(self.sensor(sensory), state)
        return self.action(features), state, features

    def parameter_card(self):
        original = dict(self.core.named_parameters())
        return dict(original_language_parameter_entries=sum(p.numel() for p in original.values()),
            frozen_core_parameter_entries_used=sum(original[name].numel() for name in USED_CORE_PARAMETERS),
            frozen_core_parameter_names_used=list(USED_CORE_PARAMETERS),
            unused_language_parameter_names=[name for name in original if name not in USED_CORE_PARAMETERS],
            trainable_adapter_parameter_entries=sum(p.numel() for p in (*self.sensor.parameters(), *self.action.parameters())),
            adapter_seed=self.adapter_seed, action_names=list(ACTION_NAMES),
            state_entries_per_sequence=2*self.core.config.neurons,
            core_frozen=not any(p.requires_grad for p in self.core.parameters()),
            scope='Fresh untrained interfaces. Lexical embedding/readout are retained in the copied Python object but unused by sensory inference. Core updates have no intrinsic biological time unit.')
