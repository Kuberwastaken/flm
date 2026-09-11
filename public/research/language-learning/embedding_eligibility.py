"""Window-bounded embedding traces for FLM's diagonal temporal approximation.

This is a research primitive, not a language trainer. The input projection and
all model weights must remain fixed during the window. Cross-neuron temporal
credit is omitted, as in local_learning.py. One trace column per token occurrence
avoids a vocabulary-by-embedding-dimension state Jacobian.
"""
from __future__ import annotations

import torch


class EmbeddingEligibility:
    @torch.no_grad()
    def __init__(self, projection, vocabulary, batch, capacity):
        if projection.ndim != 2 or not projection.is_floating_point():
            raise ValueError('Projection must be a floating [neuron, embedding] matrix')
        if min(vocabulary, batch, capacity, *projection.shape) < 1:
            raise ValueError('All dimensions must be positive')
        if not torch.isfinite(projection).all():
            raise ValueError('Projection must be finite')
        self.projection = projection.detach().clone()
        self.vocabulary = vocabulary
        self.capacity = capacity
        self.used = 0
        self.fast = projection.new_zeros(batch, projection.shape[0], capacity)
        self.slow = torch.zeros_like(self.fast)
        self.tokens = torch.zeros(batch, capacity, dtype=torch.long, device=projection.device)

    @torch.no_grad()
    def step(self, tokens, jacobian, direct, beta):
        """Append one input occurrence; direct = alpha * (1 - tanh(drive)**2)."""
        batch, neurons, _ = self.fast.shape
        if self.used == self.capacity:
            raise ValueError('Window capacity exhausted; start a fresh trace window')
        if tokens.shape != (batch,) or tokens.dtype != torch.long:
            raise ValueError('Tokens must be a long [batch] tensor')
        if (tokens < 0).any() or (tokens >= self.vocabulary).any():
            raise ValueError('Token outside vocabulary')
        if jacobian.shape != (batch, neurons) or direct.shape != (batch, neurons) or beta.shape != (neurons,):
            raise ValueError('Invalid local derivative shapes')
        for value in (jacobian, direct, beta):
            if value.device != self.fast.device or value.dtype != self.fast.dtype or not torch.isfinite(value).all():
                raise ValueError('Derivatives must be finite and match trace dtype/device')
        if tokens.device != self.fast.device or ((beta < 0) | (beta > 1)).any():
            raise ValueError('Invalid token device or slow mixing coefficient')
        end = self.used + 1
        self.fast[:, :, :self.used].mul_(jacobian.unsqueeze(-1))
        self.fast[:, :, self.used].copy_(direct)
        self.slow[:, :, :end].mul_((1 - beta)[None, :, None])
        self.slow[:, :, :end].add_(beta[None, :, None] * self.fast[:, :, :end])
        self.tokens[:, self.used].copy_(tokens)
        self.used = end

    @torch.no_grad()
    def gradient(self, signals, projection):
        """Return input-mediated embedding gradient; add the tied-output gradient.

        Signals already contain the objective's reduction. Do not average again.
        The projection check rejects using this factorization across its updates.
        """
        if not torch.equal(projection, self.projection):
            raise ValueError('Input projection changed within the eligibility window')
        for value in signals:
            if value.shape != self.fast.shape[:2] or value.device != self.fast.device or value.dtype != self.fast.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid instantaneous learning signals')
        if len(signals) != 2:
            raise ValueError('Fast and slow learning signals are required')
        lh, ls = signals
        weighted = lh.unsqueeze(-1) * self.fast[:, :, :self.used] + ls.unsqueeze(-1) * self.slow[:, :, :self.used]
        occurrence_gradients = weighted.transpose(1, 2) @ self.projection
        result = self.projection.new_zeros(self.vocabulary, self.projection.shape[1])
        result.index_add_(0, self.tokens[:, :self.used].reshape(-1), occurrence_gradients.reshape(-1, result.shape[1]))
        return result

    def storage_bytes(self):
        """Allocated persistent tensors only, excluding temporary contractions."""
        return {name: value.numel() * value.element_size() for name, value in
                [('fast', self.fast), ('slow', self.slow), ('tokens', self.tokens), ('projection', self.projection)]}


def trace_accounting(neurons=1024, edges=76130, embedding=96, vocabulary=4096,
                     batch=16, length=96, float_bytes=4):
    """Explicit representation sizes; neither process peaks nor algorithmic bounds."""
    if min(neurons, embedding, vocabulary, batch, length, float_bytes) < 1 or edges < 0:
        raise ValueError('Invalid dimensions')
    core_parameters = neurons * embedding + edges + 3 * neurons + 1
    state_parameters = core_parameters + vocabulary * embedding
    return dict(neurons=neurons, edges=edges, embedding=embedding, vocabulary=vocabulary,
        batch=batch, length=length, float_bytes=float_bytes,
        core_parameters=core_parameters, teacher_forced_state_parameters=state_parameters,
        core_trace_bytes=2 * batch * (neurons * embedding + edges + 4 * neurons) * float_bytes,
        dense_embedding_trace_bytes=2 * batch * neurons * vocabulary * embedding * float_bytes,
        factorized_vocabulary_trace_bytes=2 * batch * neurons * vocabulary * float_bytes,
        occurrence_trace_bytes=2 * batch * neurons * length * float_bytes,
        occurrence_token_bytes=batch * length * 8,
        projection_snapshot_bytes=neurons * embedding * float_bytes,
        dense_full_state_jacobian_bytes=2 * batch * neurons * state_parameters * float_bytes)
