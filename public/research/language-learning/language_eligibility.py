"""Masked language-window gradients with diagonal temporal eligibility.

The forward computation is FLM's ordinary recurrence. Only credit assignment
changes: temporal paths between different neurons are omitted. Parameters stay
fixed during this synchronous call; it does not update weights or parameter
``.grad`` fields. No training corpus, optimizer or benchmark is chosen here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .embedding_eligibility import EmbeddingEligibility
from .local_learning import CORE_PARAMETERS, trace_bytes
from .model import FLM


@dataclass
class WindowGradients:
    loss: float
    scored_tokens: int
    logits: torch.Tensor
    state: tuple[torch.Tensor, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    persistent_trace_bytes: int


def _decision(model, state):
    features = torch.cat((model.pool(state[0]), model.pool(state[1])), dim=-1)
    encoded = model.readout(model.norm(features))
    if model.config.tied_readout:
        return F.linear(encoded, model.embedding.weight, model.output_bias)
    return encoded


def _validate(model, tokens, targets, mask, state):
    # This kernel intentionally targets the current compact language model.
    # Subclass transitions and large/sparse cores need their own derivations.
    if type(model) is not FLM or model.config.variant not in ('flm', 'rewired'):
        raise ValueError('Eligibility requires an unmodified full FLM recurrence')
    if model.config.neurons > 2048 or model.config.backend == 'sparse':
        raise ValueError('This eligibility kernel supports compact dense cores only')
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('All parameters must be trainable; fixed-core control is separate')
    if tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype != torch.long:
        raise ValueError('Tokens must be long [batch, nonempty time]')
    if targets.shape != tokens.shape or targets.dtype != torch.long:
        raise ValueError('Targets must match tokens')
    if mask.shape != tokens.shape or mask.dtype != torch.bool or not mask.any():
        raise ValueError('Score mask must be boolean with at least one scored position')
    for value in (tokens, targets):
        if (value < 0).any() or (value >= model.config.vocabulary).any():
            raise ValueError('Token outside vocabulary')
    reference = model.input.weight
    if any(x.device != reference.device for x in (tokens, targets, mask)):
        raise ValueError('Inputs must share the model device')
    for x in model.state_dict().values():
        if x.is_floating_point() and (x.dtype != reference.dtype or not torch.isfinite(x).all()):
            raise ValueError('Model tensors must be finite with a common dtype')
    pairs = model.row * model.config.neurons + model.col
    if len(pairs.unique()) != len(pairs):
        raise ValueError('Parallel edge entries require a separate derivative rule')
    if state is not None:
        if len(state) != 2 or any(x.shape != (tokens.shape[0], model.config.neurons)
                or x.device != reference.device or x.dtype != reference.dtype
                or not torch.isfinite(x).all() for x in state):
            raise ValueError('Initial fast and slow states must match model and batch')


@torch.no_grad()
def language_window_gradients(model, tokens, targets, score_mask, *, history=True, state=None):
    """Return mean scored-token loss and every parameter's window gradient.

    Warm-up and other unscored inputs advance states and traces. An optional
    initial state is detached: no gradient crosses this window's left boundary.
    ``history=False`` removes old eligibility, while preserving forward memory.
    The result combines instantaneous classifier derivatives (including the
    tied embedding) with local input-mediated and core derivatives. All weights
    must remain fixed until this function returns; an optimizer may then consume
    the returned gradients once. Returned tensors carry no autograd history.
    """
    _validate(model, tokens, targets, score_mask, state)
    if type(history) is not bool:
        raise ValueError('History must be a boolean')
    batch, length = tokens.shape
    count = int(score_mask.sum())
    h, slow = model.initial_state(batch) if state is None else tuple(x.detach().clone() for x in state)
    w, alpha, beta, gain = model.constants()
    diagonal = w.diag()
    embedded = model.embedding(tokens)
    drives = model.input(embedded)
    parameters = dict(model.named_parameters())
    versions = {name: p._version for name, p in parameters.items()}
    gradients = {name: torch.zeros_like(p) for name, p in parameters.items()}
    lexical = {name: p for name, p in parameters.items() if name not in CORE_PARAMETERS
               and (name != 'embedding.weight' or model.config.tied_readout)}
    shapes = {'input.weight': (batch, model.config.neurons, model.config.embedding),
              'input.bias': h.shape, 'edge_log_gain': (batch, len(model.row)),
              'alpha_logit': h.shape, 'beta_logit': h.shape, 'recurrent_logit': h.shape}
    traces = {name: (h.new_zeros(shape), h.new_zeros(shape)) for name, shape in shapes.items()}
    embedding_trace = (EmbeddingEligibility(model.input.weight, model.config.vocabulary,
                                           batch, length) if history else None)
    da = .90 * model.alpha_logit.sigmoid() * (1 - model.alpha_logit.sigmoid())
    db = .098 * model.beta_logit.sigmoid() * (1 - model.beta_logit.sigmoid())
    dg = 2.95 * model.recurrent_logit.sigmoid() * (1 - model.recurrent_logit.sigmoid())
    raw = model.base_weight * model.edge_log_gain.clamp(-3, 3).exp()
    sums = raw.new_zeros(model.config.neurons).scatter_add_(0, model.row, raw.abs())
    normalized = w[model.row, model.col]
    normalization_active = (sums[model.row] >= 1e-8).to(h.dtype)
    clamp_active = ((model.edge_log_gain >= -3) & (model.edge_log_gain <= 3)).to(h.dtype)
    all_logits = []
    objective = 0.
    for t in range(length):
        incoming = F.linear(h, w)
        z = torch.tanh(drives[:, t] + gain * incoming)
        d = alpha * (1 - z.square())
        jacobian = 1 - alpha + d * gain * diagonal
        next_h = (1 - alpha) * h + alpha * z
        # Account for both clamp derivatives, including sub-floor incoming sums.
        edge = (normalized * h[:, model.col]
                - normalized.abs() * normalization_active * incoming[:, model.row]) * clamp_active
        direct = {'input.weight': d.unsqueeze(-1) * embedded[:, t].unsqueeze(1),
                  'input.bias': d, 'edge_log_gain': d[:, model.row] * gain * edge,
                  'alpha_logit': (z - h) * da, 'beta_logit': torch.zeros_like(h),
                  'recurrent_logit': d * incoming * dg}
        for name, (eh, es) in traces.items():
            j = jacobian[:, model.row] if name == 'edge_log_gain' else jacobian
            b = beta[model.row] if name == 'edge_log_gain' else beta
            if name == 'input.weight':
                j, b = j.unsqueeze(-1), b.unsqueeze(-1)
            eh = (j * eh if history else 0) + direct[name]
            es = ((1 - b) * es if history else 0) + b * eh
            if name == 'beta_logit':
                es = es + (next_h - slow) * db
            traces[name] = (eh, es)
        if embedding_trace is not None:
            embedding_trace.step(tokens[:, t], jacobian, d, beta)
        slow, h = (1 - beta) * slow + beta * next_h, next_h
        if not score_mask[:, t].any():
            all_logits.append(_decision(model, (h, slow)))
            continue
        # Autograd sees this position's readout only, never the recurrent past.
        with torch.enable_grad():
            leaves = tuple(x.detach().requires_grad_(True) for x in (h, slow))
            logits = _decision(model, leaves)
            losses = F.cross_entropy(logits, targets[:, t], reduction='none')
            loss = (losses * score_mask[:, t]).sum() / count
            derivatives = torch.autograd.grad(loss, (*leaves, *lexical.values()))
        lh, ls = derivatives[:2]
        objective += float(loss)
        all_logits.append(logits.detach())
        for name, gradient in zip(lexical, derivatives[2:]):
            gradients[name].add_(gradient)
        for name, (eh, es) in traces.items():
            sh, ss = lh, ls
            if name == 'edge_log_gain':
                sh, ss = lh[:, model.row], ls[:, model.row]
            elif name == 'input.weight':
                sh, ss = lh.unsqueeze(-1), ls.unsqueeze(-1)
            gradient = (sh * eh + ss * es).sum(0)
            gradients[name].add_(gradient.sum() if name == 'recurrent_logit' else gradient)
        if embedding_trace is not None:
            gradients['embedding.weight'].add_(embedding_trace.gradient((lh, ls), model.input.weight))
        else:
            occurrence = ((lh + beta * ls) * d) @ model.input.weight
            gradients['embedding.weight'].index_add_(0, tokens[:, t], occurrence)
    if any(p._version != versions[name] for name, p in parameters.items()):
        raise RuntimeError('Model parameters changed inside the fixed-weight window')
    stacked = torch.stack(all_logits, dim=1)
    if not math.isfinite(objective) or not torch.isfinite(stacked).all() or any(not torch.isfinite(x).all() for x in gradients.values()):
        raise FloatingPointError('Nonfinite language eligibility result')
    stored = trace_bytes(traces)
    if embedding_trace is not None:
        stored += sum(embedding_trace.storage_bytes().values())
    return WindowGradients(objective, count, stacked, (h, slow), gradients, stored)
