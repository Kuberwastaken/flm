"""Controlled delayed context and read-only local-gradient diagnostics."""
from __future__ import annotations
import copy
import numpy as np
import torch
from .behavior_study import episode_batch
from .local_learning import CORE_PARAMETERS


def task_batch(task, rng, count, delay):
    if task == 'cue':
        sensory, cue = episode_batch(rng, count, delay)
        return sensory, cue, dict(cue=cue.tolist())
    if task != 'context': raise ValueError('Unknown task')
    if count < 4 or count % 4 or delay < 0: raise ValueError('Context batches require balanced groups of four')
    pair = np.tile(np.arange(4, dtype=np.int64), count // 4); rng.shuffle(pair)
    cue, context = pair // 2, pair % 2
    sensory = np.zeros((count, 2 * delay + 5, 4), dtype=np.float32)
    sensory[:, :2, 0] = (2 * cue - 1)[:, None]
    sensory[:, delay + 2:delay + 4, 1] = (2 * context - 1)[:, None]
    sensory[:, :, 2] = rng.normal(0., .2, sensory.shape[:2]); sensory[:, -1, 3] = 1.
    return sensory, cue ^ context, dict(cue=cue.tolist(), context=context.tolist())


def gradient_metrics(exact, approximate):
    exact, approximate = exact.flatten(), approximate.flatten()
    if exact.shape != approximate.shape or not torch.isfinite(exact).all() or not torch.isfinite(approximate).all():
        raise ValueError('Malformed gradient comparison')
    a, b = float(exact.norm()), float(approximate.norm()); comparable = (exact != 0) & (approximate != 0)
    return dict(entries=exact.numel(), exact_norm=a, approximate_norm=b,
        cosine=max(-1., min(1., float(torch.dot(exact, approximate)) / (a * b))) if a and b else None,
        relative_l2_error=float((exact - approximate).norm()) / a if a else None,
        sign_agreement=float((exact[comparable].sign() == approximate[comparable].sign()).double().mean()) if comparable.any() else None,
        comparable_sign_entries=int(comparable.sum()), exact_nonzero=int((exact != 0).sum()), approximate_nonzero=int((approximate != 0).sum()))


def gradient_probe(model, sensory, targets):
    candidate = copy.deepcopy(model).double(); sensory = sensory.double()
    candidate.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(candidate(sensory)[0], targets); loss.backward()
    parameters = dict(candidate.named_parameters())
    exact = {name: parameters[name].grad.detach().clone() for name in CORE_PARAMETERS}
    results = {}
    for condition, history in [('eligibility', True), ('instantaneous', False)]:
        candidate.zero_grad(set_to_none=True)
        state, traces = candidate.forward_eligibility(sensory, history=history)
        state = tuple(value.requires_grad_(True) for value in state)
        torch.nn.functional.cross_entropy(candidate.decision(state), targets).backward()
        candidate.assign_core_gradients(tuple(value.grad for value in state), traces)
        approximate = {name: parameters[name].grad.detach().clone() for name in CORE_PARAMETERS}
        values = {name: gradient_metrics(exact[name], approximate[name]) for name in CORE_PARAMETERS}
        values['combined_core'] = gradient_metrics(torch.cat([exact[name].flatten() for name in CORE_PARAMETERS]),
            torch.cat([approximate[name].flatten() for name in CORE_PARAMETERS]))
        results[condition] = values
    return dict(supervised_cross_entropy=float(loss.detach()), precision='float64 copy; no optimizer update',
        gradient_loss='Supervised current-mapping cross entropy, including when the network was reward-trained', comparisons=results)
