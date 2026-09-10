"""Forward local eligibility for FLM's normalized, two-timescale rate dynamics.

This deliberately drops cross-neuron temporal credit. It is not exact BPTT or a
reproduction of spiking e-prop. The instantaneous output derivative remains exact.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .model import FLM, Config

CORE_PARAMETERS = ('input.weight', 'input.bias', 'edge_log_gain', 'alpha_logit', 'beta_logit', 'recurrent_logit')


def edge_direct(weight, row, col, previous, incoming):
    """dq_i/dtheta_ij including incoming absolute-sum normalization."""
    values = weight[row, col]
    return values * previous[:, col] - values.abs() * incoming[:, row]


class ChoiceFLM(FLM):
    def __init__(self, graph):
        super().__init__(graph, Config(neurons=len(graph['body_ids']), pools=int(graph['pool'].max()) + 1,
                                      embedding=4, vocabulary=2, tied_readout=False))
        # The behavioral task supplies four numeric sensory channels directly.
        del self.embedding

    def decision(self, state):
        features = torch.cat((self.pool(state[0]), self.pool(state[1])), dim=-1)
        return self.readout(self.norm(features))

    def forward(self, sensory, state=None):
        if sensory.ndim != 3 or sensory.shape[1] < 1 or sensory.shape[2] != 4:
            raise ValueError('Sensory input must have shape [batch, nonempty time, 4]')
        state = self.initial_state(sensory.shape[0]) if state is None else state
        constants = self.constants(); drives = self.input(sensory)
        for t in range(sensory.shape[1]): state = self.transition(drives[:, t], state, constants)
        return self.decision(state), state

    @torch.no_grad()
    def forward_eligibility(self, sensory, history=True):
        if sensory.ndim != 3 or sensory.shape[1] < 1 or sensory.shape[2] != 4:
            raise ValueError('Sensory input must have shape [batch, nonempty time, 4]')
        batch = sensory.shape[0]; h, slow = self.initial_state(batch)
        w, alpha, beta, gain = self.constants()
        if w.is_sparse: w = w.to_dense()
        diagonal = w.diag(); traces = {}
        shapes = {'input.weight': (batch, self.config.neurons, 4), 'input.bias': h.shape,
                  'edge_log_gain': (batch, len(self.row)), 'alpha_logit': h.shape,
                  'beta_logit': h.shape, 'recurrent_logit': h.shape}
        for name, shape in shapes.items(): traces[name] = (h.new_zeros(shape), h.new_zeros(shape))
        da = .90 * self.alpha_logit.sigmoid() * (1 - self.alpha_logit.sigmoid())
        db = .098 * self.beta_logit.sigmoid() * (1 - self.beta_logit.sigmoid())
        dg = 2.95 * self.recurrent_logit.sigmoid() * (1 - self.recurrent_logit.sigmoid())
        for t in range(sensory.shape[1]):
            incoming = F.linear(h, w); z = torch.tanh(self.input(sensory[:, t]) + gain * incoming)
            d = alpha * (1 - z.square()); jacobian = 1 - alpha + d * gain * diagonal
            next_h = (1 - alpha) * h + alpha * z
            direct = {'input.weight': d.unsqueeze(-1) * sensory[:, t].unsqueeze(1), 'input.bias': d,
                'edge_log_gain': d[:, self.row] * gain * edge_direct(w, self.row, self.col, h, incoming),
                'alpha_logit': (z - h) * da, 'beta_logit': torch.zeros_like(h),
                'recurrent_logit': d * incoming * dg}
            for name, (eh, es) in traces.items():
                j = jacobian[:, self.row] if name == 'edge_log_gain' else jacobian
                b = beta[self.row] if name == 'edge_log_gain' else beta
                if name == 'input.weight': j = j.unsqueeze(-1); b = b.unsqueeze(-1)
                eh = (j * eh if history else 0) + direct[name]
                es = ((1 - b) * es if history else 0) + b * eh
                if name == 'beta_logit': es = es + (next_h - slow) * db
                traces[name] = (eh, es)
            slow = (1 - beta) * slow + beta * next_h; h = next_h
        return (h, slow), traces

    @torch.no_grad()
    def assign_core_gradients(self, signals, traces):
        """Signals already contain the loss's batch reduction; do not average twice."""
        parameters = dict(self.named_parameters())
        for name, (eh, es) in traces.items():
            lh, ls = signals
            if name == 'edge_log_gain': lh, ls = lh[:, self.row], ls[:, self.row]
            elif name == 'input.weight': lh, ls = lh.unsqueeze(-1), ls.unsqueeze(-1)
            gradient = (lh * eh + ls * es).sum(0)
            if name == 'recurrent_logit': gradient = gradient.sum()
            parameters[name].grad = gradient


def trace_bytes(traces):
    return sum(t.numel() * t.element_size() for pair in traces.values() for t in pair)


def learning_step(model, sensory, targets, optimizer, method, action_rng=None, baseline=.5):
    """One complete-episode batch update, with independent action sampling RNG."""
    if method not in ('bptt', 'reservoir', 'eligibility', 'instantaneous', 'reward'):
        raise ValueError('Unknown learning condition')
    optimizer.zero_grad(set_to_none=True); trace_size = 0
    if method == 'bptt': logits, state = model(sensory)
    else:
        if method == 'reservoir':
            with torch.no_grad(): _, state = model(sensory)
        else:
            state, traces = model.forward_eligibility(sensory, history=method != 'instantaneous')
            trace_size = trace_bytes(traces)
        state = tuple(value.detach().requires_grad_(True) for value in state)
        logits = model.decision(state)
    reward = None
    if method == 'reward':
        if action_rng is None: raise ValueError('Reward learning requires a separate action generator')
        actions = torch.multinomial(logits.softmax(-1), 1, generator=action_rng).squeeze(1)
        rewards = (actions == targets).to(logits.dtype)
        loss = -((rewards - baseline).detach() * logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)).mean()
        reward = float(rewards.mean()); next_baseline = .95 * baseline + .05 * reward
    else:
        loss = F.cross_entropy(logits, targets); next_baseline = baseline
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite behavioral loss')
    loss.backward()
    if method not in ('bptt', 'reservoir'):
        model.assign_core_gradients(tuple(value.grad for value in state), traces)
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite behavioral gradient')
    optimizer.step()
    with torch.no_grad(): model.edge_log_gain.clamp_(-3, 3)
    return dict(objective=float(loss.detach()), supervised_cross_entropy=float(F.cross_entropy(logits.detach(), targets)),
        greedy_accuracy=float((logits.argmax(-1) == targets).float().mean()), reward=reward,
        baseline=next_baseline, gradient_norm=float(gradient), eligibility_tensor_bytes=trace_size)
