"""Fixed-drive dynamics probes for original and acutely intervened FLM cores."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F


CORE_GROUPS = ('edge_log_gain', 'alpha_logit', 'beta_logit', 'recurrent_logit')
SWAPS = {'trained': CORE_GROUPS, 'edges_gain_only': ('edge_log_gain', 'recurrent_logit'),
         'time_constants_only': ('alpha_logit', 'beta_logit')}
SNAPSHOT_TIMES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def core_hash(model):
    digest = hashlib.sha256()
    for name in CORE_GROUPS:
        value = model.get_parameter(name).detach().cpu().contiguous().numpy()
        for payload in (json.dumps([name, list(value.shape), value.dtype.str]).encode(), value.tobytes()):
            digest.update(len(payload).to_bytes(8, 'little')); digest.update(payload)
    return digest.hexdigest()


def intervene(initial, trained, mode):
    if mode not in SWAPS or initial.config != trained.config:
        raise ValueError('Matching configurations and a declared dynamics intervention are required')
    for name, value in initial.named_buffers():
        if not torch.equal(value, dict(trained.named_buffers())[name]):
            raise ValueError('Intervention graph/pooling buffer changed: ' + name)
    result = copy.deepcopy(initial)
    with torch.no_grad():
        for name in SWAPS[mode]:
            source = trained.get_parameter(name)
            if source.shape != result.get_parameter(name).shape or not torch.isfinite(source).all():
                raise ValueError('Invalid source dynamics tensor: ' + name)
            result.get_parameter(name).copy_(source)
    return result.eval()


def pulse_directions(neurons, count=8, seed=91011):
    if min(neurons, count) < 1:
        raise ValueError('Pulse directions require positive dimensions')
    return (np.random.Generator(np.random.PCG64(seed)).integers(0, 2, (count, neurons), dtype=np.int8) * 2 - 1).astype(np.float32)


def linearization(constants):
    w, alpha, beta, gain = constants
    w = w.to_dense() if w.is_sparse else w
    w, alpha, beta, gain = (value.detach().to(dtype=torch.float64, device='cpu') for value in (w, alpha, beta, gain))
    a = torch.diag(1 - alpha) + gain * alpha[:, None] * w
    return a, alpha, beta


def spectrum(constants):
    a, alpha, beta = linearization(constants)
    values, vectors = torch.linalg.eig(a)
    if not torch.isfinite(values).all() or not torch.isfinite(vectors).all():
        raise ValueError('Nonfinite fast-state eigensystem')
    index = int(values.abs().argmax()); value = values[index]; vector = vectors[:, index]
    residual = torch.linalg.vector_norm(a.to(torch.complex128) @ vector - value * vector)
    residual /= (torch.linalg.matrix_norm(a, ord='fro') + value.abs()) * torch.linalg.vector_norm(vector)
    if not torch.isfinite(residual) or float(residual) > 1e-10:
        raise ValueError('Leading zero-state eigenpair failed its residual check')
    fast = float(values.abs().max()); slow = float((1 - beta).max())
    return dict(fast_spectral_radius=fast, slow_diagonal_spectral_radius=slow,
                joint_spectral_radius=max(fast, slow), leading_relative_residual=float(residual)), values.numpy()


def rms(state):
    fast, slow = (value.detach().double() for value in state)
    first = fast.square().mean(dim=-1); second = slow.square().mean(dim=-1)
    return torch.stack((first.sqrt(), second.sqrt(), ((first + second) / 2).sqrt()), dim=-1)


def response_summary(curves, amplitude):
    # curves: [time, direction, fast/slow/joint RMS]. Summaries are per direction.
    result = []
    for probe in range(curves.shape[1]):
        joint = curves[:, probe, 2]
        peak = int(joint.argmax()); energy = joint ** 2
        total = float(energy.sum())
        if total <= 0 or not math.isfinite(total):
            raise ValueError('A nonzero finite pulse response is required')
        below = np.maximum.accumulate(joint[::-1])[::-1] <= .01 * joint[peak]
        crossings = np.flatnonzero(below & (np.arange(len(joint)) >= peak))
        result.append(dict(probe=probe, peak_joint_rms=float(joint[peak]), peak_time=peak,
            final_joint_rms=float(joint[-1]), normalized_response_energy=total / amplitude ** 2,
            energy_fraction_at_64_plus=float(energy[64:].sum()) / total,
            observed_one_percent_settling_time=int(crossings[0]) if len(crossings) else None))
    return result


@torch.no_grad()
def pulse_response(model, directions, amplitude, silence=256, snapshot_times=SNAPSHOT_TIMES):
    if model.config.variant != 'flm' or model.input.weight.device.type != 'cpu':
        raise ValueError('This diagnostic is declared for full fast/slow FLM dynamics on CPU')
    if (not math.isfinite(amplitude) or amplitude <= 0 or silence < 0 or
            directions.ndim != 2 or directions.shape[1] != model.config.neurons or
            not np.isfinite(directions).all() or not snapshot_times or
            tuple(snapshot_times) != tuple(sorted(set(snapshot_times))) or
            not set(snapshot_times).issubset(set(range(silence + 1)))):
        raise ValueError('Invalid pulse or observation schedule')
    pulse = torch.as_tensor(directions, dtype=model.input.weight.dtype, device=model.input.weight.device) * amplitude
    constants = model.constants()
    state = model.initial_state(len(directions))
    zeros = torch.zeros_like(pulse)
    a, alpha, beta = linearization(constants)
    linear = (torch.zeros_like(pulse, dtype=torch.float64), torch.zeros_like(pulse, dtype=torch.float64))
    curves = []; deviations = []; snapshots = []
    for time in range(silence + 1):
        drive = pulse if time == 0 else zeros
        state = model.transition(drive, state, constants)
        h = F.linear(linear[0], a) + alpha * drive.double()
        linear = h, (1 - beta) * linear[1] + beta * h
        if any(not torch.isfinite(value).all() or float(value.abs().max()) > 1 + 1e-6 for value in state):
            raise ValueError('Nonlinear pulse response is nonfinite or outside the state bound')
        curves.append(rms(state).cpu().numpy())
        deviation = rms(tuple(value.double() - reference for value, reference in zip(state, linear)))[:, 2] / amplitude
        if not torch.isfinite(deviation).all():
            raise ValueError('Nonfinite zero-state linear response comparison')
        deviations.append(deviation.cpu().numpy())
        if time in snapshot_times:
            snapshots.append(torch.stack([value[0] for value in state]).cpu().numpy())
    curves = np.stack(curves)
    return dict(curves=curves, normalized_linear_deviation=np.stack(deviations),
                snapshot_times=np.array(snapshot_times, dtype=np.int64), snapshots=np.stack(snapshots),
                summary=response_summary(curves, amplitude))


def parameter_summary(model):
    _, alpha, beta, gain = model.constants()
    def describe(value):
        array = value.detach().cpu().double().numpy()
        return dict(minimum=float(array.min()), median=float(np.median(array)), maximum=float(array.max()))
    return dict(alpha=describe(alpha), beta=describe(beta), recurrent_gain=float(gain.detach()),
                bare_fast_leak_half_life=describe(math.log(.5) / torch.log1p(-alpha.double())),
                bare_slow_leak_half_life=describe(math.log(.5) / torch.log1p(-beta.double())))
