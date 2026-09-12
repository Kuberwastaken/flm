"""Independent capacity probe. Never imported by the frozen selection study."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from importlib import metadata
import math
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.autograd.function import once_differentiable

from .model import FLM, Config, load_graph
from .provenance import sha256, write_json


def full_graph(source: Path, output: Path, pools: int = 512) -> dict:
    """Retain every acquired neuron; omit only modeled zero-sign outgoing edges."""
    acquisition = json.loads((source / 'acquisition.json').read_text())
    for name, digest in acquisition['outputs'].items():
        if sha256(source / f'{name}.npy') != digest:
            raise ValueError(f'Acquisition hash mismatch: {name}')
    offsets = np.load(source / 'offsets.npy')
    col = np.load(source / 'sources.npy')
    contacts = np.load(source / 'counts.npy')
    ids = np.load(source / 'body_ids.npy')
    signs = np.load(source / 'fast_sign.npy')
    n = len(ids)
    row = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
    keep = signs[col] != 0
    kept_contacts = int(contacts[keep].sum(dtype=np.uint64))
    row, col, contacts = row[keep], col[keep].astype(np.int32), contacts[keep]
    order = np.lexsort((col, row))
    row, col, contacts = row[order], col[order], contacts[order]
    raw = np.log1p(contacts.astype(np.float32))
    sums = np.bincount(row, weights=raw, minlength=n)
    weight = (raw / np.maximum(sums[row], 1e-12) * signs[col]).astype(np.float32)
    # Stable source-order pooling is an engineering readout, not a circuit claim.
    pool = np.minimum(np.arange(n, dtype=np.int64) * pools // n, pools - 1).astype(np.int32)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'graph.npz', row=row, col=col, weight=weight,
                        body_ids=ids, source_sign=signs, pool=pool)
    card = dict(neurons=n, edges=len(row), pools=pools, source_revision=acquisition['revision'],
                source_edges=acquisition['directed_edges'], source_contacts=acquisition['contacts'],
                retained_contacts=kept_contacts, excluded_neurons=0, boundary_contacts_cut=0,
                excluded_zero_sign_edges=acquisition['directed_edges'] - len(row),
                graph_sha256=sha256(output / 'graph.npz'), source_acquisition_sha256=sha256(source / 'acquisition.json'),
                scope='All acquired neurons, not a physiologically complete brain',
                pooling='Balanced contiguous source order; not anatomical circuit pooling',
                sign_rule='Acquired modeled source sign; unknown/modulatory zero-sign outputs omitted',
                license='CC-BY-4.0', attribution='MaleCNS / Janelia; acquired via Xenova/fruit-fly-simulation')
    write_json(output / 'graph-card.json', card)
    return card


def sized_config(graph: dict, vocabulary: int, target_millions: float) -> Config:
    n, edges = len(graph['body_ids']), len(graph['row'])
    pools = int(graph['pool'].max()) + 1
    # Exact count for the inherited tied-readout FLM, rounded to tensor-friendly width.
    fixed = edges + 3 * n + 1 + 4 * pools + vocabulary
    per_width = vocabulary + n + 2 * pools + 1
    width = max(8, round((target_millions * 1e6 - fixed) / per_width / 8) * 8)
    return Config(neurons=n, pools=pools, embedding=width, vocabulary=vocabulary,
                  tied_readout=True, backend='sparse')


class EdgeMM(torch.autograd.Function):
    """First-order sparse product with an explicitly edge-sized weight gradient.

    Avoid the dense N x N weight-gradient temporary used by some sparse backend
    paths. Edge chunks bound gathered B x E temporaries. No higher derivatives.
    """
    @staticmethod
    def forward(ctx, values, indices, state):
        ctx.save_for_backward(values, indices, state)
        n = state.shape[1]
        w = torch.sparse_coo_tensor(indices, values, (n, n), is_coalesced=True)
        return torch.sparse.mm(w, state.T).T

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        values, indices, state = ctx.saved_tensors
        dvalues = torch.empty_like(values) if ctx.needs_input_grad[0] else None
        dstate = torch.zeros_like(state) if ctx.needs_input_grad[2] else None
        for start in range(0, len(values), 131072):
            stop = start + 131072
            rows, cols = indices[:, start:stop]
            gathered = grad[:, rows]
            if dvalues is not None:
                dvalues[start:stop] = (gathered * state[:, cols]).sum(0)
            if dstate is not None:
                dstate.scatter_add_(1, cols.expand(len(state), -1), gathered * values[start:stop])
        return dvalues, None, dstate


class CapacityFLM(FLM):
    """Same rate equations, cached ordered COO indices and checkpointed loss.

    Sparse recurrence/normalization stay float32. CUDA autocast is applied only
    to the dense lexical path by the caller. No learned attention or bypass.
    """
    def __init__(self, graph: dict, config: Config, chunk: int = 8, sparse_backend: str = 'csr'):
        keys = graph['row'].astype(np.int64) * config.neurons + graph['col']
        if len(keys) and (np.any(keys[1:] <= keys[:-1]) or keys[0] < 0 or keys[-1] >= config.neurons**2):
            raise ValueError('Expected unique sorted graph edges')
        if chunk < 1:
            raise ValueError('chunk must be positive')
        if sparse_backend not in ('csr', 'edge'):
            raise ValueError('Unknown sparse backend')
        super().__init__(graph, config)
        self.chunk = chunk
        self.sparse_backend = sparse_backend
        self.register_buffer('edge_indices', torch.stack((self.row, self.col)), persistent=False)
        crow = np.r_[0, np.cumsum(np.bincount(graph['row'], minlength=config.neurons))]
        self.register_buffer('crow', torch.tensor(crow, dtype=torch.long), persistent=False)

    def constants(self):
        # Avoid sorting the same graph on every forward; checked once above.
        with torch.autocast(self.input.weight.device.type, enabled=False):
            values = self.base_weight * self.edge_log_gain.clamp(-3, 3).exp()
            sums = values.new_zeros(self.config.neurons).scatter_add(0, self.row, values.abs())
            values = values / sums[self.row].clamp_min(1e-8)
            alpha = 0.05 + 0.90 * self.alpha_logit.sigmoid()
            beta = 0.002 + 0.098 * self.beta_logit.sigmoid()
            gain = 0.05 + 2.95 * self.recurrent_logit.sigmoid()
            route = torch.sparse_csr_tensor(self.crow, self.col, values, (self.config.neurons,) * 2) if self.sparse_backend == 'csr' else values
        return route, alpha, beta, gain

    def transition(self, drive, state, constants):
        with torch.autocast(drive.device.type, enabled=False):
            values, alpha, beta, gain = constants
            h, slow = state
            drive = drive.float()
            if self.config.variant != 'no_recurrence':
                incoming = torch.sparse.mm(values, h.T).T if self.sparse_backend == 'csr' else EdgeMM.apply(values, self.edge_indices, h)
                drive = drive + gain * incoming
            h = (1 - alpha) * h + alpha * torch.tanh(drive)
            slow = torch.zeros_like(slow) if self.config.variant == 'no_slow' else (1-beta) * slow + beta * h
            return h, slow

    def loss_sum(self, x, labels, checkpointed: bool = True):
        """One target per input position. No detach at checkpoint boundaries."""
        if x.shape != labels.shape:
            raise ValueError('Inputs and shifted targets must have matching shape')
        h, slow = self.initial_state(len(x))
        total = self.input.weight.new_zeros(())

        def block(tokens, targets, fast, trace):
            logits, (fast, trace) = self(tokens, (fast, trace))
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocabulary),
                                   targets.reshape(-1), ignore_index=-100, reduction='sum')
            return loss, fast, trace

        for start in range(0, x.shape[1], self.chunk):
            args = (x[:, start:start+self.chunk], labels[:, start:start+self.chunk], h, slow)
            loss, h, slow = checkpoint(block, *args, use_reentrant=False) if checkpointed and self.training else block(*args)
            total = total + loss
        return total


def environment() -> dict:
    cuda = torch.cuda.is_available()
    packages = {}
    for name in ('numpy', 'scipy', 'tokenizers', 'datasets', 'transformers'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(platform=platform.platform(), torch=torch.__version__, cuda_version=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if cuda else 'CPU',
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else None,
                fp16_dense=cuda, fp32_sparse_state=True, packages=packages)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def batch_at(examples, cursor, batch_size, length, pad, device):
    """Deterministic cyclic order; a window never crosses documents/trees."""
    x = torch.full((batch_size, length), pad, dtype=torch.long)
    y = torch.full_like(x, -100)
    for b in range(batch_size):
        record = examples[(cursor + b) % len(examples)]
        tokens, targets = record['tokens'], record['labels']
        take = min(length, len(tokens) - 1)
        if take > 0:
            x[b, :take] = torch.tensor(tokens[:take])
            y[b, :take] = torch.tensor(targets[1:take+1])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, examples, pad, length, batches=4):
    was_training = model.training
    model.eval()
    loss, count = 0., 0
    for i in range(min(batches, len(examples))):
        x, y = batch_at(examples, i, 1, length, pad, model.input.weight.device)
        n = int((y != -100).sum())
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.is_cuda):
            loss += float(model.loss_sum(x, y, checkpointed=False))
        count += n
    model.train(was_training)
    if not count:
        raise ValueError('Evaluation has no scored targets')
    return dict(nats_per_token=loss/count, perplexity=math.exp(min(50, loss/count)), scored_tokens=count,
                examples=min(batches, len(examples)), scope='Fixed small development panel; not a benchmark')


def save_checkpoint(path, model, optimizer, scaler, identity, cursor, steps, tokens):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    identity=identity, cursor=cursor, steps=steps, scored_tokens=tokens,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), temporary)
    temporary.replace(path)
    write_json(path.with_suffix('.sha256.json'), dict(sha256=sha256(path)))


def train(model, examples, validation, *, output: Path, identity: dict, pad: int,
          seconds: float = 2400, max_steps: int = 100000, batch_size: int = 1,
          accumulation: int = 4, length: int = 128, lr: float = 3e-4, resume: bool = False):
    """Budget includes probe/validation/saving after entry, excluding data/setup.

    Soft deadline checked between optimizer updates; an active update/save cannot
    be preempted. Every run probes three real updates before continuing.
    """
    if seconds <= 0 or max_steps < 1 or min(batch_size, accumulation, length) < 1:
        raise ValueError('Invalid training budget')
    if not examples or not validation:
        raise ValueError('Training and validation must both be nonempty')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'last.pt'
    started = time.monotonic()
    cuda = model.input.weight.is_cuda
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda, init_scale=128.)
    binding = dict(experiment=identity, model=asdict(model.config), chunk=model.chunk, sparse_backend=model.sparse_backend,
                   batch=batch_size, accumulation=accumulation, length=length, lr=lr,
                   environment=environment(), runtime_sha256=sha256(Path(__file__)),
                   source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                                  ('model.py', 'colab_data.py', 'colab_reference.py')})
    cursor, steps, tokens = 0, 0, 0
    if resume:
        digest = json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
        if sha256(path) != digest:
            raise ValueError('Checkpoint checksum mismatch')
        # Only load our own saved training artifact with the matching receipt.
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['identity'] != binding:
            raise ValueError('Resume settings/data/runtime differ; use a new run directory')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        cursor, steps, tokens = saved['cursor'], saved['steps'], saved['scored_tokens']
        torch.set_rng_state(saved['rng'])
        if cuda:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    elif path.exists():
        raise ValueError('Run exists: set resume=True or select a new output directory')
    baseline = evaluate(model, validation, pad, length, batches=2)
    model.train()
    durations, records = [], []
    initial_steps, initial_tokens = steps, tokens
    checkpoint_time = 0.
    last_save = time.monotonic()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    while steps - initial_steps < max_steps:
        reserve = max(15., 2 * checkpoint_time)
        estimate = max(durations[-3:], default=0.)
        if time.monotonic() - started + reserve + estimate >= seconds:
            break
        batches = [batch_at(examples, cursor + b*batch_size, batch_size, length, pad, model.input.weight.device)
                   for b in range(accumulation)]
        count = sum(int((y != -100).sum()) for _, y in batches)
        if not count:
            raise ValueError('Batch contains no scored targets')
        sync(); step_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = lr * min(1., (steps + 1) / 10.)
        loss_value = 0.
        for x, y in batches:
            with torch.autocast(x.device.type, dtype=torch.float16, enabled=cuda):
                loss = model.loss_sum(x, y) / count
            loss_value += float(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
        if not torch.isfinite(grad_norm) or not math.isfinite(loss_value):
            # Do not silently count skipped AMP updates as trained steps.
            raise FloatingPointError('Nonfinite update; stop and retain the previous checkpoint')
        scaler.step(optimizer); scaler.update()
        sync(); duration = time.monotonic() - step_start
        durations.append(duration)
        steps += 1; tokens += count; cursor += batch_size * accumulation
        row = dict(step=steps, loss=loss_value, scored_tokens=count, input_tokens=sum(int((x != pad).sum()) for x, _ in batches),
                   learning_rate=optimizer.param_groups[0]['lr'], seconds=duration,
                   tokens_per_second=count/duration, elapsed_seconds=time.monotonic()-started)
        records.append(row)
        if len(records) <= 3 or steps % 10 == 0:
            print(json.dumps(row), flush=True)
        if len(records) == 3 or time.monotonic() - last_save >= 300:
            save_start = time.monotonic()
            save_checkpoint(path, model, optimizer, scaler, binding, cursor, steps, tokens)
            checkpoint_time = time.monotonic() - save_start; last_save = time.monotonic()
            write_json(output / 'progress.json', dict(binding=binding, steps=steps, scored_tokens=tokens,
                        recent=records[-10:], checkpoint_seconds=checkpoint_time))
    final = evaluate(model, validation, pad, length, batches=2)
    save_checkpoint(path, model, optimizer, scaler, binding, cursor, steps, tokens)
    report = dict(status='completed_budget' if records else 'no_updates_within_budget', binding=binding,
                  parameters=model.parameter_card(), environment=environment(), initial_dev=baseline,
                  final_dev=final, steps_this_session=steps-initial_steps, scored_tokens_this_session=tokens-initial_tokens,
                  total_steps=steps, total_scored_tokens=tokens, elapsed_seconds=time.monotonic()-started,
                  budget_seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None, updates=records,
                  gpu_fit_verified=cuda and bool(records), quality_claim='None: engineering pilot only')
    write_json(output / 'report.json', report)
    return report
