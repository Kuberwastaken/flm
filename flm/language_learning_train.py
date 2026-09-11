"""Resumable training-only runner for four FLM credit-assignment conditions.

Supplied data and bindings are recorded, not acquired here. This module chooses
no official dataset, training budget, validation selection or test evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .inference import state_hash
from .language_eligibility import language_window_gradients
from .local_learning import CORE_PARAMETERS
from .model import FLM
from .provenance import sha256, write_json
from .train import Sampler

METHODS = ('bptt', 'fixed_core', 'eligibility', 'no_history')


@dataclass(frozen=True)
class Settings:
    steps: int
    batch: int
    sequence: int
    warmup: int
    learning_rate: float
    final_learning_rate: float
    lr_warmup_updates: int
    weight_decay: float
    gradient_clip: float
    checkpoint_interval: int
    seed: int
    threads: int
    score_boundaries: bool = False

    def validate(self):
        for value in (self.steps, self.batch, self.sequence, self.checkpoint_interval, self.threads):
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer training dimensions required')
        if (type(self.seed) is not int or not 0 <= self.seed < 2**32 or
                type(self.warmup) is not int or not 0 <= self.warmup < self.sequence or
                type(self.lr_warmup_updates) is not int or not 0 <= self.lr_warmup_updates < self.steps or
                self.steps % self.checkpoint_interval or type(self.score_boundaries) is not bool):
            raise ValueError('Invalid training seed, masks or update schedule')
        numbers = (self.learning_rate, self.final_learning_rate, self.weight_decay, self.gradient_clip)
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in numbers)
                or not 0 <= self.final_learning_rate <= self.learning_rate or self.learning_rate <= 0
                or self.weight_decay < 0 or self.gradient_clip <= 0):
            raise ValueError('Invalid optimizer settings')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')


def learning_rate(settings, step):
    progress = max(0., (step - settings.lr_warmup_updates) / (settings.steps - settings.lr_warmup_updates))
    rate = settings.final_learning_rate + (settings.learning_rate-settings.final_learning_rate) * .5 * (1+math.cos(math.pi*min(progress, 1.)))
    return rate * min(step/max(1, settings.lr_warmup_updates), 1.)


def configure(model, method):
    if method not in METHODS or type(model) is not FLM or model.config.variant != 'flm':
        raise ValueError('Choose a registered learning method and an ordinary full FLM')
    if model.config.backend == 'sparse' or model.config.neurons > 2048:
        raise ValueError('All conditions currently require the same compact dense core')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(method != 'fixed_core' or name not in CORE_PARAMETERS)
        parameter.grad = None


def names(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def optimizer_for(model, settings):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=settings.learning_rate, weight_decay=settings.weight_decay)


def score_mask(targets, settings):
    mask = torch.ones_like(targets, dtype=torch.bool) if settings.score_boundaries else targets >= 2
    mask[:, :settings.warmup] = False
    if not mask.any():
        raise ValueError('Sampled window has no scored target; do not silently resample')
    return mask


def update(model, optimizer, x, y, settings, method, step):
    if method not in METHODS:
        raise ValueError('Unknown learning method')
    expected = [name for name, _ in model.named_parameters() if method != 'fixed_core' or name not in CORE_PARAMETERS]
    if names(model) != expected:
        raise ValueError('Condition trainable-parameter inventory changed')
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(settings, step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    mask = score_mask(y, settings)
    trace_size = 0
    if method in ('eligibility', 'no_history'):
        result = language_window_gradients(model, x, y, mask, history=method == 'eligibility')
        loss = result.loss
        trace_size = result.persistent_trace_bytes
        for name, p in model.named_parameters():
            p.grad = result.gradients[name]
    else:
        logits, _ = model(x)
        losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').reshape_as(y)
        objective = losses[mask].mean()
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite training loss')
        objective.backward()
        loss = float(objective.detach())
    gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings.gradient_clip)
    if not math.isfinite(loss) or not torch.isfinite(gradient):
        raise FloatingPointError('Nonfinite training loss or gradient')
    optimizer.step()
    # FLM.constants applies a differentiable forward clamp. Do not project the
    # parameter after AdamW; all four conditions share the existing convention.
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite updated weights')
    return dict(step=step, loss=loss, gradient_norm=float(gradient),
                learning_rate=learning_rate(settings, step), persistent_trace_bytes=trace_size)


def declaration(model, documents, lexicon, settings, method, binding):
    settings.validate()
    if not isinstance(binding, dict) or not binding:
        raise ValueError('An explicit study/data binding is required')
    if (not isinstance(lexicon.sha256, str) or len(lexicon.sha256) != 64 or
            any(x not in '0123456789abcdef' for x in lexicon.sha256)):
        raise ValueError('Tokenizer SHA-256 required')
    if (lexicon.vocabulary != model.config.vocabulary or
            np.asarray(lexicon.lengths).shape != (lexicon.vocabulary,) or
            not np.issubdtype(np.asarray(lexicon.lengths).dtype, np.integer) or
            np.any(np.asarray(lexicon.lengths) < 0)):
        raise ValueError('Tokenizer vocabulary and byte lengths must agree')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('The fitting runner currently supports CPU only')
    if any(not torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError('Finite initial model required')
    if not documents or len({identity for identity, _ in documents}) != len(documents):
        raise ValueError('Unique nonempty training document inventory required')
    digest = hashlib.sha256()
    for identity, array in documents:
        if (not isinstance(identity, str) or not identity or not isinstance(array, np.ndarray)
                or array.ndim != 1 or not len(array) or not np.issubdtype(array.dtype, np.integer)
                or int(array.min()) < 0 or int(array.max()) >= lexicon.vocabulary):
            raise ValueError('Invalid training document')
        for part in (canonical([identity, len(array)]), array.astype('<i8', copy=False).tobytes()):
            digest.update(len(part).to_bytes(8, 'little')); digest.update(part)
    # Check sampling eligibility before opening a run directory.
    sampler = Sampler(documents, settings.seed, settings.sequence)
    sources = ('language_learning_train.py', 'language_eligibility.py', 'embedding_eligibility.py',
               'local_learning.py', 'model.py', 'train.py', 'inference.py', 'provenance.py')
    parameters = dict(model.named_parameters())
    return dict(format='flm-language-learning-train-v1', method=method,
        binding=json.loads(canonical(binding)), settings=asdict(settings), config=asdict(model.config),
        initial_state_sha256=state_hash(model), tokenizer_sha256=lexicon.sha256,
        byte_lengths_sha256=hashlib.sha256(np.asarray(lexicon.lengths, dtype='<i8').tobytes()).hexdigest(),
        ordered_training_documents_sha256=digest.hexdigest(), training_documents=len(documents),
        eligible_training_documents=len(sampler.documents), optimizer_parameter_names=names(model),
        frozen_parameter_names=[name for name, p in parameters.items() if not p.requires_grad],
        trainable_parameters=sum(p.numel() for p in parameters.values() if p.requires_grad),
        torch=str(torch.__version__), numpy=str(np.__version__),
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in sources})


def add_exposure(exposure, digest, x, y, lengths, settings):
    mask = score_mask(y, settings)
    exposure['presented_tokens'] += x.numel()
    exposure['scored_tokens'] += int(mask.sum())
    exposure['presented_bytes'] += int(lengths[x.numpy()].sum())
    exposure['scored_bytes'] += int(lengths[y[mask].numpy()].sum())
    digest.update(x.numpy().astype('<i8', copy=False).tobytes())
    digest.update(y.numpy().astype('<i8', copy=False).tobytes())


def replay(documents, lengths, settings, steps):
    sampler = Sampler(documents, settings.seed, settings.sequence)
    exposure = dict(presented_tokens=0, scored_tokens=0, presented_bytes=0, scored_bytes=0)
    digest = hashlib.sha256()
    for _ in range(steps):
        x, y = sampler.sample(settings.batch, 'cpu')
        add_exposure(exposure, digest, x, y, lengths, settings)
    return sampler, exposure, digest


@contextmanager
def training_lease(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'writer.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another language-learning writer holds this directory') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def inventory(directory, declared):
    committed = []
    for index, path in enumerate(sorted(directory.glob('saved-*.json')), 1):
        step = index * declared['settings']['checkpoint_interval']
        record = json.loads(path.read_text(encoding='utf8'))
        filename = f'checkpoint-{step:06d}.pt'
        if (step > declared['settings']['steps'] or path.name != f'saved-{step:06d}.json'
                or record.get('step') != step or record.get('declaration') != declared
                or record.get('checkpoint') != filename or record.get('checkpoint_sha256') != sha256(directory/filename)):
            raise ValueError('Committed checkpoint or declaration changed')
        committed.append(record)
    return committed


def restore(path, model, optimizer, declared, documents, lengths, settings):
    saved = torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location='cpu')
    if saved['declaration'] != declared:
        raise ValueError('Saved declaration changed')
    step = saved['step']
    if type(step) is not int or not 0 < step <= settings.steps or step % settings.checkpoint_interval:
        raise ValueError('Invalid saved update')
    expected = model.state_dict(); actual = saved['model']
    if actual.keys() != expected.keys() or any(x.shape != expected[k].shape or x.dtype != expected[k].dtype
            or not torch.isfinite(x).all() for k, x in actual.items()):
        raise ValueError('Invalid saved model tensors')
    immutable = set(dict(model.named_buffers())) | set(declared['frozen_parameter_names'])
    if any(not torch.equal(expected[name], actual[name]) for name in immutable):
        raise ValueError('Saved graph, pooling or frozen core changed')
    sampler, exposure, digest = replay(documents, lengths, settings, step)
    if (saved['sampler_rng'] != sampler.rng.bit_generator.state or saved['exposure'] != exposure
            or saved['sampled_token_sha256'] != digest.hexdigest()):
        raise ValueError('Saved sampled tokens, RNG or exposure changed')
    history = saved['history']
    if len(history) != step or any(row['step'] != i or row['learning_rate'] != learning_rate(settings, i)
            or any(not math.isfinite(row[key]) or row[key] < 0 for key in ('loss', 'gradient_norm', 'persistent_trace_bytes'))
            for i, row in enumerate(history, 1)):
        raise ValueError('Saved training history changed')
    groups = optimizer.state_dict()['param_groups']
    groups[0]['lr'] = learning_rate(settings, step)
    stored = saved['optimizer']
    if stored['param_groups'] != groups or set(stored['state']) != set(groups[0]['params']):
        raise ValueError('Saved optimizer inventory or settings changed')
    trainable = [p for p in model.parameters() if p.requires_grad]
    for index, p in enumerate(trainable):
        state = stored['state'][index]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != step:
            raise ValueError('Saved optimizer steps changed')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            if value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid saved optimizer moments')
    generator = torch.Generator()
    generator.set_state(saved['torch_rng'])
    model.load_state_dict(actual); optimizer.load_state_dict(stored)
    torch.set_rng_state(saved['torch_rng'])
    return saved, sampler, digest


def fit(model, documents, lexicon, settings, method, binding, directory, *, until=None):
    """Fit or resume supplied training data; provide a fresh initial model.

    All arguments and source identities remain bound across continuation. A
    bounded stop preserves the full learning-rate schedule. Only a checkpoint
    with its final atomic commit record can resume; other payloads are ignored.
    """
    settings.validate()
    target = settings.steps if until is None else until
    if type(target) is not int or not 0 < target <= settings.steps or target % settings.checkpoint_interval:
        raise ValueError('Stop only at a declared checkpoint boundary')
    configure(model, method)
    declared = declaration(model, documents, lexicon, settings, method, binding)
    directory = Path(directory)
    with training_lease(directory):
        declared_path = directory/'declaration.json'
        if declared_path.exists() and json.loads(declared_path.read_text(encoding='utf8')) != declared:
            raise ValueError('Run directory declaration changed')
        if not declared_path.exists():
            write_json(declared_path, declared)
        committed = inventory(directory, declared)
        torch.set_num_threads(settings.threads); torch.manual_seed(settings.seed)
        optimizer = optimizer_for(model, settings)
        lengths = np.asarray(lexicon.lengths, dtype=np.int64)
        sampler, exposure, digest = replay(documents, lengths, settings, 0)
        start = 0; history = []
        if committed:
            saved, sampler, digest = restore(directory/committed[-1]['checkpoint'], model, optimizer,
                                             declared, documents, lengths, settings)
            start, history, exposure = saved['step'], saved['history'], saved['exposure']
            if start > target:
                raise ValueError('Saved update exceeds requested stop')
        for step in range(start + 1, target + 1):
            x, y = sampler.sample(settings.batch, 'cpu')
            history.append(update(model, optimizer, x, y, settings, method, step))
            add_exposure(exposure, digest, x, y, lengths, settings)
            if step % settings.checkpoint_interval == 0:
                path = directory/f'checkpoint-{step:06d}.pt'
                marker = directory/f'saved-{step:06d}.json'
                if marker.exists():
                    raise ValueError('Refusing to overwrite a committed checkpoint')
                if any(not torch.isfinite(v).all() for s in optimizer.state.values()
                       for v in s.values() if isinstance(v, torch.Tensor)):
                    raise FloatingPointError('Nonfinite optimizer state cannot be saved')
                temporary = path.with_suffix('.pt.tmp')
                torch.save(dict(declaration=declared, step=step, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), sampler_rng=sampler.rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), exposure=exposure,
                    sampled_token_sha256=digest.hexdigest(), history=history), temporary)
                temporary.replace(path)
                write_json(marker, dict(step=step, checkpoint=path.name, checkpoint_sha256=sha256(path), declaration=declared))
        final = inventory(directory, declared)[-1]
        result = dict(declaration=declared, step=target, checkpoint=final['checkpoint'],
            checkpoint_sha256=final['checkpoint_sha256'], exposure=exposure,
            selection='Fixed training endpoint only; no validation or test selection', complete=target == settings.steps)
        complete = directory/'complete.json'
        if complete.exists() and json.loads(complete.read_text(encoding='utf8')) != result:
            raise ValueError('Completed run changed')
        if result['complete'] and not complete.exists():
            write_json(complete, result)
        return result
