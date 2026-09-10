"""Resumable CPU fitting primitives for a later declared instruction study.

No corpus loading, pretrained-model selection, evaluation or automatic launch.
Callers must establish official partitions and study/source identities first.
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

from .baselines import GRU, Transformer
from .inference import state_hash
from .model import FLM
from .provenance import sha256, write_json
from .scan_runtime import batch_loss, instruction_batch
from .scan_task import encode_example


@dataclass(frozen=True)
class Settings:
    steps: int
    batch_size: int
    learning_rate: float
    final_learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    checkpoint_interval: int
    sampling_seed: int
    threads: int
    context_limit: int = 96

    def validate(self):
        integers = (self.steps, self.batch_size, self.checkpoint_interval, self.threads, self.context_limit)
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError('Positive integer training dimensions are required')
        if (type(self.sampling_seed) is not int or not 0 <= self.sampling_seed < 2**32 or
                type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps or
                self.steps % self.checkpoint_interval):
            raise ValueError('Invalid sampler seed, warmup or checkpoint schedule')
        rates = (self.learning_rate, self.final_learning_rate, self.weight_decay, self.gradient_clip)
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in rates) or
                not 0 <= self.final_learning_rate <= self.learning_rate or self.learning_rate <= 0 or
                self.weight_decay < 0 or self.gradient_clip <= 0):
            raise ValueError('Invalid optimizer settings')


def rate(settings, step):
    progress = max(0., (step-settings.warmup_steps) / (settings.steps-settings.warmup_steps))
    decay = settings.final_learning_rate + (settings.learning_rate-settings.final_learning_rate) * .5 * (1 + math.cos(math.pi*min(progress, 1.)))
    return decay * min(1., step/max(1, settings.warmup_steps))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')


@contextmanager
def training_lease(directory):
    """The OS-held lock, not the presence of a file, identifies an active writer."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'writer.lock').open('a+b') as handle:
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
            raise RuntimeError('Another instruction-training writer holds this directory') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def declaration(model, records, lexicon, settings, binding):
    settings.validate()
    if not isinstance(binding, dict) or not binding or not isinstance(records, (list, tuple)) or not records:
        raise ValueError('An explicit study binding and nonempty training rows are required')
    if (not isinstance(lexicon.sha256, str) or len(lexicon.sha256) != 64 or
            any(character not in '0123456789abcdef' for character in lexicon.sha256)):
        raise ValueError('A tokenizer identity is required')
    expected_variant = {FLM: 'flm', GRU: 'gru', Transformer: 'transformer'}.get(type(model))
    if (expected_variant is None or model.config.variant != expected_variant or
            any(p.device.type != 'cpu' or not p.requires_grad for p in model.parameters()) or
            model.config.vocabulary != lexicon.vocabulary):
        raise ValueError('The declared full-parameter CPU model and tokenizer must agree')
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Initial model tensors must be finite')
    examples = [encode_example(row, lexicon) for row in records]
    lengths = np.array([len(row['input']) for row in examples], dtype=np.int64)
    scored = np.array([row['supervised_tokens'] for row in examples], dtype=np.int64)
    limit = min(settings.context_limit, getattr(model.config, 'window', settings.context_limit))
    if int(lengths.max()) > limit:
        raise ValueError('A complete training example exceeds the common model context')
    sources = ('scan_train.py', 'scan_runtime.py', 'scan_task.py', 'scan.py', 'model.py', 'baselines.py',
               'inference.py', 'provenance.py', 'tokenizer.py')
    record = dict(format='flm-instruction-training-v1', binding=json.loads(canonical(binding)), settings=asdict(settings),
        model_type=type(model).__module__ + '.' + type(model).__qualname__,
        model_config=asdict(model.config), initial_state_sha256=state_hash(model),
        training_rows_sha256=hashlib.sha256(canonical(records)).hexdigest(), training_rows=len(records),
        tokenizer_sha256=lexicon.sha256, torch=str(torch.__version__), numpy=str(np.__version__),
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in sources})
    return record, lengths, scored


def sampling_state(settings, lengths, scored, step):
    rng = np.random.Generator(np.random.PCG64(settings.sampling_seed))
    digest = hashlib.sha256()
    exposure = dict(examples=0, input_tokens=0, supervised_tokens=0)
    for _ in range(step):
        indices = rng.integers(0, len(lengths), size=settings.batch_size, dtype=np.int64)
        digest.update(indices.astype('<i8', copy=False).tobytes())
        exposure['examples'] += settings.batch_size
        exposure['input_tokens'] += int(lengths[indices].sum())
        exposure['supervised_tokens'] += int(scored[indices].sum())
    return rng.bit_generator.state, exposure, digest.hexdigest()


def inventory(directory, declared):
    settings = declared['settings']; records = []
    for index, path in enumerate(sorted(directory.glob('saved-*.json')), 1):
        step = index * settings['checkpoint_interval']
        record = json.loads(path.read_text(encoding='utf8'))
        expected_name = f'checkpoint-{step:06d}.pt'
        if (step > settings['steps'] or path.name != f'saved-{step:06d}.json' or
                record.get('step') != step or record.get('declaration') != declared or
                record.get('checkpoint') != expected_name or
                record.get('checkpoint_sha256') != sha256(directory / expected_name)):
            raise ValueError('Committed instruction checkpoint or declaration changed')
        records.append(record)
    return records


def optimizer_for(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)


def restore_payload(path, model, optimizer, declared, lengths, scored):
    payload = torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location='cpu')
    if payload['declaration'] != declared:
        raise ValueError('Checkpoint training declaration changed')
    expected = model.state_dict(); actual = payload['model']
    if actual.keys() != expected.keys() or any(value.shape != expected[key].shape or value.dtype != expected[key].dtype
            or not torch.isfinite(value).all() for key, value in actual.items()):
        raise ValueError('Checkpoint tensor inventory, shape, dtype or finiteness changed')
    for name, value in model.named_buffers():
        if not torch.equal(actual[name], value):
            raise ValueError('Checkpoint graph or pooling buffer changed')
    settings = Settings(**declared['settings']); step = payload['step']
    if type(step) is not int or not 0 < step <= settings.steps or step % settings.checkpoint_interval:
        raise ValueError('Checkpoint update count changed')
    rng, exposure, digest = sampling_state(settings, lengths, scored, step)
    if payload['sampler_rng'] != rng or payload['exposure'] != exposure or payload['sampled_row_sha256'] != digest:
        raise ValueError('Checkpoint sampled rows, RNG or exact exposure changed')
    if (len(payload['history']) != step or any(row['step'] != index for index, row in enumerate(payload['history'], 1)) or
            any(not math.isfinite(row['loss']) or row['loss'] < 0 or not math.isfinite(row['gradient_norm'])
                or row['gradient_norm'] < 0 for row in payload['history'])):
        raise ValueError('Checkpoint training history is incomplete or invalid')
    expected_groups = optimizer.state_dict()['param_groups']
    expected_groups[0]['lr'] = rate(settings, step)
    saved_optimizer = payload['optimizer']
    if saved_optimizer['param_groups'] != expected_groups or set(saved_optimizer['state']) != set(expected_groups[0]['params']):
        raise ValueError('Checkpoint optimizer settings or moment inventory changed')
    for index, parameter in enumerate(model.parameters()):
        state = saved_optimizer['state'][index]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != step:
            raise ValueError('Checkpoint optimizer step or state fields changed')
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = state[name]
            if moment.shape != parameter.shape or moment.dtype != parameter.dtype or not torch.isfinite(moment).all():
                raise ValueError('Checkpoint optimizer moment shape, dtype or finiteness changed')
    model.load_state_dict(actual); optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(payload['torch_rng'])
    return payload


def save(directory, model, optimizer, declared, sampler, step, exposure, history, lengths, scored):
    marker = directory / f'saved-{step:06d}.json'
    if marker.exists():
        raise ValueError('Refusing to overwrite a committed instruction checkpoint')
    expected_rng, expected_exposure, digest = sampling_state(Settings(**declared['settings']), lengths, scored, step)
    if sampler.bit_generator.state != expected_rng or exposure != expected_exposure:
        raise ValueError('Live sampled rows or exposure differ from the declared stream')
    tensors = list(model.state_dict().values())
    tensors += [value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)]
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError('Nonfinite model or optimizer state cannot be committed')
    checkpoint = directory / f'checkpoint-{step:06d}.pt'
    temporary = checkpoint.with_suffix('.pt.tmp')
    torch.save(dict(declaration=declared, model=model.state_dict(), optimizer=optimizer.state_dict(),
        sampler_rng=sampler.bit_generator.state, torch_rng=torch.get_rng_state(), step=step,
        exposure=exposure, sampled_row_sha256=digest, history=history), temporary)
    temporary.replace(checkpoint)
    write_json(marker, dict(step=step, checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint), declaration=declared))


def fit(model, records, lexicon, settings, binding, directory, *, until=None):
    """Fit or resume one supplied condition; a fresh initial model is required.

    A bounded `until` retains the full declared learning-rate schedule and writes
    no completion record. No uncommitted checkpoint payload is eligible to resume.
    Official test evaluation and cross-condition completion gates are separate.
    """
    settings.validate()
    target = settings.steps if until is None else until
    if type(target) is not int or not 0 < target <= settings.steps or target % settings.checkpoint_interval:
        raise ValueError('A bounded fit must stop at a declared checkpoint boundary')
    torch.set_num_threads(settings.threads)
    torch.manual_seed(settings.sampling_seed)
    declared, lengths, scored = declaration(model, records, lexicon, settings, binding)
    directory = Path(directory)
    with training_lease(directory):
        committed = inventory(directory, declared)
        optimizer = optimizer_for(model, settings)
        sampler = np.random.Generator(np.random.PCG64(settings.sampling_seed))
        start = 0; history = []; exposure = dict(examples=0, input_tokens=0, supervised_tokens=0)
        if committed:
            latest = committed[-1]
            saved = restore_payload(directory / latest['checkpoint'], model, optimizer, declared, lengths, scored)
            if saved['step'] != latest['step'] or saved['step'] > target:
                raise ValueError('Saved update exceeds the requested training boundary')
            start = saved['step']; history = saved['history']; exposure = saved['exposure']
            sampler.bit_generator.state = saved['sampler_rng']
        for step in range(start + 1, target + 1):
            indices = sampler.integers(0, len(records), size=settings.batch_size, dtype=np.int64)
            batch = instruction_batch([records[index] for index in indices], lexicon, context_limit=settings.context_limit)
            model.train(); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group['lr'] = rate(settings, step)
            loss = batch_loss(model, batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite instruction training loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite instruction gradient')
            optimizer.step()
            exposure['examples'] += batch['examples']; exposure['input_tokens'] += batch['input_tokens']
            exposure['supervised_tokens'] += batch['supervised_tokens']
            history.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(gradient)))
            if step % settings.checkpoint_interval == 0:
                save(directory, model, optimizer, declared, sampler, step, exposure, history, lengths, scored)
        committed = inventory(directory, declared)
        result = dict(declaration=declared, step=target, checkpoint=committed[-1]['checkpoint'],
            checkpoint_sha256=committed[-1]['checkpoint_sha256'], exposure=exposure,
            selection='Fixed terminal update; no validation or test selection', complete=target == settings.steps)
        if result['complete']:
            complete = directory / 'complete.json'
            if complete.exists() and json.loads(complete.read_text(encoding='utf8')) != result:
                raise ValueError('Completed instruction condition changed')
            if not complete.exists(): write_json(complete, result)
        return result
