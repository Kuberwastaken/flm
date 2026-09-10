"""Resumable language computation controls with committed checkpoint records.

This harness reuses the frozen language loss/evaluator without modifying it.
Only checkpoint records committed last are eligible for resume or selection.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .language_core_controls import construct
from .language_train import evaluate
from .provenance import sha256, write_json
from .train import Sampler, save_checkpoint


def learning_rate(protocol, step):
    warmup = protocol['lr_warmup_updates']
    progress = max(0., (step - warmup) / max(1, protocol['steps'] - warmup))
    return (protocol['final_learning_rate'] +
            (protocol['learning_rate'] - protocol['final_learning_rate']) * .5 *
            (1 + math.cos(math.pi * min(progress, 1.)))) * min(step / warmup, 1.)


def optimizer_names(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def optimizer_for(model, protocol):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])


def update(model, optimizer, x, y, protocol, step):
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(protocol, step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x)
    warmup = protocol['warmup']
    loss = F.cross_entropy(logits[:, warmup:].reshape(-1, model.config.vocabulary),
                           y[:, warmup:].reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite loss at update {step}')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(gradient):
        raise FloatingPointError(f'Nonfinite gradient at update {step}')
    optimizer.step()
    return float(loss.detach()), float(gradient)


def verify_tensors(model, values):
    expected = model.state_dict()
    if values.keys() != expected.keys():
        raise ValueError('Checkpoint model inventory changed')
    for name, value in values.items():
        initial = expected[name]
        if value.dtype != initial.dtype or value.shape != initial.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid checkpoint tensor: ' + name)
    for name, value in model.named_buffers():
        if not torch.equal(values[name], value):
            raise ValueError('Checkpoint graph/pooling buffer changed: ' + name)
    if model.config.control == 'fixed_dynamics':
        model.verify_frozen_dynamics(values)


def restore(path, graph, lexicon, binding):
    payload = path.read_bytes()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    run = saved['run']
    if any(run.get(key) != value for key, value in binding.items()):
        raise ValueError('Checkpoint study/run binding changed')
    if run['graph_sha256'] != sha256(graph) or run['tokenizer_sha256'] != lexicon.sha256:
        raise ValueError('Checkpoint graph/tokenizer identity changed')
    # The expected seed and mechanism come from the frozen study, not this file.
    model = construct(binding['control'], graph, lexicon.vocabulary, binding['seed'])
    if asdict(model.config) != saved['config']:
        raise ValueError('Checkpoint control configuration changed')
    if run['optimizer_parameter_names'] != optimizer_names(model):
        raise ValueError('Checkpoint optimizer parameter order changed')
    if run['parameter_card'] != model.parameter_card():
        raise ValueError('Checkpoint parameter card changed')
    verify_tensors(model, saved['model'])
    model.load_state_dict(saved['model'])
    saved['_file_sha256'] = hashlib.sha256(payload).hexdigest()
    return model, saved


def restore_optimizer(model, saved, protocol):
    optimizer = optimizer_for(model, protocol)
    expected = dict(optimizer.param_groups[0])
    expected['lr'] = learning_rate(protocol, saved['step'])
    groups = saved['optimizer']['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(expected['params']):
        raise ValueError('Checkpoint optimizer group changed')
    if set(groups[0]) != set(expected) or any(groups[0][k] != v for k, v in expected.items() if k != 'params'):
        raise ValueError('Checkpoint optimizer settings changed')
    if groups[0]['params'] != list(range(len(expected['params']))):
        raise ValueError('Checkpoint optimizer parameter indices changed')
    disconnected = {'edge_log_gain', 'recurrent_logit'} if model.config.control in ('no_lateral', 'no_temporal_state') else set()
    expected_states = {index for index, name in enumerate(optimizer_names(model)) if name not in disconnected}
    if set(saved['optimizer']['state']) != expected_states:
        raise ValueError('Checkpoint optimizer moment inventory changed')
    optimizer.load_state_dict(saved['optimizer'])
    for parameter, state in optimizer.state.items():
        for name in ('exp_avg', 'exp_avg_sq'):
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid optimizer moment: ' + name)
        if float(state['step']) != saved['step']:
            raise ValueError('Optimizer update count changed')
    return optimizer


def checkpoint_records(directory, binding, interval=500, steps=6000):
    """Read a contiguous sequence of atomically committed checkpoint manifests.

    Uncommitted files from an interrupted save are ignored, then overwritten by
    deterministic replay. A committed file with altered bytes fails closed.
    """
    records = []
    paths = sorted(directory.glob('saved-*.json'))
    for index, path in enumerate(paths, 1):
        record = json.loads(path.read_text(encoding='utf8'))
        step = index * interval
        if step > steps or path.name != f'saved-{step:06d}.json' or record['step'] != step:
            raise ValueError('Checkpoint record sequence changed')
        if record['binding'] != binding:
            raise ValueError('Checkpoint record study/run binding changed')
        names = {'checkpoint': f'checkpoint-{step:06d}.pt',
                 'validation': f'validation-{step:06d}.json', 'history': f'history-{step:06d}.json'}
        for key, name in names.items():
            if record[key] != name or sha256(directory / name) != record[key + '_sha256']:
                raise ValueError('Committed checkpoint artifact changed: ' + key)
        score = json.loads((directory / names['validation']).read_text(encoding='utf8'))
        if not math.isfinite(score['bits_per_byte']) or score['bits_per_byte'] != record['validation_bpb']:
            raise ValueError('Checkpoint selection score changed')
        records.append(record)
    return records


def commit_checkpoint(directory, model, optimizer, sampler, step, best, run, binding, score, history):
    if model.config.control == 'fixed_dynamics':
        model.verify_frozen_dynamics()
    names = dict(checkpoint=f'checkpoint-{step:06d}.pt', validation=f'validation-{step:06d}.json',
                 history=f'history-{step:06d}.json')
    marker = directory / f'saved-{step:06d}.json'
    if marker.exists():
        raise ValueError('Refusing to overwrite a committed checkpoint')
    save_checkpoint(directory / names['checkpoint'], model, optimizer, sampler, step, best, run)
    write_json(directory / names['validation'], score)
    write_json(directory / names['history'], history)
    record = dict(step=step, binding=binding, validation_bpb=score['bits_per_byte'], **names,
                  **{key + '_sha256': sha256(directory / name) for key, name in names.items()})
    # This is the transaction boundary. All three payloads precede this marker.
    write_json(marker, record)
    return record


def verify_exposure(saved, identity):
    step = saved['step']
    seed = str(saved['run']['seed'])
    audit = identity['sampling'][seed].get(str(step))
    if audit is None or saved['sampler_rng'] != audit['sampler_rng'] or saved['run']['exposure'] != audit['exposure']:
        raise ValueError('Checkpoint sampled-text exposure or RNG changed')


def train(root, condition, identity):
    # Late import keeps the independent restore/update functions fixture-testable.
    from .language_core_study import (GRAPH, LEXICON, run_binding,
                                      verify_identity, verify_complete)
    from .tokenizer import Lexicon, read_cache
    verify_identity(root, identity)
    if condition['reference']:
        raise ValueError('Previously trained references must not be retrained here')
    protocol = identity['training_protocol']
    torch.set_num_threads(protocol['threads'])
    lexicon = Lexicon(root / LEXICON)
    graph = root / GRAPH
    binding = run_binding(root, condition, identity)
    directory = root / condition['output']
    records = checkpoint_records(directory, binding, steps=protocol['steps'])
    documents = read_cache(root / 'data/processed/wikitext2-bpe/train.npz')
    validation = read_cache(root / 'data/processed/wikitext2-bpe/validation.npz')
    sampler = Sampler(documents, condition['seed'], protocol['sequence'])
    if records:
        model, saved = restore(directory / records[-1]['checkpoint'], graph, lexicon, binding)
        verify_exposure(saved, identity)
        if saved['step'] != records[-1]['step'] or saved['best'] != min(r['validation_bpb'] for r in records):
            raise ValueError('Checkpoint cumulative selection changed')
        optimizer = restore_optimizer(model, saved, protocol)
        sampler.rng.bit_generator.state = saved['sampler_rng']
        torch.set_rng_state(saved['torch_rng'])
        start, best, run = saved['step'], saved['best'], saved['run']
        presented_bytes = run['exposure']['presented_bytes']
        scored_bytes = run['exposure']['scored_bytes']
    else:
        model = construct(condition['control'], graph, lexicon.vocabulary, condition['seed'])
        optimizer = optimizer_for(model, protocol)
        start, best, presented_bytes, scored_bytes = 0, float('inf'), 0, 0
        run = dict(binding, parameter_card=model.parameter_card(), optimizer_parameter_names=optimizer_names(model),
                   source_commit=identity['source_commit'], python_torch=str(torch.__version__),
                   test_set_used_for_training=False)
    directory.mkdir(parents=True, exist_ok=True)
    # This metadata is descriptive; committed checkpoint records govern resume.
    write_json(directory / 'run.json', run)
    if not records:
        initial = evaluate(model, validation, lexicon, protocol['eval_tokens'])
        write_json(directory / 'initial-validation.json', initial)
        print(json.dumps(dict(event='initial', label=condition['label'], validation_bpb=initial['bits_per_byte'])), flush=True)
    started = interval_started = time.perf_counter()
    loss_sum = 0.
    interval_steps = 0
    history = []
    for step in range(start + 1, protocol['steps'] + 1):
        x, y = sampler.sample(protocol['batch'], 'cpu')
        loss, gradient = update(model, optimizer, x, y, protocol, step)
        loss_sum += loss
        interval_steps += 1
        presented_bytes += int(lexicon.lengths[x.numpy()].sum())
        scored_bytes += int(lexicon.lengths[y[:, protocol['warmup']:].numpy()].sum())
        if step % 100 == 0:
            row = dict(step=step, train_loss=loss_sum / interval_steps,
                       tokens_per_second=interval_steps * protocol['batch'] * protocol['sequence'] /
                                        (time.perf_counter() - interval_started),
                       learning_rate=learning_rate(protocol, step), gradient_norm=gradient,
                       presented_tokens=step * protocol['batch'] * protocol['sequence'],
                       presented_bytes=presented_bytes, scored_bytes=scored_bytes,
                       attempt_seconds=time.perf_counter() - started, resumed_from=start)
            history.append(row)
            print(json.dumps(dict(label=condition['label'], **row)), flush=True)
            interval_started = time.perf_counter()
            loss_sum, interval_steps = 0., 0
        if step % 500 == 0:
            verify_identity(root, identity)
            score = evaluate(model, validation, lexicon, protocol['eval_tokens'])
            best = min(best, score['bits_per_byte'])
            run['exposure'] = dict(presented_tokens=step * protocol['batch'] * protocol['sequence'],
                                   presented_bytes=presented_bytes, scored_bytes=scored_bytes)
            verify_exposure(dict(step=step, run=run, sampler_rng=sampler.rng.bit_generator.state), identity)
            record = commit_checkpoint(directory, model, optimizer, sampler, step, best, run, binding, score, history)
            records.append(record)
            print(json.dumps(dict(label=condition['label'], step=step, validation_bpb=score['bits_per_byte'], best=best)), flush=True)
            history = []
            interval_started = time.perf_counter()
    verify_identity(root, identity)
    selected = min(records, key=lambda r: (r['validation_bpb'], r['step']))
    complete = dict(binding=binding, steps=protocol['steps'], selected_step=selected['step'],
                    best_validation_bpb=selected['validation_bpb'], checkpoint=selected['checkpoint'],
                    checkpoint_sha256=selected['checkpoint_sha256'], last_checkpoint_sha256=records[-1]['checkpoint_sha256'])
    write_json(directory / 'complete.json', complete)
    verify_complete(root, condition, identity, lexicon)
