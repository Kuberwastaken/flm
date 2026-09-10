"""Check numerical compatibility with the already completed WikiText references."""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import types
import numpy as np
import torch
from . import language_train
from .provenance import sha256
from .tokenizer import Lexicon, read_cache
from .train import Sampler


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def historical_module(root, commit, filename):
    source = subprocess.check_output(['git', 'show', f'{commit}:{filename}'], cwd=root)
    module = types.ModuleType('flm._reference_' + Path(filename).stem)
    module.__package__ = 'flm'
    exec(compile(source, f'{commit}:{filename}', 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def compatibility(root, commit, seed):
    """Actual train text, ten AdamW updates; no test data or control fitting."""
    torch.set_num_threads(4)
    for filename in ('flm/model.py', 'flm/tokenizer.py'):
        historical = subprocess.check_output(['git', 'show', f'{commit}:{filename}'], cwd=root)
        if historical != (root / filename).read_bytes():
            raise ValueError(f'Reference numerical dependency changed: {filename}')
    old, old_sha = historical_module(root, commit, 'flm/language_train.py')
    old_sampler, sampler_sha = historical_module(root, commit, 'flm/train.py')
    graph = root / 'data/graphs/central-1024/graph.npz'
    lexicon = Lexicon(root / 'data/tokenizers/wikitext2-4096/tokenizer.json')
    docs = read_cache(root / 'data/processed/wikitext2-bpe/train.npz')
    models = [module.construct('flm', graph, lexicon.vocabulary, seed) for module in (old, language_train)]
    initial = parameter_hash(models[0])
    if initial != parameter_hash(models[1]): raise ValueError('Reference initialization changed')
    samplers = [old_sampler.Sampler(docs, seed, 96), Sampler(docs, seed, 96)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01) for model in models]
    losses = []
    for step in range(1, 11):
        batches = [sampler.sample(16, 'cpu') for sampler in samplers]
        if any(not torch.equal(a, b) for a, b in zip(*batches)):
            raise ValueError('Reference sampled tokens changed')
        pair = []
        for model, optimizer, (x, y) in zip(models, optimizers, batches):
            optimizer.param_groups[0]['lr'] = .002 * step / 100
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x)[0][:, 16:].reshape(-1, lexicon.vocabulary), y[:, 16:].reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); pair.append(float(loss.detach()))
        if pair[0] != pair[1] or parameter_hash(models[0]) != parameter_hash(models[1]):
            raise ValueError(f'Reference optimizer trajectory changed at step {step}')
        losses.append(pair[0])
    if samplers[0].rng.bit_generator.state != samplers[1].rng.bit_generator.state:
        raise ValueError('Reference sampler RNG changed')
    validation = read_cache(root / 'data/processed/wikitext2-bpe/validation.npz')[:4]
    scored = [module.evaluate(model, validation, lexicon, token_limit=128)
              for module, model in zip((old, language_train), models)]
    for key in ('nll', 'bytes', 'tokens', 'documents', 'bits_per_byte'):
        if scored[0][key] != scored[1][key]: raise ValueError(f'Reference evaluation changed: {key}')
    return dict(seed=seed, reference_commit=commit, updates=10, exact_parameter_match=True,
        exact_sampler_match=True, exact_validation_match=True, losses=losses,
        initial_parameter_sha256=initial, final_parameter_sha256=parameter_hash(models[0]),
        historical_trainer_sha256=old_sha, historical_sampler_sha256=sampler_sha,
        current_trainer_sha256=sha256(root / 'flm/language_train.py'))


def sampling_audit(documents, lexicon, seed, steps=6000):
    sampler = Sampler(documents, seed, 96); digest = hashlib.sha256()
    presented_bytes = scored_bytes = 0; checkpoints = {}
    for step in range(1, steps + 1):
        x, y = sampler.sample(16, 'cpu')
        digest.update(x.numpy().astype('<i8').tobytes()); digest.update(y.numpy().astype('<i8').tobytes())
        presented_bytes += int(lexicon.lengths[x.numpy()].sum())
        scored_bytes += int(lexicon.lengths[y[:, 16:].numpy()].sum())
        if step % 500 == 0 or step == steps:
            checkpoints[str(step)] = dict(stream_sha256=digest.hexdigest(), sampler_rng=sampler.rng.bit_generator.state,
                exposure=dict(presented_tokens=step * 1536, presented_bytes=presented_bytes, scored_bytes=scored_bytes))
    return checkpoints
