"""Read-only audit of the paused BabyLM 10M GRU checkpoint; no fitting or test access.

Run from the repository root with ``python -m scripts.audit_babylm_resume``.
The report goes to stdout. Run files, model weights and optimizer state are never
written. This is a dated readiness check, not a writer lock or a bitwise training
reproduction claim. Rerun immediately before resuming the serial queue.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from flm.language_train import load_training_data, restore
from flm.provenance import sha256
from flm.tokenizer import Lexicon
from flm.train import Sampler


SOURCES = ('language_train', 'model', 'baselines', 'train', 'tokenizer',
           'corpus_cache', 'graph', 'provenance')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    torch.set_num_threads(1)
    folder = Path('runs/babylm-10m/gru-s42')
    graph = Path('data/graphs/central-1024/graph.npz')
    data = Path('data/processed/babylm-2026-bpe')
    tokenizer = Path('data/tokenizers/babylm-2026-4096/tokenizer.json')
    panel = tokenizer.with_name('validation-panel.json')
    before = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    require('complete.json' not in before, 'This audit is for an incomplete run')
    lexicon = Lexicon(tokenizer)
    model, saved = restore(folder / 'last.pt', graph, lexicon)
    _, selected = restore(folder / 'best.pt', graph, lexicon)
    run = saved['run']; step = saved['step']; seed = run['seed']
    require(seed == 42 and model.config.variant == 'gru', 'Unexpected run identity')
    require(isinstance(step, int) and 0 < step < 12000 and step % 500 == 0,
            'Unexpected saved checkpoint step')
    require(run['test_set_used_for_training'] is False, 'Test use flag changed')
    require(str(torch.__version__) == run['python_torch'], 'Torch version changed')
    source_hashes = {}
    for name in SOURCES:
        path = Path(f'flm/{name}.py')
        original = subprocess.check_output(['git', 'show', f"{run['source_commit']}:{path.as_posix()}"])
        digest = hashlib.sha256(original).hexdigest()
        require(sha256(path) == digest, f'Numerical source changed: {path}')
        source_hashes[path.as_posix()] = digest
    documents, validation, identities = load_training_data(
        data, lexicon, panel, data / 'train-10m', data / 'validation')
    expected = dict(steps=12000, batch=16, sequence=96, warmup=16,
        learning_rate=.002, final_learning_rate=.0002, lr_warmup_updates=100,
        weight_decay=.01, eval_tokens=49152, threads=4,
        tokenizer_sha256=lexicon.sha256, train_cache_sha256=identities['train'],
        validation_cache_sha256=identities['validation'], graph_sha256=sha256(graph),
        validation_panel_sha256=sha256(panel), evaluation_unit='block',
        cache_identity='SHA-256 of verified mmap manifest')
    require(run['protocol'] == expected, 'Resume protocol/input mismatch')
    study = json.loads((folder.parent / 'study.json').read_text(encoding='utf8'))
    require(study == dict(dataset='BabyLM 2026 10m', steps=12000, seeds=[42, 43],
        variants=['flm', 'gru', 'transformer'], protocol_sha256=sha256(Path('docs/BABYLM-PROTOCOL.md')),
        tokenizer_sha256=lexicon.sha256, train_manifest_sha256=identities['train'],
        validation_manifest_sha256=identities['validation'], validation_panel_sha256=sha256(panel)),
        'Registered study declaration changed')
    require(len(validation) * json.loads(panel.read_text())['target_tokens_per_block'] == 49152,
            'Validation panel exposure changed')
    require(model.parameter_card() == run['parameter_card'], 'Parameter card changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer'])
    require(len(optimizer.param_groups) == 1, 'Unexpected optimizer groups')
    require(len(optimizer.state) == len(list(model.parameters())), 'Missing optimizer state')
    group = optimizer.param_groups[0]
    defaults = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01).param_groups[0]
    require({k: v for k, v in group.items() if k not in ('lr', 'params')}
            == {k: v for k, v in defaults.items() if k not in ('lr', 'params')},
            'Saved optimizer settings differ from registered AdamW defaults')
    rate = .0002 + .0018 * .5 * (1 + math.cos(math.pi * (step - 100) / 11900))
    require(optimizer.param_groups[0]['lr'] == rate, 'Saved learning rate mismatch')
    require(optimizer.param_groups[0]['weight_decay'] == .01, 'Weight decay mismatch')
    for parameter in model.parameters():
        require(bool(torch.isfinite(parameter).all()), 'Nonfinite parameter')
        state = optimizer.state[parameter]
        require(float(state['step']) == step, 'Optimizer step mismatch')
        for name in ('exp_avg', 'exp_avg_sq'):
            require(state[name].shape == parameter.shape and bool(torch.isfinite(state[name]).all()),
                    f'Invalid optimizer moment: {name}')
        require(bool((state['exp_avg_sq'] >= 0).all()), 'Negative squared moment')
    torch.set_rng_state(saved['torch_rng'])
    require(torch.equal(torch.get_rng_state(), saved['torch_rng']), 'Torch RNG restore failed')
    sampler = Sampler(documents, seed, 96)
    presented = scored = 0
    for _ in range(step):
        x, y = sampler.sample(16, 'cpu')
        presented += int(lexicon.lengths[x.numpy()].sum())
        scored += int(lexicon.lengths[y[:, 16:].numpy()].sum())
    exposure = dict(presented_tokens=step * 1536, presented_bytes=presented, scored_bytes=scored)
    require(exposure == run['exposure'], 'Replayed exposure differs from checkpoint')
    require(sampler.rng.bit_generator.state == saved['sampler_rng'], 'Replayed sampler RNG differs')
    scores = []
    for at in range(500, step + 1, 500):
        result = json.loads((folder / f'validation-{at:06d}.json').read_text())
        require(result['tokenizer_sha256'] == lexicon.sha256, 'Validation tokenizer mismatch')
        value = result['bits_per_byte']
        require(math.isfinite(value) and value > 0, 'Invalid validation value')
        scores.append((value, at))
    best, best_step = min(scores)
    require(saved['best'] == best == selected['best'] and selected['step'] == best_step,
            'Selected checkpoint differs from validation history')
    require(selected['run']['protocol'] == expected and selected['run']['seed'] == seed,
            'Selected checkpoint identity mismatch')
    history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
    at_checkpoint = [row for row in history if row['step'] == step]
    require(bool(at_checkpoint) and all(all(row[k] == v for k, v in exposure.items())
                                      for row in at_checkpoint), 'Saved-step history exposure mismatch')
    beyond = [row['step'] for row in history if row['step'] > step]
    after = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
    require(before == after, 'Run files changed during the audit; retry with no writer')
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Paused BabyLM 10M GRU seed 42; read-only readiness audit',
        audit_source_sha256=sha256(Path(__file__)),
        checkpoint_step=step, selected_step=best_step, selected_validation_bpb=best,
        last_checkpoint_sha256=saved['_file_sha256'], best_checkpoint_sha256=selected['_file_sha256'],
        source_commit=run['source_commit'], numerical_source_sha256=source_hashes,
        protocol=expected, replayed_exposure=exposure, sampler_rng_exact=True,
        optimizer_restored_and_finite=True, torch_rng_restored=True,
        all_run_files_unchanged=True, run_file_sha256=before,
        train_and_validation_cache_files_verified=True, test_cache_opened=False,
        training_or_evaluation_updates=0, validation_records_checked=len(scores),
        unsaved_history_steps=beyond,
        recovery='Resume last.pt. Retain log rows beyond its step as unsaved attempts; they are not checkpoint exposure.',
        limitations=['Not a writer lock; confirm the control queue has ended before resuming.',
            'Checks saved states and deterministic data-window replay, not historical optimizer trajectories.',
            'No four-thread bitwise training guarantee. Run files are not modified.'],
        software=dict(torch=str(torch.__version__), numpy=np.__version__))


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
