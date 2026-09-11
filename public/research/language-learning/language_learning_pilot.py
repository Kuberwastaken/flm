"""Disposable train-only cost pilot for the four language learning rules.

Uses the actual update implementation. Completion files are prerequisites, not
process locks: confirm the priority training jobs have exited before launch.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .inference import state_hash
from .language_learning_inputs import conditions, load_inputs, prepare_condition
from .language_learning_train import (Settings, add_exposure, configure, declaration,
                                      optimizer_for, training_lease, update)
from .local_learning import CORE_PARAMETERS
from .provenance import sha256, write_json
from .train import Sampler


@dataclass(frozen=True)
class Pilot:
    warmup_updates: int = 3
    measured_updates: int = 12
    batch: int = 16
    sequence: int = 96
    context_warmup: int = 16
    threads: int = 4

    def settings(self, seed):
        if any(type(value) is not int or value < 1 for value in
               (self.warmup_updates, self.measured_updates, self.batch, self.sequence, self.threads)):
            raise ValueError('Positive integer pilot dimensions required')
        total = self.warmup_updates + self.measured_updates
        settings = Settings(steps=total, batch=self.batch, sequence=self.sequence,
            warmup=self.context_warmup, learning_rate=.002, final_learning_rate=.002,
            lr_warmup_updates=0, weight_decay=.01, gradient_clip=1.,
            checkpoint_interval=total, seed=seed, threads=self.threads, score_boundaries=False)
        settings.validate()
        return settings


def measure(source_model, documents, lexicon, condition, binding, pilot=Pilot()):
    """Return timing/exposure only; never modify supplied tensors or save weights."""
    if condition not in conditions(): raise ValueError('Unknown learning-rule condition')
    settings = pilot.settings(condition['seed']); method = condition['method']
    original = state_hash(source_model); previous_threads = torch.get_num_threads()
    model = copy.deepcopy(source_model); configure(model, method)
    declared = declaration(model, documents, lexicon, settings, method, binding)
    initial = state_hash(model)
    fixed = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
             if method == 'fixed_core' and name in CORE_PARAMETERS}
    optimizer = optimizer_for(model, settings)
    sampler = Sampler(documents, settings.seed, settings.sequence)
    digest = hashlib.sha256(); measured_digest = hashlib.sha256()
    observations = []
    try:
        torch.set_num_threads(settings.threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed)
            for step in range(1, settings.steps + 1):
                started = time.perf_counter()
                x, y = sampler.sample(settings.batch, 'cpu')
                result = update(model, optimizer, x, y, settings, method, step)
                seconds = time.perf_counter() - started
                # This bookkeeping is not part of the measured update path.
                exposure = dict(presented_tokens=0, scored_tokens=0, presented_bytes=0, scored_bytes=0)
                add_exposure(exposure, digest, x, y, lexicon.lengths, settings)
                warmup = step <= pilot.warmup_updates
                if not warmup:
                    measured_digest.update(x.numpy().astype('<i8', copy=False).tobytes())
                    measured_digest.update(y.numpy().astype('<i8', copy=False).tobytes())
                if any(not torch.isfinite(v).all() for state in optimizer.state.values()
                       for v in state.values() if isinstance(v, torch.Tensor)):
                    raise FloatingPointError('Nonfinite optimizer state in cost pilot')
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError('Invalid pilot timing')
                observations.append(dict(step=step, warmup=warmup, seconds=seconds,
                    persistent_trace_bytes=result['persistent_trace_bytes'], **exposure))
        changed = state_hash(model) != initial
        fixed_preserved = all(torch.equal(dict(model.named_parameters())[name], value)
                              for name, value in fixed.items())
        if not changed: raise ValueError('Disposable parameters did not change')
        if not fixed_preserved: raise ValueError('Fixed recurrent parameters changed')
    finally:
        torch.set_num_threads(previous_threads)
        if state_hash(source_model) != original:
            raise ValueError('Pilot changed the supplied source model')
    timed = [row for row in observations if not row['warmup']]
    elapsed = sum(row['seconds'] for row in timed)
    exposure = {key: sum(row[key] for row in timed) for key in
                ('presented_tokens', 'scored_tokens', 'presented_bytes', 'scored_bytes')}
    return dict(condition=dict(condition), settings=asdict(settings), declaration=declared,
        initial_state_sha256=initial, source_model_unchanged=True, disposable_parameters_changed=True,
        fixed_core_preserved=fixed_preserved if fixed else None,
        sampled_windows_sha256=digest.hexdigest(), measured_windows_sha256=measured_digest.hexdigest(),
        observations=observations, measured_exposure=exposure, measured_seconds=elapsed,
        median_update_seconds=float(np.median([row['seconds'] for row in timed])),
        measured_input_tokens_per_second=exposure['presented_tokens']/elapsed,
        measured_scored_tokens_per_second=exposure['scored_tokens']/elapsed,
        peak_memory_measured=False, checkpoint_written=False, language_scores_reported=False,
        timing_scope='Sampler plus the actual language_learning_train.update path, including its parameter inventory, masks, finite checks, gradient calculation, clipping and AdamW. Excludes initialization, input verification, exposure/hashing, optimizer-state audit, checkpoint I/O and validation.',
        trace_scope='Persistent eligibility tensor accounting returned by the kernel; zero for BPTT/fixed_core means no explicit eligibility trace, not zero activation memory. Not peak or total process memory.')


def ordered_conditions():
    rows = conditions()
    order = np.random.Generator(np.random.PCG64(617)).permutation(len(rows))
    return [rows[int(index)] for index in order]


def prerequisites(root):
    paths = [root/f'runs/babylm-{scale}/{variant}-s{seed}/complete.json'
             for scale in ('10m', '100m') for variant in ('flm', 'gru', 'transformer') for seed in (42, 43)]
    if any(not path.is_file() for path in paths):
        raise ValueError('Priority BabyLM training must finish before learning-rule timing')
    for path in paths:
        record = json.loads(path.read_text(encoding='utf8'))
        if record['steps'] != 12000 or record['best_checkpoint_sha256'] != sha256(path.with_name('best.pt')):
            raise ValueError('Priority BabyLM completion identity changed')


def run(root):
    root = Path(root); prerequisites(root)
    destination = root/'reports/language-eligibility/timing-pilot.json'
    with training_lease(root/'runs/language-learning-timing-pilot'):
        if destination.exists(): raise ValueError('Timing pilot already recorded; preserve the observed run')
        inputs = load_inputs(root)
        attempt = root/'runs/language-learning-timing-pilot'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt.mkdir(); results = []
        for condition in ordered_conditions():
            try:
                model, binding = prepare_condition(inputs, condition)
                result = measure(model, inputs.documents, inputs.lexicon, condition, binding)
                for previous in results:
                    if previous['condition']['seed'] == condition['seed']:
                        for key in ('initial_state_sha256', 'sampled_windows_sha256',
                                    'measured_windows_sha256', 'measured_exposure'):
                            if result[key] != previous[key]: raise ValueError('Unmatched pilot initialization or text exposure')
                        if result['declaration']['ordered_training_documents_sha256'] != previous['declaration']['ordered_training_documents_sha256']:
                            raise ValueError('Unmatched pilot training inventory')
            except Exception as error:
                write_json(attempt/'failure.json', dict(condition=condition, completed_conditions=len(results),
                    error_type=type(error).__name__, error=str(error), source_sha256=sha256(Path(__file__))))
                raise
            results.append(result); write_json(attempt/(condition['label']+'.json'), result)
            print('Measured disposable learning-rule pilot: '+condition['label'], flush=True)
        report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
            torch=str(torch.__version__), numpy=str(np.__version__), order_seed=617,
            source_sha256={name:sha256(root/'flm'/name) for name in
                ('language_learning_pilot.py', 'language_learning_inputs.py', 'language_learning_train.py',
                 'language_eligibility.py', 'embedding_eligibility.py', 'local_learning.py',
                 'model.py', 'train.py', 'inference.py', 'corpus_cache.py', 'tokenizer.py', 'provenance.py')},
            input_binding=inputs.binding, attempt=attempt.relative_to(root).as_posix(), conditions=results,
            validation_or_test_payloads_opened=False, benchmark_budget_selected=False, training_protocol_frozen=False,
            limitations=['Verify actual process handles before launch; completion records are not process locks.',
                'One shuffled pass of eight conditions; no guarantee of long-run throughput or thermal stability.',
                'Constant pilot learning rate measures implementation cost, not a future training schedule or convergence.',
                'No checkpoint, raw text, predictions or language loss values are saved; all pilot weights are discarded.',
                'Boundary targets are excluded for every rule; a fresh matched BPTT fit is required for quality comparisons.'])
        write_json(destination, report)
    return destination


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(run(Path.cwd()))
