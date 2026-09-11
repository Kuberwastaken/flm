"""Train-only timing preparation for the later instruction comparison.

Pilot weights are disposable copies and never become benchmark initializations.
The official pilot must run only after the priority training queues have exited.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .inference import state_hash
from .provenance import sha256, write_json
from .scan_conditions import conditions, prepare_condition
from .scan_runtime import batch_loss, instruction_batch
from .scan_train import Settings, optimizer_for, rate, training_lease


@dataclass(frozen=True)
class Pilot:
    warmup_updates: int = 3
    measured_updates: int = 12
    batch_size: int = 16
    threads: int = 4

    def settings(self, seed):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('Positive integer pilot dimensions are required')
        total = self.warmup_updates + self.measured_updates
        settings = Settings(total, self.batch_size, .002, .0002, self.warmup_updates,
                            .01, 1., total, seed, self.threads)
        settings.validate()
        return settings


def pilot_conditions():
    return [row for row in conditions() if row['seed'] == 42]


def measure(prepared, pilot=Pilot()):
    settings = pilot.settings(prepared.binding['condition']['seed'])
    binding = prepared.training_binding(settings)
    original_hash = state_hash(prepared.model); original_threads = torch.get_num_threads()
    # Copying, source verification and optimizer allocation precede timed updates.
    model = copy.deepcopy(prepared.model); optimizer = optimizer_for(model, settings)
    generator = np.random.Generator(np.random.PCG64(settings.sampling_seed))
    digest = hashlib.sha256(); observations = []
    try:
        torch.set_num_threads(settings.threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.sampling_seed)
            for step in range(1, settings.steps + 1):
                started = time.perf_counter()
                indices = generator.integers(0, len(prepared.records), size=settings.batch_size, dtype=np.int64)
                batch = instruction_batch([prepared.records[index] for index in indices], prepared.lexicon)
                model.train(); optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups: group['lr'] = rate(settings, step)
                loss = batch_loss(model, batch)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite pilot loss')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
                if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite pilot gradient')
                optimizer.step()
                seconds = time.perf_counter() - started
                # Measurement bookkeeping and these integrity checks are outside timing.
                tensors = list(model.state_dict().values())
                tensors += [value for state in optimizer.state.values() for value in state.values()
                            if isinstance(value, torch.Tensor)]
                if any(not torch.isfinite(value).all() for value in tensors):
                    raise FloatingPointError('Nonfinite pilot update')
                digest.update(indices.astype('<i8', copy=False).tobytes())
                observations.append(dict(step=step, warmup=step <= pilot.warmup_updates,
                    seconds=seconds, examples=batch['examples'], input_tokens=batch['input_tokens'],
                    supervised_tokens=batch['supervised_tokens']))
    finally:
        torch.set_num_threads(original_threads)
    if state_hash(prepared.model) != original_hash:
        raise ValueError('Pilot changed the original language source')
    timed = [row for row in observations if not row['warmup']]
    elapsed = sum(row['seconds'] for row in timed)
    if elapsed <= 0 or not all(np.isfinite(row['seconds']) and row['seconds'] > 0 for row in observations):
        raise ValueError('Invalid pilot timing')
    return dict(binding=binding, settings=asdict(settings), pilot=asdict(pilot),
        dtype=str(next(model.parameters()).dtype),
        original_state_sha256=original_hash, original_state_unchanged=True,
        sampled_row_sha256=digest.hexdigest(), observations=observations,
        measured_seconds=elapsed, median_update_seconds=float(np.median([row['seconds'] for row in timed])),
        measured_examples=sum(row['examples'] for row in timed),
        measured_input_tokens=sum(row['input_tokens'] for row in timed),
        measured_supervised_tokens=sum(row['supervised_tokens'] for row in timed),
        measured_updates_per_second=len(timed) / elapsed,
        checkpoint_written=False, model_predictions_scored=False,
        timing_scope='Sampling, batch encoding/padding, forward/backward, clipping and AdamW; excludes preparation, checkpoint I/O and verification.',
        boundary='Short disposable train-only timing run; not an optimizer comparison, instruction accuracy result or completed benchmark fit.')


def run(root):
    root = Path(root)
    # These prerequisites do not substitute for checking live process handles.
    required = [root / f'runs/language-core-v1/{control}-s{seed}/complete.json'
                for control in ('fixed_dynamics', 'no_lateral', 'no_temporal_state') for seed in (42, 43)]
    required += [root / f'runs/babylm-{scale}/{variant}-s{seed}/complete.json'
                 for scale in ('10m', '100m') for variant in ('flm', 'gru', 'transformer') for seed in (42, 43)]
    if any(not path.is_file() for path in required):
        raise ValueError('Priority language-control and BabyLM training queues must finish before the SCAN timing pilot')
    destination = root / 'reports/scan-runtime/timing-pilot.json'
    with training_lease(root / 'runs/scan-timing-pilot'):
        if destination.exists(): raise ValueError('Timing pilot already recorded; do not overwrite an observed run')
        attempt = root / 'runs/scan-timing-pilot' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt.mkdir()
        results = []
        for condition in pilot_conditions():
            try:
                prepared = prepare_condition(root, condition)
                measured = measure(prepared)
            except Exception as error:
                write_json(attempt / 'failure.json', dict(condition=condition, completed_conditions=len(results),
                    error_type=type(error).__name__, error=str(error), source_sha256=sha256(Path(__file__))))
                raise
            results.append(measured)
            write_json(attempt / (condition['label'].replace('/', '--') + '.json'), measured)
            print('Measured disposable timing pilot: ' + condition['label'], flush=True)
        write_json(destination, dict(verified_utc=datetime.now(timezone.utc).isoformat(),
            platform=platform.platform(), torch=str(torch.__version__), numpy=str(np.__version__),
            source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                          ('scan_pilot.py','scan_conditions.py','scan_train.py','scan_runtime.py','scan_inputs.py')},
            attempt=attempt.relative_to(root).as_posix(), conditions=results,
            test_partitions_opened=False, benchmark_budget_selected=False,
            limitations=['Call only after verifying all training processes have exited; completion files are not process locks.',
                'Short-run throughput does not include checkpoint/verification overhead, long-run thermal effects or peak memory.',
                'No pilot model, optimizer or modified source weights are saved or reused.']))
    return destination


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(run(Path.cwd()))
