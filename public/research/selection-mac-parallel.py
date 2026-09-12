"""Bounded concurrency probe and isolated-fit supervisor for the native Mac study."""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import selection_mac_runtime as runtime
from flm.inference import state_hash
from flm.selection_language import parameter_hash
from flm.language_learning_inputs import fingerprint
from flm.language_learning_train import optimizer_for, update
from flm.provenance import sha256, write_json

DOCUMENT = Path('docs/SELECTION-MAC-PARALLELISM.md')
PROBE = runtime.REPORTS/'parallel-probe-v1.json'
EXECUTION = runtime.REPORTS/'parallel-execution-v1.json'
CTX = None
BARRIER = None


def immutable(path, value):
    if path.exists():
        runtime.require(runtime.read(path) == value, 'Existing record differs: '+str(path))
    else:
        write_json(path, value)


def initialize_probe(barrier):
    global CTX, BARRIER
    torch.set_num_threads(4)
    identity = runtime.read(ROOT/runtime.IDENTITY)
    corpus = SimpleNamespace(binding=identity['corpus_binding'], lexicon=SimpleNamespace(
        vocabulary=4096, sha256=identity['corpus_binding']['tokenizer_sha256']))
    CTX = identity, runtime.coordinator.load_catalog(ROOT), corpus
    BARRIER = barrier


def probe_worker(index):
    identity, catalog, corpus = CTX
    row = next(r for r in identity['conditions'] if r['condition']['label'] == 'KCg-d-R-t5/candidate/measured/s42')
    c = row['condition']
    model, binding = runtime.coordinator.prepare_case(catalog, corpus, c['graph'], c['seed'])
    runtime.require(binding == row['base_binding'], 'Probe initialization differs from native cohort')
    settings = runtime.coordinator.Settings(**row['settings'])
    optimizer = optimizer_for(model, settings)
    x = ((torch.arange(16*96).reshape(16, 96)*37+11) % 4094)+2
    y = ((x+13-2) % 4094)+2
    for step in range(1, 11):
        update(model, optimizer, x, y, settings, 'bptt', step)
    BARRIER.wait(timeout=60)
    started = time.perf_counter()
    for step in range(11, 51):
        update(model, optimizer, x, y, settings, 'bptt', step)
    ended = time.perf_counter()
    moments = {f'{index}/{key}': value.detach().cpu().numpy()
               for index, fields in optimizer.state_dict()['state'].items()
               for key, value in fields.items() if torch.is_tensor(value)}
    return dict(worker=index, started=started, ended=ended, timed_updates=40,
                final_state_sha256=state_hash(model), optimizer_tensors_sha256=fingerprint(moments),
                parameter_sha256=parameter_hash(model), threads=torch.get_num_threads())


def choose_workers(rounds):
    baseline = rounds[0]
    runtime.require(baseline['workers'] == 1, 'Missing serial baseline')
    eligible = [r for r in rounds if r['exact_final_state_and_optimizer']]
    fastest = max(r['updates_per_second'] for r in eligible)
    if fastest < baseline['updates_per_second']*1.15:
        return 1
    # Avoid extra load for gains smaller than the precision of a short probe.
    return min(r['workers'] for r in eligible if r['updates_per_second'] >= fastest*.90)


def benchmark():
    runtime.install_adapter()
    runtime.verified_context(ROOT)
    runtime.require(not (ROOT/PROBE).exists(), 'Preserve the earlier concurrency probe')
    rounds = []
    context = mp.get_context('spawn')
    with runtime.coordinator.training_lease(ROOT/runtime.STUDY):
        for count in (1, 2, 3, 4):
            barrier = context.Barrier(count)
            with context.Pool(count, initializer=initialize_probe, initargs=(barrier,)) as pool:
                results = pool.map_async(probe_worker, range(count)).get(timeout=120)
            elapsed = max(r['ended'] for r in results)-min(r['started'] for r in results)
            baseline = results[0] if not rounds else rounds[0]['results'][0]
            exact = all(all(r[k] == baseline[k] for k in
                        ('final_state_sha256', 'optimizer_tensors_sha256', 'parameter_sha256')) for r in results)
            rounds.append(dict(workers=count, elapsed_seconds=elapsed, updates_per_second=40*count/elapsed,
                               exact_final_state_and_optimizer=exact, results=results))
            print(json.dumps({k:rounds[-1][k] for k in ('workers','updates_per_second','exact_final_state_and_optimizer')}), flush=True)
    record = dict(format='flm-mac-parallel-probe-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        study_identity_sha256=sha256(ROOT/runtime.IDENTITY), driver_sha256=sha256(Path(__file__)),
        document_sha256=sha256(ROOT/DOCUMENT), rounds=rounds, recommended_workers=choose_workers(rounds),
        scope='One fixed native 540-neuron model; ten warmup and forty timed synthetic-token updates per worker. '
              'No corpus or held-out scoring. Exact final model and optimizer tensor hashes are a bounded check, '
              'not a proof of equality for every full trajectory. No official writer overlaps the probe.')
    immutable(ROOT/PROBE, record)
    print('Recommended workers:', record['recommended_workers'], flush=True)


def initialize_worker():
    global CTX
    try:
        runtime.install_adapter()
        identity, catalog, corpus, _ = runtime.verified_context(ROOT)
        CTX = identity, catalog, corpus
    except BaseException:
        # Return the error through an assigned task instead of causing Pool to
        # respawn failing initializers indefinitely.
        CTX = dict(initialization_error=traceback.format_exc())


def fit_worker(index):
    if isinstance(CTX, dict):
        raise RuntimeError(CTX['initialization_error'])
    identity, catalog, corpus = CTX
    row = identity['conditions'][index]
    condition = row['condition']
    identity_hash = sha256(ROOT/runtime.IDENTITY)
    runtime.unchanged(ROOT, identity, identity_hash)
    execution = runtime.read(ROOT/EXECUTION)
    runtime.require(execution['driver_sha256'] == sha256(Path(__file__))
                    and execution['document_sha256'] == sha256(ROOT/DOCUMENT), 'Parallel execution source changed')
    directory = ROOT/runtime.STUDY/condition['label']
    previous = sha256(directory/'complete.json') if (directory/'complete.json').exists() else None
    print('Worker fitting/auditing: '+condition['label'], flush=True)
    started = time.perf_counter()
    result = runtime.coordinator.fit_case(catalog, corpus, condition['graph'],
        runtime.coordinator.Settings(**row['settings']), directory, dict(study_identity_sha256=identity_hash))
    runtime.require(result['complete'] is True, 'Incomplete assigned fit')
    runtime.unchanged(ROOT, identity, identity_hash)
    digest = sha256(directory/'complete.json')
    runtime.require(previous is None or previous == digest, 'Previously completed fit changed')
    receipt = dict(condition=condition, complete_sha256=digest, previously_complete=previous is not None,
                   elapsed_seconds=time.perf_counter()-started, threads=torch.get_num_threads())
    print('Worker completed: '+condition['label'], flush=True)
    return receipt


def run():
    runtime.install_adapter()
    identity, _, _, _ = runtime.verified_context(ROOT)
    identity_hash = sha256(ROOT/runtime.IDENTITY)
    probe = runtime.read(ROOT/PROBE)
    runtime.require(probe['study_identity_sha256'] == identity_hash
                    and probe['driver_sha256'] == sha256(Path(__file__))
                    and probe['document_sha256'] == sha256(ROOT/DOCUMENT), 'Concurrency qualification changed')
    count = choose_workers(probe['rounds'])
    runtime.require(count == probe['recommended_workers'] and count in (1, 2, 3, 4), 'Unqualified worker count')
    conditions = [r['condition'] for r in identity['conditions']]
    runtime.require(len(conditions) == 128 and len({r['label'] for r in conditions}) == 128, 'Require the entire unique native matrix')
    record = dict(format='flm-mac-parallel-execution-v1', study_identity_sha256=identity_hash,
        workers=count, threads_per_fit=4, conditions=conditions,
        probe_sha256=sha256(ROOT/PROBE), driver_sha256=sha256(Path(__file__)), document_sha256=sha256(ROOT/DOCUMENT),
        scheduling='Each original condition submitted once to spawned isolated processes; completion order may vary',
        checkpoint_policy='Audit existing completed fits; resume compatible committed checkpoints; preserve all 3000 updates',
        evaluation_policy='Join all training workers and verify all 128 fits before original serial validation and held-out stages')
    with runtime.coordinator.training_lease(ROOT/runtime.STUDY):
        immutable(ROOT/EXECUTION, record)
        context = mp.get_context('spawn')
        try:
            with context.Pool(count, initializer=initialize_worker) as pool:
                receipts = list(pool.imap_unordered(fit_worker, range(128), chunksize=1))
        except BaseException:
            failure = 'parallel-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
            write_json(ROOT/runtime.REPORTS/failure,
                       dict(recorded_at=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc(),
                            policy='No validation or held-out stage launched. Preserve per-fit checkpoints for recovery.'))
            raise
        by_label = {r['condition']['label']:r for r in receipts}
        runtime.require(set(by_label) == {c['label'] for c in conditions} and len(receipts) == 128, 'Incomplete worker result inventory')
        runtime.verified_context(ROOT)
        runtime.require(sha256(Path(__file__)) == record['driver_sha256']
                        and sha256(ROOT/DOCUMENT) == record['document_sha256'], 'Execution source changed before evaluation')
        destination = ROOT/runtime.REPORTS/'parallel-training-complete.json'
        completed = dict(execution_sha256=sha256(ROOT/EXECUTION), receipts=[by_label[c['label']] for c in conditions])
        if destination.exists():
            previous = runtime.read(destination)
            runtime.require(previous['execution_sha256'] == completed['execution_sha256']
                and [(r['condition'], r['complete_sha256']) for r in previous['receipts']]
                == [(r['condition'], r['complete_sha256']) for r in completed['receipts']], 'Completed training inventory changed')
        else:
            write_json(destination, completed)
    # All children have exited and the root writer lease has been released.
    runtime.coordinator.freeze_selection(ROOT)
    runtime.evaluator.score_study(ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('benchmark', 'run'))
    operation = parser.parse_args().operation
    benchmark() if operation == 'benchmark' else run()
