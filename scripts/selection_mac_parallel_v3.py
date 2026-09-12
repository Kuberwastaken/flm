"""User-authorized concurrent supervisor for the existing native Mac study."""
import argparse
import os
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import selection_mac_runtime as runtime
from flm.provenance import sha256, write_json

DOCUMENT = Path('docs/SELECTION-MAC-PARALLELISM-V3.md')
PROBE = runtime.REPORTS/'parallel-probe-v1.json'
EXECUTION = runtime.REPORTS/'parallel-execution-v3.json'
CTX = None


def immutable(path, value):
    if path.exists():
        runtime.require(runtime.read(path) == value, 'Existing record differs: '+str(path))
    else:
        write_json(path, value)


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
    waiting = {'OMP_WAIT_POLICY': 'PASSIVE', 'KMP_BLOCKTIME': '0'}
    runtime.require(all(os.environ.get(k) == v for k, v in waiting.items())
                    and os.environ.get('OMP_THREAD_LIMIT') is None, 'Require declared waiting policy and unchanged thread cap')
    diagnostic_path = ROOT/runtime.REPORTS/'thread-wait-diagnostic-v1.json'
    diagnostic = runtime.read(diagnostic_path)
    runtime.require(all(diagnostic['exact_match'].values()), 'Waiting-policy diagnostic failed numerical equality')
    runtime.install_adapter()
    identity, _, _, _ = runtime.verified_context(ROOT)
    identity_hash = sha256(ROOT/runtime.IDENTITY)
    # Explicit user override of the earlier operational speedup cutoff.
    count = 8
    probe = runtime.read(ROOT/PROBE)
    runtime.require(probe['study_identity_sha256'] == identity_hash, 'Historical probe belongs to another cohort')
    conditions = [r['condition'] for r in identity['conditions']]
    runtime.require(len(conditions) == 128 and len({r['label'] for r in conditions}) == 128, 'Require the entire unique native matrix')
    record = dict(format='flm-mac-parallel-execution-v3', study_identity_sha256=identity_hash,
        workers=count, threads_per_fit=4, conditions=conditions,
        waiting_environment=waiting, waiting_diagnostic_sha256=sha256(diagnostic_path),
        authorization='User explicitly requested parallel execution and superseded the earlier scheduling adoption cutoff.',
        scope='Eight workers are an operational choice, not a new qualification result or a guaranteed speedup.',
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
            failure = 'parallel-v3-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
            write_json(ROOT/runtime.REPORTS/failure,
                       dict(recorded_at=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc(),
                            policy='No validation or held-out stage launched. Preserve per-fit checkpoints for recovery.'))
            raise
        by_label = {r['condition']['label']:r for r in receipts}
        runtime.require(set(by_label) == {c['label'] for c in conditions} and len(receipts) == 128, 'Incomplete worker result inventory')
        runtime.verified_context(ROOT)
        runtime.require(sha256(Path(__file__)) == record['driver_sha256']
                        and sha256(ROOT/DOCUMENT) == record['document_sha256'], 'Execution source changed before evaluation')
        destination = ROOT/runtime.REPORTS/'parallel-training-complete-v3.json'
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
    parser.add_argument('operation', choices=('run',))
    parser.parse_args()
    run()
