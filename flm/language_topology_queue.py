"""Run independent frozen language conditions with bounded process concurrency."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .language_topology import (REPORTS, STUDY, LEXICON, conditions, read_json,
                               verify_source_identity, verify_saved, verify_complete)
from .language_train import restore
from .provenance import sha256
from .tokenizer import Lexicon


@contextmanager
def queue_lease(path):
    """An OS-held lock; a leftover file alone never implies an active process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
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
            raise RuntimeError('Another language topology scheduler holds the queue lease') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def existing_trainers():
    """Check actual process commands before acquiring any checkpoint output."""
    if os.name == 'nt':
        command = (
            "@(Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -match '^python.*\\.exe$' -and "
            "$_.CommandLine -match ' -m flm\\.language_(train(?:\\s|$)|topology\\s+train(?:\\s|$))' "
            "} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
        )
        text = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        rows = json.loads(text) if text.strip() else []
        return rows if isinstance(rows, list) else [rows]
    result = []
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = process.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):
            continue
        if b'-m' in arguments:
            position = arguments.index(b'-m') + 1
            if position < len(arguments) and arguments[position] in (b'flm.language_train', b'flm.language_topology'):
                result.append(dict(ProcessId=int(process.parent.name)))
    return result


def available_memory_mib():
    if os.name != 'nt':
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                    ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended_virtual', ctypes.c_ulonglong)]
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot inspect available memory before starting a second trainer')
    return status.available_physical / 2**20


def training_command(root, condition, identity, lexicon):
    verify_source_identity(root, identity)
    output = root / condition['output']
    command = [sys.executable, '-X', 'utf8', '-m', 'flm.language_train',
               '--variant', condition['variant'], '--graph', condition['graph'],
               '--seed', str(condition['seed']), '--steps', '6000', '--threads', '4',
               '--output', condition['output']]
    if (output / 'last.pt').exists():
        _, saved = restore(output / 'last.pt', root / condition['graph'], lexicon)
        verify_saved(saved, condition, identity, root / condition['graph'])
        command += ['--resume', str(output / 'last.pt')]
    elif (output / 'run.json').exists():
        raise RuntimeError(f'Run exists without a saved checkpoint: {condition["label"]}')
    return command


def schedule(jobs, workers, start, finish, *, can_add=lambda: True, pause=time.sleep):
    """A failed child prevents further launches; already-running children finish."""
    if workers not in (1, 2):
        raise ValueError('At most two four-thread training processes are supported')
    pending = deque(jobs); active = []; failed = []
    try:
        while pending or active:
            # Reap before launching: a reported failure cannot start more jobs.
            for job, process in list(active):
                code = process.poll()
                if code is None:
                    continue
                active.remove((job, process))
                if code:
                    failed.append((job['label'], code))
                else:
                    finish(job)
            while pending and not failed and len(active) < workers:
                if active and not can_add():
                    break
                job = pending.popleft()
                active.append((job, start(job)))
            if failed and not active:
                raise RuntimeError(f'Training failed; remaining jobs were not launched: {failed}')
            if active:
                pause(1)
    finally:
        # Stop only owned, still-live children if the scheduler itself fails.
        # Their last atomic checkpoint remains resumable.
        for _, process in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()


def run(root, workers, score):
    with queue_lease(root / STUDY / 'queue.lock'):
        running = existing_trainers()
        if running:
            raise RuntimeError(f'Existing language training processes must finish or be paused first: {running}')
        identity = read_json(root / REPORTS / 'identity.json')
        verify_source_identity(root, identity)
        lexicon = Lexicon(root / LEXICON)
        pending = []
        for condition in conditions():
            if (root / condition['output'] / 'complete.json').exists():
                verify_complete(root, condition, identity, lexicon)
                print(f'Already verified: {condition["label"]}', flush=True)
            elif condition['reference']:
                raise RuntimeError('A frozen measured reference is incomplete')
            else:
                training_command(root, condition, identity, lexicon)
                pending.append(condition)
        record = dict(event='scheduler-start', pid=os.getpid(), maximum_workers=workers,
                      threads_per_worker=4, minimum_available_mib_for_additional_worker=2048,
                      study_identity_sha256=sha256(root / REPORTS / 'identity.json'),
                      scheduler_sha256=sha256(Path(__file__)),
                      pending=[row['label'] for row in pending], test_scoring_after_completion=score)
        with (root / STUDY / 'scheduling.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

        def start(condition):
            command = training_command(root, condition, identity, lexicon)
            process = subprocess.Popen(command, cwd=root)
            print(json.dumps(dict(event='trainer-start', label=condition['label'], pid=process.pid,
                                  resumed='--resume' in command)), flush=True)
            return process

        def finish(condition):
            verify_source_identity(root, identity)
            verify_complete(root, condition, identity, lexicon)
            print(f'Completed and verified: {condition["label"]}', flush=True)

        def can_add():
            available = available_memory_mib()
            return available is None or available >= 2048

        schedule(pending, workers, start, finish, can_add=can_add)
        print('Every registered language condition is complete and verified.', flush=True)
        if score:
            from .language_topology_test import score_study, snapshot
            score_study(root); snapshot(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    run(Path('.').resolve(), args.workers, args.score)


if __name__ == '__main__':
    main()
