"""Durable episode boundaries; interrupted physics restarts from original inputs.

This store owns one episode, not an experimental inventory or training budget.
It never resumes a simulator from a controller snapshot. A durable failed result
is returned as a failure on retry, rather than rerun until it succeeds.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

from .food_episode import FoodEpisode, plain, run_episode


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(item) for item in value]
    return value


def canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, allow_nan=False,
                      ensure_ascii=False, separators=(',', ':')).encode('utf8')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


@contextmanager
def episode_lease(directory):
    """Only a held operating-system lock denotes a writer; stale files do not."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'writer.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another food-episode writer holds this directory') from error
        try: yield
        finally:
            handle.seek(0)
            if os.name == 'nt': msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_new(path, value, *, compressed=False):
    """Atomic publication under the caller's lease; interrupted staging survives."""
    if path.exists(): raise FileExistsError(path)
    content = canonical(value)
    if compressed: content = gzip.compress(content, mtime=0)
    pending = path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    with pending.open('xb') as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())
    os.replace(pending, path)


def read(path, *, compressed=False):
    content = path.read_bytes()
    if compressed: content = gzip.decompress(content)
    value = json.loads(content)
    canonical(value)  # Reject nonfinite values accepted by Python's JSON reader.
    return value


def replay_result(template, record, identity, attempt):
    """Check every saved controller sample; physical truth remains a separate audit."""
    if set(record) != {'identity_sha256','attempt','finished_utc','rollout','cleanup_error','readout'}:
        raise ValueError('Unknown stored food episode result inventory')
    if record['identity_sha256'] != digest(identity) or record['attempt'] != attempt:
        raise ValueError('Stored food result identity or attempt changed')
    result = record['rollout']; state = result['controller']
    if state['initial_readout'] != template.initial_readout or state['identity'] != template.identity:
        raise ValueError('Stored food result does not start from the declared episode')
    restored = FoodEpisode.restore(template.core.weights, state, binding=template.identity['binding'])
    if restored.status == 'active' or result['outcome'] != restored.outcome:
        raise ValueError('Stored food result is not a consistent terminal outcome')
    actor = FoodEpisode.restore(template.core.weights, identity['initial_controller'],
                               binding=template.identity['binding'])
    observations = [event for event in state['events'] if event['kind'] == 'observation']
    if len(result['samples']) != len(observations):
        raise ValueError('Stored food result omitted controller observations')
    for row, event in zip(result['samples'], observations):
        if set(row) != {'control','environment'} or not isinstance(row['environment'], dict):
            raise ValueError('Unknown physical/controller sample structure')
        expected = actor.observe(event['sensory'], event['physics_step'])
        if canonical(expected) != canonical(row['control']):
            raise ValueError('Stored food sample differs from controller replay')
    if record['cleanup_error'] is not None and (set(record['cleanup_error']) != {'type','message'}
            or not all(isinstance(value, str) for value in record['cleanup_error'].values())):
        raise ValueError('Invalid environment cleanup failure record')
    accepted = restored.status in ('contact','timeout') and record['cleanup_error'] is None
    expected_readout = restored.learner.state() if accepted else template.initial_readout
    if record['readout'] != expected_readout:
        raise ValueError('Stored food readout differs from the accepted terminal update')
    return record


def stored_episode(directory, template, environment_factory):
    """Return a durable terminal record, or restart an unfinished attempt once.

    The caller supplies a fresh FoodEpisode and a fresh-environment factory whose
    identity() must equal template.identity['binding']['environment']. Neither
    the supplied template nor its readout is mutated. Carry returned ['readout']
    to a distinct next episode only under the caller's complete study schedule.
    Failed/invalid/cleanup-failed outcomes return the original readout unchanged.
    BaseException (including cancellation) propagates; its unfinished attempt is
    retained and classified on the next call after acquiring the actual OS lock.
    """
    if not isinstance(template, FoodEpisode) or template.events or template.status != 'active':
        raise ValueError('A fresh declared food episode is required')
    environment_identity = template.identity['binding'].get('environment')
    if not isinstance(environment_identity, dict) or not environment_identity or not callable(environment_factory):
        raise ValueError('An explicit environment identity and fresh-environment factory are required')
    initial = template.state()
    identity = plain(dict(format='flm-food-episode-store-v1', initial_controller=initial,
        store_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        restart='Original inputs and random seeds; no simulator or gait checkpoint resumption'))
    directory = Path(directory)
    with episode_lease(directory):
        identity_path = directory/'identity.json'
        if identity_path.exists():
            if read(identity_path) != identity: raise ValueError('Stored episode identity changed')
        else:
            if any(path.name != 'writer.lock' and not path.name.startswith('identity.json.pending-') for path in directory.iterdir()):
                raise ValueError('Existing episode artifacts have no durable identity')
            write_new(identity_path, identity)
        attempts = []
        for path in directory.iterdir():
            if path.name in ('writer.lock','identity.json') or path.name.startswith('identity.json.pending-'): continue
            if not re.fullmatch(r'attempt-\d{6}', path.name) or not path.is_dir():
                raise ValueError('Unexpected stored episode artifact: '+path.name)
            attempts.append(path)
        attempts.sort()
        if [path.name for path in attempts] != [f'attempt-{number:06d}' for number in range(1,len(attempts)+1)]:
            raise ValueError('Stored food episode attempt inventory has a gap')
        for number, path in enumerate(attempts, 1):
            allowed = ('start.json','interrupted.json','result.json.gz')
            if any(item.name not in allowed and not any(item.name.startswith(name+'.pending-') for name in allowed)
                   for item in path.iterdir()):
                raise ValueError('Unknown food attempt artifact')
            start_path = path/'start.json'
            if start_path.exists():
                start = read(start_path)
                if (set(start) != {'attempt','identity_sha256','started_utc'} or start['attempt'] != number
                        or start['identity_sha256'] != digest(identity)):
                    raise ValueError('Food attempt starting identity changed')
            result_path = path/'result.json.gz'
            if result_path.exists():
                if number != len(attempts) or not start_path.exists() or (path/'interrupted.json').exists():
                    raise ValueError('Terminal food attempt has an inconsistent inventory')
                return replay_result(template, read(result_path, compressed=True), identity, number)
            interruption = dict(attempt=number, identity_sha256=digest(identity),
                status='unfinished', physics_steps_and_reward='Unknown: no durable terminal result',
                action='Restart original inputs; preserve this attempt; apply no readout update')
            interrupted = path/'interrupted.json'
            if interrupted.exists():
                if read(interrupted) != interruption: raise ValueError('Food interruption record changed')
            else: write_new(interrupted, interruption)
        number = len(attempts)+1
        if number > 999999: raise ValueError('Food episode attempt inventory exhausted')
        attempt = directory/f'attempt-{number:06d}'; attempt.mkdir()
        write_new(attempt/'start.json', dict(attempt=number, identity_sha256=digest(identity), started_utc=utc()))
        actor = FoodEpisode.restore(template.core.weights, initial, binding=template.identity['binding'])
        environment = None; cleanup_error = None
        try:
            environment = environment_factory()
            if plain(environment.identity()) != environment_identity:
                raise ValueError('Constructed food environment differs from the declared identity')
            result = run_episode(actor, environment)
        except Exception as error:
            if actor.status != 'active': raise
            actor.abort(type(error).__name__, str(error))
            result = dict(outcome=plain(actor.outcome), samples=[], controller=actor.state(),
                          scope='Environment construction failed before physical rollout')
        finally:
            if environment is not None:
                try: environment.close()
                except Exception as error: cleanup_error = dict(type=type(error).__name__, message=str(error))
        accepted = actor.status in ('contact','timeout') and cleanup_error is None
        record = json.loads(canonical(dict(identity_sha256=digest(identity), attempt=number, finished_utc=utc(),
            rollout=result, cleanup_error=cleanup_error,
            readout=actor.learner.state() if accepted else template.initial_readout)))
        replay_result(template, record, identity, number)
        write_new(attempt/'result.json.gz', record, compressed=True)
        # Read back the committed bytes before returning a state for the next episode.
        return replay_result(template, read(attempt/'result.json.gz', compressed=True), identity, number)
