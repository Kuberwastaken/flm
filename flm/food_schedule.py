"""Immutable multi-condition food schedules over durable episode boundaries.

This execution layer does not choose an official protocol, cost budget, layouts
or study prerequisites. The caller must register those before physical fitting.
All declared training finishes before any evaluation environment is constructed.
"""
import hashlib
from pathlib import Path
import re

import numpy as np

from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from .food_episode import Clock, FoodEpisode, Rewards, array_digest, plain
from .food_episode_store import canonical, digest, episode_lease, read, stored_episode, utc, write_new
from .food_readout_learning import EpisodicReadout, Rule


SOURCES = ('flm/food_schedule.py','flm/food_episode.py','flm/food_episode_store.py',
           'flm/food_readout_learning.py','flm/food_motor.py','experiments/embodiment/food_core_runtime.py')
ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def identifiers(rows, keys):
    if not isinstance(rows,list) or not rows: raise ValueError('Explicit nonempty inventory required')
    for row in rows:
        if (not isinstance(row,dict) or set(row) != set(keys)
                or not isinstance(row['id'],str) or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,79}',row['id'])):
            raise ValueError('Unknown inventory fields or unsafe identifier')
    if len({row['id'] for row in rows}) != len(rows): raise ValueError('Repeated inventory identifier')


def declaration(request, weights):
    """Bind an explicit complete schedule and paired frozen interfaces to sources."""
    request = plain(request)
    if (set(request) != {'purpose','protocol','clock','rewards','models','methods','environments','streams'}
            or not isinstance(request['purpose'],str) or not request['purpose']
            or not isinstance(request['protocol'],dict) or not request['protocol']):
        raise ValueError('An explicit purpose, protocol binding and complete schedule are required')
    clock = Clock(**request['clock']); clock.validate()
    rewards = Rewards(**request['rewards']); rewards.validate()
    models = request['models']; methods = request['methods']; streams = request['streams']
    identifiers(models,('id','pair','role','source'))
    identifiers(methods,('id','learning','rule'))
    identifiers(streams,('id','training','probes','checkpoints'))
    if set(weights) != {model['id'] for model in models}: raise ValueError('Frozen core inventory differs from request')
    pairs = {}; core_hashes = {}; checked = {}
    for model in models:
        if (model['role'] not in ('initial','language') or not isinstance(model['pair'],str) or not model['pair']
                or not isinstance(model['source'],dict) or not model['source']):
            raise ValueError('Each core needs an initial/language pair and a source binding')
        pair = pairs.setdefault(model['pair'],{})
        if model['role'] in pair: raise ValueError('A core pair contains repeated roles')
        pair[model['role']] = model['id']
        checked[model['id']] = FoodCoreRuntime(weights[model['id']]).weights
        core_hashes[model['id']] = array_digest(checked[model['id']])
    for pair in pairs.values():
        if set(pair) != {'initial','language'}: raise ValueError('Every initial core needs its paired language core')
        initial,trained = (checked[pair[role]] for role in ('initial','language'))
        for name in ('sensor_weight','sensor_bias','action_weight','action_bias','body_ids','pool_index','pool_sizes'):
            if not np.array_equal(initial[name],trained[name]):
                raise ValueError('Paired cores must share the sensory/action interface and neuron/pool identity: '+name)
    for method in methods:
        rule = Rule(**method['rule']); rule.validate()
        if type(method['learning']) is not bool or rule.maximum_decisions != clock.maximum_physics_steps//clock.steps_per_action:
            raise ValueError('Explicit learning flags and matching per-episode decision budgets required')
    environments = request['environments']
    if (not isinstance(environments,dict) or not environments
            or any(not isinstance(key,str) or not key or not isinstance(value,dict) or not value
                   for key,value in environments.items())):
        raise ValueError('Explicit environment identities required')
    used = set(); common = None
    for stream in streams:
        for split in ('training','probes'):
            episodes = stream[split]
            identifiers(episodes,('id','phase','environment','action_seed'))
            for episode in episodes:
                if (not isinstance(episode['phase'],str) or not episode['phase']
                        or episode['environment'] not in environments or type(episode['action_seed']) is not int
                        or not 0 <= episode['action_seed'] < 2**64):
                    raise ValueError('Every episode needs an explicit phase, environment and action seed')
                used.add(episode['environment'])
        points = stream['checkpoints']; length = len(stream['training'])
        if (not isinstance(points,list) or any(type(point) is not int for point in points)
                or points != sorted(set(points)) or not points or points[0] != 0 or points[-1] != length):
            raise ValueError('Predeclare ordered evaluation checkpoints including initial and final episode counts')
        # Model, rule and stream comparisons get equal declared opportunities.
        # Outcomes can still shorten episodes or cause failures; report actual exposure.
        structure = dict(training=[row['phase'] for row in stream['training']],
                         probes=[row['phase'] for row in stream['probes']],checkpoints=points)
        if common is None: common = structure
        elif structure != common: raise ValueError('Streams must share phase order, episode counts and checkpoints')
    if used != set(environments): raise ValueError('Environment inventory contains unused entries')
    conditions = [dict(id=f'condition-{index:06d}',model=model['id'],method=method['id'],stream=stream['id'])
                  for index,(model,method,stream) in enumerate(
                      ((model,method,stream) for model in models for method in methods for stream in streams),1)]
    return dict(format='flm-food-schedule-v1',request=request,core_sha256=core_hashes,conditions=conditions,
        source_sha256={name:sha(ROOT/name) for name in SOURCES},numpy=str(np.__version__),
        policy=dict(training='Declared episode order; failures consume their slot without a readout update',
            evaluation='All training conditions complete first; each probe starts from the declared checkpoint readout',
            restart='Fresh core state and declared action/gait seeds; unfinished physical attempts restart original inputs',
            selection='No reward-based stopping, checkpoint choice or evaluation-to-training update',
            scope='Schedule execution and records. Official scientific membership, cost and prerequisite evidence belong to the registering caller'))


def initialize(directory, request, weights):
    identity = declaration(request,weights); directory = Path(directory)
    with episode_lease(directory):
        path = directory/'identity.json'
        if path.exists():
            if read(path) != identity: raise ValueError('A different food schedule is already frozen')
        else:
            if any(item.name != 'writer.lock' and not item.name.startswith('identity.json.pending-') for item in directory.iterdir()):
                raise ValueError('Food schedule artifacts have no durable identity')
            write_new(path,identity)
    return identity


def context(directory, request, weights):
    directory = Path(directory)
    if not (directory/'identity.json').is_file(): raise ValueError('Freeze the complete food schedule before executing it')
    identity = read(directory/'identity.json')
    if identity != declaration(request,weights): raise ValueError('Frozen food schedule, interfaces or implementation changed')
    return identity


def lookup(identity, field, key):
    return next(row for row in identity['request'][field] if row['id']==key)


def initial_readout(identity, condition, weights):
    core = weights[condition['model']]; method = lookup(identity,'methods',condition['method'])
    return EpisodicReadout(core['action_weight'],core['action_bias'],Rule(**method['rule'])).state()


def template_for(identity,condition,episode,readout,weights,*,phase,checkpoint=None):
    request = identity['request']; method = lookup(identity,'methods',condition['method'])
    return FoodEpisode(weights[condition['model']],readout,Clock(**request['clock']),Rewards(**request['rewards']),
        seed=episode['action_seed'],learning=method['learning'] if phase=='training' else False,
        binding=dict(schedule_sha256=digest(identity),condition=condition,phase=phase,checkpoint=checkpoint,
                     episode=episode,environment=request['environments'][episode['environment']]))


def terminal(path, template, factory, *, allow_new):
    exists = list(path.glob('attempt-*/result.json.gz')) if path.exists() else []
    if not exists and not allow_new: raise ValueError('Complete every declared training episode before evaluation')
    def forbidden(): raise AssertionError('A declared existing terminal episode attempted new simulation')
    result = stored_episode(path,template,(lambda:factory()) if allow_new else forbidden)
    number = result['attempt']; record_path = path/f'attempt-{number:06d}'/'result.json.gz'
    return result,dict(terminal_sha256=sha(record_path),attempt=number,
        outcome=result['rollout']['outcome'],cleanup_error=result['cleanup_error'],readout_sha256=digest(result['readout']))


def inventory(directory, identity, phase):
    base = Path(directory)/phase
    if not base.exists(): return
    expected = {}
    for condition in identity['conditions']:
        stream = lookup(identity,'streams',condition['stream'])
        count = len(stream['training']) if phase=='training' else len(stream['probes'])*len(stream['checkpoints'])
        expected[condition['id']] = {f'episode-{index:06d}' for index in range(1,count+1)}
    for condition_path in base.iterdir():
        if not condition_path.is_dir() or condition_path.name not in expected:
            raise ValueError('Unexpected food condition artifact')
        if any(not path.is_dir() or path.name not in expected[condition_path.name] for path in condition_path.iterdir()):
            raise ValueError('Unexpected food episode artifact')


def training_records(directory, identity, weights, factory, *, max_new=None, verify_only=False):
    inventory(directory,identity,'training')
    rows = []; checkpoints = {}; new = 0
    for condition in identity['conditions']:
        readout = initial_readout(identity,condition,weights)
        stream = lookup(identity,'streams',condition['stream']); saved = {'0':plain(readout)}
        chain = []
        for index,episode in enumerate(stream['training'],1):
            path = Path(directory)/'training'/condition['id']/f'episode-{index:06d}'
            complete = path.exists() and bool(list(path.glob('attempt-*/result.json.gz')))
            if not complete and max_new is not None and new >= max_new: return None,new
            template = template_for(identity,condition,episode,readout,weights,phase='training')
            result,row = terminal(path,template,lambda:factory(episode['environment']),allow_new=not verify_only)
            if not complete: new += 1
            readout = result['readout']; chain.append(dict(index=index,episode=episode['id'],**row))
            if index in stream['checkpoints']: saved[str(index)] = plain(readout)
        rows.append(dict(condition=condition,episodes=chain,final_readout_sha256=digest(readout)))
        checkpoints[condition['id']] = saved
    inventory(directory,identity,'training')
    return dict(schedule_sha256=digest(identity),conditions=rows,checkpoints=checkpoints),new


def check_limit(max_new):
    if max_new is not None and (type(max_new) is not int or max_new < 1):
        raise ValueError('A positive optional episode limit is required')


def train(directory, request, weights, factory, *, max_new=None):
    """Bounded calls stop only between episodes; every saved failed slot is retained."""
    check_limit(max_new); directory = Path(directory)
    with episode_lease(directory):
        identity = context(directory,request,weights)
        complete = directory/'training-complete.json'
        result,new = training_records(directory,identity,weights,factory,
            max_new=max_new,verify_only=complete.exists())
        context(directory,request,weights)
        if result is None: return dict(status='partial',new_episodes=new)
        if complete.exists():
            if read(complete)['records'] != result: raise ValueError('Frozen food training completion changed')
        else: write_new(complete,dict(completed_utc=utc(),records=result))
        return dict(status='complete',new_episodes=new,conditions=len(result['conditions']))


def evaluate(directory, request, weights, factory, *, max_new=None):
    """Run every predeclared checkpoint/probe after whole-training verification."""
    check_limit(max_new); directory = Path(directory)
    with episode_lease(directory):
        identity = context(directory,request,weights)
        completion = directory/'training-complete.json'
        if not completion.exists(): raise ValueError('Complete all declared food training conditions before evaluation')
        training,_ = training_records(directory,identity,weights,factory,verify_only=True)
        if read(completion)['records'] != training: raise ValueError('Frozen food training completion changed')
        training_hash = sha(completion); inventory(directory,identity,'evaluation')
        rows = []; new = 0; final = directory/'evaluation-complete.json'
        for condition in identity['conditions']:
            stream = lookup(identity,'streams',condition['stream']); position = 0
            for checkpoint in stream['checkpoints']:
                readout = training['checkpoints'][condition['id']][str(checkpoint)]
                for episode in stream['probes']:
                    position += 1
                    path = directory/'evaluation'/condition['id']/f'episode-{position:06d}'
                    complete = path.exists() and bool(list(path.glob('attempt-*/result.json.gz')))
                    if not complete and max_new is not None and new >= max_new:
                        context(directory,request,weights)
                        return dict(status='partial',new_episodes=new)
                    template = template_for(identity,condition,episode,readout,weights,phase='evaluation',checkpoint=checkpoint)
                    result,row = terminal(path,template,lambda:factory(episode['environment']),allow_new=not final.exists())
                    if result['readout'] != readout: raise ValueError('An evaluation probe changed its starting readout')
                    if not complete: new += 1
                    rows.append(dict(condition=condition,checkpoint=checkpoint,episode=episode['id'],**row))
        inventory(directory,identity,'evaluation')
        context(directory,request,weights)
        if sha(completion) != training_hash: raise ValueError('Food training completion changed during evaluation')
        record = dict(schedule_sha256=digest(identity),training_complete_sha256=training_hash,episodes=rows)
        if final.exists():
            if read(final)['records'] != record: raise ValueError('Frozen food evaluation completion changed')
        else: write_new(final,dict(completed_utc=utc(),records=record))
        return dict(status='complete',new_episodes=new,evaluated_episodes=len(rows))
