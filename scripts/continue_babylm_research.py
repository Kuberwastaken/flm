"""Wait for the bound BabyLM trainer, then evaluate and measure selection costs.

No training budget or group is chosen here. This supervisor never restarts a
failed stage, modifies training sources, commits files or publishes results.
Install requirements-operations.txt when psutil is not already available.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

STAGES = (
    ('babylm-test','flm.babylm_test','reports/babylm/summary.json'),
    ('babylm-samples','flm.babylm_samples','reports/babylm/samples.json'),
    ('selection-pilot','flm.selection_pilot','reports/selection-pilot/timing.json'))
SOURCES = (
    'scripts/continue_babylm_research.py','requirements-operations.txt',
    'flm/babylm_suite.py','flm/babylm_test.py','flm/babylm_samples.py','flm/babylm.py','flm/corpus_evaluation.py',
    'flm/generation_audit.py','flm/language_train.py','flm/model.py','flm/baselines.py',
    'flm/provenance.py','flm/tokenizer.py','flm/graph.py','flm/train.py','flm/corpus_cache.py',
    'flm/selection_pilot.py','flm/scan_train.py','flm/inference.py','flm/language_learning_train.py',
    'flm/language_eligibility.py','flm/embedding_eligibility.py','flm/local_learning.py',
    'docs/BABYLM-PROTOCOL.md','docs/BABYLM-EVALUATION.md','docs/SELECTION-TIMING-PILOT.md',
    'data/prompts/babylm-original.json')
BUSY_MODULES = frozenset(('flm.babylm_suite','flm.language_train','flm.babylm_test','flm.babylm_samples',
    'flm.selection_pilot','flm.selection_language_study','flm.selection_language_test',
    'flm.language_learning_study','flm.scan_study','flm.scan_evaluate','unittest'))
RUNS = {(scale,variant,seed) for scale in ('10m','100m') for variant in ('flm','gru','transformer') for seed in (42,43)}


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def package_versions(): return {name:importlib.metadata.version(name) for name in ('numpy','torch','scipy','tokenizers','psutil')}


def module_name(argv):
    return argv[argv.index('-m')+1] if '-m' in argv and argv.index('-m')+1 < len(argv) else None


def process_identity(process):
    return dict(pid=process.pid,created=process.create_time(),argv=process.cmdline(),cwd=process.cwd())


def same_directory(first,second): return Path(first).resolve() == Path(second).resolve()


def capture_trainer(root,pid):
    identity = process_identity(psutil.Process(pid))
    if module_name(identity['argv']) != 'flm.babylm_suite' or not same_directory(identity['cwd'],root):
        raise ValueError('Bind the live BabyLM suite process in this exact repository')
    return identity


def live_process(identity):
    try:
        process = psutil.Process(identity['pid'])
        if process.create_time() != identity['created']: return None  # The original PID has exited and been reused.
        if process.cmdline() != identity['argv'] or not same_directory(process.cwd(),identity['cwd']):
            raise ValueError('A bound process changed its command or working directory')
        return process
    except psutil.NoSuchProcess: return None
    # AccessDenied and inspection errors propagate; they never mean "exited".


def wait_for_exit(target,event,*,sleep=time.sleep):
    tracked = {(target['pid'],target['created']):target}; announced = set()
    while True:
        active = []
        for identity in list(tracked.values()):
            process = live_process(identity)
            if process is None: continue
            active.append(identity)
            try:
                for child in process.children(recursive=True):
                    try: observed = process_identity(child)
                    except psutil.NoSuchProcess: continue
                    tracked[(observed['pid'],observed['created'])] = observed
            except psutil.NoSuchProcess: continue
        fresh = set(tracked)-announced
        if fresh:
            event('watching_processes',processes=[tracked[key] for key in sorted(fresh)])
            announced.update(fresh)
        if not active:
            # A newly observed descendant may outlive its suite parent.
            if any(live_process(value) is not None for value in tracked.values()): continue
            event('trainer_and_observed_children_exited',processes=len(tracked)); return
        sleep(5)


def busy_processes(root):
    found = []
    for process in psutil.process_iter():
        if process.pid == os.getpid(): continue
        argv = []
        try:
            argv = process.cmdline()
            if module_name(argv) in BUSY_MODULES and same_directory(process.cwd(),root):
                found.append(process_identity(process))
        except psutil.NoSuchProcess: continue
        except psutil.AccessDenied:
            # Unreadable unrelated/system commands cannot be identified as project work.
            # If a known project command was identified, do not infer that it stopped.
            if module_name(argv) in BUSY_MODULES: raise
    return sorted(found,key=lambda row:(row['pid'],row['created']))


def wait_for_idle(root,event,*,sleep=time.sleep):
    previous = None
    while True:
        active = busy_processes(root)
        if not active: return
        if active != previous: event('waiting_for_known_project_work',processes=active); previous=active
        sleep(5)


@contextmanager
def lease(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error: raise RuntimeError('Another post-BabyLM supervisor holds the repository lease') from error
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def verify_sources(root,identity):
    for name,expected in identity['source_sha256'].items():
        if sha(root/name) != expected: raise ValueError('Bound research source changed: '+name)
    if package_versions() != identity['package_versions']: raise ValueError('Research package versions changed during the handoff')


def verify_training(root):
    for scale,variant,seed in sorted(RUNS):
        path=root/f'runs/babylm-{scale}/{variant}-s{seed}/complete.json'
        if not path.is_file(): raise ValueError('The trainer exited before all twelve BabyLM fits completed')
        record=read(path)
        if record['steps'] != 12000 or record['best_checkpoint_sha256'] != sha(path.with_name('best.pt')):
            raise ValueError('A BabyLM completion or selected-checkpoint checksum changed')
    # Full selected/final tensor and validation audits remain inside babylm_test's gate.


def verify_output(root,stage):
    name,_,relative=stage; path=root/relative; record=read(path)
    if name=='babylm-test':
        rows=record['runs']
        if len(rows)!=12 or {(row['scale'],row['variant'],row['seed']) for row in rows} != RUNS:
            raise ValueError('Incomplete BabyLM test output inventory')
    elif name=='babylm-samples':
        rows=record['models']; prompts=read(root/'data/prompts/babylm-original.json')['prompts']
        if len(prompts)!=12 or len({row['id'] for row in prompts})!=12:
            raise ValueError('Original prompt inventory changed')
        if len(rows)!=12 or {(row['selected']['scale'],row['selected']['variant'],row['selected']['seed']) for row in rows} != RUNS:
            raise ValueError('Incomplete BabyLM sampling model inventory')
        expected=[(prompt['id'],seed) for prompt in prompts for seed in (17,29)]
        if any([(item['prompt_id'],item['sampling_seed']) for item in row['passages']] != expected for row in rows):
            raise ValueError('Incomplete or reordered original continuation panel')
    elif name=='selection-pilot':
        rows=record['conditions']
        if (len(rows)!=64 or len({row['selection'] for row in rows})!=64
                or record['benchmark_budget_selected'] is not False or record['training_matrix_frozen'] is not False):
            raise ValueError('Incomplete disposable selection cost inventory')
    else: raise ValueError('Unknown post-training stage')
    return dict(path=relative,sha256=sha(path))


def execute_stage(root,stage,output,event):
    command=[sys.executable,'-u','-X','utf8','-m',stage[1]]
    with (output/(stage[0]+'.log')).open('xb') as log:
        process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try: created=psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess: created=None
        event('stage_process_started',stage=stage[0],pid=process.pid,created=created,command=command)
        code=process.wait()
    if code: raise subprocess.CalledProcessError(code,command)


def run(root,pid,output,*,runner=execute_stage,waiter=wait_for_exit,idle=wait_for_idle):
    root=Path(root).resolve(); output=Path(output).resolve()
    target=capture_trainer(root,pid)  # Must actually be live at arming, not just a state file.
    identity=dict(format='flm-after-babylm-v1',armed_utc=utc(),trainer=target,stages=STAGES,
        source_sha256={name:sha(root/name) for name in SOURCES},python=sys.version,psutil=psutil.__version__,package_versions=package_versions(),
        scope='Serial operational handoff. No group/budget choice or automatic stage retry. Known project commands are observed; general OS idleness is not guaranteed.')
    with lease(root/'work/babylm-post-training.lock'):
        output.mkdir(parents=True,exist_ok=False)
        (output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf8')
        def event(kind,**details):
            record=dict(utc=utc(),event=kind,**details)
            with (output/'events.jsonl').open('a',encoding='utf8') as log:
                log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush(); os.fsync(log.fileno())
            print(json.dumps(record,allow_nan=False),flush=True)
        try:
            event('armed',trainer=target)
            waiter(target,event)
            results=[]
            for stage in STAGES:
                idle(root,event); verify_sources(root,identity); verify_training(root)
                if (root/stage[2]).exists():
                    raise ValueError('Stage output already exists; inspect it rather than rerunning: '+stage[2])
                event('stage_start',stage=stage[0]); runner(root,stage,output,event)
                verify_sources(root,identity); result=verify_output(root,stage)
                results.append(result); event('stage_complete',stage=stage[0],result=result)
            (output/'complete.json').write_text(json.dumps(dict(completed_utc=utc(),results=results),indent=2)+'\n',encoding='utf8')
            event('complete',next_action='Review the measured selection costs, then choose complete groups and freeze the protocol before fitting')
        except BaseException as error:
            event('stopped',error_type=type(error).__name__,error=str(error),
                instruction='Inspect recorded stage child PIDs before any retry; supervisor interruption alone does not prove a child has exited')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainer-pid',type=int,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(Path(__file__).resolve().parents[1],args.trainer_pid,args.output)
