"""Audit and package every short physical restart attempt; no new physics."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.food_core_runtime import FoodCoreRuntime
from flm.food_episode import FoodEpisode
from flm.food_episode_store import canonical, stored_episode
from flm.food_sensors import FoodField, Source, body_sensor_positions

MODELS = ('initial-s42','language-s42','initial-s43','language-s43')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder,bundle):
    summary = json.loads((folder/'summary.json').read_bytes())
    identity = json.loads((folder/'identity.json').read_bytes())
    if (summary['complete'] != 4 or summary['failed'] or summary['task_training_updates']
            or [row['model_id'] for row in summary['checks']] != list(MODELS)
            or sha(folder/'identity.json') != summary['identity_sha256']
            or sha(bundle/'manifest.json') != identity['bundle_manifest_sha256']
            or identity['learning'] is not False or identity['task_training_updates'] != 0):
        raise ValueError('Complete four-core no-learning restart verification required')
    for name,value in identity['source_sha256'].items():
        if sha(ROOT/name) != value: raise ValueError('Use the declared source version: '+name)
    names = {'identity.json','summary.json'}; observations = 0
    for report in summary['checks']:
        model_id = report['model_id']; names.add(model_id+'.json')
        if (json.loads((folder/(model_id+'.json')).read_bytes()) != report or report['status'] != 'complete'
                or report['post_warmup_physics_steps'] != 500 or report['complete_episodes'] != 2
                or report['interrupted_attempts'] != 1):
            raise ValueError('Physical restart report changed')
        results = []; core = FoodCoreRuntime.load(bundle,model_id)
        for branch,attempt in (('direct',1),('restarted',2)):
            path = folder/model_id/branch
            names.update(f'{model_id}/{branch}/{name}' for name in ('identity.json','writer.lock'))
            for number in range(1,attempt+1):
                prefix = f'{model_id}/{branch}/attempt-{number:06d}/'
                names.add(prefix+'start.json')
                names.add(prefix+('result.json.gz' if number==attempt else 'interrupted.json'))
            saved = json.loads((path/'identity.json').read_bytes())['initial_controller']
            template = FoodEpisode.restore(core.weights,saved,binding=saved['identity']['binding'])
            if template.learning or saved['identity']['binding']['environment'] != report['environment_identity']:
                raise ValueError('Unexpected learning or environmental identity')
            def forbidden(): raise AssertionError('Numerical record verification must not run physics')
            result = stored_episode(path,template,forbidden)
            if (result['attempt'] != attempt or result['readout'] != template.initial_readout
                    or result['cleanup_error'] is not None or result['rollout']['outcome']['status'] != 'timeout'
                    or sha(path/f'attempt-{attempt:06d}'/'result.json.gz') != report['result_sha256'][branch]):
                raise ValueError('Physical terminal record changed')
            environment = report['environment_identity']
            field = FoodField([Source(**{key:tuple(value) if isinstance(value,list) else value
                                       for key,value in row.items()}) for row in environment['field']['sources']])
            rows = result['rollout']['samples']
            if len(rows) != 3: raise ValueError('Incomplete short physical record')
            for row in rows:
                physical = row['environment']; control = row['control']
                antennae,feet = body_sensor_positions(physical['body_names'],physical['body_positions_mm'])
                expected = field.observe(antennae,feet,missing_odor=environment['missing_odor'])
                for key,actual in (('sensory',control['sensory']),('raw_odor',physical['raw_odor']),('contact_mask',physical['contact_mask'])):
                    if not np.array_equal(expected[key],actual): raise ValueError('Stored physical sensor geometry changed')
                if physical['physics_step'] != control['physics_step'] or physical['time_s'] != control['time_s']:
                    raise ValueError('Stored physical and control clocks differ')
                observations += 1
            results.append(result)
        if results[0]['rollout'] != results[1]['rollout'] or results[0]['readout'] != results[1]['readout']:
            raise ValueError('Restarted physical record differs from direct execution')
    if {path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file()} != names:
        raise ValueError('Incomplete or extra physical restart record inventory')
    return names,dict(observations=observations,complete_episodes=8,interrupted_attempts=4,
        matched_physical_restarts=4,new_physics_steps=0,task_training_updates=0)


def package(folder,bundle,output,report_path):
    names,checks = audit(folder,bundle)
    contents = {'records/'+name:(folder/name).read_bytes() for name in sorted(names)}
    contents['LICENSE'] = (ROOT/'LICENSE').read_bytes()
    for path in (ROOT/'licenses').iterdir():
        if path.suffix in ('.md','.txt'): contents['licenses/'+path.name] = path.read_bytes()
    contents['README.md'] = (
        '# Physical food episode restart records\n\nKuber Mehta. '
        'Four fixed cores: direct 20-ms run, interrupted 10-ms attempt, fresh 20-ms restart. '
        'All durations exclude factory warmup. Learning disabled; no navigation or adaptation claim.\n\n'
        'Every store identity, attempt start, interruption marker and terminal result is retained. '
        'A writer.lock file does not mean a process is running. Interrupted physical samples are not checkpointed; '
        'the deliberate interruption boundary was observed by the verification harness. '
        'Complete result.json.gz records include actual body observations, sensor channels and neural/controller states. '
        'The known cached head-row alias and different qpos/body-cache integration stages still apply.\n\n'
        'Extract this archive and use scripts/package_food_episode_store.py with the matching research sources, '
        'recorded NumPy version and separate food-core-interface.zip to verify all terminal controller/geometry records '
        'without running physics. See https://flm.kuber.studio/research/food-episode-runner.md and '
        'https://flm.kuber.studio/research/paper-source.zip. '
        'For that audit, point --input at the extracted records/ folder.\n'
    ).encode('utf8')
    manifest = {name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in contents.items()}
    contents['record-manifest.json'] = canonical(manifest)+b'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in sorted(contents.items()):
            info = zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16; archive.writestr(info,value)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(contents):
            raise ValueError('Physical restart archive inventory or CRC failed')
        for name,value in contents.items():
            if archive.read(name) != value: raise ValueError('Physical restart archive changed source bytes')
    record = dict(verified_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
        records=len(names),archive_members=len(contents),bytes=output.stat().st_size,sha256=sha(output),
        bundle_manifest_sha256=sha(bundle/'manifest.json'),auditor_sha256=sha(Path(__file__)),
        numpy=str(np.__version__),source_identity=json.loads((folder/'identity.json').read_bytes()),
        scope='Whole physical record inventory, exact terminal controller replay and saved sensor geometry; same numerical routines, no new physics or independent simulator implementation.')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open('x',encoding='utf8',newline='\n') as handle: json.dump(record,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(checks=checks,records=len(names),bytes=record['bytes'],sha256=record['sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','bundle','output','report'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); package(args.input,args.bundle,args.output,args.report)
