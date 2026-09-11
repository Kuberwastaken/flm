"""Package the complete short, no-learning food-episode integration records."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ('initial-s42','language-s42','initial-s43','language-s43')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def package(folder,output):
    summary = json.loads((folder/'summary.json').read_bytes())
    labels = [model+f'-repeat{repeat}' for model in MODEL_IDS for repeat in (0,1)]
    if ([row['label'] for row in summary['trials']] != labels or summary['complete'] != 8
            or summary['failed'] != 0 or summary['task_training_updates'] != 0
            or summary['physical_steps_excluding_warmup'] != 1600
            or digest(folder/'identity.json') != summary['identity_sha256']
            or json.loads((folder/'identity.json').read_bytes()) != summary['identity']
            or [row['model_id'] for row in summary['paired_repeats']] != list(MODEL_IDS)
            or any(not all(row['exact_array_equal'].values()) for row in summary['paired_repeats'])):
        raise ValueError('Require the complete eight-rollout no-learning verification')
    names = {'identity.json','summary.json'}
    for row in summary['trials']:
        label = row['label']
        if (json.loads((folder/(label+'.json')).read_bytes()) != row
                or row['status'] != 'complete' or row['physics_steps'] != 200 or row['observations'] != 3
                or any(row[key] is not True for key in ('core_unchanged','readout_unchanged','controller_replay_exact'))
                or digest(folder/(label+'.npz')) != row['trajectory_sha256']
                or digest(folder/(label+'-controller.json')) != row['controller_sha256']):
            raise ValueError('A physical smoke record changed')
        names.update([label+'.json',label+'.npz',label+'-controller.json'])
    if {path.name for path in folder.iterdir()} != names:
        raise ValueError('Unexpected or missing smoke record inventory')
    contents = {name:(folder/name).read_bytes() for name in sorted(names)}
    contents['LICENSE'] = (ROOT/'LICENSE').read_bytes()
    for path in (ROOT/'licenses').iterdir():
        if path.suffix in ('.md','.txt'): contents['licenses/'+path.name] = path.read_bytes()
    contents['README.md'] = (
        '# Short physical food-episode integration records\n\n'
        'Kuber Mehta. Eight 20-ms post-warmup rollouts; four fixed exported cores, each repeated. '
        'Learning was disabled. These records demonstrate connection and repeatability, not navigation, food-task learning or language benefit.\n\n'
        'Each NPZ contains three observations and only two applied descending commands. The final action/uniform values of -1 mean no action/draw; '
        'the final zero drive is not applied. Controller JSON records the exact sensory/action history and frozen identities. '
        'The existing body-state limitation remains: the invalid c_head row aliases a hind foot. Use the recorded valid sensor origins; '
        'qpos and cached body positions describe different integration stages.\n\n'
        'The manifest checks bytes only. Source and reproduction instructions: '
        'https://flm.kuber.studio/research/food-episode-runner.md and '
        'https://flm.kuber.studio/research/paper-source.zip. '
        'The separate four-core interface package is required for numerical replay or physical reruns. '
        'Controller replay does not replay physics or restore a simulator/gait checkpoint.\n'
    ).encode('utf8')
    manifest = {name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in contents.items()}
    contents['record-manifest.json'] = (json.dumps(manifest,indent=2)+'\n').encode('utf8')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in sorted(contents.items()):
            info = zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16; archive.writestr(info,value)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(contents):
            raise ValueError('Smoke archive inventory or CRC failed')
        for name,value in contents.items():
            if archive.read(name) != value: raise ValueError('Smoke archive changed source bytes')
    print(json.dumps(dict(records=len(names),archive_members=len(contents),bytes=output.stat().st_size,sha256=digest(output))))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); package(args.input,args.output)
