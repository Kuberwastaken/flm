"""Extract and rerun the complete physical-record audit outside the repository."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def verify(executable, extract_to, output):
    if extract_to.exists() or output.exists():
        raise ValueError('Use new extraction and proof paths; preserve earlier verification')
    release = json.loads((ROOT/'reports/food-core-physical/release.json').read_bytes())
    archive_path = ROOT/'public/research/food-core-physical-records.zip'
    expected = release['files'][archive_path.relative_to(ROOT).as_posix()]
    payload = archive_path.read_bytes()
    if len(payload)!=expected['bytes'] or sha(payload)!=expected['sha256']:
        raise ValueError('Physical archive differs from its release record')
    extract_to = extract_to.resolve(); extract_to.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names)!=len(set(names)) or archive.testzip() is not None:
            raise ValueError('Repeated archive name or CRC failure')
        inventory = json.loads(archive.read('archive-manifest.json'))
        if set(names)!={*inventory,'archive-manifest.json'} or len(inventory)!=release['archive_manifest_files']:
            raise ValueError('Archive member inventory changed')
        for item in archive.infolist():
            name=item.filename
            if ('\\' in name or ':' in name or name.startswith('/') or '..' in Path(name).parts
                    or not (extract_to/name).resolve().is_relative_to(extract_to)
                    or (item.external_attr>>16)&0o170000==0o120000):
                raise ValueError('Nonlocal or symlink archive member')
        for name, record in inventory.items():
            data=archive.read(name)
            if len(data)!=record['bytes'] or sha(data)!=record['sha256']:
                raise ValueError('Archive content differs: '+name)
        archive.extractall(extract_to)
    for name,record in inventory.items():
        if sha((extract_to/name).read_bytes())!=record['sha256']:
            raise ValueError('Extracted member differs: '+name)
    for name in ('identity.json','summary.json','audit.json'):
        if (extract_to/'records'/name).read_bytes()!=(ROOT/'reports/food-core-physical'/name).read_bytes():
            raise ValueError('Extracted result record differs: '+name)
    manifest=json.loads((extract_to/'core/manifest.json').read_bytes())
    identity=json.loads((extract_to/'records/identity.json').read_bytes())
    sources={**manifest['sources'],**identity['sources']}
    for name,digest in sources.items():
        if sha((extract_to/'sources'/name).read_bytes())!=digest:
            raise ValueError('Archived bound source differs: '+name)
    env=os.environ.copy()
    env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
    env['PYTHONNOUSERSITE']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[key]='1'
    command=[str(executable.resolve()),'audit_food_core_physical.py','records','--bundle','core','--output','fresh-audit.json']
    result=subprocess.run(command,cwd=extract_to,env=env,capture_output=True,text=True,check=False)
    (extract_to/'verification-stdout.txt').write_text(result.stdout,encoding='utf8')
    (extract_to/'verification-stderr.txt').write_text(result.stderr,encoding='utf8')
    if result.returncode!=0:
        raise RuntimeError('Extracted physical audit failed; inspect preserved verification-stdout/stderr.txt')
    fresh=json.loads((extract_to/'fresh-audit.json').read_bytes())
    prior=json.loads((ROOT/'reports/food-core-physical/audit.json').read_bytes())
    for key in ('identity_sha256','summary_sha256','auditor_sha256','complete_cases','failed_cases',
                'observations_replayed','paired_replays','physics_rerun','training_updates','scope'):
        if fresh[key]!=prior[key]: raise ValueError('Isolated audit disagrees: '+key)
    if [r['label'] for r in fresh['trials']]!=[r['label'] for r in prior['trials']]:
        raise ValueError('Isolated trial inventory differs')
    proof=dict(verified_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=expected['sha256'],
        archive_manifest_files=len(inventory),bound_source_files_verified=len(sources),
        isolated_extraction_cli_returncode=result.returncode,python_executable=str(executable.resolve()),
        extraction_directory=str(extract_to),audit=fresh,verification_source_sha256=sha(Path(__file__).read_bytes()),
        scope='Fresh-extraction NumPy audit of all archived physical records and core traces. Body physics and learning are not rerun.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8',newline='\n') as handle:
        json.dump(proof,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(archive_files=len(inventory),complete_cases=fresh['complete_cases'],
        failed_cases=fresh['failed_cases'],observations_replayed=fresh['observations_replayed'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--extract-to',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); verify(args.python,args.extract_to,args.output)
