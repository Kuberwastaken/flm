"""Reconstruct visual geometry from all audited physical records, without stepping physics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.calibrate import make_simulation
from experiments.embodiment.food_approach import lower_own_priority


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    with path.open('x',encoding='utf8',newline='\n') as handle:
        json.dump(record,handle,indent=2); handle.write('\n')


def export(output):
    if output.exists(): raise ValueError('Use a fresh replay directory')
    lower_own_priority()
    run=ROOT/'runs/embodiment/food-core-physical-v1'
    identity=json.loads((run/'identity.json').read_bytes())
    summary=json.loads((run/'summary.json').read_bytes())
    audit_path=ROOT/'reports/food-core-physical/audit.json'; checked=json.loads(audit_path.read_bytes())
    if (checked['identity_sha256']!=sha(run/'identity.json') or checked['summary_sha256']!=sha(run/'summary.json')
            or checked['complete_cases']!=24 or checked['failed_cases']!=0 or checked['observations_replayed']!=4824):
        raise ValueError('Require the audited complete 24-case reference for this replay version')
    for name,digest in identity['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Physical source changed: '+name)
    for name,digest in identity['installed_sources'].items():
        if sha(__import__(name,fromlist=['__file__']).__file__)!=digest: raise ValueError('Installed source changed: '+name)
    if any(importlib.metadata.version(name)!=version for name,version in identity['environment']['versions'].items()):
        raise ValueError('Physical environment differs')
    asset_path=ROOT/'public/body/recorded-food/model.json'; assets=json.loads(asset_path.read_bytes())
    if assets['physical_identity_sha256']!=checked['identity_sha256']: raise ValueError('Asset identity differs')
    geometry_path=asset_path.with_name('geometry.bin')
    if sha(geometry_path)!=assets['geometry_sha256']: raise ValueError('Compiled mesh bytes differ')
    anatomy_path=ROOT/'public/models/flm-wikitext/anatomy.json'; anatomy=json.loads(anatomy_path.read_bytes())
    bundle=ROOT/'work/food-core-interface-v1'; manifest=json.loads((bundle/'manifest.json').read_bytes())
    if sha(bundle/'manifest.json')!=identity['bundle_manifest_sha256']: raise ValueError('Core bundle differs')
    for model in manifest['models']:
        path=bundle/model['file']
        if sha(path)!=model['sha256']: raise ValueError('Core payload differs')
        with np.load(path,allow_pickle=False) as core:
            if [str(x) for x in core['body_ids']]!=anatomy['body_ids']: raise ValueError('Brain display ordering differs')
    fly,sim,_=make_simulation(); model=sim.mj_model; data=sim.mj_data
    names=[segment.name for segment in fly.get_bodysegs_order()]
    geom_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,fly.name+'/'+row['segment']) for row in assets['geometries']]
    if geom_ids!=[row['geom_id'] for row in assets['geometries']] or model.nq!=73: raise ValueError('Compiled geometry order differs')
    output.mkdir(parents=True); trials=[]; maximum_cast_error=0.
    for record in summary['trials']:
        label=record['label']; path=run/(label+'.npz')
        if sha(path)!=record['trajectory_sha256'] or record['body_names']!=names: raise ValueError('Physical record differs')
        with np.load(path,allow_pickle=False) as archive: arrays={key:archive[key] for key in archive.files}
        poses=[]
        for qpos in arrays['qpos']:
            data.qpos[:]=qpos
            mujoco.mj_kinematics(model,data)
            matrices=data.geom_xmat[geom_ids].reshape(-1,3,3)
            np.testing.assert_allclose(matrices@matrices.transpose(0,2,1),np.broadcast_to(np.eye(3),matrices.shape),atol=1e-12,rtol=0)
            poses.append(np.concatenate((data.geom_xpos[geom_ids],matrices.reshape(-1,9)),axis=1))
        native=np.asarray(poses); poses=native.astype('<f4')
        maximum_cast_error=max(maximum_cast_error,float(np.abs(native-poses).max()))
        values=dict(geometry=poses,fast=arrays['fast'].astype('<f4'),slow=arrays['slow'].astype('<f4'))
        specs={}; payload=b''
        for name,array in values.items():
            if not np.isfinite(array).all(): raise ValueError('Nonfinite replay array')
            specs[name]=dict(offset=len(payload),shape=list(array.shape),dtype='float32')
            payload+=array.tobytes()
        binary=output/(label+'.bin'); binary.write_bytes(payload)
        for name,array in values.items():
            spec=specs[name]; decoded=np.frombuffer(binary.read_bytes(),dtype='<f4',count=array.size,offset=spec['offset']).reshape(array.shape)
            np.testing.assert_array_equal(decoded,array)
        metadata=dict(label=label,model_id=record['model_id'],case=record['case'],status=record['status'],
            metrics=record['metrics'],trajectory_sha256=record['trajectory_sha256'],model_payload_sha256=record['model_payload_sha256'],
            binary_file=binary.name,binary_sha256=sha(binary),binary_bytes=len(payload),arrays=specs,
            time_s=arrays['time_s'].tolist(),sensory=arrays['sensory'].tolist(),probabilities=arrays['probabilities'].tolist(),
            descending_signal=arrays['descending_signal'].tolist(),
            cached_thorax_mm=arrays['body_positions_mm'][:,names.index('c_thorax')].tolist(),
            field=identity['fields'][record['case']['label']])
        target=output/(label+'.json'); write_json(target,metadata)
        trials.append(dict(label=label,model_id=record['model_id'],case=record['case']['label'],metadata_file=target.name,
            metadata_sha256=sha(target),binary_sha256=sha(binary),binary_bytes=len(payload)))
        print(label,flush=True)
    result=dict(format='flm-food-recorded-replay-v1',exported_utc=datetime.now(timezone.utc).isoformat(),
        exporter_sha256=sha(__file__),physical_identity_sha256=checked['identity_sha256'],audit_sha256=sha(audit_path),
        summary_sha256=sha(run/'summary.json'),body_manifest_sha256=sha(asset_path),anatomy_sha256=sha(anatomy_path),
        trials=trials,observations=4824,frames_per_trial=201,neurons=1024,geometry_count=69,byte_order='little',
        geometry_layout='World position xyz in mm, then row-major 3x3 world rotation; native compiled mesh basis.',
        geometry_stage='Kinematics recomputed from recorded qpos; original cached sensor positions belong to the preceding integration stage.',
        maximum_float32_geometry_cast_error=maximum_cast_error,physics_steps_run=0,training_updates=0,
        scope='Every recorded sample of the 24-case unadapted core reference. Actual neural states; reconstructed visual geometry. No food learning or biological transfer established.')
    write_json(output/'manifest.json',result)
    print(json.dumps(dict(trials=len(trials),observations=4824,maximum_float32_geometry_cast_error=maximum_cast_error)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
