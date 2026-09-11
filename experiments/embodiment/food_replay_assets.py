"""Export the actual compiled visual meshes and audit body lookup identities.

No physics steps, neural inference or training are run. Existing trial records
are never rewritten. These assets support a separate recorded-state viewer.
"""
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


def export(output):
    if output.exists(): raise ValueError('Use a new asset directory')
    lower_own_priority()
    identity_path=ROOT/'runs/embodiment/food-core-physical-v1/identity.json'
    identity=json.loads(identity_path.read_bytes())
    for name,digest in identity['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Original physical source changed: '+name)
    fly,sim,_=make_simulation(); model=sim.mj_model
    for module_name,digest in identity['installed_sources'].items():
        module=__import__(module_name,fromlist=['__file__'])
        if sha(module.__file__)!=digest: raise ValueError('Installed physical source changed: '+module_name)
    if model.nq!=73 or int(model.opt.integrator)!=0 or model.opt.timestep!=.0001:
        raise ValueError('Compiled state or integration identity differs')
    if any(importlib.metadata.version(name)!=version for name,version in identity['environment']['versions'].items()):
        raise ValueError('Use the original isolated physical environment')
    rows=[]
    for segment,element in fly.bodyseg_to_mjcfbody.items():
        index=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,element.name)
        rows.append(dict(segment=segment.name,declared_body_name=element.name,compiled_body_id=int(index),
            resolved_body_name=None if index<0 else mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,index)))
    missing=[r['segment'] for r in rows if r['compiled_body_id']<0]
    if missing!=['c_head']: raise ValueError('Body lookup inventory differs from the inspected issue')
    used=['c_thorax',*identity['odor_body_origins'],*identity['contact_body_origins']]
    if any(next(r for r in rows if r['segment']==name)['compiled_body_id']<0 for name in used):
        raise ValueError('A measured sensor or outcome uses an unresolved body')
    alias=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.nbody-1)
    if alias!='flm_body/rh_tarsus5': raise ValueError('Unresolved-row alias differs')
    geoms=[]; blobs=[]; offset=0
    for row in rows:
        name=fly.name+'/'+row['segment']; geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
        if geom<0 or int(model.geom_type[geom])!=int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError('Missing compiled visual mesh: '+name)
        mesh=int(model.geom_dataid[geom]); vstart=int(model.mesh_vertadr[mesh]); vcount=int(model.mesh_vertnum[mesh])
        fstart=int(model.mesh_faceadr[mesh]); fcount=int(model.mesh_facenum[mesh])
        vertices=np.asarray(model.mesh_vert[vstart:vstart+vcount],dtype='<f4')
        faces=np.asarray(model.mesh_face[fstart:fstart+fcount],dtype='<u4')
        if not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=vcount:
            raise ValueError('Compiled mesh array inventory differs')
        vertex_bytes=vertices.tobytes(); face_bytes=faces.tobytes()
        geoms.append(dict(segment=row['segment'],geom_id=int(geom),mesh_id=mesh,body_id=int(model.geom_bodyid[geom]),
            vertices=dict(offset=offset,count=vcount,dtype='float32',components=3),
            faces=dict(offset=offset+len(vertex_bytes),count=fcount,dtype='uint32',components=3),
            local_position_mm=model.geom_pos[geom].tolist(),local_quaternion_wxyz=model.geom_quat[geom].tolist()))
        blobs.extend((vertex_bytes,face_bytes)); offset+=len(vertex_bytes)+len(face_bytes)
    if len(geoms)!=69 or len({row['geom_id'] for row in geoms})!=69: raise ValueError('Expected 69 unique visual geometries')
    output.mkdir(parents=True)
    geometry=output/'geometry.bin'; geometry.write_bytes(b''.join(blobs))
    # Decode each written slice back to its compiled source before recording it.
    payload=geometry.read_bytes()
    for row in geoms:
        mesh=row['mesh_id']
        for key,source,start in (('vertices',model.mesh_vert,int(model.mesh_vertadr[mesh])),
                                 ('faces',model.mesh_face,int(model.mesh_faceadr[mesh]))):
            spec=row[key]; dtype='<f4' if key=='vertices' else '<u4'
            decoded=np.frombuffer(payload,dtype=dtype,count=3*spec['count'],offset=spec['offset']).reshape(-1,3)
            np.testing.assert_array_equal(decoded,source[start:start+spec['count']])
    record=dict(format='flm-compiled-food-body-v1',exported_utc=datetime.now(timezone.utc).isoformat(),
        physical_identity_sha256=sha(identity_path),exporter_sha256=sha(__file__),
        geometry_file='geometry.bin',geometry_bytes=len(payload),geometry_sha256=sha(geometry),
        geometries=geoms,body_lookup=rows,unresolved_body_segments=missing,
        unresolved_numpy_alias=alias,sensor_and_outcome_body_segments=used,
        all_sensor_and_outcome_body_ids_resolve=True,coordinate_units='mm',byte_order='little',
        compiled_state_dimensions=dict(nq=int(model.nq),nv=int(model.nv),bodies=int(model.nbody),geometries=int(model.ngeom)),
        environment=identity['environment'],installed_sources=identity['installed_sources'],
        physics_steps_run=0,training_updates=0,
        scope='Exact compiled mesh geometry for recorded-state rendering. The unresolved head body row in old API output aliases the final body; use named visual-geometry transforms instead. Sensor and outcome body lookups are valid.')
    (output/'model.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({key:record[key] for key in ('geometry_bytes','geometry_sha256','unresolved_body_segments','all_sensor_and_outcome_body_ids_resolve')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
