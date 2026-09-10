"""Audit recorded neural inputs, state, delayed commands and physical measurements."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .choice_runtime import ChoiceRuntime
from .feedback import cases


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def verify_trial(root, case, repeat=False):
    root=Path(root); folder=root/'runs/embodiment/closed-loop-v1'
    label=case['label']+('-repeat' if repeat else '')
    record=read(folder/(label+'.json')); identity=read(folder/'identity.json')
    if record['status']!='complete' or record['failure'] is not None or record['case']!=case or record['repeat']!=repeat:
        raise ValueError('Physical trial is incomplete or mislabeled')
    if record['study_identity_sha256']!=sha(folder/'identity.json'):
        raise ValueError('Physical study identity changed')
    for name,digest in {**identity['sources'],**identity['inputs']}.items():
        if sha(root/name)!=digest: raise ValueError('Frozen physical source or input changed: '+name)
    archive_path=folder/(label+'.npz')
    if record['trajectory_sha256']!=sha(archive_path) or record['csv_sha256']!=sha(folder/(label+'.csv')):
        raise ValueError('Recorded trajectory changed')
    with np.load(archive_path,allow_pickle=False) as archive:
        a={name:archive[name].copy() for name in archive.files}
    n=2001; neural_n=400
    shapes=dict(time_s=(n,),thorax_position_mm=(n,3),thorax_quaternion_wxyz=(n,4),yaw_rad=(n,),
        true_bearing_error_rad=(n,),target_mm=(n,2),target_distance_mm=(n,),thorax_up_z=(n,),
        contact_count=(n,),descending_signal=(n,2),neural_time_s=(neural_n,),neural_phase=(neural_n,),
        sensory=(neural_n,4),query=(neural_n,))
    if case['model']!='scripted': shapes.update(fast_state=(neural_n,256),slow_state=(neural_n,256),logits=(neural_n,2))
    if set(a)!=set(shapes)|{'qpos'} or any(a[k].shape!=shape for k,shape in shapes.items()):
        raise ValueError('Physical/neural observation dimensions changed')
    if a['qpos'].ndim!=2 or a['qpos'].shape[0]!=n or any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError('Nonfinite or missing recorded state')
    np.testing.assert_allclose(a['time_s'],np.arange(n)/1000,atol=1e-12,rtol=0)
    np.testing.assert_allclose(a['neural_time_s'],np.arange(neural_n)*.005,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(a['neural_phase'],np.arange(neural_n)%11)
    np.testing.assert_array_equal(a['query'],np.arange(neural_n)%11==10)
    np.testing.assert_allclose(np.linalg.norm(a['thorax_quaternion_wxyz'],axis=1),1,atol=1e-9,rtol=0)
    w,x,y,z=a['thorax_quaternion_wxyz'].T
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    np.testing.assert_allclose(yaw,a['yaw_rad'],atol=1e-12,rtol=0)
    target=np.tile([40.,20.],(n,1))
    if case['scenario']=='negative': target[:,1]=-20
    if case['scenario']=='switch': target[1000:,1]=-20
    np.testing.assert_array_equal(target,a['target_mm'])
    delta=target-a['thorax_position_mm'][:,:2]
    error=(np.arctan2(delta[:,1],delta[:,0])-yaw+np.pi)%(2*np.pi)-np.pi
    np.testing.assert_allclose(error,a['true_bearing_error_rad'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(delta,axis=1),a['target_distance_mm'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(1-2*(x*x+y*y),a['thorax_up_z'],atol=1e-12,rtol=0)
    model=None if case['model']=='scripted' else ChoiceRuntime(root/'data/controllers/choice-v1',case['model'])
    if record['model_checkpoint']!=(None if model is None else model.record):
        raise ValueError('Wrong controller checkpoint')
    decision_index=0; query_ticks=[]; chosen_signals=[]; maximum_state_error=0.
    for frame in range(neural_n):
        phase=frame%11; physical_index=frame*5
        if phase==0:
            observed_index=physical_index if case['pose_mode']=='live' else 0
            sampled_position=a['thorax_position_mm'][observed_index]
            sampled_yaw=yaw[observed_index]; sampled_target=target[physical_index]
            dx,dy=sampled_target-sampled_position[:2]
            sampled_error=(math.atan2(dy,dx)-sampled_yaw+math.pi)%(2*math.pi)-math.pi
            cue=0 if sampled_error>=0 else 1
            if model is not None: model.reset()
            sample_time=a['neural_time_s'][frame]
        sensory=np.zeros(4,dtype=np.float32)
        if phase<2: sensory[cue]=1
        if phase==10: sensory[3]=1
        np.testing.assert_array_equal(a['sensory'][frame],sensory)
        if model is not None:
            logits,probabilities=model.step(sensory)
            for actual,expected in [(model.fast,a['fast_state'][frame]),(model.slow,a['slow_state'][frame]),(logits,a['logits'][frame])]:
                np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)
                maximum_state_error=max(maximum_state_error,float(np.max(np.abs(actual-expected))))
        if phase==10:
            action=cue if model is None else int(logits.argmax())
            straight=abs(sampled_error)<=.10
            signal=[1.,1.] if straight else ([.4,1.2] if action==0 else [1.2,.4])
            decision=record['decisions'][decision_index]; sample=decision['sampled']
            if decision['action']!=action or decision['straight_deadband']!=straight or decision['descending_signal']!=signal:
                raise ValueError('Applied choice does not follow neural output and sampled deadband')
            np.testing.assert_allclose([decision['time_s'],sample['time_s']],[a['neural_time_s'][frame],sample_time],atol=1e-12,rtol=0)
            np.testing.assert_allclose(sample['position_mm'],sampled_position,atol=1e-12,rtol=0)
            np.testing.assert_allclose([sample['yaw_rad'],sample['bearing_error_rad']],[sampled_yaw,sampled_error],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(sample['target_mm'],sampled_target)
            if sample['cue']!=cue: raise ValueError('Recorded cue changed')
            if model is None:
                if decision['probabilities'] is not None: raise ValueError('Scripted controller has invented neural probabilities')
            else: np.testing.assert_allclose(decision['probabilities'],probabilities,atol=2e-6,rtol=2e-5)
            query_ticks.append(frame*50); chosen_signals.append(signal); decision_index+=1
    if decision_index!=36 or len(record['decisions'])!=36: raise ValueError('Decision budget changed')
    # A sample at a query boundary is the pose before that query is applied.
    indices=np.searchsorted(query_ticks,np.arange(n)*10,side='left')-1
    expected_signals=np.ones((n,2)); mask=indices>=0
    expected_signals[mask]=np.asarray(chosen_signals)[indices[mask]]
    np.testing.assert_array_equal(expected_signals,a['descending_signal'])
    metrics=dict(mean_absolute_bearing_error_rad=float(np.trapezoid(np.abs(error),a['time_s'])/2),
        final_absolute_bearing_error_rad=float(abs(error[-1])),
        final_target_distance_mm=float(np.linalg.norm(delta[-1])),
        minimum_thorax_height_mm=float(a['thorax_position_mm'][:,2].min()),
        minimum_thorax_up_z=float(a['thorax_up_z'].min()),mean_contact_count=float(a['contact_count'].mean()),
        heading_change_rad=float(np.unwrap(yaw)[-1]-yaw[0]),
        descending_signal_switches=int(np.any(np.diff(expected_signals,axis=0)!=0,axis=1).sum()))
    for key,value in metrics.items():
        if not math.isclose(record['metrics'][key],value,abs_tol=1e-10,rel_tol=1e-10):
            raise ValueError('Reported physical metric does not reproduce: '+key)
    return dict(case=case,metrics=record['metrics'],trajectory_sha256=record['trajectory_sha256'],
        controller_replay=dict(frames=neural_n,decisions=decision_index,maximum_state_logit_error=maximum_state_error,
                              sensory_and_commands_exact=True),report_sha256=sha(folder/(label+'.json'))), a


def verify_all(root):
    root=Path(root); output=[]
    for case in cases():
        report,_=verify_trial(root,case); output.append(report)
    repeated=next(case for case in cases() if case['label']=='eligibility-live-switch')
    _,first=verify_trial(root,repeated); _,second=verify_trial(root,repeated,repeat=True)
    if any(not np.array_equal(first[key],second[key]) for key in first):
        raise ValueError('Repeated physical/neural arrays are not identical')
    return output
