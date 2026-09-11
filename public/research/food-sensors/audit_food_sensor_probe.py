"""Standard-library-only replay of supplied food sensor geometry and measurements.

This recomputes sensor observations; it does not rerun MuJoCo or model learning.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path


def close(actual,expected):
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected): raise ValueError('Observation shape changed')
        for a,b in zip(actual,expected): close(a,b)
    elif type(expected) is bool:
        if type(actual) is not bool or actual!=expected: raise ValueError('Contact mask changed')
    elif not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Sensor value differs from scalar geometry replay')


def replay(field,antennae,contacts,missing):
    sources=field['sources']; raw=[]
    for point in antennae:
        row=[]
        for channel in range(2):
            row.append(math.fsum(s['odor'][channel]*math.exp(-.5*math.fsum(
                ((point[i]-s['position_mm'][i])/s['spread_mm'][i])**2 for i in range(3))) for s in sources))
        raw.append(row)
    mask=[]
    for point in contacts:
        row=[]
        for source in sources:
            delta=[a-b for a,b in zip(point,source['position_mm'])]
            row.append(math.hypot(*delta[:2])<=source['contact_radius_mm'] and 0<=delta[2]<=source['contact_height_mm'])
        mask.append(row)
    touched=[any(row[index] for row in mask) for index in range(len(sources))]
    sugar=max((s['sugar'] for s,hit in zip(sources,touched) if hit),default=0.)
    sensory=[0. if missing else raw[side][channel]/(1+raw[side][channel]) for channel in range(2) for side in range(2)]
    return dict(sensory=sensory+[sugar,float(any(touched))],raw_odor=raw,contact_mask=mask,
        diagnostic_contacted_sources=[s['name'] for s,hit in zip(sources,touched) if hit])


def audit(report):
    if (report['channels']!=['odor_a_left','odor_a_right','odor_b_left','odor_b_right','sugar_contact','source_contact']
            or report['language_model_loaded'] is not False or report['neural_policy_used'] is not False
            or report['training_updates']!=0 or report['food_approach_or_feeding_measured'] is not False
            or report['repetitions']!=2 or report['exact_repeated_geometry_and_sensor_records'] is not True
            or report['physics_steps_per_repeat']!=200 or report['physics_timestep_s']!=.0001
            or report['simulation_seconds']!=.02 or report['descending_signal']!=[1.,1.]):
        raise ValueError('Probe scope or execution inventory changed')
    names=report['body_names']
    if len(set(names))!=len(names): raise ValueError('Body identity inventory changed')
    if report['odor_body_origins']!=['l_funiculus','r_funiculus'] or report['contact_body_origins']!=[f'{leg}_tarsus5' for leg in ('lf','lm','lh','rf','rm','rh')]:
        raise ValueError('Sensor attachment identities changed')
    frames=report['frames']
    if [f['time_s'] for f in frames]!=[0.,.01,.02]: raise ValueError('Physical sample times changed')
    count=0
    def check(field,antennae,contacts,missing,observed):
        expected=replay(field,antennae,contacts,missing)
        for key in ('sensory','raw_odor','contact_mask'): close(observed[key],expected[key])
        if observed['diagnostic_contacted_sources']!=expected['diagnostic_contacted_sources']:
            raise ValueError('Diagnostic source identity changed')
    for frame in frames:
        positions=frame['body_positions_mm']
        if len(positions)!=len(names) or any(len(p)!=3 or any(not math.isfinite(v) for v in p) for p in positions):
            raise ValueError('Invalid physical body positions')
        if any(not math.isfinite(v) for v in frame['qpos']): raise ValueError('Invalid physical generalized positions')
        for field,key in (('odor_body_origins','antennae_mm'),('contact_body_origins','contact_points_mm')):
            if frame[key]!=[positions[names.index(name)] for name in report[field]]:
                raise ValueError('Sensor points do not match recorded body origins')
        check(report['field'],frame['antennae_mm'],frame['contact_points_mm'],False,frame['observation']); count+=1
    geometry=report['geometry_counterfactuals']
    if geometry['antennae_mm']!=frames[0]['antennae_mm'] or geometry['contact_points_mm']!=frames[0]['contact_points_mm']:
        raise ValueError('Counterfactuals use different physical geometry')
    expected_labels=['distant-sweet','distant-neutral','touched-sweet','touched-neutral','touched-missing-odor','touched-odorless-sugar']
    if [r['label'] for r in geometry['cases']]!=expected_labels: raise ValueError('Counterfactual inventory changed')
    for case in geometry['cases']:
        check(case['field'],geometry['antennae_mm'],geometry['contact_points_mm'],case['missing_odor'],case['observation']); count+=1
    return dict(observations_recomputed=count,physical_frames=3,counterfactuals=6,
        implementation='Independent scalar math; no NumPy, FLM or FlyGym imports',
        scope='Replays supplied positions and fields. Does not independently rerun physics or the reported exact physical repetition, and is not evidence of learned food behavior.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('record',type=Path); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve previous audits')
    raw=args.record.read_bytes(); result=audit(json.loads(raw))
    result.update(verified_utc=datetime.now(timezone.utc).isoformat(),record_sha256=hashlib.sha256(raw).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps(result,indent=2))
