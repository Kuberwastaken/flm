"""Reject self-consistently rehashed but causally incorrect physical records."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.feedback import cases
from experiments.embodiment.verify_feedback import verify_trial


def main():
    case=next(row for row in cases() if row['label']=='eligibility-live-switch')
    folder=Path('runs/embodiment/closed-loop-v1')
    identity=json.loads((ROOT/folder/'identity.json').read_text())
    base=ROOT/'work/feedback-audit'; base.mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='record-',dir=base) as directory:
        temporary=Path(directory).resolve()
        if not temporary.is_relative_to(base.resolve()): raise ValueError('Temporary audit path escaped workspace')
        files=[*identity['sources'],*identity['inputs'],(folder/'identity.json').as_posix()]
        files.extend((folder/(case['label']+suffix)).as_posix() for suffix in ('.json','.npz','.csv'))
        for name in files:
            destination=temporary/name; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,destination)
        verify_trial(temporary,case)
        original_record=(temporary/folder/(case['label']+'.json')).read_bytes()
        with np.load(temporary/folder/(case['label']+'.npz'),allow_pickle=False) as archive:
            original_arrays={name:archive[name].copy() for name in archive.files}
        for fault in ('wrong_sensory_cue','command_before_query','invented_neural_state','wrong_sampled_pose','wrong_phase_progress'):
            record=json.loads(original_record); arrays={name:value.copy() for name,value in original_arrays.items()}
            if fault=='wrong_sensory_cue': arrays['sensory'][0,:2]=arrays['sensory'][0,:2][::-1]
            if fault=='command_before_query': arrays['descending_signal'][1]=[.4,1.2]
            if fault=='invented_neural_state': arrays['fast_state'][110,0]+=.1
            if fault=='wrong_sampled_pose': record['decisions'][1]['sampled']['position_mm'][0]+=1
            if fault=='wrong_phase_progress': record['metrics']['phase_progress'][0]['progress_toward_target_mm']+=10
            path=temporary/folder/(case['label']+'.npz'); np.savez_compressed(path,**arrays)
            record['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            (temporary/folder/(case['label']+'.json')).write_text(json.dumps(record),encoding='utf8')
            try:
                verify_trial(temporary,case)
            except (ValueError,AssertionError) as error:
                rows.append(dict(fault=fault,rejected=True,error_type=type(error).__name__,
                                 self_consistent_archive_hash=True))
            else:
                raise AssertionError('Causally wrong record was accepted: '+fault)
    report=dict(original_trial_verified=True,negative_tests=rows,
        original_trajectory_sha256=hashlib.sha256((ROOT/folder/(case['label']+'.npz')).read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256((ROOT/'experiments/embodiment/verify_feedback.py').read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='Each copy has a valid refreshed archive hash; rejection must come from independent causal/state/metric checks.')
    path=ROOT/'reports/embodiment/closed-loop/audit-tests.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
