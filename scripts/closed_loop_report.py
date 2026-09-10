"""Publish only the complete, independently replayed physical feedback assay."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.embodiment.verify_feedback import verify_all, verify_trial
from experiments.embodiment.feedback import cases

NAMES=dict(initial='Untrained',bptt='BPTT',reservoir='Fixed core',eligibility='Eligibility',scripted='Scripted reference')
COLORS=dict(initial='#34322d',bptt='#77716b',reservoir='#497569',eligibility='#a74c20',scripted='#666277')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def main():
    folder=ROOT/'runs/embodiment/closed-loop-v1'
    complete_path=ROOT/'reports/embodiment/closed-loop.json'
    if not complete_path.exists(): raise ValueError('Complete all 27 physical conditions and the repeated trial before publication')
    complete=json.loads(complete_path.read_text()); identity=json.loads((folder/'identity.json').read_text())
    if complete['identity']!=identity or complete['identity_sha256']!=sha(folder/'identity.json'):
        raise ValueError('Complete report refers to another physical study')
    if {r['case']['label'] for r in complete['trials']}!={r['label'] for r in cases()} or len(complete['trials'])!=27:
        raise ValueError('The complete report is missing conditions')
    rows=verify_all(ROOT)
    for row in complete['trials']:
        if row!=json.loads((folder/(row['case']['label']+'.json')).read_text()):
            raise ValueError('Consolidated report differs from original physical trial')
    output=ROOT/'output/closed-loop-release'; output.mkdir(parents=True,exist_ok=True)
    traces={}; equivalence={}; contrasts=[]
    for row in rows:
        with np.load(folder/(row['case']['label']+'.npz'),allow_pickle=False) as archive:
            traces[row['case']['label']]={name:archive[name].copy() for name in archive.files}
        trajectory=traces[row['case']['label']]['qpos']
        physical_sha=hashlib.sha256(trajectory.tobytes()).hexdigest()
        row['physical_qpos_sha256']=physical_sha
        equivalence.setdefault(physical_sha,[]).append(row['case']['label'])
    for scenario in ('positive','negative','switch'):
        for model in ('initial','bptt','reservoir','eligibility'):
            pair={r['case']['pose_mode']:r for r in rows if r['case']['model']==model and r['case']['scenario']==scenario}
            contrasts.append(dict(scenario=scenario,model=model,
                live_minus_frozen_error_rad=pair['live']['metrics']['mean_absolute_bearing_error_rad']-pair['frozen']['metrics']['mean_absolute_bearing_error_rad']))
    report=dict(schema_version=1,complete=True,conditions=27,physical_reproducibility_repeats=1,
        identity_sha256=sha(folder/'identity.json'),protocol_sha256=identity['inputs']['docs/CLOSED-LOOP-PROTOCOL.md'],
        source_report_sha256=sha(complete_path),verifier_sha256=sha(ROOT/'experiments/embodiment/verify_feedback.py'),
        generator_sha256=sha(Path(__file__)),environment=identity['environment'],
        verification_environment=dict(python=platform.python_version(),numpy=np.__version__),
        neural_models=json.loads((ROOT/'data/controllers/choice-v1/manifest.json').read_text())['models'],
        runtime_parity=complete['runtime_parity'],repeat_arrays_exact=True,rows=rows,contrasts=contrasts,
        identical_physical_trajectory_groups=[v for v in equivalence.values() if len(v)>1],
        neural_panel='Eligibility/live/switch. Each column holds the actual state for its following 5 ms control interval; rows preserve the 256-node graph order.',
        interpretation='One training seed and one physical seed; live/frozen pose feedback through an engineered cue/deadband/gait wrapper. No language weights, learned gait or anatomical advantage.',
        specimen_note='MaleCNS graph subset and female NeuroMechFly body come from different specimens.')
    write(output/'closed-loop.json',report)
    columns=['model','pose_mode','scenario','mean_absolute_bearing_error_rad','final_absolute_bearing_error_rad',
             'final_target_distance_mm','heading_change_rad','minimum_thorax_height_mm','minimum_thorax_up_z','descending_signal_switches']
    with (output/'closed-loop.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader()
        for row in rows:
            data={**row['case'],**row['metrics']}; writer.writerow({name:data[name] for name in columns})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'path','figure.facecolor':'#faf8f5','axes.facecolor':'#faf8f5'})
    for scenario in ('positive','negative','switch'):
        fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
        for model in NAMES:
            trace=traces[f'{model}-live-{scenario}']
            axes[0].plot(trace['time_s'],np.rad2deg(np.abs(trace['true_bearing_error_rad'])),
                color=COLORS[model],label=NAMES[model],linewidth=1.3,
                linestyle=':' if model=='scripted' else ('--' if model=='reservoir' else '-'))
        axes[0].set(xlabel='Simulated time (s)',ylabel='Absolute target-bearing error (degrees)',xlim=(0,2),ylim=(0,180))
        axes[0].grid(axis='y',color='#d6d0c5',linewidth=.6)
        axes[0].legend(frameon=False,fontsize=8,ncol=3,loc='lower left',
                       bbox_to_anchor=(0,1.01),borderaxespad=0,columnspacing=.9)
        for mode,style in [('live','-'),('frozen','--')]:
            trace=traces[f'eligibility-{mode}-{scenario}']; positions=trace['thorax_position_mm']
            axes[1].plot(positions[:,0],positions[:,1],color=COLORS['eligibility'],linestyle=style,
                label='Eligibility: '+('live pose' if mode=='live' else 'frozen pose'),linewidth=1.5)
            axes[1].plot(positions[0,0],positions[0,1],marker='o',color='#34322d',markersize=3)
        targets=np.unique(trace['target_mm'],axis=0)
        axes[1].scatter(targets[:,0],targets[:,1],marker='x',s=35,color='#34322d',label='Virtual waypoint')
        axes[1].set(xlabel='World x (mm)',ylabel='World y (mm)',aspect='equal')
        axes[1].legend(frameon=False,fontsize=8,loc='best')
        if scenario=='switch':
            axes[0].axvline(1,color='#a8a094',linestyle='--',linewidth=.8)
            axes[0].text(1.03,100,'Target switches',fontsize=8,color='#625c52')
        fig.savefig(output/f'closed-loop-{scenario}.svg')
        fig.savefig(output/f'closed-loop-{scenario}.png',dpi=160); plt.close(fig)
    trace=traces['eligibility-live-switch']
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
    for axis,key,label in zip(axes,('fast_state','slow_state'),('Fast state','Slow state')):
        values=trace[key].T; bound=float(np.max(np.abs(values)))
        plot=axis.imshow(values,aspect='auto',origin='lower',interpolation='nearest',cmap='PuOr_r',
            extent=(0,2,-.5,255.5),vmin=-bound,vmax=bound)
        axis.set(ylabel='Neuron index',title=label); fig.colorbar(plot,ax=axis,label='Signed state')
        axis.axvline(1,color='#34322d',linewidth=.7,linestyle='--')
    axes[-1].set_xlabel('Simulated time (s)'); fig.savefig(output/'closed-loop-neural.svg')
    fig.savefig(output/'closed-loop-neural.png',dpi=160); plt.close(fig)
    video=folder/'eligibility-live-switch.mp4'; video_output=output/'closed-loop.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video),'-map','0:v:0',
                    '-c','copy','-movflags','+faststart',str(video_output)],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video_output),
                    '-frames:v','1',str(output/'closed-loop-poster.png')],check=True)
    files={}
    for path in sorted(folder.glob('*')):
        if path.suffix in ('.npz','.csv','.json','.mp4'): files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    graph_path=ROOT/'data/graphs/central-256/graph.npz'
    if any(row['graph_sha256']!=sha(graph_path) for row in report['neural_models']):
        raise ValueError('The anatomical identity map changed')
    for name in ['docs/CLOSED-LOOP-PROTOCOL.md','docs/LOCAL-LEARNING-PROTOCOL.md',
                 'docs/CLOSED-LOOP-REPRODUCTION.md','scripts/audit_feedback_release.py',
                 'LICENSE','licenses/CC-BY-4.0.txt','licenses/DATA-ATTRIBUTION.md',
                 'licenses/Body-MIT.txt','licenses/Body-Apache-2.0.txt',
                 'licenses/BODY-PROVENANCE.md','licenses/Neural-Canvas-LICENSE.txt',
                 'experiments/embodiment/choice_runtime.py','experiments/embodiment/feedback.py',
                 'experiments/embodiment/closed_loop.py','experiments/embodiment/calibrate.py',
                 'experiments/embodiment/verify_feedback.py','scripts/closed_loop_report.py',
                 'requirements-embodied-lock.txt','data/graphs/central-256/graph-card.json',
                 'data/graphs/central-256/graph.npz','reports/embodiment/closed-loop.json',
                 'reports/embodiment/closed-loop/first-trial.json','reports/embodiment/closed-loop/audit-tests.json',
                 'scripts/verify_feedback_faults.py']:
        files[name]=(ROOT/name).read_bytes()
    files.update({p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted((ROOT/'data/controllers/choice-v1').glob('*')) if p.is_file()})
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    files['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    archive_path=output/'closed-loop-records.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()): archive.writestr(name,data)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None: raise ValueError('Feedback archive failed CRC validation')
        for name,digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest: raise ValueError('Feedback archive payload changed')
    write(ROOT/'reports/embodiment/closed-loop/release-candidate.json',dict(
        verified_conditions=27,verified_repeat=True,archive_bytes=archive_path.stat().st_size,
        archive_sha256=sha(archive_path),files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        status='Complete candidate generated. Visually review figures/video before copying to public.'))
    print(json.dumps(dict(conditions=27,contrasts=len(contrasts),archive_bytes=archive_path.stat().st_size,
                          identical_physical_groups=report['identical_physical_trajectory_groups']),indent=2))


if __name__=='__main__': main()
