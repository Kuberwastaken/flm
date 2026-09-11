"""Publish the complete, audited unadapted-core physical inventory without filtering outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scripts.audit_food_core_physical import audit

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome_rows(summary):
    rows = []
    for trial in summary['trials']:
        m = trial['metrics']
        if trial['status'] not in ('complete', 'failed'):
            raise ValueError('Unknown physical outcome status')
        if trial['status']=='complete' and m is None:
            raise ValueError('Completed trials require measured outcomes')
        # A partial failed trial may have an observed contact, but remains failed.
        outcome = 'simulation failure' if trial['status'] == 'failed' else (
            'no source contact' if m['contact_latency_censored'] else
            'sugar contact' if m['first_contact_sugar'] > 0 else 'neutral contact')
        rows.append(dict(model=trial['model_id'], case=trial['case']['label'], status=trial['status'],
            outcome=outcome, observations=trial['observations'], physics_steps=trial['physics_steps'],
            first_contact_s=None if m is None else m['first_contact_s'],
            first_contact_sources='' if m is None else '|'.join(m['first_contact_sources']),
            first_contact_sugar=None if m is None else m['first_contact_sugar'],
            latency_censored=None if m is None else m['contact_latency_censored'],
            censor_time_s=None if m is None else m['censor_time_s'],
            final_x_mm=None if m is None else m['final_thorax_position_mm'][0],
            final_y_mm=None if m is None else m['final_thorax_position_mm'][1],
            final_z_mm=None if m is None else m['final_thorax_position_mm'][2],
            failure='' if trial['failure'] is None else trial['failure']['type']+': '+trial['failure']['message']))
    return rows


def figures(summary, declared, trajectories, output):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'svg.hashsalt':'flm-food-core-physical-v1', 'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'initial':'#226ba6', 'language':'#ad592e'}
    case_names = ['A left', 'A right', 'Sugar reversed', 'Both neutral', 'Odor removed', 'Exact repeat']
    nonempty = [path for path in trajectories.values() if len(path)]
    points = np.concatenate([np.array([[0.,0.,0.],[8.,3.,0.],[8.,-3.,0.]]), *nonempty])
    lower, upper = points[:,:2].min(axis=0)-1., points[:,:2].max(axis=0)+1.
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(3, 2, height_ratios=[1,1,1.4], left=.075, right=.96, top=.88, bottom=.12, hspace=.55, wspace=.32)
    for row, seed in enumerate((42,43)):
        for col, case in enumerate(('odor-a-left','odor-a-right')):
            ax = fig.add_subplot(grid[row,col])
            for source in declared['fields'][case]['sources']:
                xy = source['position_mm'][:2]
                ax.add_patch(Circle(xy, .75, facecolor='#bbc5bb', edgecolor='#526852', alpha=.6))
                ax.annotate(source['name'].upper(), xy, xytext=(0,10), textcoords='offset points', ha='center')
            for origin in ('initial','language'):
                label = f'{origin}-s{seed}--{case}'; trial = next(r for r in summary['trials'] if r['label']==label)
                path = trajectories[label]
                if len(path):
                    ax.plot(path[:,0], path[:,1], color=colors[origin], lw=1.8,
                        label='Initial core' if origin=='initial' else 'Language-trained core')
                    ax.plot(path[-1,0], path[-1,1], 'x' if trial['status']=='failed' else 'o', color=colors[origin], ms=5)
                if trial['status']=='failed':
                    ax.text(.02, .95-(.1 if origin=='language' else 0), origin+': simulation failed', transform=ax.transAxes,
                        color=colors[origin], va='top', fontsize=9)
            ax.set(title=f'Seed {seed} · '+('A left' if col==0 else 'A right'),
                xlabel='Thorax x (mm)', ylabel='Thorax y (mm)', xlim=(lower[0],upper[0]), ylim=(lower[1],upper[1]))
            ax.set_aspect('equal', adjustable='box'); ax.axhline(0, color='#dddddd', lw=.5, zorder=-1)
            if row==0 and col==0: ax.legend(loc='lower left', fontsize=9, frameon=False)
    for col, seed in enumerate((42,43)):
        ax = fig.add_subplot(grid[2,col])
        for origin, offset in (('initial',-.18), ('language',.18)):
            trials = [r for r in summary['trials'] if r['model_id']==f'{origin}-s{seed}']
            for i, trial in enumerate(trials):
                m = trial['metrics']; y = i+offset; color=colors[origin]
                if trial['status']=='failed':
                    value = trial['physics_steps']*.0001; marker='x'; text='failure'
                elif m['contact_latency_censored']:
                    value=m['censor_time_s']; marker='>'; text='no contact'
                else:
                    value=m['first_contact_s']; marker='o'
                    text=' / '.join(m['first_contact_sources']).upper()+(' · sugar' if m['first_contact_sugar'] > 0 else ' · neutral')
                ax.plot([0,value], [y,y], color=color, alpha=.3, lw=.8)
                ax.plot(value,y,marker=marker,color=color,ms=5,markerfacecolor='white' if marker=='>' else color)
                ax.annotate(text,(value,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=8,color=color)
        ax.set(title=f'Seed {seed} · all 12 trials', yticks=range(6), yticklabels=case_names,
            xlim=(0,2.8), ylim=(5.6,-.6), xticks=[0,.5,1,1.5,2], xlabel='First sampled contact / censoring time (s)')
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
    fig.suptitle('Initial and language-trained cores steer the same physical body', x=.075,y=.975,ha='left',fontsize=17)
    fig.text(.075,.925,'Identical unadapted sensory/action interfaces. Actual FlyGym paths above; every declared outcome below.',fontsize=11)
    fig.text(.075,.045,'Blue: initial core. Orange: language-trained core. Circles show virtual source patches; contact uses foot origins, not thorax distance.\nOpen arrows mark no contact at 2s; crosses mark simulation failures. One gait seed and one adapter seed; no food-task training or transfer benefit is established.',fontsize=9)
    for ext in ('png','svg'):
        fig.savefig(output/f'food-core-physical.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    svg = output/'food-core-physical.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')


def main(run, bundle):
    if not (run/'summary.json').is_file():
        raise ValueError('A complete physical inventory summary is required before reporting')
    checked = audit(run, bundle)
    declared = json.loads((run/'identity.json').read_bytes()); summary = json.loads((run/'summary.json').read_bytes())
    core_manifest = json.loads((bundle/'manifest.json').read_bytes())
    # Preserve the numerical and physical source identities used in these trials.
    sources = dict(core_manifest['sources'])
    for name, digest in declared['sources'].items():
        if name in sources and sources[name] != digest: raise ValueError('Inconsistent bound source: '+name)
        sources[name] = digest
    for name, digest in sources.items():
        if sha(ROOT/name) != digest: raise ValueError('Bound trial source changed: '+name)
    reports=ROOT/'reports/food-core-physical'; reports.mkdir(parents=True,exist_ok=True)
    for name in ('identity.json','summary.json'):
        target=reports/name
        if target.exists() and target.read_bytes() != (run/name).read_bytes(): raise ValueError('Preserve earlier physical record: '+name)
        shutil.copyfile(run/name,target)
    audit_path=reports/'audit.json'
    if audit_path.exists():
        previous=json.loads(audit_path.read_bytes())
        if any(previous[k]!=value for k,value in checked.items() if k!='verified_utc'): raise ValueError('Earlier audit differs')
    else: audit_path.write_text(json.dumps(checked,indent=2)+'\n',encoding='utf8',newline='\n')
    rows=outcome_rows(summary); output=ROOT/'public/research/figures'; output.mkdir(parents=True,exist_ok=True)
    csv_path=output/'food-core-physical.csv'
    with csv_path.open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    trajectories={}
    for trial in summary['trials']:
        with np.load(run/(trial['label']+'.npz'),allow_pickle=False) as archive:
            trajectories[trial['label']]=(archive['body_positions_mm'][:,trial['body_names'].index('c_thorax')]
                if trial['observations'] else np.empty((0,3)))
    figures(summary,declared,trajectories,output)
    names=['identity.json','summary.json']+[trial['label']+ext for trial in summary['trials'] for ext in ('.json','.npz')]
    contents={'records/'+name:(run/name).read_bytes() for name in names}
    contents['records/audit.json']=audit_path.read_bytes()
    contents['core/manifest.json']=(bundle/'manifest.json').read_bytes()
    for model in core_manifest['models']: contents['core/'+model['file']]=(bundle/model['file']).read_bytes()
    for name in ('audit_food_core_physical.py','audit_food_approach.py','audit_food_sensor_probe.py'):
        contents[name]=(ROOT/'scripts'/name).read_bytes()
    contents['food_core_runtime.py']=(ROOT/'experiments/embodiment/food_core_runtime.py').read_bytes()
    for name in sources: contents['sources/'+name]=(ROOT/name).read_bytes()
    # Post-hoc method evidence is separate from the frozen source inventory.
    for name in ('docs/PHYSICAL-STATE-SEMANTICS.md',
                 'reports/food-core-physical/body-assets.json',
                 'reports/food-core-physical/body-semantics.json',
                 'experiments/embodiment/food_replay_assets.py',
                 'experiments/embodiment/inspect_food_body_semantics.py'):
        contents[name]=(ROOT/name).read_bytes()
    contents['LICENSE']=(ROOT/'LICENSE').read_bytes()
    for name in ('CC-BY-4.0.txt','DATA-ATTRIBUTION.md','BODY-PROVENANCE.md','Body-Apache-2.0.txt','Body-MIT.txt'):
        contents['licenses/'+name]=(ROOT/'licenses'/name).read_bytes()
    contents['README.txt']=(
        'Unadapted FLM physical food reference: all 24 declared trials\n\n'
        'With NumPy installed, audit the extracted directory:\n'
        'python audit_food_core_physical.py records --bundle core --output fresh-audit.json\n\n'
        'Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 before launch.\n'
        'The audit replays saved neural states, geometry and commands; it does not rerun physics.\n'
        'No training corpus, PyTorch or FlyGym installation is needed for this audit.\n'
        'core/ contains the original bound manifest and four inference payloads. Its earlier\n'
        'open-loop fixture is supplied in the separate food-core-interface.zip release, not here.\n'
        'sources/ preserves all bound local sources, not a complete training repository.\n'
        'Read sources/docs/FOOD-CORE-PHYSICAL-REFERENCE.md for the prior declaration.\n'
        'Read docs/PHYSICAL-STATE-SEMANTICS.md before reusing body poses: c_head aliases\n'
        'rh_tarsus5, and cached sensor poses differ from kinematics of saved qpos.\n'
        'Keep the MIT code license and included graph/body provenance and component licenses.\n'
        'There are zero food-adaptation updates; do not treat contacts as proof of learning.\n').encode('utf8')
    inventory={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(contents.items())}
    contents['archive-manifest.json']=(json.dumps(inventory,indent=2)+'\n').encode('utf8')
    destination=ROOT/'public/research/food-core-physical-records.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(contents.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16
            archive.writestr(info,data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist())!=set(contents): raise ValueError('Physical archive integrity failure')
        for name,data in contents.items():
            if archive.read(name)!=data: raise ValueError('Archive member differs: '+name)
    artifacts=[destination,csv_path,output/'food-core-physical.png',output/'food-core-physical.svg']
    release=dict(identity_sha256=sha(run/'identity.json'),summary_sha256=sha(run/'summary.json'),
        report_source_sha256=sha(__file__),audit_sha256=sha(audit_path), archive_manifest_files=len(inventory),
        files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts},
        complete_cases=checked['complete_cases'],failed_cases=checked['failed_cases'],
        observations_audited=checked['observations_replayed'],training_updates=0,scope=declared['scope'])
    (reports/'release.json').write_text(json.dumps(release,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(release,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/embodiment/food-core-physical-v1')
    parser.add_argument('--bundle',type=Path,default=ROOT/'work/food-core-interface-v1')
    args=parser.parse_args(); main(args.run,args.bundle)
