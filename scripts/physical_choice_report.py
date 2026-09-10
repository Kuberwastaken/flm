"""Verify the complete neural-choice physics assay and publish measured artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/embodiment/learned-choice-v1'
PUBLIC = ROOT / 'public/research'
METHODS = dict(bptt='BPTT', reservoir='Fixed core', eligibility='Eligibility', instantaneous='No trace history', reward='Reward + eligibility')
COLORS = dict(bptt='#34322d', reservoir='#497569', eligibility='#a74c20', instantaneous='#77716b', reward='#666277')


def main():
    source = ROOT / 'reports/embodiment/learned-choice.json'; report = json.loads(source.read_text())
    trials = report['trials']; expected = {(m, s, c) for m in METHODS for s in (0,300,600,900) for c in (0,1)}
    identities = {(r['identity']['case']['method'], r['identity']['case']['neural_episode']['step'], r['identity']['case']['neural_episode']['cue']) for r in trials}
    if len(trials) != 40 or identities != expected: raise ValueError('Incomplete predeclared physical assay')
    action_paths = {}; geometry_rows = []; chosen_heading_correct = 0
    for row in trials:
        case = row['identity']['case']; episode = case['neural_episode']; action = episode['chosen_action']
        path = WORK / f'{case["label"]}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['physical']['trajectory_sha256']:
            raise ValueError('Physical trajectory identity changed')
        with np.load(path, allow_pickle=False) as archive:
            qpos = archive['qpos'].copy(); position = archive['thorax_position_mm'].copy()
            if qpos.shape[0] != 1001 or not np.isfinite(qpos).all(): raise ValueError('Incomplete or nonfinite physics samples')
            if action in action_paths and not np.array_equal(action_paths[action]['qpos'], qpos):
                raise ValueError('The same command and initialization did not reproduce the physical path')
            action_paths[action] = dict(qpos=qpos, position=position, time=archive['time_s'].copy(), yaw=archive['yaw_rad'].copy())
        chosen_heading_correct += int(np.sign(row['physical']['yaw_change_rad']) == (1 if action == 0 else -1))
        geometry_rows.append(dict(method=case['method'], step=episode['step'], cue=episode['cue'],
            action_probability_0=episode['action_probabilities'][0], chosen_action=action,
            expected_action=episode['expected_action'], neural_choice_correct=row['neural_choice_correct'],
            yaw_change_rad=row['physical']['yaw_change_rad'], trajectory_sha256=row['physical']['trajectory_sha256']))
    summary = dict(report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=geometry_rows,
        physical_cases=len(trials), unique_chosen_commands=len(action_paths), neural_choices_correct=sum(r['neural_choice_correct'] for r in trials),
        heading_matches_chosen_action=chosen_heading_correct, identical_command_replays='Exact equality of all recorded generalized positions within each chosen action',
        full_report='learned-choice-records.json', caution=report['caution'], coupling=report['coupling'])
    shutil.copyfile(source, PUBLIC / 'learned-choice-records.json')
    video = WORK / 'eligibility-000300-cue0.mp4'
    if not video.exists(): raise ValueError('The predeclared video case is missing')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('Install FFmpeg on PATH to package the recorded video for streaming')
    published_video = PUBLIC / 'learned-choice.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
        '-map', '0:v:0', '-c', 'copy', '-movflags', '+faststart', str(published_video)], check=True)
    summary['video'] = dict(source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        published_sha256=hashlib.sha256(published_video.read_bytes()).hexdigest(),
        transformation='MP4 stream copy with metadata first; no video re-encoding',
        ffmpeg_version=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0])
    (PUBLIC / 'learned-choice.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
    with zipfile.ZipFile(PUBLIC / 'learned-choice-trajectories.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for row in trials:
            label = row['identity']['case']['label']; archive.write(WORK / f'{label}.csv', label + '.csv')
        archive.write(source, 'neural-choices-and-physics.json')
    figures = PUBLIC / 'figures'; figures.mkdir(exist_ok=True)
    with (figures / 'learned-choice.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0])); writer.writeheader(); writer.writerows(geometry_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'path','figure.facecolor':'#faf8f5','axes.facecolor':'#faf8f5'})
    fig, ax = plt.subplots(figsize=(6.8,3.8), layout='constrained')
    for method, name in METHODS.items():
        rows = [r for r in geometry_rows if r['method']==method and r['cue']==0]
        ax.plot([r['step'] for r in rows], [r['action_probability_0'] for r in rows], label=name, color=COLORS[method],
            marker='o',markersize=3,linestyle='--' if method=='instantaneous' else '-')
    ax.set(xlabel='Checkpoint update',ylabel='Probability of action 0 for cue 0',ylim=(-.03,1.03),xticks=[0,300,600,900])
    ax.grid(axis='y',color='#d6d0c5',linewidth=.6); ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=8)
    fig.savefig(figures/'learned-choice-probability.svg'); fig.savefig(figures/'learned-choice-probability.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8,3.8),layout='constrained')
    for action, color in [(0,'#a74c20'),(1,'#497569')]:
        value=action_paths[action]; ax.plot(value['time'],value['yaw'] - value['yaw'][0],color=color,label=f'Chosen action {action}')
    ax.set(xlabel='Simulated time (s)',ylabel='Change in heading (radians)',xlim=(0,1)); ax.legend(frameon=False,fontsize=9)
    ax.grid(axis='y',color='#d6d0c5',linewidth=.6)
    fig.savefig(figures/'learned-choice-heading.svg'); fig.savefig(figures/'learned-choice-heading.png',dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__': main()
