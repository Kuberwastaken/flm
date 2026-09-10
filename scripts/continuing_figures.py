"""Publish measured neural diagnostics, physical calibration and corpus counts."""
from pathlib import Path
import csv
import json
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flm.provenance import sha256, write_json

OUT = ROOT / 'public/research'; FIG = OUT / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.dpi': 180})


def save(fig, name):
    for suffix in ('svg', 'png'): fig.savefig(FIG / f'{name}.{suffix}', bbox_inches='tight', facecolor='#faf8f5')
    plt.close(fig)


mechanism_path = ROOT / 'reports/wikitext2/mechanisms-s42.json'
mechanism = json.loads(mechanism_path.read_text(encoding='utf8'))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
names = {'untrained': 'Untrained', 'trained': 'Trained', 'recurrence_off': 'Recurrence off', 'slow_state_off': 'Slow state off'}
values = [r['score']['bits_per_byte'] for r in mechanism['conditions']]
labels = [names.get(r['condition'], r['condition']) for r in mechanism['conditions']]
axes[0].barh(labels, values, color=['#aca79d', '#a74c20', '#497569', '#666277'])
axes[0].invert_yaxis(); axes[0].set(xlim=(0, 4), xlabel='Validation bits / byte (lower is better)')
for i, value in enumerate(values): axes[0].text(value + .04, i, f'{value:.3f}', va='center', fontsize=9)
trace = mechanism['response']['records']
for state, color in [('fast', '#a74c20'), ('slow', '#497569')]:
    for version, style in [('initial', '--'), ('trained', '-')]:
        axes[1].plot([r['position'] for r in trace], [r[state][f'{version}_mean_absolute'] for r in trace],
            color=color, ls=style, label=f'{state.capitalize()} · {version}')
axes[1].set(xlabel='Position in the same original input (tokens)', ylabel='Mean absolute state')
axes[1].legend(frameon=False, fontsize=8, ncol=2); save(fig, 'mechanism-s42')
with (FIG / 'mechanism-response.csv').open('w', newline='', encoding='utf8') as stream:
    writer = csv.writer(stream); writer.writerow(['position', 'state', 'initial_mean_absolute', 'trained_mean_absolute', 'paired_rms_distance'])
    for record in trace:
        for state in ('fast', 'slow'):
            writer.writerow([record['position'], state, *[record[state][k] for k in ('initial_mean_absolute', 'trained_mean_absolute', 'paired_rms_distance')]])

calibration_path = ROOT / 'reports/embodiment/calibration.json'
calibration = json.loads(calibration_path.read_text(encoding='utf8'))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained'); rows = []
for trial, color, label in zip(calibration['trials'][:3], ['#a74c20', '#497569', '#666277'], ['Symmetric drive', 'Left drive', 'Right drive']):
    path = ROOT / f"runs/embodiment/calibration/{trial['label']}.npz"
    if sha256(path) != trial['trajectory_sha256']: raise ValueError('Physical trajectory changed')
    with np.load(path, allow_pickle=False) as arrays:
        t = arrays['time_s'] - arrays['time_s'][0]; p = arrays['thorax_position_mm'] - arrays['thorax_position_mm'][0]
        yaw = arrays['yaw_rad'] - arrays['yaw_rad'][0]
        axes[0].plot(p[:, 0], p[:, 1], color=color, label=label)
        axes[0].scatter(p[-1, 0], p[-1, 1], color=color, s=18)
        axes[1].plot(t, yaw, color=color, label=label)
        rows.extend([trial['label'], float(ti), *map(float, pi), float(yi), int(ci)]
                    for ti, pi, yi, ci in zip(t, p, yaw, arrays['contact_count']))
axes[0].set(aspect='equal', xlabel='Thorax displacement x (mm)', ylabel='Thorax displacement y (mm)')
axes[1].set(xlabel='Simulation time (seconds)', ylabel='Change in heading (radians)')
axes[1].legend(frameon=False, fontsize=9); save(fig, 'physical-calibration')
with (FIG / 'physical-trajectories.csv').open('w', newline='', encoding='utf8') as stream:
    writer = csv.writer(stream); writer.writerow(['condition', 'time_seconds', 'x_mm', 'y_mm', 'z_mm', 'yaw_rad', 'contacts']); writer.writerows(rows)
shutil.copyfile(calibration_path, OUT / 'physical-calibration.json')
video = ROOT / 'runs/embodiment/calibration/left_drive.mp4'; shutil.copyfile(video, OUT / 'physical-calibration.mp4')

acquisition = ROOT / 'data/cards/babylm-2026-acquisition.json'; overlap = ROOT / 'data/cards/babylm-2026-overlap.json'
data = json.loads(acquisition.read_text(encoding='utf8'))
dataset_report = dict(dataset=data['dataset'], status='Acquired and audited; preparation and training recorded separately.',
    acquisition_sha256=sha256(acquisition), overlap_sha256=sha256(overlap), files=data['files'],
    overlap=json.loads(overlap.read_text(encoding='utf8'))['comparisons'])
write_json(OUT / 'babylm-data.json', dataset_report)
write_json(FIG / 'provenance.json', dict(mechanism_sha256=sha256(mechanism_path), calibration_sha256=sha256(calibration_path),
    video_sha256=sha256(video), sources='Measured FLM arrays and FlyGym simulation; no generated neural activity or synthetic plot data.',
    video_note='One simulated second, displayed at quarter speed; upstream designed controller, FLM disconnected, no learning.',
    figures={p.name: sha256(p) for p in sorted(FIG.iterdir()) if p.suffix in ('.svg', '.png', '.csv')}))
print('Published measured state responses, physical trajectories, calibration video and corpus audit.')
