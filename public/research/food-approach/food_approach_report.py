"""Publish all declared physical reference outcomes with audited trajectories."""
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
from scripts.audit_food_approach import audit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/embodiment/food-approach-v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit_record = audit(RUN)
    declared = json.loads((RUN/'identity.json').read_bytes())
    summary = json.loads((RUN/'summary.json').read_bytes())
    for name, value in declared['sources'].items():
        if sha(ROOT/name) != value: raise ValueError('Physical source changed: '+name)
    if audit_record['complete_cases'] != 7:
        raise ValueError('This figure requires all completed cases; preserve failed records and report them before adapting the plot')
    for pair in audit_record['paired_replays']:
        if not all(pair['exact_array_equal'].values()): raise ValueError('Shared-path figure requires verified exact equality')
    reports = ROOT/'reports/food-approach'; reports.mkdir(parents=True, exist_ok=True)
    for name in ('identity.json', 'summary.json'): shutil.copyfile(RUN/name, reports/name)
    audit_path = reports/'audit.json'
    if not audit_path.exists(): audit_path.write_text(json.dumps(audit_record, indent=2)+'\n', encoding='utf8', newline='\n')
    else:
        prior = json.loads(audit_path.read_bytes())
        if any(prior[k] != v for k, v in audit_record.items() if k != 'verified_utc'):
            raise ValueError('Existing independent audit differs')
    trajectories = {}
    for trial in summary['trials']:
        label = trial['case']['label']
        with np.load(RUN/(label+'.npz'), allow_pickle=False) as archive:
            trajectories[label] = archive['body_positions_mm'][:, trial['body_names'].index('c_thorax')]
    output = ROOT/'public/research/figures'; output.mkdir(parents=True, exist_ok=True)
    csv_path = output/'food-approach.csv'
    with csv_path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case', 'status', 'first_contact_s', 'first_contact_sources', 'first_contact_sugar', 'latency_censored', 'censor_time_s', 'final_x_mm', 'final_y_mm', 'final_z_mm'])
        for trial in summary['trials']:
            m = trial['metrics']
            writer.writerow([trial['case']['label'], trial['status'], m['first_contact_s'], '|'.join(m['first_contact_sources']), m['first_contact_sugar'], m['contact_latency_censored'], m['censor_time_s'], *m['final_thorax_position_mm']])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'flm-food-approach-v1', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    fig.subplots_adjust(left=.045, right=.975, top=.76, bottom=.25, wspace=.63)
    for ax, key, title in zip(axes[:2], ('odor-a-left', 'odor-a-right'), ('A on the left', 'A on the right')):
        field = declared['fields'][key]
        for source in field['sources']:
            xy = source['position_mm'][:2]; color = '#226ba6' if source['name']=='a' else '#aa6b40'
            ax.add_patch(Circle(xy, .75, facecolor=color, alpha=.18, edgecolor=color))
            ax.annotate(source['name'].upper(), xy, xytext=(0, 11), textcoords='offset points', ha='center', color=color, weight='bold')
        path = trajectories[key]; ax.plot(path[:, 0], path[:, 1], color='#226ba6', lw=2, label='Odor-driven reference')
        ax.plot(path[-1, 0], path[-1, 1], 'o', color='#226ba6', ms=4)
        if key == 'odor-a-left':
            path = trajectories['odor-a-left-missing']
            ax.plot(path[:, 0], path[:, 1], color='#666666', lw=1.5, ls='--', label='Missing odor / straight')
        ax.set(title=title, xlabel='Thorax x (mm)', ylabel='Thorax y (mm)', xlim=(-1, 25), ylim=(-5, 5))
        ax.set_aspect('equal', adjustable='box'); ax.axhline(0, color='#dddddd', lw=.5, zorder=-1)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.3), frameon=False, fontsize=9)
    ax = axes[2]
    labels = ['Odor A left', 'Odor A right', 'Sugar reversed', 'Both neutral', 'Odor removed', 'Straight reference', 'Exact-seed repeat']
    for i, trial in enumerate(summary['trials']):
        m = trial['metrics']; censored = m['contact_latency_censored']
        value = m['censor_time_s'] if censored else m['first_contact_s']
        color = '#666666' if censored else '#aa6b40' if m['first_contact_sugar'] == 0 else '#226ba6'
        ax.plot([0, value], [i, i], color=color, lw=1, alpha=.5)
        ax.plot(value, i, marker='>' if censored else 'o', color=color, markerfacecolor='white' if censored else color, ms=6)
        text = 'no contact' if censored else f"{value:.2f}s · {'sugar' if m['first_contact_sugar'] else 'neutral'}"
        ax.text(value+.07, i, text, va='center', fontsize=9)
    ax.set(yticks=range(7), yticklabels=labels, xlim=(0, 2.8), ylim=(6.6, -.6), xticks=[0, .5, 1., 1.5, 2.], xlabel='First sampled source contact (seconds)', title='Every declared physical outcome')
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    fig.suptitle('Odor following works; reward reversal does not change the script', x=.045, y=.96, ha='left', fontsize=18)
    fig.text(.045, .865, 'Actual FlyGym trajectories. A fixed bilateral odor-A rule steers the existing gait controller; no neural model or training is used.', fontsize=11)
    fig.text(.045, .065, 'Left paths also cover reversed sugar, neutral sources and the exact repeat: their physical arrays are identical. Circles mark virtual contact patches.\nNo-contact trials are censored at 2s. Seven declared cases, one gait seed; these are engineering observations, not independent biological replications or feeding.', fontsize=10)
    for ext in ('png', 'svg'): fig.savefig(output/f'food-approach.{ext}', dpi=140, metadata={'Date': None} if ext=='svg' else None)
    plt.close(fig)
    svg = output/'food-approach.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n', encoding='utf8', newline='\n')
    contents = {p.name: p.read_bytes() for p in sorted(RUN.iterdir()) if p.suffix in ('.json', '.npz')}
    if len(contents) != 16: raise ValueError('Physical archive must contain identity, summary and seven complete trial pairs')
    for name in ('audit_food_approach.py', 'audit_food_sensor_probe.py'):
        contents[name] = (ROOT/'scripts'/name).read_bytes()
    contents['audit.json'] = audit_path.read_bytes()
    for name in declared['sources']: contents[name] = (ROOT/name).read_bytes()
    contents['README.txt'] = ('Scripted physical food-approach reference v1\n\n'
        'Install NumPy, then audit the extracted complete directory:\n'
        'python audit_food_approach.py . --output fresh-audit.json\n\n'
        'The audit replays supplied geometry and actions, not physics or learning.\n'
        'Read docs/FOOD-APPROACH-REFERENCE.md for the prior declaration and scope.\n'
        'The physical runner requires the pinned isolated FlyGym environment.\n').encode('utf8')
    manifest = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(contents.items())}
    contents['source-manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode('utf8')
    destination = ROOT/'public/research/food-approach-records.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None: raise ValueError('Physical archive CRC mismatch')
        for name, record in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != record['sha256']: raise ValueError('Physical archive hash mismatch')
    artifacts = [destination, csv_path, svg, output/'food-approach.png']
    release = dict(summary_sha256=sha(RUN/'summary.json'), identity_sha256=sha(RUN/'identity.json'),
        report_script_sha256=sha(__file__), files={p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in artifacts},
        physical_archive_manifest_files=len(manifest), observations_audited=audit_record['observations_replayed'],
        complete_cases=7, failed_cases=0, training_updates=0, scope=declared['scope'])
    (reports/'release.json').write_text(json.dumps(release, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps(release, indent=2))


if __name__ == '__main__': main()
