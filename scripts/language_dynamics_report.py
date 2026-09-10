"""Draw the complete, replay-verified recurrent dynamics diagnostic."""
import csv
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flm.language_dynamics_study import FOLDER, verified_results
from flm.provenance import sha256, write_json


def csv_bytes(rows):
    target = io.StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return target.getvalue().encode('utf8')


def generate(root):
    report = verified_results(root)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'path',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.6), sharey=True)
    figure.subplots_adjust(left=.075, right=.98, bottom=.21, top=.69, wspace=.12)
    colors = dict(initial='#37424b', trained='#295e9b', edges_gain_only='#b47428', time_constants_only='#55836b')
    names = dict(initial='Initial (shared)', trained='Trained', edges_gain_only='Edges + gain only', time_constants_only='Time constants only')
    curves = []; models = []; handles = []; labels = []
    for case in report['cases']:
        condition = case['condition']; metrics = case['metrics']; mode = condition['mode']; seed = condition['seed']
        label = names[mode] + (f' / seed {seed}' if seed is not None else '')
        models.append(dict(condition=condition['label'], fast_spectral_radius=metrics['spectrum']['fast_spectral_radius'],
            joint_spectral_radius=metrics['spectrum']['joint_spectral_radius'], recurrent_gain=metrics['parameters']['recurrent_gain'],
            median_alpha=metrics['parameters']['alpha']['median'], median_beta=metrics['parameters']['beta']['median'],
            mean_small_pulse_energy=float(np.mean([row['normalized_response_energy'] for row in metrics['pulses'][0]['directions']])),
            mean_large_pulse_energy=float(np.mean([row['normalized_response_energy'] for row in metrics['pulses'][1]['directions']]))))
        with np.load(root / FOLDER / case['arrays'], allow_pickle=False) as archive:
            for index, amplitude in enumerate((.001, 1.)):
                means = archive[f'amp_{index}_curves'].mean(axis=1)
                deviations = archive[f'amp_{index}_normalized_linear_deviation'].mean(axis=1)
                time = np.arange(len(means))
                line, = axes[index].plot(time, means[:, 2] / amplitude, color=colors[mode],
                    linestyle='--' if seed == 43 else '-', linewidth=2 if mode in ('initial', 'trained') else 1.5, label=label)
                if index == 0:
                    handles.append(line); labels.append(label)
                for step, values in enumerate(means):
                    curves.append(dict(condition=condition['label'], amplitude=amplitude, post_pulse_update=step,
                        mean_fast_rms=float(values[0]), mean_slow_rms=float(values[1]), mean_joint_rms=float(values[2]),
                        mean_joint_rms_per_amplitude=float(values[2] / amplitude),
                        mean_linear_deviation_per_amplitude=float(deviations[step])))
    for axis, amplitude in zip(axes, (.001, 1.)):
        axis.set_title(f'Pulse amplitude {amplitude:g}', loc='left', fontsize=11, pad=12)
        axis.set_yscale('log'); axis.set_xlim(0, 256); axis.set_xticks([0, 64, 128, 192, 256])
        axis.set_xlabel('Model updates after the pulse', labelpad=9)
        axis.grid(axis='y', which='major', color='#dfe4e7', linewidth=.6)
        axis.tick_params(colors='#37424b')
    axes[0].set_ylabel('Mean joint state RMS / pulse amplitude', labelpad=10)
    figure.text(.075, .96, 'FLM: response to a synthetic pulse', fontsize=20, weight='bold', color='#26323b')
    figure.text(.075, .915, 'Measured 1,024-neuron subset · two trained cores and four acute parameter swaps', fontsize=11, color='#56626c')
    figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(.066, .88), frameon=False,
                  ncol=3, fontsize=9, columnspacing=1.9, handlelength=3.5, labelspacing=.7)
    figure.text(.075, .098, 'Each line averages eight fixed pulse directions. External drive is zero after the pulse.', fontsize=9, color='#56626c')
    figure.text(.075, .067, 'Lexical input/readout bypassed. Response persistence does not measure language quality or memory capacity.', fontsize=9, color='#56626c')
    destination = root / 'public/research/figures/language-dynamics-pulses'
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        figure.savefig(destination.with_suffix(suffix), dpi=150, facecolor='white')
    plt.close(figure)
    for name, rows in [('language-dynamics-curves.csv', curves), ('language-dynamics-summary.csv', models)]:
        (root / 'public/research' / name).write_bytes(csv_bytes(rows))
    write_json(root / 'public/research/language-dynamics-results.json', report)
    paths = [destination.with_suffix(suffix) for suffix in ('.png', '.svg')]
    paths += [root / 'public/research' / name for name in ('language-dynamics-curves.csv', 'language-dynamics-summary.csv', 'language-dynamics-results.json')]
    record = dict(study_identity_sha256=report['study_identity_sha256'],
        summary_sha256=sha256(root / FOLDER / 'summary.json'), plot_source_sha256=sha256(Path(__file__)),
        conditions=7, plotted_curve_rows=len(curves), directions_per_line=8,
        scope='Arithmetic mean of per-direction joint RMS; no inferential band or biological time conversion',
        files={path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in paths})
    write_json(root / FOLDER / 'figure.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    generate(ROOT)
