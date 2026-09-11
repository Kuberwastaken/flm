"""Plot all descriptive subset candidates, without implying a matched experiment."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flm.provenance import sha256, write_json

LABELS = {
    'frozen_connectivity_1024': 'Current connectivity ranking',
    'named_mb_bilateral': 'Named MB families, bilateral',
    'named_mb_left': 'Named MB families, left',
    'named_mb_right': 'Named MB families, right',
    'named_compass_families': 'EPG / PEN / PEG families',
    'uniform_1024_s201': 'Uniform random, seed 201',
    'uniform_1024_s203': 'Uniform random, seed 203',
    'uniform_1024_s207': 'Uniform random, seed 207',
}


def main():
    source = Path('reports/selection-feasibility/summary.json')
    report = json.loads(source.read_text(encoding='utf8'))
    if (report['audit_source_sha256'] != sha256(Path('flm/selection_feasibility.py')) or
            report['candidate_body_ids_sha256'] != sha256(source.with_name('candidate-body-ids.json')) or
            list(report['candidates']) != list(LABELS)):
        raise ValueError('Selection audit identity or candidate inventory changed')
    rows = [dict(candidate=name, neurons=row['neurons'], fast_edges=row['inside_fast']['connections'],
                 incoming_cut_percent=100*row['incoming_cut_fraction'], outgoing_cut_percent=100*row['outgoing_cut_fraction'])
            for name, row in report['candidates'].items()]
    output = Path('public/research/figures'); output.mkdir(parents=True, exist_ok=True)
    csv_path = output/'selection-feasibility.csv'
    with csv_path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'path'})
    fig, ax = plt.subplots(figsize=(11, 5.7))
    fig.subplots_adjust(left=.34, right=.965, top=.76, bottom=.19)
    y = np.arange(len(rows))
    incoming = [r['incoming_cut_percent'] for r in rows]
    outgoing = [r['outgoing_cut_percent'] for r in rows]
    for i, (first, second) in enumerate(zip(incoming, outgoing)):
        ax.plot([first, second], [i, i], color='#b9b4ab', linewidth=1.2, zorder=1)
    ax.scatter(incoming, y, marker='o', s=42, color='#aa572d', label='Incoming cut', zorder=3)
    ax.scatter(outgoing, y, marker='s', s=34, facecolors='white', edgecolors='#353936', label='Outgoing cut', zorder=4)
    ax.set_yticks(y, [f"{LABELS[r['candidate']]}  (N={r['neurons']:,})" for r in rows], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel('Raw contacts crossing the subset boundary (%)', labelpad=10)
    ax.grid(axis='x', color='#dedbd5', linewidth=.6); ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    fig.text(.04, .94, 'Selection feasibility: severed raw contacts', fontsize=15, weight='bold')
    fig.text(.04, .887, 'Descriptive anatomy audit; candidate sizes differ. No language models trained.', fontsize=10)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.025), frameon=False, ncol=2, fontsize=9)
    fig.text(.04, .075, 'Named families are not certified functional circuits. Lines connect incoming/outgoing fractions, not confidence intervals.', fontsize=8.8)
    fig.text(.04, .04, 'Each fraction uses its own total-contact denominator. Contact counts do not measure functional current.', fontsize=8.8)
    artifacts = [csv_path]
    for suffix in ('svg', 'png'):
        path = output/f'selection-feasibility.{suffix}'
        fig.savefig(path, dpi=160, facecolor='white')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n', encoding='utf8')
        artifacts.append(path)
    plt.close(fig)
    write_json(source.with_name('figures.json'), dict(audit_sha256=sha256(source),
        plot_source_sha256=sha256(Path(__file__)), candidates=len(rows),
        artifacts={str(path).replace('\\', '/'): sha256(path) for path in artifacts},
        scope='All eight anatomy-only candidate inventories; unequal node counts, no trained scores'))


if __name__ == '__main__': main()
