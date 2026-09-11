"""Plot every KC type's body-level coverage; this is not a selected model graph."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flm.provenance import sha256, write_json


def main():
    directory = Path('reports/selection-pathways')
    report = json.loads((directory/'summary.json').read_text(encoding='utf8'))
    source = directory/'kenyon-coverage.csv'
    if (sha256(source) != report['tables'][source.name] or
            sha256(Path('flm/selection_pathways.py')) != report['audit_source_sha256']):
        raise ValueError('Pathway audit identities changed')
    with source.open(encoding='utf8', newline='') as handle:
        raw = list(csv.DictReader(handle))
    totals = {}
    for row in raw:
        key = row['type'], int(row['minimum_contacts_per_edge'])
        if key not in totals:
            totals[key] = dict(type=key[0], minimum_contacts_per_edge=key[1], kenyon_cells=0,
                              with_alpn_input=0, with_mbon_output=0, with_both=0, without_either=0)
        for field in ('kenyon_cells', 'with_alpn_input', 'with_mbon_output', 'with_both', 'without_either'):
            totals[key][field] += int(row[field])
    for threshold in (1, 5):
        for field, count in report['kc_pathway_coverage'][str(threshold)].items():
            if sum(row[field] for (name, t), row in totals.items() if t == threshold) != count:
                raise ValueError('KC strata fail to reproduce aggregate coverage')
    names = sorted({name for name, threshold in totals})
    rows = [totals[name, threshold] for name in names for threshold in (1, 5)]
    output = Path('public/research/figures'); output.mkdir(parents=True, exist_ok=True)
    csv_path = output/'selection-pathways.csv'
    with csv_path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.6), sharey=True)
    fig.subplots_adjust(left=.23, right=.965, top=.77, bottom=.17, wspace=.2)
    y = np.arange(len(names))
    for ax, field, title in zip(axes, ('with_alpn_input', 'with_mbon_output'), ('Has an ALPN input', 'Has an MBON output')):
        for threshold, offset, color, marker in ((1, -.12, '#aa572d', 'o'), (5, .12, '#353936', 's')):
            values = [100*totals[name, threshold][field]/totals[name, threshold]['kenyon_cells'] for name in names]
            ax.scatter(values, y+offset, s=25, color=color if threshold == 1 else 'white',
                       edgecolors=color, marker=marker, label=f'≥{threshold} contacts per edge', zorder=3)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlim(-3, 103); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Cells with a qualifying connection (%)', fontsize=9)
        ax.grid(axis='x', color='#dedbd5', linewidth=.6); ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    axes[0].set_yticks(y, [f"{name}  (N={totals[name, 1]['kenyon_cells']:,})" for name in names], fontsize=9)
    axes[0].invert_yaxis()
    fig.text(.04, .95, 'Kenyon-cell pathway coverage in the acquired graph', fontsize=15, weight='bold')
    fig.text(.04, .902, 'All literal KC types; left and right combined. Anatomical contacts, not measured signal propagation.', fontsize=10)
    axes[1].legend(loc='lower right', bbox_to_anchor=(1, 1.12), frameon=False, ncol=2, fontsize=8.5)
    fig.text(.04, .085, 'Absent ALPN input does not imply absent sensory input: other pathways are outside the left panel.', fontsize=9)
    fig.text(.04, .048, 'Counts use the full acquired source. No new model subset or language result is shown.', fontsize=9)
    artifacts = [csv_path]
    for suffix in ('png', 'svg'):
        path = output/f'selection-pathways.{suffix}'
        fig.savefig(path, dpi=160, facecolor='white')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n', encoding='utf8')
        artifacts.append(path)
    plt.close(fig)
    write_json(directory/'figures.json', dict(summary_sha256=sha256(directory/'summary.json'),
        plot_source_sha256=sha256(Path(__file__)), types=len(names), rows=len(rows),
        artifacts={str(path).replace('\\', '/'): sha256(path) for path in artifacts},
        scope='Every literal KC type; sides combined, two prespecified descriptive edge thresholds'))


if __name__ == '__main__': main()
