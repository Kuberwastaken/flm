"""Replot audited computation contrasts on one axis; no fitting or resampling."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public/research/figures/language-core-test.csv'
PAIRS = [
    ('full', 'fixed_dynamics', 'Full − fixed dynamics', 'Primary'),
    ('full', 'no_lateral', 'Full − no lateral communication', 'Primary'),
    ('full', 'no_temporal_state', 'Full − no temporal state', 'Primary'),
    ('no_lateral', 'no_temporal_state', 'No lateral − no temporal state', 'Secondary'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((ROOT / 'reports/language-core/figures.json').read_text())
    expected = original['files'][SOURCE.relative_to(ROOT).as_posix()]
    if isinstance(expected, dict):
        expected = expected['sha256']
    if digest(SOURCE) != expected:
        raise ValueError('Audited contrast CSV changed')
    with SOURCE.open(newline='', encoding='utf8') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(r['first'], r['second'], int(r['training_seed'])): r for r in rows}
    if len(rows) != 8 or set(indexed) != {(a, b, s) for a, b, _, _ in PAIRS for s in (42, 43)}:
        raise ValueError('Require the complete eight-contrast inventory')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'svg.hashsalt': 'flm-predictive-computation-v1'})
    fig, ax = plt.subplots(figsize=(12.8, 6.3), facecolor='#faf8f5')
    ax.set_facecolor('#faf8f5')
    fig.subplots_adjust(left=.31, right=.96, top=.78, bottom=.24)
    fig.text(.035, .935, 'Temporal state makes the larger difference', fontsize=21, weight='bold', color='#272522')
    fig.text(.035, .875, 'WikiText · retrained controls · all eight seed-level effects on the same BPB scale', fontsize=12)
    for i, (a, b, label, family) in enumerate(PAIRS):
        for seed, offset, color, marker in ((42, .11, '#a95326', 'o'), (43, -.11, '#24776b', '^')):
            row = indexed[a, b, seed]
            if row['family'] != family.lower():
                raise ValueError('Contrast classification changed')
            value, lower, upper = (float(row[k]) for k in ('difference_bpb', 'lower_95', 'upper_95'))
            ax.plot([lower, upper], [3-i+offset]*2, color=color, lw=2)
            ax.plot(value, 3-i+offset, marker, color=color, ms=7,
                    label=f'Training seed {seed}' if i == 0 else None)
        values = [float(indexed[a, b, seed]['difference_bpb']) for seed in (42, 43)]
        ax.text(.997, 3-i+.29, f'Mean {sum(values)/2:+.6f}', transform=ax.get_yaxis_transform(),
                ha='right', fontsize=10, color='#58534c')
    ax.set_yticks([3, 2, 1, 0], [f'{label}\n{family}' for _, _, label, family in PAIRS])
    ax.tick_params(axis='y', length=0, pad=14)
    ax.set_ylim(-.4, 3.55)
    ax.set_xlim(-.145, .008)
    ax.set_xticks([-.14, -.12, -.10, -.08, -.06, -.04, -.02, 0])
    ax.axvline(0, color='#58534c', lw=1)
    ax.grid(axis='x', color='#ded9d1', lw=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('First model − second model, bits per UTF-8 byte  (negative favors first)', labelpad=12)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.16), frameon=False, ncol=2, fontsize=10)
    fig.text(.035, .075, 'Lines: original conditional 95% paired-article intervals; 60 shared test articles, two training seeds.\n'
             'Means are descriptive, without a new interval. Different effective parameter allocations; effects are not additive.',
             fontsize=10, linespacing=1.6, color='#58534c')
    outputs = []
    for suffix in ('png', 'svg'):
        path = ROOT / f'public/research/figures/language-core-shared-scale.{suffix}'
        fig.savefig(path, dpi=180, metadata={'Date': None} if suffix == 'svg' else {})
        outputs.append(path)
    plt.close(fig)
    record = dict(format='flm-predictive-computation-figure-v1',
                  source_sha256={SOURCE.relative_to(ROOT).as_posix(): digest(SOURCE)},
                  script_sha256=digest(Path(__file__)),
                  method='Replot all eight existing differences and intervals on one common axis; no fitting or resampling.',
                  files={p.relative_to(ROOT).as_posix(): digest(p) for p in outputs})
    (ROOT / 'reports/language-core/shared-scale-figure.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
