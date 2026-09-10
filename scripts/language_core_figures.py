"""Generate the complete computation comparison after its final-score gate."""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flm.provenance import sha256, write_json
from scripts.audit_language_core_report import audit
from scripts.language_core_report import verified_report

NAMES = dict(full='Full FLM', fixed_dynamics='Fixed recurrent dynamics',
             no_lateral='No lateral recurrence', no_temporal_state='No temporal state')
PANELS = (('primary', 'full', 'fixed_dynamics'), ('primary', 'full', 'no_lateral'),
          ('primary', 'full', 'no_temporal_state'), ('secondary', 'no_lateral', 'no_temporal_state'))


def contrast_rows(report):
    """Audit all supplied arithmetic, then retain every declared fitted pair."""
    audit(report)
    runs = {run['label']: run for run in report['runs']}
    rows = []
    for family, first, second in PANELS:
        contrasts = report['primary_contrasts' if family == 'primary' else 'independent_unit_memory_contrasts']
        for seed in (42, 43):
            contrast = next(row for row in contrasts if (row['first'], row['second'], row['training_seed']) == (first, second, seed))
            a, b = runs[f'{first}-s{seed}'], runs[f'{second}-s{seed}']
            rows.append(dict(family=family, first=first, second=second, training_seed=seed,
                first_selected_update=a['checkpoint_step'], second_selected_update=b['checkpoint_step'],
                first_bpb=a['score']['bits_per_byte'], second_bpb=b['score']['bits_per_byte'],
                difference_bpb=contrast['difference_bpb'], lower_95=contrast['lower_95'], upper_95=contrast['upper_95']))
    return rows


def draw(rows, output, *, title='WikiText-2 · all language computation contrasts'):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': '#292820', 'axes.labelcolor': '#292820',
        'svg.fonttype': 'path', 'svg.hashsalt': 'flm-language-core-v1'})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    figure.subplots_adjust(left=.10, right=.965, top=.83, bottom=.25, hspace=1.0, wspace=.35)
    figure.text(.045, .96, title, fontsize=17, weight='bold')
    figure.text(.045, .915, 'Each panel has its own horizontal scale; negative favors the first named model.', fontsize=10)
    for axis, (family, first, second) in zip(axes.flat, PANELS):
        subset = [r for r in rows if (r['family'], r['first'], r['second']) == (family, first, second)]
        if len(subset) != 2 or {r['training_seed'] for r in subset} != {42, 43}:
            raise ValueError('Both declared seeds are required for every contrast panel')
        lower = min(0., *(min(r['lower_95'], r['difference_bpb']) for r in subset))
        upper = max(0., *(max(r['upper_95'], r['difference_bpb']) for r in subset))
        margin = max(upper - lower, 1e-5) * .12
        for y, row in enumerate(subset):
            color = '#a74c20' if family == 'primary' else '#4b6862'
            axis.hlines(y, row['lower_95'], row['upper_95'], color=color, linewidth=1.8)
            axis.plot(row['difference_bpb'], y, marker='o' if row['training_seed'] == 42 else 's', color=color, markersize=6)
        axis.set(yticks=[0, 1], yticklabels=[f'Seed {r["training_seed"]}' for r in subset],
                 ylim=(1.65, -.65), xlim=(lower-margin, upper+margin))
        label = f'{NAMES[first]} − {NAMES[second]}'
        axis.set_title(('Secondary: ' if family == 'secondary' else '') + label, loc='left', fontsize=10, pad=12)
        axis.set_xlabel('Difference in test bits / UTF-8 byte', fontsize=9, labelpad=8)
        axis.axvline(0, color='#77776e', linewidth=.8, linestyle='--')
        axis.grid(axis='x', alpha=.13)
        axis.ticklabel_format(axis='x', style='plain', useOffset=False)
        axis.locator_params(axis='x', nbins=5)
        axis.spines['left'].set_visible(False); axis.tick_params(axis='y', length=0)
    figure.text(.045, .145, 'Six new fits and two reused full-model references · 1,024-neuron measured subset · 6,000 updates/run.', fontsize=9)
    figure.text(.045, .105, 'Lines: 95% paired-article bootstrap intervals, conditional on fitted pairs; 60 articles and 10,000 resamples.', fontsize=9)
    figure.text(.045, .065, 'Shared references and articles; two initializations do not establish training uncertainty. No topology interaction is tested.', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        figure.savefig(output.with_suffix(suffix), dpi=160, metadata={'Date': None} if suffix == '.svg' else {})
    plt.close(figure)


def paper_inputs(report):
    runs = {run['label']: run for run in report['runs']}
    lines = ['% Generated from all eight verified computation conditions; do not hand-edit measurements.',
             r'\newcommand{\CoreLossRows}{%']
    for control, name in NAMES.items():
        values = [runs[f'{control}-s{seed}']['score']['bits_per_byte'] for seed in (42, 43)]
        lines.append(name + ' & ' + ' & '.join(f'{v:.6f}' for v in (*values, math.fsum(values)/2)) + r' \\')
    lines.append('}')
    lines.append(r'\newcommand{\CoreContrastRows}{%')
    for row in contrast_rows(report):
        label = NAMES[row['first']] + ' $-$ ' + NAMES[row['second']]
        lines.append(f'{label} & {row["training_seed"]} & {row["difference_bpb"]:+.6f} & '
                     f'[{row["lower_95"]:+.6f}, {row["upper_95"]:+.6f}]' + r' \\')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def findings(report, rows):
    runs = {run['label']: run for run in report['runs']}
    lines = ['# Complete language computation comparison', '',
        'All six new fits completed before the eight checkpoint selections were frozen. '
        'The two full-FLM references reuse their previously published test scores. '
        'This is an exploratory follow-up on the measured 1,024-neuron subset.', '',
        '![Every declared fitted-pair effect; panels use different horizontal scales](../public/research/figures/language-core-test.svg)', '',
        '## Held-out prediction loss', '',
        'Bits per UTF-8 byte; lower is better. Each model is scored on all 60 test articles.', '',
        '| Model | Seed 42 | Seed 43 | Descriptive mean |', '|---|---:|---:|---:|']
    for control, name in NAMES.items():
        values = [runs[f'{control}-s{seed}']['score']['bits_per_byte'] for seed in (42, 43)]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.6f}' for v in (*values, math.fsum(values)/2)) + ' |')
    lines += ['', '## Every declared contrast', '',
        'Differences are the first model minus the second. Negative favors the first model. '
        'Intervals resample paired articles, conditional on these fitted checkpoints; they '
        'do not include training or graph uncertainty. The comparisons share articles and references.', '',
        '| Family | First minus second | Seed | Difference | 95% article interval |',
        '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["family"].title()} | {NAMES[row["first"]]} − {NAMES[row["second"]]} | '
            f'{row["training_seed"]} | {row["difference_bpb"]:+.6f} | '
            f'[{row["lower_95"]:+.6f}, {row["upper_95"]:+.6f}] |')
    lines += ['', 'The three primary descriptive means are:', '']
    for row in report['primary_means']:
        lines.append(f'- Full FLM minus {NAMES[row["control"]].lower()}: {row["mean_difference_bpb"]:+.6f} BPB.')
    lines += ['', f'The secondary no-lateral minus no-temporal mean is '
        f'{report["independent_unit_memory_mean_difference_bpb"]:+.6f} BPB. No interval on these descriptive means is declared.', '',
        '## What each comparison tests', '',
        '- **Fixed dynamics:** the four recurrent parameter groups remain at initialization; the lexical interface is trained through time.',
        '- **No lateral recurrence:** removes graph-mediated communication while retaining independent fast/slow temporal states.',
        '- **No temporal state:** also resets both states at every token. It retains the per-token nonlinear and lexical machinery.', '',
        'Every condition allocates 600,003 parameter entries. Fixed dynamics freezes 78,179 '
        'and trains 521,824. Both recurrence-disabled variants mark 600,003 entries trainable, '
        'but 76,131 edge/gain entries are disconnected. These counts do not imply equal effective capacity.', '',
        'A difference for one of these implemented mechanisms is not an anatomical-topology '
        'advantage, equivalence result, or evidence of language-to-behavior transfer. The separate '
        'topology study and its negative result remain distinct. Two initializations and one corpus '
        'do not establish a general ranking.', '',
        '## Inspect and reproduce', '',
        '- [Frozen protocol](LANGUAGE-CORE-PROTOCOL.md)',
        '- [Every selected checkpoint and article score](../public/research/language-core-results.json)',
        '- [Complete records and standalone arithmetic checker](https://flm.kuber.studio/research/language-core-records.zip)',
        '- [All plotted values and selected update counts](../public/research/figures/language-core-test.csv)',
        '- [Frozen selection](../reports/language-core/selection.json)', '',
        'Figures and tables are generated from the verified complete score report. The arithmetic '
        'check does not independently retrain models or rerun held-out inference.', '']
    return '\n'.join(lines)


def generate(root):
    root = Path(root).resolve()
    # No directories, plotted data or paper outputs before the full study gate.
    report = verified_report(root)
    rows = contrast_rows(report)
    figure = root / 'public/research/figures/language-core-test'
    draw(rows, figure)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    figure.with_suffix('.csv').write_bytes(buffer.getvalue().encode('utf8'))
    paper = root / 'papers/language-core-measured.tex'
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_bytes(paper_inputs(report).encode('utf8'))
    paper_figure = root / 'papers/figures/language-core-test.png'
    paper_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(figure.with_suffix('.png'), paper_figure)
    note = root / 'docs/LANGUAGE-CORE-RESULTS.md'
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_bytes(findings(report, rows).encode('utf8'))
    outputs = [figure.with_suffix(s) for s in ('.png', '.svg', '.csv')] + [paper, paper_figure, note]
    record = dict(study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        summary_sha256=sha256(root/'reports/language-core/summary.json'), source_sha256=sha256(Path(__file__)),
        complete_conditions=8, primary_contrasts=6, secondary_contrasts=2,
        horizontal_scales='Independent panel limits, labeled explicitly; each includes zero and both complete intervals',
        files={p.relative_to(root).as_posix(): sha256(p) for p in outputs})
    write_json(root / 'reports/language-core/figures.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = generate(args.root)
    print(f'Generated all {result["primary_contrasts"] + result["secondary_contrasts"]} declared contrasts.')
