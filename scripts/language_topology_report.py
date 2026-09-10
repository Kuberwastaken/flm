"""Plot every final language control only after checking the completed study."""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flm.language_topology import conditions, verify_complete, verify_source_identity
from flm.language_topology_test import SCORING_SOURCES, summarize, validate_score
from flm.provenance import sha256
from flm.tokenizer import Lexicon, read_cache


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def verified_report(root):
    """Read no test cache until the full selection and checkpoint gate passes."""
    folder = root / 'reports/language-topology'
    if not (folder / 'summary.json').is_file() or not (folder / 'selection.json').is_file():
        raise ValueError('Final language control scores and frozen selections are not available')
    identity = read_json(folder / 'identity.json')
    verify_source_identity(root, identity)
    for condition in conditions():
        if not (root / condition['output'] / 'complete.json').is_file():
            raise ValueError('Registered language condition incomplete: ' + condition['label'])
    selection = read_json(folder / 'selection.json')
    lexicon = Lexicon(root / 'data/tokenizers/wikitext2-4096/tokenizer.json')
    selected = [verify_complete(root, condition, identity, lexicon) for condition in conditions()]
    if selected != selection['runs']:
        raise ValueError('Selected checkpoint identities changed')
    identity_hash = sha256(folder / 'identity.json')
    if selection['study_identity_sha256'] != identity_hash or selection['tokenizer_sha256'] != lexicon.sha256:
        raise ValueError('Frozen selection belongs to a different study or tokenizer')
    if selection['scoring_sources'] != {name: sha256(root / name) for name in SCORING_SOURCES}:
        raise ValueError('Frozen scoring implementation changed')
    report = read_json(folder / 'summary.json')
    selection_hash = sha256(folder / 'selection.json')
    if report['selection_sha256'] != selection_hash or report['study_identity_sha256'] != identity_hash:
        raise ValueError('Final report belongs to a different selection or study')
    test_path = root / 'data/processed/wikitext2-bpe/test.npz'
    if sha256(test_path) != selection['test_cache_sha256']:
        raise ValueError('Frozen test cache changed')
    # All real conditions and selections have passed before any test decoding.
    documents = read_cache(test_path)
    expected = {row['label']: row for row in selected}
    if len(report['runs']) != 10 or {r['label'] for r in report['runs']} != set(expected):
        raise ValueError('All ten unique language conditions are required')
    for run in report['runs']:
        bound = dict(**expected[run['label']], selection_sha256=selection_hash,
                     test_cache_sha256=selection['test_cache_sha256'])
        if any(run.get(key) != value for key, value in bound.items()):
            raise ValueError('Final score identity changed: ' + run['label'])
        if run != read_json(folder / ('test-' + run['label'] + '.json')):
            raise ValueError('Summary differs from the individual score: ' + run['label'])
        validate_score(run['score'], documents, lexicon)
    for key, expected_value in summarize(report['runs']).items():
        if report[key] != expected_value:
            raise ValueError('Recomputed paired results differ: ' + key)
    return report


def contrast_rows(report):
    runs = {run['label']: run for run in report['runs']}
    rows = []
    for mechanism, key in [('topology', 'topology_contrasts'), ('slow_state', 'slow_state_contrasts')]:
        points = sorted(report[key], key=lambda r: (r.get('graph_seed', 0), r['training_seed']))
        for point in points:
            seed = point['training_seed']
            label = f'null{point["graph_seed"]}-s{seed}' if mechanism == 'topology' else f'no-slow-s{seed}'
            measured = runs[f'measured-s{seed}']; control = runs[label]
            difference = measured['score']['bits_per_byte'] - control['score']['bits_per_byte']
            if not math.isclose(difference, point['difference_bpb'], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Plotted contrast differs from its paired scores')
            rows.append(dict(mechanism=mechanism, control=label, training_seed=seed,
                graph_seed=point.get('graph_seed', ''),
                measured_selected_update=measured['checkpoint_step'], control_selected_update=control['checkpoint_step'],
                measured_bpb=measured['score']['bits_per_byte'], control_bpb=control['score']['bits_per_byte'],
                difference_bpb=point['difference_bpb'], lower_95=point['lower_95'], upper_95=point['upper_95']))
    return rows


def draw(rows, output, *, title='WikiText-2 · all language topology and slow-state contrasts'):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#292820', 'text.color': '#292820',
        'svg.fonttype': 'none', 'svg.hashsalt': 'flm-language-topology-v1'})
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.4]})
    fig.subplots_adjust(left=.23, right=.97, top=.88, bottom=.24, hspace=.58)
    fig.suptitle(title, fontsize=12, x=.04, ha='left', y=.98)
    lower = min(0, *(min(row['lower_95'], row['difference_bpb']) for row in rows))
    upper = max(0, *(max(row['upper_95'], row['difference_bpb']) for row in rows))
    margin = max(upper - lower, 1e-5) * .1
    for axis, mechanism, heading in zip(axes, ('topology', 'slow_state'),
        ('Measured wiring vs independently rewired graphs', 'Fast + slow state vs retrained fast-only dynamics')):
        subset = [row for row in rows if row['mechanism'] == mechanism]
        for y, row in enumerate(subset):
            color = '#a74c20'
            axis.hlines(y, row['lower_95'], row['upper_95'], color=color, linewidth=1.6)
            axis.plot(row['difference_bpb'], y, marker='o' if row['training_seed'] == 42 else 's',
                      color=color, markersize=5)
        labels = [(f'Graph {r["graph_seed"]} · seed {r["training_seed"]}' if mechanism == 'topology'
                   else f'Training seed {r["training_seed"]}') for r in subset]
        axis.set(yticks=range(len(subset)), yticklabels=labels, ylim=(len(subset)-.5, -.5),
                 xlim=(lower-margin, upper+margin))
        axis.set_title(heading, loc='left', fontsize=10)
        axis.axvline(0, color='#77776e', linewidth=.8, linestyle='--')
        axis.grid(axis='x', alpha=.12)
        axis.ticklabel_format(axis='x', style='plain', useOffset=False)
        axis.spines['left'].set_visible(False); axis.tick_params(axis='y', length=0)
    axes[-1].set_xlabel('Measured − control test loss (bits / UTF-8 byte)', labelpad=9)
    fig.text(.04, .12, '1,024-neuron subset · 6,000 updates/run. Negative favors measured fast/slow. Shared horizontal scale.', fontsize=8)
    fig.text(.04, .08, 'Lines: 95% paired-article bootstrap intervals, conditional on each trained pair; 60 articles, 10,000 resamples.', fontsize=8)
    fig.text(.04, .04, 'Six topology contrasts share two measured references. Intervals omit graph, initialization and dataset uncertainty.', fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=180, metadata={'Date': None} if suffix == 'svg' else {})
    plt.close(fig)


def write_paper_inputs(root, report, figure):
    """Use the same verified ten-run report for the paper table and figure."""
    runs = {run['label']: run for run in report['runs']}
    text = ['% Generated by scripts/language_topology_report.py; do not hand-edit measurements.',
            r'\newcommand{\TopologyRows}{%']
    for prefix, title in [('measured', 'Measured, fast + slow'),
                          ('null101', 'Rewired 101, fast + slow'),
                          ('null103', 'Rewired 103, fast + slow'),
                          ('null107', 'Rewired 107, fast + slow'),
                          ('no-slow', 'Measured, retrained without slow')]:
        values = [runs[f'{prefix}-s{seed}']['score']['bits_per_byte'] for seed in (42, 43)]
        text.append(title + ' & ' + ' & '.join(f'{value:.6f}' for value in [*values, math.fsum(values)/2]) + r' \\')
    text.append('}')
    topology = [row['difference_bpb'] for row in report['topology_contrasts']]
    for name, value in [('TopologyMeanDifference', report['topology_mean_difference_bpb']),
                        ('TopologyMinimumDifference', min(topology)),
                        ('TopologyMaximumDifference', max(topology)),
                        ('SlowStateMeanDifference', report['slow_state_mean_difference_bpb'])]:
        text.append('\\newcommand{\\' + name + '}{' + f'{value:+.6f}' + '}')
    destination = root/'papers/language-topology-measured.tex'
    destination.write_text('\n'.join(text) + '\n', encoding='utf8')
    copied_figure = root/'papers/figures/language-topology-test.png'
    shutil.copyfile(figure.with_suffix('.png'), copied_figure)
    record = dict(summary_sha256=sha256(root/'reports/language-topology/summary.json'),
        study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        generator_sha256=sha256(Path(__file__)), complete_conditions=len(runs),
        files={path.relative_to(root).as_posix(): sha256(path) for path in (destination, copied_figure)})
    (root/'reports/language-topology/paper-inputs.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(); root = args.root.resolve()
    report = verified_report(root)
    rows = contrast_rows(report)
    output = root / 'public/research/figures/language-topology-test'
    draw(rows, output)
    with output.with_suffix('.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write_paper_inputs(root, report, output)
    print('Verified all final score identities and generated all eight contrasts: ' + str(output))


if __name__ == '__main__':
    main()
