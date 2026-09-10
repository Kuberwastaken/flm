"""Build the wiring note from verified complete records, retaining source hashes."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flm.provenance import sha256, write_json
from flm.wiring_report import collect


def main():
    report = collect(ROOT)
    saved = json.loads((ROOT / 'reports/wiring-learning/summary.json').read_text(encoding='utf8'))
    if saved != report: raise ValueError('Published wiring summary differs from verified records')
    commands = []; rows = []
    def score(task, method, topology, delay):
        return next(row['current_accuracy'] for row in report['curves'] if
            (row['task'], row['method'], row['topology'], row['step'], row['delay']) == (task, method, topology, 900, delay))
    def macro(name, value): commands.append(f'\\newcommand{{\\{name}}}{{{value:.2f}}}')
    for method, label in report['methods'].items():
        values = [score(task, method, topology, delay)['mean']*100 for task, delay in [('cue', 48), ('context', 8)]
                  for topology in ('measured', 'null')]
        rows.append(' & '.join([label] + [f'{value:.2f}' for value in values]) + r' \\')
    commands.append('\\newcommand{\\WiringRows}{%\n' + '\n'.join(rows) + '\n}')
    for task, prefix, delay in [('cue', 'Cue', 48), ('context', 'Context', 8)]:
        for topology, suffix in [('measured', 'Measured'), ('null', 'Null')]:
            macro(prefix + suffix, score(task, 'bptt', topology, delay)['mean']*100)
    gradient_rows = []
    for step in (0, 300, 600, 900):
        values = []
        for topology in ('measured', 'null'):
            for approximation in ('eligibility', 'instantaneous'):
                rows = [r['cosine'] for r in report['gradient_diagnostics'] if
                    (r['task'], r['method'], r['topology'], r['step'], r['approximation'], r['group']) ==
                    ('context', 'bptt', topology, step, approximation, 'combined_core')]
                if len(rows) != 3 or any(value is None for value in rows): raise ValueError('Undefined or missing cosine')
                values.append(float(np.mean(rows)))
        gradient_rows.append(' & '.join([str(step)] + [f'{value:.3f}' for value in values]) + r' \\')
    commands.append('\\newcommand{\\GradientRows}{%\n' + '\n'.join(gradient_rows) + '\n}')
    path = ROOT / 'papers/wiring-measured.tex'
    path.write_text('\n'.join(commands) + '\n', encoding='utf8')
    figures = ['wiring-matrix.png', 'wiring-context-delay8.png', 'wiring-gradient-alignment.png']
    for name in figures:
        shutil.copyfile(ROOT / 'public/research/figures' / name, ROOT / 'papers/figures' / name)
    write_json(ROOT / 'reports/wiring-learning/paper-inputs.json', dict(
        summary_sha256=sha256(ROOT / 'reports/wiring-learning/summary.json'), runs=report['runs'],
        numerical_include_sha256=sha256(path), generator_sha256=sha256(Path(__file__)),
        figures={name: sha256(ROOT / 'papers/figures' / name) for name in figures}))
    print('Verified sixty runs and generated the paper tables and three figures.')


if __name__ == '__main__': main()
