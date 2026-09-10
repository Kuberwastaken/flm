"""Verify every declared choice-learning run and publish all outcomes."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile
import numpy as np
from .behavior_study import METHODS, SEEDS, DELAYS, GRAPH, PROTOCOL
from .provenance import sha256, write_json

NAMES = dict(bptt='BPTT', reservoir='Fixed core', eligibility='Eligibility', instantaneous='No trace history', reward='Reward + eligibility')


def collect():
    runs = []
    for seed in SEEDS:
        matching = []
        for method in METHODS:
            path = Path(f'reports/local-learning/{method}-s{seed}.json')
            complete_path = Path(f'runs/local-learning-v1/{method}-s{seed}/complete.json')
            if not path.exists() or not complete_path.exists(): raise ValueError('Registered choice study is incomplete')
            report = json.loads(path.read_text(encoding='utf8')); complete = json.loads(complete_path.read_text(encoding='utf8'))
            if complete['report_sha256'] != sha256(path) or complete['identity'] != report['identity']:
                raise ValueError('Choice report changed or does not belong to its completed run')
            identity = report['identity']
            if identity['method'] != method or identity['seed'] != seed or identity['updates'] != 900 or identity['graph_sha256'] != sha256(GRAPH) or identity['protocol_sha256'] != sha256(PROTOCOL):
                raise ValueError('Wrong registered choice experiment')
            if [r['step'] for r in report['probes']] != list(range(0, 901, 100)):
                raise ValueError('Missing declared diagnostic checkpoint')
            for record in report['probes']:
                if [p['delay'] for p in record['panels']] != list(DELAYS): raise ValueError('Missing declared delay panel')
                for panel in record['panels']:
                    probabilities = np.array(panel['action_probabilities']); labels = np.array(panel['cue_ids'])
                    if probabilities.shape != (256, 2) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1., atol=1e-6):
                        raise ValueError('Invalid diagnostic probability panel')
                    for mapping, targets in [('original', labels), ('reversed', 1 - labels)]:
                        actual = float((probabilities.argmax(1) == targets).mean())
                        if actual != panel['scores'][mapping]['accuracy']: raise ValueError('Reported accuracy differs from saved decisions')
            matching.append(report); runs.append(report)
        if len({r['training_stream_sha256'] for r in matching}) != 1:
            raise ValueError('Methods saw different sensory/target streams')
        if any(r['probes'][0] != matching[0]['probes'][0] or r['choices'][0] != matching[0]['choices'][0] for r in matching):
            raise ValueError('Methods did not start with identical diagnostics and neural state')
    return runs


def main():
    runs = collect(); curves = []
    parameter_card = runs[0]['parameters']
    if any(r['parameters'] != parameter_card for r in runs): raise ValueError('Architecture sizes differ across conditions')
    readout_parameters = sum(count for name, count in parameter_card['parameter_groups'].items() if name.startswith(('readout.', 'norm.')))
    trace_sizes = {p['eligibility_tensor_bytes'] for r in runs if r['identity']['method'] == 'eligibility' for p in r['progress']}
    if len(trace_sizes) != 1: raise ValueError('Eligibility storage was not fixed across delays')
    for method in METHODS:
        chosen = [r for r in runs if r['identity']['method'] == method]
        for step in range(0, 901, 100):
            for delay in DELAYS:
                panels = [next(p for p in r['probes'][step // 100]['panels'] if p['delay'] == delay) for r in chosen]
                mapping = chosen[0]['probes'][step // 100]['current_mapping']
                original = [p['scores']['original']['accuracy'] for p in panels]
                current = [p['scores'][mapping]['accuracy'] for p in panels]
                curves.append(dict(method=method, step=step, delay=delay, current_mapping=mapping,
                    seeds=list(SEEDS), original_accuracy=original, original_mean=float(np.mean(original)),
                    current_accuracy=current, current_mean=float(np.mean(current)),
                    current_seed_sd=float(np.std(current, ddof=1)),
                    current_cross_entropy=[p['scores'][mapping]['cross_entropy'] for p in panels]))
    summary = dict(dataset='Synthetic delayed cue with distractor; original, reversed, then restored association',
        methods=NAMES, seeds=list(SEEDS), updates=900, episodes_per_run=7200, panel_episodes=256, delays=list(DELAYS),
        graph=dict(neurons=256, edges=6678, pools=32, sha256=sha256(GRAPH)), total_parameters=parameter_card['trainable_parameters'],
        learned_parameters={method: readout_parameters if method == 'reservoir' else parameter_card['trainable_parameters'] for method in METHODS},
        eligibility_tensor_bytes_batch8=next(iter(trace_sizes)), recurrent_state_tensor_bytes_batch8=2 * parameter_card['config']['neurons'] * 8 * 4,
        protocol_sha256=sha256(PROTOCOL), curves=curves,
        source_reports=[dict(file=f'{r["identity"]["method"]}-s{r["identity"]["seed"]}.json',
            sha256=sha256(Path(f'reports/local-learning/{r["identity"]["method"]}-s{r["identity"]["seed"]}.json')),
            training_stream_sha256=r['training_stream_sha256']) for r in runs],
        cautions=['Repeated diagnostic panels, not untouched test data or language evaluation.',
            'Original and reversed mappings are mutually exclusive; accuracy under both mappings is not independent evidence.',
            'Three seeds give limited variability estimates; the shaded range in figures is not a confidence interval.',
            'The fixed-core control solves much of the task, so success alone does not demonstrate useful internal plasticity.',
            'No rewired-graph comparison establishes an anatomical advantage.',
            'Eligibility stores per-synapse traces and uses engineered readout feedback; it is neither cost-free nor fully biological.'])
    write_json(Path('reports/local-learning/summary.json'), summary)
    write_json(Path('public/research/choice-learning.json'), summary)
    destination = Path('public/research/choice-learning-records.zip')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for row in summary['source_reports']: archive.write(Path('reports/local-learning') / row['file'], 'measurements/' + row['file'])
        archive.write(PROTOCOL, 'LOCAL-LEARNING-PROTOCOL.md')
        archive.write(GRAPH, 'graph/graph.npz'); archive.write(GRAPH.with_name('graph-card.json'), 'graph/graph-card.json')
    print(json.dumps([r for r in curves if r['step'] == 900 and r['delay'] in (8, 12, 48)], indent=2))


if __name__ == '__main__': main()
