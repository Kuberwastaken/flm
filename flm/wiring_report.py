"""Verify all sixty factorial runs and expose paired outcomes without selection."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
from .behavior_study import METHODS, SEEDS, DELAYS, reversed_mapping
from .model import load_graph
from .provenance import sha256, write_json
from .wiring_controls import validate_control
from .wiring_study import TASKS, TOPOLOGIES, NULL_SEEDS, PROTOCOL, read

METHOD_NAMES = dict(bptt='BPTT',reservoir='Fixed core',eligibility='Eligibility',instantaneous='No trace history',reward='Reward + eligibility')


def statistics(values):
    values = [float(value) for value in values]
    if not values or not np.isfinite(values).all(): raise ValueError('Invalid seed observations')
    return dict(values=values,mean=float(np.mean(values)),minimum=min(values),maximum=max(values),
        sd=float(np.std(values,ddof=1)) if len(values)>1 else None)


def paired_statistics(measured, randomized):
    if len(measured)!=len(randomized) or len(measured)!=len(SEEDS): raise ValueError('Unpaired seeds')
    return statistics(np.array(measured)-np.array(randomized))


def validate_panels(report):
    if [p['step'] for p in report['probes']] != list(range(0,901,100)): raise ValueError('Incomplete checkpoint panel')
    for probe in report['probes']:
        expected_mapping='reversed' if reversed_mapping(probe['step']) else 'original'
        if probe['current_mapping']!=expected_mapping: raise ValueError('Incorrect phase')
        if [p['delay'] for p in probe['panels']]!=list(DELAYS): raise ValueError('Incomplete delay panel')
        for panel in probe['panels']:
            probabilities=np.array(panel['action_probabilities']); labels=np.array(panel['labels'])
            if probabilities.shape!=(256,2) or labels.shape!=(256,) or panel['episodes']!=256:
                raise ValueError('Incomplete prediction panel')
            if not np.isfinite(probabilities).all() or np.any(probabilities<0) or not np.allclose(probabilities.sum(-1),1,atol=2e-7,rtol=0):
                raise ValueError('Invalid action probabilities')
            if not np.isin(labels,[0,1]).all() or labels.sum()!=128: raise ValueError('Unbalanced targets')
            for mapping,target in [('original',labels),('reversed',1-labels)]:
                measured=float(np.mean(probabilities.argmax(-1)==target))
                if measured!=panel['scores'][mapping]['accuracy']: raise ValueError('Accuracy differs from retained predictions')
                # Probabilities were exported from float32 log-softmax; allow only its rounding error.
                selected=probabilities[np.arange(256),target]
                if np.any(selected<=0) or abs(float(-np.log(selected).mean())-panel['scores'][mapping]['cross_entropy'])>2e-6:
                    raise ValueError('Cross entropy differs from retained predictions')


def collect(root=Path('.')):
    reports={}; provenance=[]; graph_cards={}
    source_graph=load_graph(root/'data/graphs/central-256/graph.npz')
    for seed in SEEDS:
        folder=root/f'data/graphs/central-256-null{NULL_SEEDS[seed]}'
        card=read(folder/'graph-card.json')
        if card['graph_sha256']!=sha256(folder/'graph.npz') or card['source_graph_sha256']!=sha256(root/'data/graphs/central-256/graph.npz'):
            raise ValueError('Null graph identity changed')
        validate_control(source_graph,load_graph(folder/'graph.npz')); graph_cards[str(seed)]=card
    for task in TASKS:
        for topology in TOPOLOGIES:
            for method in METHODS:
                for seed in SEEDS:
                    label=f'{task}-{topology}-{method}-s{seed}'; path=root/f'reports/wiring-learning/{label}.json'
                    folder=root/f'runs/wiring-learning-v1/{label}'
                    if not path.exists() or not (folder/'complete.json').exists(): raise ValueError(f'Incomplete factorial study: {label}')
                    report=read(path); complete=read(folder/'complete.json'); identity=report['identity']
                    if complete['identity']!=identity or complete['report_sha256']!=sha256(path) or complete['checkpoint_sha256']!=sha256(folder/'checkpoint-000900.pt'):
                        raise ValueError('Completed run artifacts changed')
                    for name,value in dict(task=task,topology=topology,method=method,seed=seed,updates=900,batch=8,learning_rate=.03,threads=1,protocol_sha256=sha256(root/PROTOCOL)).items():
                        if identity[name]!=value: raise ValueError(f'Changed run identity: {name}')
                    for name,digest in identity['source_sha256'].items():
                        if sha256(Path(__file__).with_name(name))!=digest: raise ValueError('Changed numerical implementation')
                    expected_graph=graph_cards[str(seed)]['graph_sha256'] if topology=='null' else sha256(root/'data/graphs/central-256/graph.npz')
                    if identity['graph_sha256']!=expected_graph: raise ValueError('Run uses an undeclared graph')
                    if report['parameters']['trainable_parameters']!=8729: raise ValueError('Run parameter budget changed')
                    if complete['training_stream_sha256']!=report['training_stream_sha256']: raise ValueError('Training stream record changed')
                    if task=='cue' and topology=='measured':
                        bridge=report['bridge']
                        if not bridge or not bridge['identical_model_tensors'] or not bridge['identical_training_stream'] or bridge['reference_report_sha256']!=sha256(root/f'reports/local-learning/{method}-s{seed}.json'):
                            raise ValueError('Prior cue bridge was not reproduced')
                    validate_panels(report); reports[(task,topology,method,seed)]=report
                    provenance.append(dict(label=label,path=path.relative_to(root).as_posix(),sha256=sha256(path),checkpoint_sha256=complete['checkpoint_sha256']))
    for seed in SEEDS:
        if len({r['initial_parameter_sha256'] for key,r in reports.items() if key[-1]==seed})!=1:
            raise ValueError('Initial parameter tensors differ across conditions')
        for task in TASKS:
            paired=[r for key,r in reports.items() if key[0]==task and key[-1]==seed]
            if len({r['training_stream_sha256'] for r in paired})!=1: raise ValueError('Sensory exposure differs across graphs or methods')
            for index in range(10):
                for panel in range(5):
                    if len({r['probes'][index]['panels'][panel]['stimulus_sha256'] for r in paired})!=1:
                        raise ValueError('Diagnostic stimuli differ across paired conditions')
    curves=[]; contrasts=[]; gradients=[]
    for task in TASKS:
        for method in METHODS:
            for index,step in enumerate(range(0,901,100)):
                mapping='reversed' if reversed_mapping(step) else 'original'
                for panel_index,delay in enumerate(DELAYS):
                    current={}
                    for topology in TOPOLOGIES:
                        panels=[reports[(task,topology,method,seed)]['probes'][index]['panels'][panel_index] for seed in SEEDS]
                        values=[p['scores'][mapping]['accuracy'] for p in panels]; current[topology]=values
                        curves.append(dict(task=task,method=method,topology=topology,step=step,delay=delay,
                            current_accuracy=statistics(values),original_accuracy=statistics([p['scores']['original']['accuracy'] for p in panels]),
                            current_cross_entropy=statistics([p['scores'][mapping]['cross_entropy'] for p in panels])))
                    contrasts.append(dict(task=task,method=method,step=step,delay=delay,
                        measured_minus_null=paired_statistics(current['measured'],current['null'])))
                if step in (0,300,600,900):
                    for topology in TOPOLOGIES:
                        for seed in SEEDS:
                            probe=reports[(task,topology,method,seed)]['probes'][index]['gradient_probe']
                            for approximation,groups in probe['comparisons'].items():
                                for group,metrics in groups.items():
                                    gradients.append(dict(task=task,method=method,topology=topology,seed=seed,step=step,
                                        approximation=approximation,group=group,**metrics))
    return dict(name='Wiring, delayed context and forward credit',runs=60,seeds=list(SEEDS),methods=METHOD_NAMES,
        tasks=dict(cue='Cue reversal',context='Delayed context'),topologies=dict(measured='Measured wiring',null='Artificial degree/sign control'),
        protocol_sha256=sha256(root/PROTOCOL),graphs=graph_cards,provenance=provenance,curves=curves,contrasts=contrasts,
        gradient_diagnostics=gradients,checks=dict(shared_initial_parameters=True,shared_sensory_streams=True,prior_cue_reproductions=15),
        limitations=['Repeated diagnostic panels, no untouched test or checkpoint selection.',
            'Three paired model/null seeds do not separately estimate topology and initialization uncertainty.',
            'Finite swap chains preserve declared invariants, not all anatomical statistics or proven uniform sampling.',
            'The context task has more sensory frames than the cue task; equal episodes do not mean equal computation.',
            'No language, real animal behavior, learned gait or general anatomical superiority is established.'])


def main():
    root=Path('.'); report=collect(root)
    write_json(root/'reports/wiring-learning/summary.json',report)
    public=root/'public/research'; public.mkdir(exist_ok=True)
    compact={key:value for key,value in report.items() if key!='gradient_diagnostics'}
    compact['gradient_diagnostics']=[row for row in report['gradient_diagnostics'] if row['group']=='combined_core']
    write_json(public/'wiring-learning.json',compact)
    with (public/'figures/wiring-gradients.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(report['gradient_diagnostics'][0])); writer.writeheader(); writer.writerows(report['gradient_diagnostics'])
    with zipfile.ZipFile(public/'wiring-learning-records.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for record in report['provenance']: archive.write(root/record['path'],record['path'])
        archive.write(root/PROTOCOL,PROTOCOL.as_posix())
        for folder in ['central-256']+[f'central-256-null{s}' for s in NULL_SEEDS.values()]:
            for name in ('graph.npz','graph-card.json'):
                path=root/'data/graphs'/folder/name; archive.write(path,path.as_posix())
        archive.write(root/'reports/wiring-learning/summary.json','summary.json')
        archive.write(root/'LICENSE','LICENSE')
    for task in TASKS:
        for method in METHODS:
            print(json.dumps(dict(task=task,method=method,final={topology:{delay:next(row['current_accuracy']['mean'] for row in report['curves'] if row['task']==task and row['method']==method and row['topology']==topology and row['step']==900 and row['delay']==delay) for delay in (8,12,48)} for topology in TOPOLOGIES})))


if __name__=='__main__': main()
