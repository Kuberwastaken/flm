"""Run the declared factorial wiring/learning/context study with exact recovery."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .behavior_study import METHODS, SEEDS, DELAYS, reversed_mapping, save
from .local_learning import ChoiceFLM, learning_step
from .model import load_graph
from .provenance import sha256, write_json
from .wiring_diagnostics import task_batch, gradient_probe

TASKS = ('cue', 'context')
TOPOLOGIES = ('measured', 'null')
NULL_SEEDS = dict(zip(SEEDS, (101, 103, 107)))
PROTOCOL = Path('docs/WIRING-LEARNING-PROTOCOL.md')


def read(path): return json.loads(path.read_text(encoding='utf8'))


def tensor_hash(parameters):
    digest = hashlib.sha256()
    for name, value in parameters:
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def probes(model, task):
    records = []
    for delay in DELAYS:
        sensory, targets, metadata = task_batch(task, np.random.default_rng((7001 if task == 'cue' else 17001) + delay), 256, delay)
        logits = model(torch.from_numpy(sensory))[0]; logp = logits.log_softmax(-1)
        labels = torch.from_numpy(targets); rows = torch.arange(len(labels)); predictions = logits.argmax(-1)
        scores = {}
        for mapping, y in [('original', labels), ('reversed', 1-labels)]:
            scores[mapping] = dict(accuracy=float((predictions == y).double().mean()), cross_entropy=float(-logp[rows, y].double().mean()))
        records.append(dict(delay=delay, episodes=len(labels), stimulus_sha256=hashlib.sha256(sensory.tobytes()+targets.tobytes()).hexdigest(),
            labels=targets.tolist(), **metadata, scores=scores, action_probabilities=logp.exp().tolist()))
    return records


def verify_bridge(root, method, seed, model, chain):
    previous = root / f'runs/local-learning-v1/{method}-s{seed}'
    complete = read(previous / 'complete.json'); checkpoint = previous / 'checkpoint-000900.pt'
    report_path = root / f'reports/local-learning/{method}-s{seed}.json'
    if sha256(checkpoint) != complete['checkpoint_sha256'] or sha256(report_path) != complete['report_sha256']:
        raise ValueError('Bridge reference changed')
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if saved['training_stream_sha256'] != chain: raise ValueError('Bridge sensory stream differs')
    for name, value in model.state_dict().items():
        if not torch.equal(value, saved['model'][name]): raise ValueError(f'Bridge model differs: {name}')
    return dict(reference_report_sha256=sha256(report_path), identical_model_tensors=True,
        identical_training_stream=True, note='Repeated earlier condition, not an independent replicate')


def run(task, topology, method, seed, root=Path('.'), stop_after=900):
    if task not in TASKS or topology not in TOPOLOGIES or method not in METHODS or seed not in SEEDS:
        raise ValueError('Undeclared experimental condition')
    if stop_after not in range(100, 901, 100): raise ValueError('Stop only at a recoverable 100-update boundary')
    graph_path = root / ('data/graphs/central-256' + (f'-null{NULL_SEEDS[seed]}' if topology == 'null' else '') + '/graph.npz')
    if sha256(graph_path) != read(graph_path.with_name('graph-card.json'))['graph_sha256']:
        raise ValueError('Graph no longer matches its card')
    label = f'{task}-{topology}-{method}-s{seed}'; folder = root / f'runs/wiring-learning-v1/{label}'
    report_path = root / f'reports/wiring-learning/{label}.json'
    identity = dict(task=task, topology=topology, null_seed=NULL_SEEDS[seed] if topology == 'null' else None,
        method=method, seed=seed, graph_sha256=sha256(graph_path), protocol_sha256=sha256(root / PROTOCOL),
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('wiring_study.py','wiring_diagnostics.py','behavior_study.py','local_learning.py','model.py')},
        updates=900, batch=8, learning_rate=.03, threads=1, torch_version=str(torch.__version__))
    if (folder / 'complete.json').exists():
        completed = read(folder / 'complete.json')
        if completed['identity'] != identity or completed['checkpoint_sha256'] != sha256(folder / 'checkpoint-000900.pt') or completed['report_sha256'] != sha256(report_path):
            raise ValueError('Completed factorial run identity changed')
        return completed
    torch.manual_seed(seed); model = ChoiceFLM(load_graph(graph_path))
    initial = {name:value.detach().clone() for name,value in model.named_parameters()}
    initial_hash = tensor_hash(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    rng = np.random.default_rng(seed); action_rng = torch.Generator().manual_seed(seed+10000)
    start=0; baseline=.5; chain='00'*32; previous_seconds=0.; checkpoint=folder/'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if saved['identity'] != identity: raise ValueError('Resume changes the registered factorial experiment')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['sensory_rng']; action_rng.set_state(saved['action_rng'])
        start=saved['step']; baseline=saved['baseline']; chain=saved['training_stream_sha256']; previous_seconds=saved['seconds']
    elif (folder/'run.json').exists(): raise ValueError('Run exists without a recoverable checkpoint')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write_json(folder/'run.json',dict(identity=identity,parameters=model.parameter_card(),initial_parameter_sha256=initial_hash,source_commit=commit))
    started=time.perf_counter()

    def payload(step):
        return dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            sensory_rng=rng.bit_generator.state,action_rng=action_rng.get_state(),baseline=baseline,
            training_stream_sha256=chain,seconds=previous_seconds+time.perf_counter()-started)

    def measure(step):
        measured=dict(step=step,current_mapping='reversed' if reversed_mapping(step) else 'original',panels=probes(model,task),
            parameter_l2_change={name:float((value.detach()-initial[name]).norm()) for name,value in model.named_parameters()})
        if step in (0,300,600,900):
            sensory,targets,_=task_batch(task,np.random.default_rng(27009 if task=='cue' else 37009),8,8)
            if reversed_mapping(step): targets=1-targets
            measured['gradient_probe']=dict(stimulus_sha256=hashlib.sha256(sensory.tobytes()+targets.tobytes()).hexdigest(),
                **gradient_probe(model,torch.from_numpy(sensory),torch.from_numpy(targets)))
            save(folder/f'checkpoint-{step:06d}.pt',payload(step))
        write_json(folder/f'probe-{step:06d}.json',measured); save(checkpoint,payload(step))
        print(json.dumps(dict(condition=label,step=step,accuracy={p['delay']:p['scores'][measured['current_mapping']]['accuracy'] for p in measured['panels']})),flush=True)

    if start==0: measure(0)
    for step in range(start+1,stop_after+1):
        delay=int(rng.integers(4,13)); sensory,targets,_=task_batch(task,rng,8,delay)
        if reversed_mapping(step): targets=1-targets
        chain=hashlib.sha256(bytes.fromhex(chain)+sensory.tobytes()+targets.tobytes()).hexdigest()
        metric=learning_step(model,torch.from_numpy(sensory),torch.from_numpy(targets),optimizer,method,action_rng,baseline)
        baseline=metric.pop('baseline')
        if step%25==0: write_json(folder/f'progress-{step:06d}.json',dict(step=step,delay=delay,training_stream_sha256=chain,**metric))
        if step%100==0: measure(step)
    if stop_after<900: return dict(complete=False,identity=identity,step=max(start,stop_after),training_stream_sha256=chain)
    bridge=verify_bridge(root,method,seed,model,chain) if task=='cue' and topology=='measured' else None
    report=dict(identity=identity,source_commit=commit,parameters=model.parameter_card(),initial_parameter_sha256=initial_hash,
        training_stream_sha256=chain,bridge=bridge,probes=[read(path) for path in sorted(folder.glob('probe-*.json'))],
        progress=[read(path) for path in sorted(folder.glob('progress-*.json'))],
        note='Repeated fixed-panel diagnostics; no checkpoint selection. Null topology and initialization are paired, not independently crossed. Artificial cue/context task, not language or gait.')
    write_json(report_path,report)
    completed=dict(identity=identity,checkpoint_sha256=sha256(folder/'checkpoint-000900.pt'),report_sha256=sha256(report_path),
        training_stream_sha256=chain,initial_parameter_sha256=initial_hash)
    write_json(folder/'complete.json',completed); return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--topologies',nargs='+',choices=TOPOLOGIES,default=list(TOPOLOGIES))
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    args=parser.parse_args(); torch.set_num_threads(1)
    for task in args.tasks:
        for seed in args.seeds:
            results=[run(task,topology,method,seed) for topology in args.topologies for method in args.methods]
            if len({r['training_stream_sha256'] for r in results})!=1 or len({r['initial_parameter_sha256'] for r in results})!=1:
                raise ValueError('Paired factorial conditions differ in exposure or initial parameters')


if __name__=='__main__': main()
