"""Run the declared delayed-cue reversal study with recoverable checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .local_learning import ChoiceFLM, learning_step
from .model import load_graph
from .provenance import sha256, write_json

METHODS = ('bptt', 'reservoir', 'eligibility', 'instantaneous', 'reward')
SEEDS = (17, 29, 41)
DELAYS = (4, 8, 12, 24, 48)
GRAPH = Path('data/graphs/central-256/graph.npz')
PROTOCOL = Path('docs/LOCAL-LEARNING-PROTOCOL.md')


def episode_batch(rng, count, delay, zero_distractor=False):
    if count < 2 or count % 2 or delay < 0: raise ValueError('Require a positive even batch and nonnegative delay')
    cue = np.tile(np.array([0, 1], dtype=np.int64), count // 2); rng.shuffle(cue)
    sensory = np.zeros((count, delay + 3, 4), dtype=np.float32)
    sensory[np.arange(count), 0, cue] = 1.; sensory[np.arange(count), 1, cue] = 1.
    if not zero_distractor: sensory[:, :, 2] = rng.normal(0., .2, (count, delay + 3))
    sensory[:, -1, 3] = 1.
    return sensory, cue


def reversed_mapping(step): return 300 < step <= 600


@torch.no_grad()
def probes(model):
    records = []
    for delay in DELAYS:
        sensory, cue = episode_batch(np.random.default_rng(7001 + delay), 256, delay)
        logits = model(torch.from_numpy(sensory))[0]; logp = logits.log_softmax(-1)
        predicted = logits.argmax(-1).numpy(); rows = torch.arange(len(cue)); labels = torch.from_numpy(cue)
        scores = {}
        for mapping, targets in [('original', labels), ('reversed', 1 - labels)]:
            scores[mapping] = dict(accuracy=float(np.mean(predicted == targets.numpy())),
                                  cross_entropy=float(-logp[rows, targets].double().mean()))
        records.append(dict(delay=delay, episodes=len(cue), stimulus_sha256=hashlib.sha256(sensory.tobytes() + cue.tobytes()).hexdigest(),
            scores=scores, cue_ids=cue.tolist(), action_probabilities=logp.exp().tolist()))
    return records


@torch.no_grad()
def physical_choices(model, step):
    sensory, cue = episode_batch(np.random.default_rng(7009), 2, 8, zero_distractor=True)
    logits, (h, slow) = model(torch.from_numpy(sensory)); probabilities = logits.softmax(-1)
    result = []
    for i in np.argsort(cue):
        action = int(probabilities[i].argmax())
        result.append(dict(cue=int(cue[i]), step=step, delay=8, distractor='zero',
            expected_action=int(1 - cue[i] if reversed_mapping(step) else cue[i]),
            chosen_action=action, action_probabilities=probabilities[i].tolist(),
            descending_signal=[.4, 1.2] if action == 0 else [1.2, .4],
            fast_state=h[i].tolist(), slow_state=slow[i].tolist()))
    return result


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def run(method, seed):
    folder = Path(f'runs/local-learning-v1/{method}-s{seed}')
    identity = dict(method=method, seed=seed, protocol_sha256=sha256(PROTOCOL), graph_sha256=sha256(GRAPH),
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in ('behavior_study.py', 'local_learning.py', 'model.py')},
        updates=900, batch=8, learning_rate=.03, threads=1, torch_version=str(torch.__version__))
    if (folder / 'complete.json').exists():
        complete = json.loads((folder / 'complete.json').read_text(encoding='utf8'))
        if complete['identity'] != identity or complete['checkpoint_sha256'] != sha256(folder / 'checkpoint-000900.pt'):
            raise ValueError('Completed behavior run identity changed')
        return complete
    torch.manual_seed(seed); graph = load_graph(GRAPH); model = ChoiceFLM(graph)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    rng = np.random.default_rng(seed); action_rng = torch.Generator().manual_seed(seed + 10000)
    start = 0; baseline = .5; chain = '00' * 32; previous_seconds = 0.
    checkpoint = folder / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if saved['identity'] != identity: raise ValueError('Resume changes the declared behavior experiment')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['sensory_rng']; action_rng.set_state(saved['action_rng'])
        start = saved['step']; baseline = saved['baseline']; chain = saved['training_stream_sha256']; previous_seconds = saved['seconds']
    elif (folder / 'run.json').exists(): raise ValueError('Run exists without a recoverable checkpoint')
    write_json(folder / 'run.json', dict(identity=identity, parameters=model.parameter_card(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        name='Delayed cue, distractor and reversal; synthetic sensory task, not language training'))
    started = time.perf_counter()

    def checkpoint_payload(step):
        return dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sensory_rng=rng.bit_generator.state, action_rng=action_rng.get_state(), step=step, baseline=baseline,
            training_stream_sha256=chain, seconds=previous_seconds + time.perf_counter() - started)

    def measure(step):
        measured = dict(step=step, current_mapping='reversed' if reversed_mapping(step) else 'original',
            panels=probes(model), parameter_l2_change={name: float((value.detach() - initial[name]).norm()) for name, value in model.named_parameters()})
        write_json(folder / f'probe-{step:06d}.json', measured)
        if step in (0, 300, 600, 900):
            save(folder / f'checkpoint-{step:06d}.pt', checkpoint_payload(step))
            write_json(folder / f'choices-{step:06d}.json', dict(step=step, graph_sha256=identity['graph_sha256'],
                body_ids=graph['body_ids'].tolist(), episodes=physical_choices(model, step)))
        save(checkpoint, checkpoint_payload(step))
        print(json.dumps(dict(event='probe', method=method, seed=seed, step=step,
            current_mapping=measured['current_mapping'], accuracy_by_delay={p['delay']: p['scores'][measured['current_mapping']]['accuracy'] for p in measured['panels']})), flush=True)

    if start == 0: measure(0)
    for step in range(start + 1, 901):
        delay = int(rng.integers(4, 13)); sensory, cue = episode_batch(rng, 8, delay)
        targets = 1 - cue if reversed_mapping(step) else cue
        chain = hashlib.sha256(bytes.fromhex(chain) + sensory.tobytes() + targets.tobytes()).hexdigest()
        measured = learning_step(model, torch.from_numpy(sensory), torch.from_numpy(targets), optimizer, method, action_rng, baseline)
        baseline = measured.pop('baseline')
        if step % 25 == 0:
            write_json(folder / f'progress-{step:06d}.json', dict(step=step, delay=delay,
                training_stream_sha256=chain, **measured))
        if step % 100 == 0: measure(step)
    report = dict(identity=identity, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        parameters=model.parameter_card(), training_stream_sha256=chain,
        probes=[json.loads(path.read_text(encoding='utf8')) for path in sorted(folder.glob('probe-*.json'))],
        choices=[json.loads(path.read_text(encoding='utf8')) for path in sorted(folder.glob('choices-*.json'))],
        progress=[json.loads(path.read_text(encoding='utf8')) for path in sorted(folder.glob('progress-*.json'))],
        note='Repeated fixed-panel diagnostics; no checkpoint selection. Forward core credit omits cross-neuron temporal paths. No language training, rewired-graph control, or learned gait is implied.')
    report_path = Path(f'reports/local-learning/{method}-s{seed}.json'); write_json(report_path, report)
    complete = dict(identity=identity, checkpoint_sha256=sha256(folder / 'checkpoint-000900.pt'),
        report_sha256=sha256(report_path), training_stream_sha256=chain)
    write_json(folder / 'complete.json', complete); return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args(); torch.set_num_threads(1)
    for seed in args.seeds:
        results = [run(method, seed) for method in args.methods]
        if len({r['training_stream_sha256'] for r in results}) != 1:
            raise ValueError('Conditions did not receive identical stimuli and targets')


if __name__ == '__main__': main()
