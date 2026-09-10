"""Export frozen cue-learning models and independent PyTorch state fixtures."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flm.local_learning import ChoiceFLM
from flm.model import load_graph
from flm.provenance import sha256, write_json
from experiments.embodiment.choice_runtime import verify_parity


def main():
    torch.set_num_threads(1)
    graph_path = ROOT / 'data/graphs/central-256/graph.npz'; graph = load_graph(graph_path)
    destination = ROOT / 'data/controllers/choice-v1'; destination.mkdir(parents=True, exist_ok=True)
    methods = ('bptt', 'reservoir', 'eligibility'); initial_states = []
    for method in methods:
        folder = ROOT / f'runs/local-learning-v1/{method}-s17'
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['report_sha256'] != sha256(ROOT / f'reports/local-learning/{method}-s17.json'):
            raise ValueError('Original behavioral report changed')
        if complete['checkpoint_sha256'] != sha256(folder / 'checkpoint-000900.pt'):
            raise ValueError('Selected final behavioral checkpoint changed')
        initial_states.append(torch.load(folder / 'checkpoint-000000.pt', weights_only=True, map_location='cpu')['model'])
    if any(not torch.equal(value, other[name]) for other in initial_states[1:] for name,value in initial_states[0].items()):
        raise ValueError('The shared initialization differs across methods')
    rng = np.random.default_rng(8911)
    sensory = rng.normal(0, .2, (128,4)).astype(np.float32)
    reset = np.zeros(128, dtype=bool); reset[[0,37,74,106,117]] = True
    for start, cue in ((106,0), (117,1)):
        sensory[start:start+11] = 0; sensory[start:start+2, cue] = 1; sensory[start+10,3] = 1
    fixtures = dict(sensory=sensory, reset=reset); records = []
    for identity, method, step in [('initial','bptt',0), *[(m,m,900) for m in methods]]:
        checkpoint = ROOT / f'runs/local-learning-v1/{method}-s17/checkpoint-{step:06d}.pt'
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if saved['step'] != step or saved['identity']['method'] != method or saved['identity']['seed'] != 17 or saved['identity']['graph_sha256'] != sha256(graph_path):
            raise ValueError('Original model identity mismatch')
        model = ChoiceFLM(graph); model.load_state_dict(saved['model']); model.eval()
        with torch.no_grad():
            matrix, alpha, beta, gain = model.constants()
            if matrix.is_sparse: matrix = matrix.to_dense()
            values = dict(recurrent=matrix, alpha=alpha, beta=beta, gain=gain,
                input_weight=model.input.weight, input_bias=model.input.bias,
                pool_index=model.pool_index, pool_sizes=model.pool_sizes,
                norm_weight=model.norm.weight, norm_bias=model.norm.bias,
                readout_weight=model.readout.weight, readout_bias=model.readout.bias)
            arrays = {name: value.detach().cpu().numpy().copy() for name,value in values.items()}
            arrays['norm_epsilon'] = np.asarray(model.norm.eps, dtype=np.float32)
            path = destination / (identity + '.npz'); np.savez_compressed(path, **arrays)
            states = dict(fast=[], slow=[], logits=[]); state = None
            for i, frame in enumerate(sensory):
                if reset[i]: state = None
                logits, state = model(torch.from_numpy(frame).reshape(1,1,4), state)
                for name,value in [('logits',logits[0]), ('fast',state[0][0]), ('slow',state[1][0])]:
                    states[name].append(value.numpy().copy())
            fixtures.update({identity + '_' + name: np.asarray(rows) for name,rows in states.items()})
        records.append(dict(id=identity, method=method, training_seed=17, checkpoint_step=step,
            file=path.name, sha256=sha256(path), source_checkpoint_sha256=sha256(checkpoint),
            graph_sha256=sha256(graph_path), parameters=model.parameter_card()['trainable_parameters']))
    path = destination / 'parity.npz'; np.savez_compressed(path, **fixtures)
    write_json(destination / 'manifest.json', dict(format='flm-choice-inference-v1', models=records,
        fixture=dict(file=path.name,sha256=sha256(path),random_seed=8911,frames=128,
                     tolerance=dict(atol=2e-6,rtol=2e-5),includes_original_zero_distractor_cues=True),
        runtime_source_sha256=sha256(ROOT / 'experiments/embodiment/choice_runtime.py'),
        exporter_sha256=sha256(Path(__file__)), torch=str(torch.__version__), numpy=np.__version__,
        initial_tensors_identical_across_methods=True,
        scope='Fixed sensory-choice models; no language weights, new training or learned gait.'))
    result = verify_parity(destination)
    write_json(ROOT / 'reports/embodiment/choice-runtime-parity.json', dict(environment='Original PyTorch training environment', models=result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
