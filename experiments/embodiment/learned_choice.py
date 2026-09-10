"""Replay every predeclared learned high-level choice in real MuJoCo physics.

Uses the isolated FlyGym environment. Neural inference has already been recorded
by behavior_study; no PyTorch dependency or invented neural activity is needed.
The gait/contact controller is designed, and FLM receives no proprioception here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
from calibrate import trial

METHODS = ('bptt', 'reservoir', 'eligibility', 'instantaneous', 'reward')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf8'))
def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf8'); temporary.replace(path)


def cases():
    output = []
    for method in METHODS:
        folder = Path(f'runs/local-learning-v1/{method}-s17')
        if not (folder / 'complete.json').exists(): raise ValueError(f'Neural condition is incomplete: {method}')
        complete = read(folder / 'complete.json'); report_path = Path(f'reports/local-learning/{method}-s17.json')
        if complete['report_sha256'] != sha(report_path): raise ValueError('Neural report identity changed')
        report = read(report_path)
        if report['identity'] != complete['identity'] or report['identity']['seed'] != 17 or report['identity']['method'] != method:
            raise ValueError('Wrong neural experiment identity')
        if [item['step'] for item in report['choices']] != [0, 300, 600, 900]: raise ValueError('Missing neural assay checkpoints')
        for checkpoint in report['choices']:
            if [item['cue'] for item in checkpoint['episodes']] != [0, 1]: raise ValueError('Missing cue condition')
            weight_path = folder / f'checkpoint-{checkpoint["step"]:06d}.pt'
            for episode in checkpoint['episodes']:
                probabilities = np.array(episode['action_probabilities'])
                if probabilities.shape != (2,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1., atol=1e-6):
                    raise ValueError('Invalid neural action distribution')
                action = int(probabilities.argmax()); signal = [.4, 1.2] if action == 0 else [1.2, .4]
                if episode['chosen_action'] != action or episode['descending_signal'] != signal: raise ValueError('Neural choice/command mismatch')
                output.append(dict(label=f'{method}-{checkpoint["step"]:06d}-cue{episode["cue"]}', method=method,
                    training_seed=17, neural_report_sha256=sha(report_path), checkpoint_sha256=sha(weight_path),
                    neural_episode=episode))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--video', action='store_true')
    args = parser.parse_args(); declared = cases(); root = Path('runs/embodiment/learned-choice-v1'); results = []
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        versions={name: importlib.metadata.version(name) for name in ('flygym', 'mujoco', 'numpy', 'scipy', 'numba')})
    source = {name: sha(Path(__file__).with_name(name)) for name in ('learned_choice.py', 'calibrate.py')}
    for case in declared:
        episode = case['neural_episode']; video = args.video and case['method'] == 'eligibility' and episode['step'] == 300 and episode['cue'] == 0
        identity = dict(case=case, source_sha256=source, environment=environment, physics_seed=17, simulation_seconds=1., video=video)
        path = root / f'{case["label"]}.json'
        if path.exists():
            measured = read(path)
            if measured['identity'] != identity or measured['physical']['trajectory_sha256'] != sha(root / f'{case["label"]}.npz'):
                raise ValueError('Physical replay identity changed')
        else:
            physical, _ = trial(episode['descending_signal'], 17, 1., case['label'], root, video)
            measured = dict(identity=identity, physical=physical,
                neural_choice_correct=episode['chosen_action'] == episode['expected_action'],
                actual_heading_sign=int(np.sign(physical['yaw_change_rad'])))
            write(path, measured)
        results.append(measured)
        print(json.dumps(dict(event='learned_physical_replay', complete=len(results), total=len(declared),
            case=case['label'], chosen_action=episode['chosen_action'], correct=measured['neural_choice_correct'],
            yaw_change_rad=measured['physical']['yaw_change_rad'])), flush=True)
    write(Path('reports/embodiment/learned-choice.json'), dict(environment=environment, source_sha256=source, trials=results,
        model='256-neuron FLM choice network, trained on synthetic cue episodes; no language weights',
        coupling='A recorded learned action selects the descending command for a designed hybrid gait controller; every case is simulated separately',
        flm_connected_at_action_selection=True, proprioceptive_feedback_to_flm=False, learned_gait=False,
        caution='A learned high-level direction changes the physical trajectory. The same two descending commands and initial state can reproduce identical paths across many cases. These are not forty independent motor skills or language-to-body transfer.'))


if __name__ == '__main__': main()
