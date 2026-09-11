"""Bind the four paired core interfaces to released checkpoints and real sensors.

This exports zero-adaptation interfaces and checks numerical compatibility. It
does not open a training corpus, fit food behavior or run physical feedback.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from flm.food_core import FoodCore, ACTION_NAMES
from flm.food_core_export import arrays
from flm.inference import load_model, state_hash, verify_bundle, RUNTIME_FILES
from flm.language_train import construct
from flm.model import load_graph
from experiments.embodiment.food_core_runtime import FoodCoreRuntime

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (*RUNTIME_FILES, 'flm/food_core.py', 'flm/food_core_export.py',
    'experiments/embodiment/food_core_runtime.py', 'experiments/embodiment/verify_food_core.py',
    'scripts/prepare_food_core.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_card(path): return dict(bytes=path.stat().st_size, sha256=sha(path))


def reference(model, sensory):
    """Native bridge outputs plus the complete native transition-state trace."""
    x = torch.from_numpy(sensory)[None]
    logits, final, features = model(x)
    constants = model.core.constants(); drive = model.core.input(model.sensor(x))
    state = model.initial_state(1); fast = []; slow = []
    for t in range(x.shape[1]):
        state = model.core.transition(drive[:,t], state, constants)
        fast.append(state[0][0].numpy().copy()); slow.append(state[1][0].numpy().copy())
    for a, b in zip(state, final): torch.testing.assert_close(a, b, atol=0, rtol=0)
    return dict(fast=np.asarray(fast), slow=np.asarray(slow), features=features[0].numpy(),
        logits=logits[0].numpy(), probabilities=logits[0].softmax(-1).numpy())


@torch.no_grad()
def run(output):
    torch.set_num_threads(1)
    if output.exists(): raise ValueError('Use a new output directory; preserve prior interfaces')
    output.mkdir(parents=True)
    release = json.loads((ROOT/'public/research/inference-release.json').read_bytes())
    source_zip = ROOT/release['archive']
    if sha(source_zip) != release['archive_sha256']: raise ValueError('Released language bundle changed')
    source_folder = ROOT/'work/food-core-language-source-v1'
    if not source_folder.exists():
        source_folder.mkdir(parents=True)
        with zipfile.ZipFile(source_zip) as archive:
            for name in archive.namelist():
                if not (source_folder/name).resolve().is_relative_to(source_folder.resolve()): raise ValueError('Source bundle path leaves its directory')
            archive.extractall(source_folder)
    bundle = verify_bundle(source_folder)
    graph_path = source_folder/bundle['graph']; graph = load_graph(graph_path)
    physical = ROOT/'runs/embodiment/food-approach-v1'
    physical_summary = json.loads((physical/'summary.json').read_bytes())
    streams = []; segments = []
    for name in ('odor-a-left', 'odor-a-right'):
        record = next(r for r in physical_summary['trials'] if r['case']['label'] == name)
        path = physical/(name+'.npz')
        if sha(path) != record['trajectory_sha256'] or record['status'] != 'complete': raise ValueError('Physical source trace differs')
        with np.load(path, allow_pickle=False) as archive: stream = archive['sensory'].astype(np.float32)
        if stream.shape != (201,6): raise ValueError('Expected complete physical sensor episode')
        streams.append(stream); segments.append(dict(label=name, frames=201, source_trajectory_sha256=sha(path), start=len(segments)*201))
    stress = np.random.default_rng(711).uniform(size=(64,6)).astype(np.float32)
    stress[0] = 0; stress[1] = 1; stress[2:8] = np.eye(6, dtype=np.float32)
    streams.append(stress); segments.append(dict(label='synthetic-normalized-channel-stress', frames=64, start=402, pcg64_seed=711))
    sensory = np.concatenate(streams); reset = np.zeros(len(sensory), dtype=bool); reset[[0,201,402]] = True
    fixture = dict(sensory=sensory, reset=reset); models = []; parity = []; shared_adapters = None
    for seed in (42,43):
        trained, lexicon, selected = load_model(source_folder, f'flm-s{seed}')
        initial = construct('flm', graph_path, lexicon.vocabulary, seed)
        for label, language in (('initial', initial), ('language', trained)):
            identity = f'{label}-s{seed}'; source_state = state_hash(language); bridge = FoodCore(language, 711).eval()
            for name, expected in (('row', graph['row']), ('col', graph['col']), ('base_weight', graph['weight']), ('pool_index', graph['pool'])):
                np.testing.assert_array_equal(getattr(bridge.core, name).numpy(), expected)
            tokens = torch.tensor([[0, *lexicon.encode('Small signals can guide the fly.')]])
            native_logits, native_state = language(tokens)
            features, state = bridge.encode_projected(language.embedding(tokens))
            recovered = F.linear(language.readout(features), language.embedding.weight, language.output_bias)
            torch.testing.assert_close(recovered, native_logits, atol=0, rtol=0)
            for a, b in zip(state, native_state): torch.testing.assert_close(a, b, atol=0, rtol=0)
            exported = arrays(bridge, graph['body_ids']); current = {k: v for k,v in exported.items() if k.startswith(('sensor_', 'action_'))}
            if shared_adapters is None: shared_adapters = current
            elif any(not np.array_equal(v, shared_adapters[k]) for k,v in current.items()): raise ValueError('Fresh adapters differ across cores')
            references = [reference(bridge, stream) for stream in streams]
            for key in references[0]: fixture[identity+'_'+key] = np.concatenate([r[key] for r in references])
            runtime = FoodCoreRuntime(exported); maxima = {k:0. for k in references[0]}
            for i, x in enumerate(sensory):
                if reset[i]: runtime.reset()
                logits, probabilities, encoded = runtime.step(x)
                for key, actual in (('fast',runtime.fast), ('slow',runtime.slow), ('features',encoded), ('logits',logits), ('probabilities',probabilities)):
                    expected = fixture[identity+'_'+key][i]
                    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)
                    maxima[key] = max(maxima[key], float(np.max(np.abs(actual-expected))))
                if int(logits.argmax()) != int(fixture[identity+'_logits'][i].argmax()): raise ValueError('Greedy action differs')
            if state_hash(language) != source_state or state_hash(bridge.core) != source_state: raise ValueError('Source language tensors were modified')
            payload = output/(identity+'.npz'); np.savez_compressed(payload, **exported)
            models.append(dict(id=identity, file=payload.name, **file_card(payload), training_seed=seed,
                core_state_sha256=source_state, source_selected_language_model=selected,
                origin='reconstructed initialization using released constructor and original seed' if label=='initial' else 'released validation-selected WikiText checkpoint',
                parameter_card=bridge.parameter_card()))
            parity.append(dict(id=identity, frames=len(sensory), maximum_absolute_error=maxima,
                lexical_forward_exact=True, greedy_action_exact=True, source_model_unchanged=True))
            print(json.dumps(parity[-1]), flush=True)
    np.savez_compressed(output/'fixtures.npz', **fixture)
    manifest = dict(format='flm-food-core-interface-v1', prepared_utc=datetime.now(timezone.utc).isoformat(),
        models=models, fixture=dict(file='fixtures.npz', **file_card(output/'fixtures.npz'), segments=segments),
        action_names=list(ACTION_NAMES), adaptation_updates=0, physical_trials_run=0, corpus_payloads_opened=False,
        source_language_archive_sha256=sha(source_zip), graph_sha256=sha(graph_path),
        source_physical_summary_sha256=sha(physical/'summary.json'), sources={n:sha(ROOT/n) for n in SOURCES},
        native_environment=dict(torch=torch.__version__, numpy=np.__version__, numerical_threads=1),
        parity_tolerance=dict(atol=5e-6, rtol=5e-5), native_parity=parity,
        scope='Four unadapted sensory interfaces on matched original/random and language-trained cores. Sensor replay is open loop; action logits are not physical outcomes or evidence of beneficial transfer.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8', newline='\n')
    for name in ('food_core_runtime.py','verify_food_core.py'):
        shutil.copyfile(ROOT/'experiments/embodiment'/name, output/name)
    shutil.copyfile(ROOT/'LICENSE', output/'LICENSE')
    for name in ('CC-BY-4.0.txt', 'DATA-ATTRIBUTION.md', 'BODY-PROVENANCE.md', 'Body-Apache-2.0.txt', 'Body-MIT.txt'):
        shutil.copyfile(ROOT/'licenses'/name, output/name)
    shutil.copyfile(ROOT/'data/graphs/central-1024/graph-card.json', output/'graph-card.json')
    print(json.dumps(dict(models=4, frames_per_model=len(sensory), adaptation_updates=0, output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
