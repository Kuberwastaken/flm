"""Inspect and run the complete released language computation-control selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path

import torch

from .inference import RUNTIME_FILES as COMPARISON_RUNTIME, bundle_path, generate, state_hash
from .language_core_controls import construct as construct_control
from .language_train import construct as construct_original
from .provenance import sha256
from .tokenizer import Lexicon


RUNTIME_FILES = (*COMPARISON_RUNTIME, 'flm/language_core_controls.py', 'flm/core_inference.py')
CHECKPOINT_FORMAT = 'flm-language-core-inference-checkpoint-v1'
BUNDLE_FORMAT = 'flm-language-core-inference-v1'


def condition_specs():
    return {f'{control}-s{seed}': dict(control=control, seed=seed, reference=control == 'full')
            for seed in (42, 43)
            for control in ('full', 'fixed_dynamics', 'no_lateral', 'no_temporal_state')}


def parameter_counts(model):
    allocated = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(allocated_parameters=allocated, trainable_parameters=trainable,
                frozen_parameters=allocated - trainable)


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'bundle.json').read_text(encoding='utf8'))
    if manifest.get('format') != BUNDLE_FORMAT:
        raise ValueError('Unsupported language computation inference bundle')
    if manifest.get('runtime_sources') != list(RUNTIME_FILES):
        raise ValueError('Computation inference runtime inventory changed')
    files = manifest['files']
    for name, record in files.items():
        path = bundle_path(root, name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Bundle file failed integrity verification: ' + name)
    runtime_root = Path(__file__).resolve().parents[1]
    for name in RUNTIME_FILES:
        if name not in files or sha256(runtime_root / name) != files[name]['sha256']:
            raise ValueError('Run with the matching bundled computation runtime: ' + name)
    for name in (manifest['study_identity'], manifest['selection'], manifest['tokenizer']):
        if name not in files:
            raise ValueError('Required study input is absent from the manifest')
    identity_path = bundle_path(root, manifest['study_identity'])
    selection_path = bundle_path(root, manifest['selection'])
    if sha256(identity_path) != manifest['study_identity_sha256'] or sha256(selection_path) != manifest['selection_sha256']:
        raise ValueError('Study or selection identity changed')
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    if selection['study_identity_sha256'] != manifest['study_identity_sha256']:
        raise ValueError('Selection belongs to a different computation study')
    for name in RUNTIME_FILES:
        if name in identity['sources'] and files[name]['sha256'] != identity['sources'][name]:
            raise ValueError('A frozen numerical source changed: ' + name)
    if (files[manifest['tokenizer']]['sha256'] != selection['tokenizer_sha256'] or
            selection['tokenizer_sha256'] != identity['inputs'].get(manifest['tokenizer'])):
        raise ValueError('Selected tokenizer changed')
    expected = condition_specs()
    inventories = {}
    for field, rows, key in [('declared', identity['conditions'], 'label'),
                             ('selected', selection['runs'], 'label'), ('models', manifest['models'], 'id')]:
        lookup = {row[key]: row for row in rows}
        if len(rows) != 8 or set(lookup) != set(expected):
            raise ValueError('The complete eight-condition computation selection is required: ' + field)
        inventories[field] = lookup
    graph_paths = set()
    checkpoint_paths = set()
    for name, record in inventories['models'].items():
        selected = inventories['selected'][name]
        declared = inventories['declared'][name]
        if any(declared.get(key) != value for key, value in expected[name].items()):
            raise ValueError('Unexpected declared computation condition: ' + name)
        if any(selected.get(key) != value for key, value in declared.items()):
            raise ValueError('Selection changed its declared condition: ' + name)
        binding = dict(control=selected['control'], training_seed=selected['seed'],
                       reference=selected['reference'], graph=selected['graph'],
                       source_checkpoint_sha256=selected['checkpoint_sha256'],
                       checkpoint_step=selected['checkpoint_step'],
                       **{key: selected[key] for key in ('allocated_parameters', 'trainable_parameters', 'frozen_parameters')})
        if any(record.get(key) != value for key, value in binding.items()):
            raise ValueError('Released model differs from its selected condition: ' + name)
        step = record['checkpoint_step']
        if type(step) is not int or not 500 <= step <= 6000 or step % 500:
            raise ValueError('Checkpoint is outside the declared selection schedule')
        for path in (record['file'], record['graph']):
            if path not in files:
                raise ValueError('A required checkpoint or graph is absent from the manifest')
        graph_hash = files[record['graph']]['sha256']
        if graph_hash != identity['inputs'].get(record['graph']) or record.get('graph_sha256') != graph_hash:
            raise ValueError('Selected measured graph changed: ' + name)
        graph_paths.add(record['graph'])
        checkpoint_paths.add(record['file'])
        if record['reference'] and identity['reference_checkpoints'].get(name) != record['source_checkpoint_sha256']:
            raise ValueError('The original full-model reference changed: ' + name)
    if len(graph_paths) != 1 or len(checkpoint_paths) != 8:
        raise ValueError('Use the same measured graph and eight distinct checkpoint files')
    return manifest


def load_model(root, name):
    root = Path(root).resolve()
    manifest = verify_bundle(root)
    record = next((row for row in manifest['models'] if row['id'] == name), None)
    if record is None:
        raise ValueError('Unknown language computation condition')
    payload = bundle_path(root, record['file']).read_bytes()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    keys = {'format', 'control', 'model', 'config', 'step', 'run', 'source_checkpoint_sha256', 'state_sha256'}
    if (set(saved) != keys or saved['format'] != CHECKPOINT_FORMAT or
            set(saved['run']) != {'seed', 'graph_sha256', 'tokenizer_sha256'}):
        raise ValueError('Expected a model-only computation inference checkpoint')
    expected_run = dict(seed=record['training_seed'], graph_sha256=record['graph_sha256'],
                        tokenizer_sha256=manifest['files'][manifest['tokenizer']]['sha256'])
    if (saved['run'] != expected_run or saved['control'] != record['control'] or
            saved['source_checkpoint_sha256'] != record['source_checkpoint_sha256'] or
            saved['step'] != record['checkpoint_step'] or saved['state_sha256'] != record['state_sha256']):
        raise ValueError('Checkpoint identity or mechanism differs from its selected model')
    lexicon = Lexicon(bundle_path(root, manifest['tokenizer']))
    graph = bundle_path(root, record['graph'])
    if record['control'] == 'full':
        # Preserve the original reference class/configuration. Its checkpoint
        # predates the extended control field; it is not silently converted.
        model = construct_original('flm', graph, lexicon.vocabulary, record['training_seed'])
    else:
        model = construct_control(record['control'], graph, lexicon.vocabulary, record['training_seed'])
    if saved['config'] != asdict(model.config):
        raise ValueError('Checkpoint configuration does not implement the selected mechanism')
    values = saved['model']
    expected = model.state_dict()
    if values.keys() != expected.keys():
        raise ValueError('Checkpoint model tensor inventory changed')
    for key, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.dtype != expected[key].dtype or
                value.shape != expected[key].shape or not torch.isfinite(value).all()):
            raise ValueError('Invalid inference tensor: ' + key)
    for key, value in model.named_buffers():
        if not torch.equal(values[key], value):
            raise ValueError('Checkpoint graph/pooling buffer changed: ' + key)
    if record['control'] == 'fixed_dynamics':
        model.verify_frozen_dynamics(values)
    model.load_state_dict(values)
    if state_hash(model) != record['state_sha256']:
        raise ValueError('Inference tensor state differs from the release record')
    if any(record[key] != value for key, value in parameter_counts(model).items()):
        raise ValueError('Allocated/trainable/frozen parameter inventory changed')
    if record['control'] == 'fixed_dynamics':
        model.verify_frozen_dynamics()
    model.eval()
    return model, lexicon, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('.'))
    parser.add_argument('--condition', choices=tuple(condition_specs()), default='full-s42')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--tokens', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, lexicon, record = load_model(args.bundle, args.condition)
    result = generate(model, lexicon, args.prompt, seed=args.sampling_seed,
                      temperature=args.temperature, top_k=args.top_k, maximum_tokens=args.tokens)
    print(json.dumps(dict(model=record, sampling=dict(seed=args.sampling_seed,
        temperature=args.temperature, top_k=args.top_k, maximum_tokens=args.tokens,
        threads=1, engine='PyTorch CPU multinomial; temperature zero uses argmax'), **result),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
