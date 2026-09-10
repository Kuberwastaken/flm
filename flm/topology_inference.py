"""Run the complete released language-topology selection with its exact graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .inference import RUNTIME_FILES as COMPARISON_RUNTIME, bundle_path, generate, state_hash
from .language_train import restore
from .model import load_graph
from .provenance import sha256
from .tokenizer import Lexicon


RUNTIME_FILES = (*COMPARISON_RUNTIME, 'flm/topology_inference.py')


def condition_specs():
    result = {}
    for seed in (42, 43):
        result[f'measured-s{seed}'] = dict(seed=seed, graph_seed=None, variant='flm', reference=True, topology='measured')
        for graph_seed in (101, 103, 107):
            result[f'null{graph_seed}-s{seed}'] = dict(seed=seed, graph_seed=graph_seed, variant='flm', reference=False, topology='rewired')
        result[f'no-slow-s{seed}'] = dict(seed=seed, graph_seed=None, variant='no_slow', reference=False, topology='measured')
    return result


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'bundle.json').read_text(encoding='utf8'))
    if manifest.get('format') != 'flm-language-topology-inference-v1':
        raise ValueError('Unsupported language topology inference bundle')
    if manifest.get('runtime_sources') != list(RUNTIME_FILES):
        raise ValueError('Topology inference runtime inventory changed')
    for name, record in manifest['files'].items():
        path = bundle_path(root, name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Bundle file failed integrity verification: ' + name)
    runtime_root = Path(__file__).resolve().parents[1]
    for name in RUNTIME_FILES:
        if name not in manifest['files'] or sha256(runtime_root/name) != manifest['files'][name]['sha256']:
            raise ValueError('Run with the matching bundled topology runtime: ' + name)
    for name in (manifest['study_identity'], manifest['selection'], manifest['tokenizer']):
        if name not in manifest['files']:
            raise ValueError('Required study input is absent from the manifest')
    identity_path = bundle_path(root, manifest['study_identity'])
    selection_path = bundle_path(root, manifest['selection'])
    if sha256(identity_path) != manifest['study_identity_sha256'] or sha256(selection_path) != manifest['selection_sha256']:
        raise ValueError('Study or selection identity changed')
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    if selection['study_identity_sha256'] != manifest['study_identity_sha256']:
        raise ValueError('Selection belongs to a different study')
    for name in RUNTIME_FILES:
        if name in identity['sources'] and manifest['files'][name]['sha256'] != identity['sources'][name]:
            raise ValueError('A frozen numerical source changed: ' + name)
    if (manifest['files'][manifest['tokenizer']]['sha256'] != selection['tokenizer_sha256']
            or selection['tokenizer_sha256'] != identity['inputs'].get(manifest['tokenizer'])):
        raise ValueError('Selected tokenizer changed')
    expected = condition_specs()
    inventories = {}
    for field, rows, key in [('declared', identity['conditions'], 'label'),
                              ('selected', selection['runs'], 'label'), ('models', manifest['models'], 'id')]:
        lookup = {row[key]: row for row in rows}
        if len(rows) != 10 or set(lookup) != set(expected):
            raise ValueError('The complete ten-condition topology selection is required: ' + field)
        inventories[field] = lookup
    for name, record in inventories['models'].items():
        selected = inventories['selected'][name]; declared = inventories['declared'][name]
        if any(declared.get(k) != value for k, value in expected[name].items()):
            raise ValueError('Unexpected declared condition: ' + name)
        if any(selected.get(k) != value for k, value in declared.items()):
            raise ValueError('Selection changed its declared condition: ' + name)
        binding = dict(training_seed=selected['seed'], variant=selected['variant'],
            graph_seed=selected['graph_seed'], topology=selected['topology'], reference=selected['reference'],
            graph=selected['graph'], source_checkpoint_sha256=selected['checkpoint_sha256'],
            checkpoint_step=selected['checkpoint_step'], parameters=selected['parameters'])
        if any(record.get(k) != value for k, value in binding.items()):
            raise ValueError('Released model differs from its selected condition: ' + name)
        for path in (record['file'], record['graph']):
            if path not in manifest['files']:
                raise ValueError('A required model or graph is absent from the manifest')
        graph_hash = manifest['files'][record['graph']]['sha256']
        if graph_hash != identity['inputs'].get(record['graph']) or record.get('graph_sha256') != graph_hash:
            raise ValueError('Selected anatomical/null graph changed: ' + name)
        if record['reference'] and identity['reference_checkpoints'].get(name) != record['source_checkpoint_sha256']:
            raise ValueError('The original measured reference changed: ' + name)
    return manifest


def load_model(root, name):
    root = Path(root).resolve(); manifest = verify_bundle(root)
    record = next((row for row in manifest['models'] if row['id'] == name), None)
    if record is None:
        raise ValueError('Unknown language topology condition')
    lexicon = Lexicon(bundle_path(root, manifest['tokenizer']))
    graph_path = bundle_path(root, record['graph'])
    model, saved = restore(bundle_path(root, record['file']), graph_path, lexicon)
    expected_keys = {'format', 'model', 'config', 'step', 'run', 'source_checkpoint_sha256', 'state_sha256', '_file_sha256'}
    if (saved.get('format') != 'flm-inference-checkpoint-v1' or set(saved) != expected_keys
            or set(saved['run']) != {'seed', 'graph_sha256', 'tokenizer_sha256'}):
        raise ValueError('Expected a model-only inference checkpoint')
    if (saved['source_checkpoint_sha256'] != record['source_checkpoint_sha256']
            or saved['step'] != record['checkpoint_step'] or saved['run']['seed'] != record['training_seed']
            or model.config.variant != record['variant'] or saved['state_sha256'] != record['state_sha256']
            or state_hash(model) != record['state_sha256']
            or model.parameter_card()['trainable_parameters'] != record['parameters']):
        raise ValueError('Inference tensors or mechanism differ from the selected model')
    graph = load_graph(graph_path)
    for target, source in [('row', 'row'), ('col', 'col'), ('base_weight', 'weight'), ('pool_index', 'pool')]:
        actual = getattr(model, target)
        if not torch.equal(actual, torch.as_tensor(graph[source], dtype=actual.dtype)):
            raise ValueError('Checkpoint graph buffer differs from its selected graph: ' + target)
    sizes = torch.bincount(torch.as_tensor(graph['pool'], dtype=torch.long), minlength=model.config.pools).clamp_min(1).float()
    if not torch.equal(model.pool_sizes, sizes):
        raise ValueError('Checkpoint pooling sizes differ from the selected graph')
    model.eval()
    return model, lexicon, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('.'))
    parser.add_argument('--condition', choices=tuple(condition_specs()), default='measured-s42')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--tokens', type=int, default=128)
    args = parser.parse_args(); torch.set_num_threads(1)
    model, lexicon, record = load_model(args.bundle, args.condition)
    result = generate(model, lexicon, args.prompt, seed=args.sampling_seed,
                      temperature=args.temperature, top_k=args.top_k, maximum_tokens=args.tokens)
    print(json.dumps(dict(model=record, sampling=dict(seed=args.sampling_seed, temperature=args.temperature,
        top_k=args.top_k, maximum_tokens=args.tokens, threads=1,
        engine='PyTorch CPU multinomial; temperature zero uses argmax'), **result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
