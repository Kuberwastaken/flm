"""Prepare and serially train the declared language topology/slow-state controls."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .language_bridge import compatibility, parameter_hash, sampling_audit
from .language_train import construct, restore
from .model import load_graph
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache
from .wiring_controls import GRAPH_SEEDS, rewire_graph, validate_control

PROTOCOL = Path('docs/LANGUAGE-TOPOLOGY-PROTOCOL.md')
STUDY = Path('runs/language-topology-v1')
REPORTS = Path('reports/language-topology')
MEASURED = Path('data/graphs/central-1024/graph.npz')
LEXICON = Path('data/tokenizers/wikitext2-4096/tokenizer.json')
SEEDS = (42, 43)
SOURCES = ('flm/language_topology.py', 'flm/language_bridge.py', 'flm/language_train.py',
    'flm/model.py', 'flm/train.py', 'flm/tokenizer.py', 'flm/corpus_cache.py',
    'flm/provenance.py', 'flm/wiring_controls.py')


def read_json(path): return json.loads(path.read_text(encoding='utf8'))


def conditions():
    records = []
    for seed in SEEDS:
        records.append(dict(label=f'measured-s{seed}', seed=seed, topology='measured', graph_seed=None,
            variant='flm', graph=MEASURED.as_posix(), output=f'runs/wikitext2/flm-s{seed}', reference=True))
        for graph_seed in GRAPH_SEEDS:
            label = f'null{graph_seed}-s{seed}'
            records.append(dict(label=label, seed=seed, topology='rewired', graph_seed=graph_seed,
                variant='flm', graph=f'data/graphs/central-1024-null{graph_seed}/graph.npz',
                output=(STUDY / label).as_posix(), reference=False))
        label = f'no-slow-s{seed}'
        records.append(dict(label=label, seed=seed, topology='measured', graph_seed=None,
            variant='no_slow', graph=MEASURED.as_posix(), output=(STUDY / label).as_posix(), reference=False))
    return records


def verify_source_identity(root, identity):
    if sha256(root / PROTOCOL) != identity['protocol_sha256']:
        raise ValueError('Language topology protocol changed')
    for relative, expected in identity['sources'].items():
        if sha256(root / relative) != expected: raise ValueError(f'Numerical source changed: {relative}')
    for relative, expected in identity['inputs'].items():
        if sha256(root / relative) != expected: raise ValueError(f'Study input changed: {relative}')
    if identity['torch'] != str(torch.__version__): raise ValueError('PyTorch version changed')


def build_graphs(root):
    source = root / MEASURED; original = load_graph(source)
    for seed in GRAPH_SEEDS:
        directory = source.parent.with_name(source.parent.name + f'-null{seed}')
        card_path = directory / 'graph-card.json'; graph_path = directory / 'graph.npz'
        if card_path.exists():
            card = read_json(card_path)
            expected = dict(seed=seed, source_graph_sha256=sha256(source),
                graph_sha256=sha256(graph_path), protocol_sha256=sha256(root / PROTOCOL),
                implementation_sha256=sha256(root / 'flm/wiring_controls.py'))
            if any(card.get(k) != v for k, v in expected.items()): raise ValueError(f'Graph identity changed: {directory}')
            validate_control(original, load_graph(graph_path)); continue
        if graph_path.exists(): raise ValueError(f'Graph exists without a verified card: {directory}')
        graph, report = rewire_graph(original, seed)
        directory.mkdir(parents=True, exist_ok=True); np.savez_compressed(graph_path, **graph)
        write_json(card_path, dict(name=f'Language topology control, graph seed {seed}',
            source_graph=MEASURED.as_posix(), source_graph_sha256=sha256(source), graph_sha256=sha256(graph_path),
            protocol_sha256=sha256(root / PROTOCOL), implementation_sha256=sha256(root / 'flm/wiring_controls.py'),
            graph_type='Artificial rewiring of 1024 retained MaleCNS nodes; not measured synapses',
            magnitude_meaning='Transferred incoming weights/contact counts at fixed postsynaptic slots, not measured new edges',
            orientation='row postsynaptic, column presynaptic', license='CC-BY-4.0', **report))
        print(json.dumps(dict(event='graph-prepared', graph_seed=seed, overlap=report['original_edge_overlap_fraction'])), flush=True)


def verify_saved(saved, condition, identity, graph):
    expected_protocol = dict(identity['training_protocol'], graph_sha256=sha256(graph))
    if saved['run']['protocol'] != expected_protocol or saved['run']['seed'] != condition['seed']:
        raise ValueError('Run protocol or training seed changed')
    if saved['config']['variant'] != condition['variant']: raise ValueError('Run mechanism changed')
    if saved['step'] % 500 or not 500 <= saved['step'] <= 6000: raise ValueError('Invalid saved step')
    audit = identity['sampling'][str(condition['seed'])][str(saved['step'])]
    if saved['sampler_rng'] != audit['sampler_rng'] or saved['run'].get('exposure') != audit['exposure']:
        raise ValueError('Sampled text or exposure differs from the matched reference')
    measured = load_graph(graph)
    for buffer, source in (('row', 'row'), ('col', 'col'), ('base_weight', 'weight'), ('pool_index', 'pool')):
        expected = torch.as_tensor(measured[source], dtype=saved['model'][buffer].dtype)
        if not torch.equal(saved['model'][buffer], expected): raise ValueError(f'Checkpoint graph buffer changed: {buffer}')


def verify_complete(root, condition, identity, lexicon):
    directory = root / condition['output']; graph = root / condition['graph']
    record = read_json(directory / 'complete.json')
    if record['steps'] != 6000 or record['protocol'] != dict(identity['training_protocol'], graph_sha256=sha256(graph)):
        raise ValueError(f'Incomplete or mismatched run: {condition["label"]}')
    selections = []
    for step in range(500, 6001, 500):
        score = read_json(directory / f'validation-{step:06d}.json')
        value = score['bits_per_byte']
        if not math.isfinite(value) or score['tokenizer_sha256'] != lexicon.sha256:
            raise ValueError('Invalid checkpoint selection score')
        selections.append((value, step))
    value, step = min(selections)
    model, saved = restore(directory / 'best.pt', graph, lexicon)
    verify_saved(saved, condition, identity, graph)
    if saved['step'] != step or record['best_validation_bpb'] != value or record['best_checkpoint_sha256'] != saved['_file_sha256']:
        raise ValueError('Checkpoint is not the declared validation selection')
    _, last = restore(directory / 'last.pt', graph, lexicon)
    verify_saved(last, condition, identity, graph)
    if last['step'] != 6000: raise ValueError('Final checkpoint missing')
    if condition['reference']:
        frozen = identity['reference_checkpoints'][condition['label']]
        if saved['_file_sha256'] != frozen: raise ValueError('Previously frozen reference changed')
    return dict(**condition, checkpoint=(Path(condition['output']) / 'best.pt').as_posix(),
        checkpoint_sha256=saved['_file_sha256'], checkpoint_step=step,
        selection_validation_bpb=value, parameters=model.parameter_card()['trainable_parameters'])


def prepare(root):
    path = root / REPORTS / 'identity.json'
    if path.exists():
        identity = read_json(path); verify_source_identity(root, identity); return identity
    build_graphs(root)
    frozen = read_json(root / 'reports/wikitext2/selection.json')
    protocol = frozen['protocol']; lexicon = Lexicon(root / LEXICON)
    if protocol['steps'] != 6000 or protocol['threads'] != 4: raise ValueError('Reference budget changed')
    paths = [PROTOCOL, Path('docs/WIKITEXT-PROTOCOL.md'), MEASURED, LEXICON,
        Path('data/processed/wikitext2-bpe/train.npz'), Path('data/processed/wikitext2-bpe/validation.npz'),
        Path('reports/wikitext2/selection.json'), Path('data/tokenizers/wikitext2-4096/tokenization-card.json')]
    bridges = []; sampling = {}; initial = {}; refs = {}
    documents = read_cache(root / 'data/processed/wikitext2-bpe/train.npz')
    for seed in SEEDS:
        ref = next(r for r in frozen['runs'] if r['variant'] == 'flm' and r['seed'] == seed)
        run = read_json(root / f'runs/wikitext2/flm-s{seed}/run.json')
        bridge = compatibility(root, run['source_commit'], seed); bridges.append(bridge)
        sampling[str(seed)] = sampling_audit(documents, lexicon, seed)
        initial[str(seed)] = bridge['initial_parameter_sha256']; refs[f'measured-s{seed}'] = ref['checkpoint_sha256']
        print(json.dumps(dict(event='reference-compatible', seed=seed, exact_updates=10)), flush=True)
    for condition in conditions():
        graph = root / condition['graph']
        model = construct(condition['variant'], graph, lexicon.vocabulary, condition['seed'])
        if parameter_hash(model) != initial[str(condition['seed'])]: raise ValueError('Conditions do not share initial parameters')
        if model.parameter_card()['trainable_parameters'] != 600003: raise ValueError('Condition parameter count changed')
        if not condition['reference'] and condition['topology'] == 'rewired':
            paths += [Path(condition['graph']), Path(condition['graph']).with_name('graph-card.json')]
    identity = dict(study='WikiText language topology and retrained slow-state controls v1',
        protocol_sha256=sha256(root / PROTOCOL), training_protocol=protocol,
        sources={p: sha256(root / p) for p in SOURCES}, inputs={p.as_posix(): sha256(root / p) for p in paths},
        torch=str(torch.__version__), conditions=conditions(), reference_checkpoints=refs,
        compatibility=bridges, initial_parameter_sha256=initial, sampling=sampling,
        test_status='New control test losses prohibited until all eight new runs and ten selections are complete')
    verify_source_identity(root, identity)
    for key, relative in [('tokenizer_sha256', LEXICON), ('graph_sha256', MEASURED),
        ('train_cache_sha256', Path('data/processed/wikitext2-bpe/train.npz')),
        ('validation_cache_sha256', Path('data/processed/wikitext2-bpe/validation.npz'))]:
        if protocol[key] != sha256(root / relative): raise ValueError('Reference data changed')
    for condition in conditions():
        if condition['reference']: verify_complete(root, condition, identity, lexicon)
    write_json(path, identity); return identity


def run(root, identity):
    lexicon = Lexicon(root / LEXICON)
    for condition in conditions():
        verify_source_identity(root, identity)
        output = root / condition['output']; graph = root / condition['graph']
        if (output / 'complete.json').exists():
            verify_complete(root, condition, identity, lexicon)
            print(f'Already verified: {condition["label"]}', flush=True); continue
        if condition['reference']: raise ValueError('An existing reference is incomplete')
        command = [sys.executable, '-X', 'utf8', '-m', 'flm.language_train', '--variant', condition['variant'],
            '--graph', condition['graph'], '--seed', str(condition['seed']), '--steps', '6000',
            '--threads', '4', '--output', condition['output']]
        last = output / 'last.pt'
        if last.exists():
            _, saved = restore(last, graph, lexicon); verify_saved(saved, condition, identity, graph)
            command += ['--resume', str(last)]
        print(f'Starting {condition["label"]}', flush=True)
        subprocess.run(command, check=True, cwd=root)
        verify_source_identity(root, identity); verify_complete(root, condition, identity, lexicon)
    print('All language topology controls completed; no test scoring was performed by this trainer.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('graphs', 'prepare', 'train'))
    args = parser.parse_args(); root = Path('.').resolve()
    if args.action == 'graphs': build_graphs(root); return
    identity = prepare(root)
    if args.action == 'train': run(root, identity)


if __name__ == '__main__': main()
