"""Declare, verify and serially fit the measured-graph language computation controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import torch

from .language_bridge import parameter_hash, sampling_audit
from .language_core_controls import CONTROLS, construct
from .language_core_train import (checkpoint_records, optimizer_for, optimizer_names,
                                  restore, restore_optimizer, update, verify_exposure)
from .language_train import construct as original_construct, evaluate
from .language_topology import (conditions as topology_conditions, verify_complete as topology_complete,
                                verify_source_identity as verify_topology_identity)
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache
from .train import Sampler


PROTOCOL = Path('docs/LANGUAGE-CORE-PROTOCOL.md')
REPORTS = Path('reports/language-core')
STUDY = Path('runs/language-core-v1')
GRAPH = Path('data/graphs/central-1024/graph.npz')
LEXICON = Path('data/tokenizers/wikitext2-4096/tokenizer.json')
SEEDS = (42, 43)
SOURCES = (
    'flm/language_core_controls.py', 'flm/language_core_train.py', 'flm/language_core_study.py',
    'flm/language_core_test.py', 'flm/language_train.py', 'flm/language_bridge.py',
    'flm/language_topology.py', 'flm/language_topology_test.py', 'flm/language_test.py',
    'flm/model.py', 'flm/train.py', 'flm/baselines.py', 'flm/graph.py',
    'flm/tokenizer.py', 'flm/corpus_cache.py', 'flm/provenance.py', 'flm/wiring_controls.py',
    'flm/ngram.py', 'flm/study_index.py')


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def conditions():
    return [dict(label=f'{control}-s{seed}', control=control, seed=seed,
                 reference=control == 'full', graph=GRAPH.as_posix(),
                 output=f'runs/wikitext2/flm-s{seed}' if control == 'full' else (STUDY / f'{control}-s{seed}').as_posix())
            for seed in SEEDS for control in CONTROLS]


def verify_identity(root, identity):
    if identity['conditions'] != conditions() or identity['protocol_sha256'] != sha256(root / PROTOCOL):
        raise ValueError('Language computation study declaration changed')
    for group in ('sources', 'inputs'):
        for relative, expected in identity[group].items():
            if sha256(root / relative) != expected:
                raise ValueError(f'Frozen {group} changed: {relative}')
    if identity['torch'] != str(torch.__version__):
        raise ValueError('Frozen PyTorch version changed')
    previous = read_json(root / 'reports/language-topology/identity.json')
    verify_topology_identity(root, previous)
    if identity['training_protocol'] != previous['training_protocol'] or set(identity['sources']) != set(SOURCES):
        raise ValueError('Historical training protocol or source inventory changed')


def run_binding(root, condition, identity):
    return dict(study_identity_sha256=sha256(root / REPORTS / 'identity.json'),
                label=condition['label'], control=condition['control'], seed=condition['seed'],
                protocol=identity['training_protocol'], graph_sha256=sha256(root / GRAPH),
                tokenizer_sha256=identity['training_protocol']['tokenizer_sha256'],
                initial_parameter_sha256=identity['initial_parameter_sha256'][str(condition['seed'])])


def reference_record(root, condition, lexicon):
    previous = read_json(root / 'reports/language-topology/identity.json')
    original = next(row for row in topology_conditions() if row['reference'] and row['seed'] == condition['seed'])
    checked = topology_complete(root, original, previous, lexicon)
    return dict(**condition, checkpoint=checked['checkpoint'], checkpoint_sha256=checked['checkpoint_sha256'],
                checkpoint_step=checked['checkpoint_step'], selection_validation_bpb=checked['selection_validation_bpb'],
                allocated_parameters=checked['parameters'], trainable_parameters=checked['parameters'], frozen_parameters=0)


def verify_complete(root, condition, identity, lexicon):
    if condition['reference']:
        selected = reference_record(root, condition, lexicon)
        if selected['checkpoint_sha256'] != identity['reference_checkpoints'][condition['label']]:
            raise ValueError('Frozen full-model reference changed')
        return selected
    directory = root / condition['output']
    binding = run_binding(root, condition, identity)
    records = checkpoint_records(directory, binding)
    if len(records) != 12 or records[-1]['step'] != 6000:
        raise ValueError('Language computation run is incomplete: ' + condition['label'])
    best = float('inf')
    # Verify every saved fixed core and every sampler/optimizer, not just best/last.
    for record in records:
        model, saved = restore(directory / record['checkpoint'], root / GRAPH, lexicon, binding)
        verify_exposure(saved, identity)
        restore_optimizer(model, saved, identity['training_protocol'])
        best = min(best, record['validation_bpb'])
        if saved['step'] != record['step'] or saved['best'] != best:
            raise ValueError('Saved step or cumulative validation selection changed')
        score = read_json(directory / record['validation'])
        if score['tokenizer_sha256'] != lexicon.sha256 or score['tokens'] != identity['validation_denominators']['tokens']:
            raise ValueError('Validation tokenizer or token coverage changed')
        if score['bytes'] != identity['validation_denominators']['bytes']:
            raise ValueError('Validation byte coverage changed')
        coverage = [{key: row[key] for key in ('document', 'tokens', 'bytes')} for row in score['documents']]
        if coverage != identity['validation_denominators']['documents']:
            raise ValueError('Validation article coverage changed')
        for row in score['documents']:
            if not math.isfinite(row['nll']) or row['nll'] < 0 or not math.isclose(
                    row['bits_per_byte'], row['nll'] / max(1, row['bytes']) / math.log(2), rel_tol=1e-10):
                raise ValueError('Invalid validation article likelihood')
        if not math.isclose(score['nll'], sum(row['nll'] for row in score['documents']), rel_tol=1e-10):
            raise ValueError('Validation aggregate likelihood changed')
        if not math.isclose(score['bits_per_byte'], score['nll'] / score['bytes'] / math.log(2), rel_tol=1e-10):
            raise ValueError('Validation aggregate BPB changed')
        if not math.isclose(score['token_perplexity'], math.exp(score['nll'] / score['tokens']), rel_tol=1e-10):
            raise ValueError('Validation perplexity changed')
    selected = min(records, key=lambda row: (row['validation_bpb'], row['step']))
    expected = dict(binding=binding, steps=6000, selected_step=selected['step'],
                    best_validation_bpb=selected['validation_bpb'], checkpoint=selected['checkpoint'],
                    checkpoint_sha256=selected['checkpoint_sha256'], last_checkpoint_sha256=records[-1]['checkpoint_sha256'])
    if read_json(directory / 'complete.json') != expected:
        raise ValueError('Completion record differs from committed checkpoint selection')
    card = model.parameter_card()
    return dict(**condition, checkpoint=(Path(condition['output']) / selected['checkpoint']).as_posix(),
                checkpoint_sha256=selected['checkpoint_sha256'], checkpoint_step=selected['step'],
                selection_validation_bpb=selected['validation_bpb'],
                **{key: card[key] for key in ('allocated_parameters', 'trainable_parameters', 'frozen_parameters')})


def compatibility(root, seed, protocol, lexicon, documents, validation):
    """Ten full-model updates only; the six new control fits start after freezing."""
    reference = original_construct('flm', root / GRAPH, lexicon.vocabulary, seed)
    reference_rng = torch.get_rng_state()
    initial = parameter_hash(reference)
    cards = {}
    for control in CONTROLS:
        model = construct(control, root / GRAPH, lexicon.vocabulary, seed)
        if parameter_hash(model) != initial or not torch.equal(torch.get_rng_state(), reference_rng):
            raise ValueError('Control initialization or construction RNG differs')
        card = model.parameter_card()
        if card['allocated_parameters'] != 600003 or card['frozen_parameters'] != (78179 if control == 'fixed_dynamics' else 0):
            raise ValueError('Declared control parameter inventory differs')
        probe = model.gradient_probe(model(torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]]))[0].square().mean())
        disconnected = {name for name, row in probe['groups'].items() if not row['connected']}
        if disconnected != ({'edge_log_gain', 'recurrent_logit'} if control in ('no_lateral', 'no_temporal_state') else set()):
            raise ValueError('Declared gradient connectivity differs')
        if control == 'no_temporal_state':
            tokens = torch.tensor([[0, 2, 3, 7], [5, 6, 8, 7]])
            changed = tokens.clone()
            changed[:, :-1] = tokens.flip(0)[:, :-1]
            with torch.no_grad():
                if not torch.equal(model(tokens)[0][:, -1], model(changed)[0][:, -1]):
                    raise ValueError('Memoryless control retains prefix dependence')
        cards[control] = dict(parameter_card=card, optimizer_parameter_names=optimizer_names(model),
                              gradient_probe=probe)
    replays = []
    try:
        for threads in (1, protocol['threads']):
            torch.set_num_threads(threads)
            reference = original_construct('flm', root / GRAPH, lexicon.vocabulary, seed)
            model = construct('full', root / GRAPH, lexicon.vocabulary, seed)
            samplers = [Sampler(documents, seed, protocol['sequence']) for _ in range(2)]
            optimizers = [torch.optim.AdamW(reference.parameters(), lr=.002, weight_decay=.01), optimizer_for(model, protocol)]
            losses = []
            maximum_parameter_difference = maximum_loss_difference = 0.
            for step in range(1, 11):
                batches = [sampler.sample(protocol['batch'], 'cpu') for sampler in samplers]
                if any(not torch.equal(a, b) for a, b in zip(*batches)):
                    raise ValueError('Reference sampled windows differ')
                x, y = batches[0]
                # Keep the completed trainer's exact expression, including float
                # operation order; do not simplify its warmup algebra.
                progress = max(0., (step - 100) / max(1, 6000 - 100))
                optimizers[0].param_groups[0]['lr'] = (.0002 + .0018 * .5 *
                    (1 + math.cos(math.pi * min(progress, 1.)))) * min(step / 100, 1.)
                optimizers[0].zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(reference(x)[0][:, 16:].reshape(-1, lexicon.vocabulary), y[:, 16:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                optimizers[0].step()
                actual, _ = update(model, optimizers[1], *batches[1], protocol, step)
                loss_difference = abs(float(loss.detach()) - actual)
                differences = [float((value - model.get_parameter(name)).detach().abs().max())
                               for name, value in reference.named_parameters()]
                if not math.isfinite(loss_difference) or any(not math.isfinite(value) for value in differences):
                    raise ValueError('Nonfinite full-model replay difference')
                maximum_parameter_difference = max(maximum_parameter_difference, *differences)
                maximum_loss_difference = max(maximum_loss_difference, loss_difference)
                if threads == 1:
                    if loss_difference or parameter_hash(reference) != parameter_hash(model):
                        raise ValueError(f'Single-thread full-model replay differs at update {step}')
                elif max(differences) > 1e-7 or loss_difference > 1e-6:
                    raise ValueError(f'Four-thread full-model replay exceeds numerical bounds at update {step}')
                losses.append(actual)
            scores = [evaluate(item, validation[:4], lexicon, token_limit=128) for item in (reference, model)]
            for key in ('bytes', 'tokens', 'tokenizer_sha256'):
                if scores[0][key] != scores[1][key]:
                    raise ValueError('Full-model evaluation identity differs: ' + key)
            coverage = [[{key: row[key] for key in ('document', 'tokens', 'bytes')} for row in score['documents']] for score in scores]
            if coverage[0] != coverage[1]:
                raise ValueError('Full-model evaluation article coverage differs')
            nll_difference = max([abs(scores[0]['nll'] - scores[1]['nll'])] +
                                 [abs(a['nll'] - b['nll']) for a, b in zip(scores[0]['documents'], scores[1]['documents'])])
            if not math.isfinite(nll_difference) or nll_difference > (0. if threads == 1 else 1e-4):
                raise ValueError('Full-model validation replay exceeds numerical bounds')
            if samplers[0].rng.bit_generator.state != samplers[1].rng.bit_generator.state:
                raise ValueError('Full-model sampler RNG replay differs')
            replays.append(dict(threads=threads, updates=10, exact_sampler=True, losses=losses,
                maximum_parameter_absolute_difference=maximum_parameter_difference,
                maximum_loss_absolute_difference=maximum_loss_difference,
                maximum_validation_nll_absolute_difference=nll_difference,
                exact_parameter_match=parameter_hash(reference) == parameter_hash(model),
                final_parameter_sha256=parameter_hash(model),
                bounds=dict(parameter_absolute=0. if threads == 1 else 1e-7,
                            loss_absolute=0. if threads == 1 else 1e-6,
                            validation_nll_absolute=0. if threads == 1 else 1e-4)))
    finally:
        torch.set_num_threads(protocol['threads'])
    return dict(seed=seed, initial_parameter_sha256=initial, controls=cards, replays=replays,
                interpretation='Single-thread exact numerical replay; four-thread bounded numerical replay. Training retains the historical four-thread protocol.')


def prepare(root):
    path = root / REPORTS / 'identity.json'
    if path.exists():
        identity = read_json(path)
        verify_identity(root, identity)
        return identity
    # Freeze only committed numerical sources. Unrelated presentation edits may coexist.
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for name in (*SOURCES, PROTOCOL.as_posix()):
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root) != (root / name).read_bytes():
            raise ValueError('Commit the experiment source before freezing: ' + name)
    previous = read_json(root / 'reports/language-topology/identity.json')
    verify_topology_identity(root, previous)
    final = read_json(root / 'reports/language-topology/final-release.json')
    if not final['final_test_result_published'] or final['workflow']['conclusion'] != 'success':
        raise ValueError('Finish the public topology release before starting this study')
    protocol = previous['training_protocol']
    torch.set_num_threads(protocol['threads'])
    lexicon = Lexicon(root / LEXICON)
    documents = read_cache(root / 'data/processed/wikitext2-bpe/train.npz')
    validation = read_cache(root / 'data/processed/wikitext2-bpe/validation.npz')
    bridges = []
    sampling = {}
    for seed in SEEDS:
        bridge = compatibility(root, seed, protocol, lexicon, documents, validation)
        if bridge['initial_parameter_sha256'] != previous['initial_parameter_sha256'][str(seed)]:
            raise ValueError('Historical reference initialization differs')
        bridges.append(bridge)
        sampling[str(seed)] = sampling_audit(documents, lexicon, seed)
        if sampling[str(seed)] != previous['sampling'][str(seed)]:
            raise ValueError('Historical matched token stream differs')
        print(json.dumps(dict(event='reference-compatible', seed=seed, replay_updates=10,
                              exact_replay_threads=1, bounded_replay_threads=protocol['threads'])), flush=True)
    references = {row['label']: reference_record(root, row, lexicon)['checkpoint_sha256']
                  for row in conditions() if row['reference']}
    sample = read_json(root / 'runs/wikitext2/flm-s42/validation-006000.json')
    denominators = {key: sample[key] for key in ('tokens', 'bytes')}
    denominators['documents'] = [{key: row[key] for key in ('document', 'tokens', 'bytes')} for row in sample['documents']]
    inputs = [PROTOCOL, GRAPH, LEXICON, Path('data/graphs/central-1024/graph-card.json'),
              Path('data/tokenizers/wikitext2-4096/tokenization-card.json'),
              *[Path('data/processed/wikitext2-bpe') / (split + '.npz') for split in ('train', 'validation', 'test')],
              *[Path('reports/language-topology') / (name + '.json') for name in ('identity', 'selection', 'summary', 'final-release')],
              *[REPORTS / (name + '.json') for name in ('software-preflight', 'replay-diagnostic')],
              *[Path(f'reports/wikitext2/test-flm-s{seed}.json') for seed in SEEDS]]
    identity = dict(study='Measured-graph language computation controls v1', declared_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit=commit, protocol_sha256=sha256(root / PROTOCOL), training_protocol=protocol,
                    sources={name: sha256(root / name) for name in SOURCES},
                    inputs={name.as_posix(): sha256(root / name) for name in inputs}, torch=str(torch.__version__),
                    conditions=conditions(), reference_checkpoints=references, compatibility=bridges,
                    initial_parameter_sha256={str(row['seed']): row['initial_parameter_sha256'] for row in bridges},
                    sampling=sampling, validation_denominators=denominators,
                    test_gate='All six new runs and eight selections must be complete before decoding new-control test scores',
                    prior_test_access='WikiText baseline and topology test results already inspected; exploratory extension')
    verify_identity(root, identity)
    write_json(path, identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train'))
    args = parser.parse_args()
    root = Path('.').resolve()
    if args.action == 'prepare':
        prepare(root)
        return
    identity = read_json(root / REPORTS / 'identity.json')
    from .language_core_train import train
    for condition in conditions():
        verify_identity(root, identity)
        if condition['reference'] or (root / condition['output'] / 'complete.json').exists():
            verify_complete(root, condition, identity, Lexicon(root / LEXICON))
            print('Already verified: ' + condition['label'], flush=True)
        else:
            print('Starting ' + condition['label'], flush=True)
            train(root, condition, identity)
    print('All computation controls complete. No new test scores were decoded by this trainer.', flush=True)


if __name__ == '__main__':
    main()
