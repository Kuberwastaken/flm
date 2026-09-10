"""Validation-only acute interventions and identical-input state responses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .language_train import construct, evaluate, restore
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache

STIMULUS = ('A small bird lives in a garden. Each morning it leaves the tree and returns to the same branch. '
            'The gardener watches from a window. Later, rain falls on the leaves and the bird shelters beneath them. '
            'When the sky clears, it crosses the garden again.')


@torch.no_grad()
def state_response(initial, trained, lexicon):
    tokens = [0] + lexicon.encode(STIMULUS); models = (initial, trained)
    constants = [model.constants() for model in models]; states = [model.initial_state(1) for model in models]
    records = []
    for position, token in enumerate(tokens):
        for i, model in enumerate(models):
            drive = model.input(model.embedding(torch.tensor([token])))
            states[i] = model.transition(drive, states[i], constants[i])
        record = dict(position=position, token=token, piece='<BOS>' if token == 0 else lexicon.pieces[token].decode('utf8', errors='replace'))
        for j, label in enumerate(('fast', 'slow')):
            record[label] = dict(initial_mean_absolute=float(states[0][j].abs().mean()),
                trained_mean_absolute=float(states[1][j].abs().mean()),
                paired_rms_distance=float((states[0][j] - states[1][j]).square().mean().sqrt()))
        records.append(record)
    return dict(stimulus=STIMULUS, tokens=len(tokens), records=records,
        interpretation='Identical teacher-forced input from zero state. State distance is not a measure of semantic knowledge or animal behavior.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=(42, 43), default=42)
    p.add_argument('--runs', type=Path, default=Path('runs/wikitext2'))
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    folder = a.runs / f'flm-s{a.seed}'; checkpoint = folder / 'best.pt'
    complete = json.loads((folder / 'complete.json').read_text(encoding='utf8'))
    if complete['steps'] != 6000 or sha256(checkpoint) != complete['best_checkpoint_sha256']:
        raise ValueError('A completed unchanged main FLM checkpoint is required')
    lexicon = Lexicon(Path('data/tokenizers/wikitext2-4096/tokenizer.json'))
    graph = Path('data/graphs/central-1024/graph.npz'); validation = Path('data/processed/wikitext2-bpe/validation.npz')
    if sha256(validation) != complete['protocol']['validation_cache_sha256']: raise ValueError('Validation cache changed')
    documents = read_cache(validation)
    model, saved = restore(checkpoint, graph, lexicon)
    if saved['_file_sha256'] != complete['best_checkpoint_sha256']: raise ValueError('Checkpoint changed during read')
    initial = construct('flm', graph, lexicon.vocabulary, a.seed)
    results = []
    for condition, candidate in [('untrained', initial), ('trained', model), ('recurrence_off', model), ('slow_state_off', model)]:
        candidate.config.variant = {'recurrence_off':'no_recurrence', 'slow_state_off':'no_slow'}.get(condition, 'flm')
        score = evaluate(candidate, documents, lexicon, complete['protocol']['eval_tokens'])
        if condition in ('untrained', 'trained'):
            reference = folder / ('initial-validation.json' if condition == 'untrained' else f'validation-{saved["step"]:06d}.json')
            expected = json.loads(reference.read_text(encoding='utf8'))
            if score['bytes'] != expected['bytes'] or abs(score['bits_per_byte'] - expected['bits_per_byte']) > 1e-5:
                raise ValueError(f'Reconstructed validation differs from the recorded run: {condition}')
        results.append(dict(condition=condition, score=score))
        print(json.dumps(dict(condition=condition, bits_per_byte=score['bits_per_byte'])), flush=True)
    model.config.variant = 'flm'; initial.eval(); model.eval()
    differences = []
    before = dict(initial.named_parameters())
    for name, parameter in model.named_parameters():
        differences.append(dict(name=name, parameters=parameter.numel(),
            initial_l2=float(before[name].detach().norm()), trained_l2=float(parameter.detach().norm()),
            change_l2=float((parameter.detach() - before[name].detach()).norm())))
    result = dict(training_seed=a.seed, checkpoint_step=saved['step'], checkpoint_sha256=saved['_file_sha256'],
        tokenizer_sha256=lexicon.sha256, graph_sha256=sha256(graph), validation_cache_sha256=sha256(validation),
        conditions=results, response=state_response(initial, model, lexicon), parameter_changes=differences,
        protocol='docs/MECHANISM-PROTOCOL.md',
        caution='Exploratory validation diagnostics after training. Acute removal without retraining is not a matched architectural ablation or a topology advantage test. No held-out test data is used.')
    write_json(Path(f'reports/wikitext2/mechanisms-s{a.seed}.json'), result)
    write_json(Path(f'public/research/mechanisms-s{a.seed}.json'), result)


if __name__ == '__main__': main()
