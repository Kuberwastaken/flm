"""Consolidate immutable validation artifacts and sample matched checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .language_train import restore
from .provenance import sha256, write_json
from .tokenizer import Lexicon

PROMPTS = [
    'The history of science',
    'In the summer, the village',
    'The small animal moved through',
    'Language is a way to',
]
SAMPLING = dict(seed=17, temperature=.8, top_k=40, maximum_tokens=128, engine='PyTorch CPU multinomial')


@torch.no_grad()
def sample(model, lexicon, prompt):
    generator = torch.Generator().manual_seed(SAMPLING['seed']); model.eval()
    logits, state = model(torch.tensor([[0] + lexicon.encode(prompt)])); generated = []
    allowed = torch.tensor([False, True] + [all(b in (9, 10) or b >= 32 for b in piece) for piece in lexicon.pieces[2:]])
    for _ in range(SAMPLING['maximum_tokens']):
        scores = logits[0, -1].clone(); scores[~allowed] = -torch.inf
        values, indices = scores.topk(SAMPLING['top_k'])
        chosen = int(indices[torch.multinomial(torch.softmax(values / SAMPLING['temperature'], dim=0), 1, generator=generator)])
        if chosen == 1: break
        generated.append(chosen); logits, state = model(torch.tensor([[chosen]]), state)
    payload = b''.join(lexicon.pieces[t] for t in generated)
    return dict(prompt=prompt, continuation=payload.decode('utf8', errors='replace'), tokens=generated, bytes=len(payload))


def collect(root):
    records = []
    for folder in sorted(root.glob('*-s*')):
        if not (folder / 'run.json').exists(): continue
        run = json.loads((folder / 'run.json').read_text(encoding='utf8')); points = []
        paths = [folder / 'initial-validation.json'] + sorted(folder.glob('validation-*.json'))
        for path in paths:
            if not path.exists(): continue
            data = json.loads(path.read_text(encoding='utf8'))
            step = 0 if path.name.startswith('initial') else int(path.stem.split('-')[-1])
            points.append(dict(step=step, presented_tokens=step * 1536, bits_per_byte=data['bits_per_byte'],
                token_perplexity=data['token_perplexity'], evaluated_tokens=data['tokens'], evaluated_bytes=data['bytes'], artifact_sha256=sha256(path)))
        if not points: continue
        records.append(dict(id=folder.name, variant=run['parameter_card']['config']['variant'], seed=run['seed'],
            parameters=run['parameter_card']['trainable_parameters'], protocol=run['protocol'],
            complete=(folder / 'complete.json').exists(), validation=points, source_commit=run['source_commit']))
    return dict(dataset='WikiText-2 raw', status='validation progress; test results not included',
        measurement='Fixed article prefixes; lower bits per UTF-8 byte is better. Compare models at the same update count and seed.',
        runs=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('runs/wikitext2'))
    p.add_argument('--output', type=Path, default=Path('public/research'))
    p.add_argument('--sample-step', type=int)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); write_json(a.output / 'validation.json', collect(a.runs))
    if a.sample_step is not None:
        torch.set_num_threads(1)
        lexicon = Lexicon(Path('data/tokenizers/wikitext2-4096/tokenizer.json')); samples = []
        for variant in ('flm', 'gru', 'transformer'):
            checkpoint = a.runs / f'{variant}-s{a.seed}' / f'checkpoint-{a.sample_step:06d}.pt'
            model, saved = restore(checkpoint, Path('data/graphs/central-1024/graph.npz'), lexicon)
            samples.append(dict(variant=variant, training_seed=a.seed, step=saved['step'], checkpoint_sha256=saved['_file_sha256'],
                parameters=model.parameter_card()['trainable_parameters'], passages=[sample(model, lexicon, prompt) for prompt in PROMPTS]))
        write_json(a.output / f'samples-{a.sample_step:06d}-s{a.seed}.json', dict(settings=SAMPLING, tokenizer_sha256=lexicon.sha256,
            note='Original prompts fixed in source. Unedited model continuations, including failures. These are not factual answers or held-out scores.', models=samples))
    write_json(a.output / 'samples-index.json', [dict(file=path.name,
        step=int(path.stem.split('-')[1]), seed=int(path.stem.split('-')[2][1:]))
        for path in sorted(a.output.glob('samples-*-s*.json'))])
    print('Published validation progress' + (' and matched samples.' if a.sample_step else '.'))


if __name__ == '__main__': main()
