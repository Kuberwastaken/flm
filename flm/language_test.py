"""Evaluate the untouched test split after every registered main run has finished."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .language_train import evaluate, restore
from .ngram import NGram
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache
from .train import read_documents

VARIANTS = ('flm', 'gru', 'transformer')
SEEDS = (42, 43)


def freeze_selection(root):
    """Fail before reading test data unless all six fixed-budget runs are complete."""
    selection, protocols = [], []
    for seed in SEEDS:
        for variant in VARIANTS:
            folder = root / f'{variant}-s{seed}'
            complete = folder / 'complete.json'; checkpoint = folder / 'best.pt'
            if not complete.exists() or not checkpoint.exists(): raise ValueError(f'Registered run incomplete: {folder}')
            record = json.loads(complete.read_text(encoding='utf8'))
            if record['steps'] != 6000 or record['protocol']['steps'] != 6000:
                raise ValueError(f'Unexpected run budget: {folder}')
            if record['best_checkpoint_sha256'] != sha256(checkpoint): raise ValueError(f'Checkpoint changed: {folder}')
            protocols.append(record['protocol'])
            selection.append(dict(variant=variant, seed=seed, checkpoint=str(checkpoint),
                checkpoint_sha256=record['best_checkpoint_sha256'], selection_validation_bpb=record['best_validation_bpb']))
    if any(p != protocols[0] for p in protocols): raise ValueError('Registered runs have inconsistent protocols')
    return dict(protocol=protocols[0], runs=selection, selection='Lowest fixed-prefix validation bits/byte during the registered 6000-update run')


def paired_interval(first, second, replicates=10000, seed=31415):
    a = {x['document']: x for x in first}; b = {x['document']: x for x in second}
    if not a or len(a) != len(first) or len(b) != len(second) or set(a) != set(b):
        raise ValueError('Paired scores must contain the same unique articles')
    ids = sorted(a)
    if any(a[i]['bytes'] != b[i]['bytes'] or a[i]['bytes'] <= 0 for i in ids): raise ValueError('Paired article byte counts differ')
    delta = np.array([a[i]['nll'] - b[i]['nll'] for i in ids]); sizes = np.array([a[i]['bytes'] for i in ids])
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(ids), (replicates, len(ids)))
    values = delta[draws].sum(axis=1) / sizes[draws].sum(axis=1) / math.log(2)
    return dict(difference_bpb=float(delta.sum() / sizes.sum() / math.log(2)),
        lower_95=float(np.quantile(values, .025)), upper_95=float(np.quantile(values, .975)),
        replicates=replicates, seed=seed, unit='paired article resampling',
        interpretation='Positive means the first model has worse loss; interval is conditional on these trained checkpoints')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('runs/wikitext2'))
    p.add_argument('--output', type=Path, default=Path('reports/wikitext2'))
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); selection = freeze_selection(a.runs)
    frozen = a.output / 'selection.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf8')) != selection:
        raise ValueError('A different test selection is already recorded; use a separately declared experiment')
    write_json(frozen, selection)
    torch.set_num_threads(a.threads)
    lexicon = Lexicon(Path('data/tokenizers/wikitext2-4096/tokenizer.json'))
    data = Path('data/processed/wikitext2-bpe/test.npz'); data_hash = sha256(data)
    documents = read_cache(data); results = []
    for selected in selection['runs']:
        path = a.output / f'test-{selected["variant"]}-s{selected["seed"]}.json'
        if path.exists():
            result = json.loads(path.read_text(encoding='utf8'))
            if result['checkpoint_sha256'] != selected['checkpoint_sha256'] or result['test_cache_sha256'] != data_hash:
                raise ValueError(f'Test cache mismatch: {path}')
        else:
            model, saved = restore(Path(selected['checkpoint']), Path('data/graphs/central-1024/graph.npz'), lexicon)
            if saved['_file_sha256'] != selected['checkpoint_sha256']: raise ValueError('Checkpoint changed after selection freeze')
            result = dict(**selected, checkpoint_step=saved['step'], test_cache_sha256=data_hash,
                parameters=model.parameter_card()['trainable_parameters'], score=evaluate(model, documents, lexicon))
            write_json(path, result)
        results.append(result)
        print(json.dumps(dict(variant=selected['variant'], seed=selected['seed'], test_bpb=result['score']['bits_per_byte'])), flush=True)
    comparisons = []
    for seed in SEEDS:
        chosen = {r['variant']: r for r in results if r['seed'] == seed}
        for variant in ('gru', 'transformer'):
            comparisons.append(dict(first='flm', second=variant, training_seed=seed,
                **paired_interval(chosen['flm']['score']['documents'], chosen[variant]['score']['documents'])))
    aggregates = []
    for variant in VARIANTS:
        values = [r['score']['bits_per_byte'] for r in results if r['variant'] == variant]
        aggregates.append(dict(variant=variant, mean_bpb=float(np.mean(values)), seed_standard_deviation=float(np.std(values, ddof=1)), seeds=list(SEEDS)))
    # Smoothing must already have been selected on validation before test evaluation.
    tuning = json.loads((a.runs / 'ngram/result.json').read_text(encoding='utf8'))
    train = Path('data/processed/wikitext2/train.jsonl'); raw_test = train.with_name('test.jsonl')
    if tuning['train_sha256'] != sha256(train): raise ValueError('N-gram training source changed')
    ngram_path = a.output / 'test-ngram.json'
    if ngram_path.exists():
        ngram = json.loads(ngram_path.read_text(encoding='utf8'))
        if ngram['test_sha256'] != sha256(raw_test) or ngram['smoothing'] != tuning['smoothing']:
            raise ValueError('N-gram test cache mismatch')
    else:
        model = NGram(4).fit(read_documents(train))
        ngram = dict(smoothing=tuning['smoothing'], train_sha256=sha256(train), test_sha256=sha256(raw_test),
            parameter_matched=False, score=model.evaluate(read_documents(raw_test), tuning['smoothing']))
        write_json(ngram_path, ngram)
    summary = dict(dataset='WikiText-2 raw', aggregates=aggregates, paired_comparisons=comparisons, runs=results, ngram=ngram,
        caution='Two seeds give limited information about training variability. Article bootstrap intervals do not account for other corpora, tuning choices or biological validity.')
    write_json(a.output / 'summary.json', summary)
    write_json(Path('public/research/test-results.json'), summary)


if __name__ == '__main__': main()
