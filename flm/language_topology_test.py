"""Freeze all language controls before test scoring and retain every paired result."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import numpy as np
import torch
from .language_test import paired_interval
from .language_topology import (REPORTS, LEXICON, conditions, read_json,
    verify_source_identity, verify_complete, verify_saved)
from .language_train import evaluate, restore
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache

SCORING_SOURCES = ('flm/language_topology_test.py', 'flm/language_test.py', 'flm/language_train.py')


def freeze_selection(root):
    identity_path = root / REPORTS / 'identity.json'; identity = read_json(identity_path)
    verify_source_identity(root, identity)
    # Completeness is checked before even decoding the test token cache.
    for condition in conditions():
        if not (root / condition['output'] / 'complete.json').exists():
            raise ValueError(f'Language control incomplete: {condition["label"]}')
    lexicon = Lexicon(root / LEXICON)
    selected = [verify_complete(root, condition, identity, lexicon) for condition in conditions()]
    test_path = root / 'data/processed/wikitext2-bpe/test.npz'
    card = read_json(root / 'data/tokenizers/wikitext2-4096/tokenization-card.json')
    if sha256(test_path) != card['partitions']['test']['cache_sha256']:
        raise ValueError('Original test cache changed')
    frozen = dict(study_identity_sha256=sha256(identity_path), runs=selected,
        test_cache_sha256=sha256(test_path), tokenizer_sha256=lexicon.sha256,
        scoring_sources={name: sha256(root / name) for name in SCORING_SOURCES},
        selection='Lowest fixed-prefix validation BPB at updates 500..6000, earliest exact tie',
        prior_test_access='Measured FLM/GRU/transformer test results inspected before this extension; new controls selected without test scores')
    path = root / REPORTS / 'selection.json'
    if path.exists() and read_json(path) != frozen: raise ValueError('A different language topology test selection is already frozen')
    write_json(path, frozen)
    return frozen, identity


def validate_score(score, documents, lexicon):
    records = score['documents']; lookup = {row['document']: row for row in records}
    if len(lookup) != len(records) or set(lookup) != {name for name, _ in documents}:
        raise ValueError('Test article coverage changed')
    for name, document in documents:
        row = lookup[name]; targets = document[1:]
        if row['tokens'] != int(np.sum(targets >= 2)) or row['bytes'] != int(lexicon.lengths[targets].sum()):
            raise ValueError('Test article denominator changed')
        if not math.isfinite(row['nll']) or row['nll'] < 0: raise ValueError('Invalid test likelihood')
        if not math.isclose(row['bits_per_byte'], row['nll'] / row['bytes'] / math.log(2), rel_tol=1e-10):
            raise ValueError('Article BPB inconsistent with likelihood')
    for key in ('bytes', 'tokens', 'nll'):
        if not math.isclose(score[key], sum(row[key] for row in records), rel_tol=1e-10):
            raise ValueError('Test aggregate does not match its articles')
    expected = score['nll'] / score['bytes'] / math.log(2)
    if not math.isclose(score['bits_per_byte'], expected, rel_tol=1e-10): raise ValueError('Aggregate BPB changed')
    if not math.isclose(score['token_perplexity'], math.exp(score['nll']/score['tokens']), rel_tol=1e-10):
        raise ValueError('Aggregate perplexity changed')
    if score['tokenizer_sha256'] != lexicon.sha256: raise ValueError('Score tokenizer changed')


def summarize(results):
    expected = {row['label'] for row in conditions()}
    lookup = {row['label']: row for row in results}
    if len(lookup) != len(results) or set(lookup) != expected: raise ValueError('All ten unique conditions are required')
    topology = []; slow = []
    for seed in (42, 43):
        measured = lookup[f'measured-s{seed}']['score']['documents']
        for graph_seed in (101, 103, 107):
            control = lookup[f'null{graph_seed}-s{seed}']['score']['documents']
            topology.append(dict(training_seed=seed, graph_seed=graph_seed, **paired_interval(measured, control)))
        slow.append(dict(training_seed=seed, **paired_interval(measured, lookup[f'no-slow-s{seed}']['score']['documents'])))
    means = lambda key, values: [dict(**{key: value}, mean_difference_bpb=float(np.mean(
        [row['difference_bpb'] for row in topology if row[key] == value]))) for value in values]
    return dict(topology_contrasts=topology, slow_state_contrasts=slow,
        topology_by_graph=means('graph_seed', (101, 103, 107)), topology_by_training_seed=means('training_seed', (42, 43)),
        topology_mean_difference_bpb=float(np.mean([row['difference_bpb'] for row in topology])),
        slow_state_mean_difference_bpb=float(np.mean([row['difference_bpb'] for row in slow])),
        sign='Measured minus control; negative favors the measured fast/slow model',
        uncertainty='Article intervals condition on each trained pair. Six topology contrasts share two measured references; they are not six independent replications.')


def score_study(root):
    frozen, identity = freeze_selection(root); torch.set_num_threads(4)
    selection_hash = sha256(root / REPORTS / 'selection.json')
    lexicon = Lexicon(root / LEXICON)
    documents = read_cache(root / 'data/processed/wikitext2-bpe/test.npz'); results = []
    for selected in frozen['runs']:
        verify_source_identity(root, identity)
        if any(sha256(root / name) != digest for name, digest in frozen['scoring_sources'].items()):
            raise ValueError('Scoring source changed after selection freeze')
        bound = dict(**selected, selection_sha256=selection_hash, test_cache_sha256=frozen['test_cache_sha256'])
        path = root / REPORTS / ('test-' + selected['label'] + '.json')
        if path.exists():
            result = read_json(path)
            if any(result.get(k) != v for k, v in bound.items()): raise ValueError('Cached test result identity changed')
        elif selected['reference']:
            previous_path = root / f'reports/wikitext2/test-flm-s{selected["seed"]}.json'
            previous = read_json(previous_path)
            for key in ('checkpoint_sha256', 'test_cache_sha256', 'seed', 'parameters', 'checkpoint_step'):
                if previous[key] != bound[key]: raise ValueError('Previously published reference score changed identity')
            result = dict(**bound, score=previous['score'], reused_score_sha256=sha256(previous_path))
            validate_score(result['score'], documents, lexicon); write_json(path, result)
        else:
            model, saved = restore(root / selected['checkpoint'], root / selected['graph'], lexicon)
            verify_saved(saved, selected, identity, root / selected['graph'])
            if saved['_file_sha256'] != selected['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
            result = dict(**bound, score=evaluate(model, documents, lexicon))
            validate_score(result['score'], documents, lexicon); write_json(path, result)
        validate_score(result['score'], documents, lexicon); results.append(result)
        print(f'{selected["label"]}: {result["score"]["bits_per_byte"]:.6f} test bits/byte', flush=True)
    report = dict(study='Language topology and retrained slow-state comparison', runs=results,
        selection_sha256=selection_hash, **summarize(results),
        limitations=['One selected 1024-neuron subset and one corpus; not a whole-brain language result.',
            'Three constrained null graphs and two training seeds; no proven chain mixing or broad topology superiority.',
            'Slow-state control is retrained on measured wiring only; no topology-by-slow-state interaction estimate.',
            'Exploratory extension after the original test scores were inspected.'])
    write_json(root / REPORTS / 'summary.json', report)
    write_json(root / 'public/research/language-topology-results.json', report)


def snapshot(root):
    identity_path = root / REPORTS / 'identity.json'; identity = read_json(identity_path)
    verify_source_identity(root, identity); torch.set_num_threads(4)
    lexicon = Lexicon(root / LEXICON); rows = []
    for condition in conditions():
        directory = root / condition['output']; step = 0; complete = False
        if (directory / 'last.pt').exists():
            _, saved = restore(directory / 'last.pt', root / condition['graph'], lexicon)
            verify_saved(saved, condition, identity, root / condition['graph']); step = saved['step']
        if (directory / 'complete.json').exists():
            verify_complete(root, condition, identity, lexicon); complete = True
        validation = []
        for update in range(500, step + 1, 500):
            score = read_json(directory / f'validation-{update:06d}.json')
            if not math.isfinite(score['bits_per_byte']) or score['tokenizer_sha256'] != lexicon.sha256:
                raise ValueError('Invalid validation snapshot')
            validation.append(dict(step=update, bits_per_byte=score['bits_per_byte'], token_perplexity=score['token_perplexity']))
        rows.append(dict(**condition, saved_step=step, complete=complete, validation=validation))
    report = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), study_identity_sha256=sha256(identity_path),
        new_runs=8, completed_new_runs=sum(r['complete'] and not r['reference'] for r in rows), registered_updates=6000,
        test_status='Published' if (root / REPORTS / 'summary.json').exists() else 'Pending all eight new runs and frozen checkpoint selections',
        comparison='Validation at matched updates only. Existing measured references are reused, not additional independent runs.',
        runs=rows)
    write_json(root / 'public/research/language-topology-progress.json', report)
    print(f'Snapshot: {report["completed_new_runs"]}/8 new controls complete; test {report["test_status"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'score', 'snapshot'))
    args = parser.parse_args(); root = Path('.').resolve()
    if args.action == 'freeze': freeze_selection(root)
    elif args.action == 'snapshot': snapshot(root)
    else: score_study(root)


if __name__ == '__main__': main()
