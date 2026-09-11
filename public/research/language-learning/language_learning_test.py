"""Score all eight frozen learning-rule checkpoints on complete held-out blocks.

No partial-condition comparison or test-driven checkpoint selection is allowed.
The source-bound study protocol fixes scoring settings before any official fit.
"""
from __future__ import annotations
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import torch

from .babylm_test import scored_blocks
from .corpus_cache import read_mmap
from .corpus_evaluation import aggregate, component_summary, paired_block_interval
from .language_learning_inputs import conditions
from .language_learning_study import (IDENTITY, REPORTS, SELECTION, STUDY, freeze_selection,
    prepared_run, read, test_metadata, verified_context)
from .language_learning_train import declaration, optimizer_for, restore, training_lease
from .language_learning_validation import document_identity
from .provenance import sha256, write_json

CACHE = Path('data/processed/babylm-2026-bpe/test')
EVALUATION = Path('runs/language-learning-evaluation-v1')
SUMMARY = REPORTS/'test-summary.json'
METHODS = ('bptt', 'fixed_core', 'eligibility', 'no_history')
POLICY = dict(batch_size=8, chunk_size=96, threads=4, bootstrap_replicates=10000,
    bootstrap_seed=31415, subsets=['official', 'overlap_filtered'],
    comparisons=[list(pair) for pair in itertools.combinations(METHODS, 2)],
    boundary_targets_scored=False, evaluation_warmup=0,
    state='Reset at each block; carry across chunks within that block',
    uncertainty='Paired block bootstrap within source, conditional on each seed/checkpoint; unadjusted descriptive intervals')


def unchanged(root, identity, selection, identity_hash, selection_hash):
    if sha256(root/IDENTITY) != identity_hash or sha256(root/SELECTION) != selection_hash:
        raise ValueError('Frozen study or selection changed during evaluation')
    for name, expected in identity['sources'].items():
        if sha256(root/'flm'/name) != expected: raise ValueError('Frozen evaluator source changed')
    for row in selection['conditions']:
        directory = root/STUDY/row['condition']['label']
        for path, expected in ((root/row['checkpoint'], row['checkpoint_sha256']),
                (directory/'complete.json', row['complete_sha256']),
                (directory/'validation-selection.json', row['validation_selection_sha256'])):
            if sha256(path) != expected: raise ValueError('Selected run changed during evaluation')


def load_test(root, lexicon, expected):
    """Called only after the whole-inventory selection gate; verify actual payloads."""
    if test_metadata(root, lexicon) != expected:
        raise ValueError('Bound test metadata changed before payload access')
    mapped, metadata, manifest = read_mmap(root/CACHE, lexicon.sha256)
    # Test payload is about 35 MiB. Own read-only copies so failure tracebacks and
    # cached evaluator callbacks cannot retain Windows file locks.
    mappings = {id(doc._mmap):doc._mmap for _,doc in mapped if isinstance(doc, np.memmap)}
    try:
        documents = [(name, np.array(doc, copy=True)) for name,doc in mapped]
        for _,doc in documents: doc.setflags(write=False)
    finally:
        for mapping in mappings.values(): mapping.close()
    if manifest != expected['manifest']: raise ValueError('Test cache differs from frozen metadata')
    coverage = document_identity(documents, lexicon)
    lookup = {r['id']:r for r in metadata}
    if len(lookup) != len(metadata) or set(lookup) != {r['document'] for r in coverage['documents']}:
        raise ValueError('Test metadata does not cover unique blocks')
    for (name, tokens), counts in zip(documents, coverage['documents']):
        row = lookup[name]
        if (not name.startswith('test/') or not isinstance(row['component'], str) or not row['component']
                or type(row['overlap_filtered_eligible']) is not bool or tokens[0] != 0 or tokens[-1] != 1
                or np.any(tokens[1:-1] < 2) or counts['tokens'] != len(tokens)-2
                or counts['bytes'] != row['utf8_bytes']):
            raise ValueError('Invalid complete test block or scoring denominator')
    if (len(documents) != manifest['blocks'] or coverage['tokens'] != manifest['text_tokens']
            or coverage['bytes'] != manifest['utf8_bytes']):
        raise ValueError('Test payloads do not cover complete frozen corpus')
    if {r['component'] for r in metadata} != set(manifest['components']):
        raise ValueError('Test source-component inventory changed')
    for component, expected_counts in manifest['components'].items():
        rows = [r for r in coverage['documents'] if lookup[r['document']]['component'] == component]
        if (len(rows) != expected_counts['blocks'] or sum(r['tokens'] for r in rows) != expected_counts['text_tokens']
                or sum(r['bytes'] for r in rows) != expected_counts['utf8_bytes']):
            raise ValueError('Test component denominators changed')
    return documents, metadata, manifest, coverage


def restore_selected(root, inputs, registered, selected):
    model, binding, settings = prepared_run(root, inputs, registered)
    declared = declaration(model, inputs.documents, inputs.lexicon, settings,
                           registered['condition']['method'], binding)
    path = root/selected['checkpoint']
    if sha256(path) != selected['checkpoint_sha256']: raise ValueError('Selected payload checksum changed')
    saved, _, _ = restore(path, model, optimizer_for(model, settings), declared,
                         inputs.documents, inputs.lexicon.lengths, settings)
    if saved['step'] != selected['checkpoint_step']: raise ValueError('Selected update changed')
    return model


def validate_scores(records, coverage, metadata):
    if [r['document'] for r in records] != [r['document'] for r in coverage['documents']]:
        raise ValueError('Scored block order or inventory changed')
    lookup = {r['id']:r for r in metadata}
    total = aggregate(records)
    for row, counts in zip(records, coverage['documents']):
        meta = lookup[row['document']]
        if (any(row[key] != counts[key] for key in ('tokens', 'bytes'))
                or row['component'] != meta['component']
                or type(row['overlap_filtered_eligible']) is not bool
                or row['overlap_filtered_eligible'] != meta['overlap_filtered_eligible']
                or not math.isclose(row['bits_per_byte'], row['nll']/row['bytes']/math.log(2), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Scored block counts, eligibility or likelihood arithmetic changed')
    if total['tokens'] != coverage['tokens'] or total['bytes'] != coverage['bytes']:
        raise ValueError('Incomplete test likelihood coverage')
    return component_summary(records)


def summarize(results, policy):
    expected = {(c['method'], c['seed']) for c in conditions()}
    observed = [(r['condition']['method'], r['condition']['seed']) for r in results]
    if len(observed) != 8 or set(observed) != expected:
        raise ValueError('Require every registered learning-rule result')
    component_keys = set(results[0]['components'])
    if any(set(r['components']) != component_keys for r in results):
        raise ValueError('Results have different source components')
    aggregates = []; comparisons = []
    for subset in policy['subsets']:
        for component in sorted(component_keys):
            for method in METHODS:
                rows = sorted((r for r in results if r['condition']['method'] == method), key=lambda r:r['condition']['seed'])
                scores = [r['components'][component][subset] for r in rows]
                values = [r['bits_per_byte'] for r in scores if r is not None]
                if values and len(values) != 2: raise ValueError('Unmatched retained blocks across seeds')
                aggregates.append(dict(method=method, component=component, subset=subset, seeds=[42,43],
                    bits_per_byte=values, mean_bpb=float(np.mean(values)) if values else None,
                    seed_standard_deviation=float(np.std(values, ddof=1)) if values else None))
        for seed in (42,43):
            rows = {r['condition']['method']:r for r in results if r['condition']['seed'] == seed}
            for first, second in policy['comparisons']:
                blocks = {method:[r for r in rows[method]['blocks'] if subset == 'official' or r['overlap_filtered_eligible']]
                          for method in (first, second)}
                if bool(blocks[first]) != bool(blocks[second]): raise ValueError('Unmatched overlap-filtered coverage')
                interval = paired_block_interval(blocks[first], blocks[second], policy['bootstrap_replicates'],
                    policy['bootstrap_seed']) if blocks[first] else None
                comparisons.append(dict(first=first, second=second, training_seed=seed, subset=subset,
                    estimate=interval, reason=None if interval else 'No eligible blocks in the frozen filtered subset'))
    return dict(aggregates=aggregates, paired_comparisons=comparisons,
        runs=[{k:v for k,v in r.items() if k != 'blocks'} for r in results],
        caution='All six pairwise differences are descriptive, without multiplicity correction. Intervals condition on fitted checkpoints and artificial blocks with unknown document dependence. Two seeds do not establish training robustness. These learning-rule results cannot establish anatomical-topology advantage, biological plausibility or conversation ability.')


def immutable_json(path, result):
    if path.exists():
        if read(path) != result: raise ValueError('Existing completed evaluation record changed')
    else: write_json(path, result)


def score_study(root):
    root = Path(root)
    previous_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            # The gate restores/audits every training payload before any test file is read.
            selection = freeze_selection(root)
            with training_lease(root/STUDY), training_lease(root/EVALUATION):
                identity, inputs, _ = verified_context(root)
                identity_hash = sha256(root/IDENTITY); selection_hash = sha256(root/SELECTION)
                if selection != read(root/SELECTION) or selection['study_identity_sha256'] != identity_hash:
                    raise ValueError('Selection no longer belongs to the frozen study')
                policy = identity['evaluation_policy']
                if policy != POLICY: raise ValueError('Frozen evaluation policy changed')
                unchanged(root, identity, selection, identity_hash, selection_hash)
                torch.set_num_threads(policy['threads'])
                documents, metadata, manifest, coverage = load_test(root, inputs.lexicon, identity['test_metadata'])
                common = dict(study_identity_sha256=identity_hash, selection_sha256=selection_hash,
                    test_metadata=identity['test_metadata'], coverage=coverage, policy=policy,
                    source_sha256=identity['sources'], torch=str(torch.__version__), numpy=str(np.__version__))
                immutable_json(root/EVALUATION/'identity.json', common)
                evaluation_hash = sha256(root/EVALUATION/'identity.json')
                registered = {r['condition']['label']:r for r in identity['conditions']}
                results = []; result_hashes = {}
                for selected in selection['conditions']:
                    unchanged(root, identity, selection, identity_hash, selection_hash)
                    label = selected['condition']['label']
                    model = restore_selected(root, inputs, registered[label], selected)
                    run_identity = dict(evaluation_identity_sha256=evaluation_hash, selected=selected)
                    destination = root/EVALUATION/label
                    records, seconds = scored_blocks(model, documents, metadata, inputs.lexicon, destination,
                        run_identity, policy['batch_size'], policy['chunk_size'])
                    if not math.isfinite(seconds) or seconds < 0: raise ValueError('Invalid scoring time')
                    components = validate_scores(records, coverage, metadata)
                    batch_paths = sorted(destination.glob('batch-*.json'))
                    if [p.name for p in batch_paths] != [f'batch-{i:05d}.json' for i in range(math.ceil(len(documents)/policy['batch_size']))]:
                        raise ValueError('Unexpected evaluation batch inventory')
                    if any(not math.isfinite(read(p)['seconds']) or read(p)['seconds'] < 0 for p in batch_paths):
                        raise ValueError('Invalid cached batch time')
                    result = dict(condition=selected['condition'], identity=run_identity,
                        checkpoint_step=selected['checkpoint_step'], components=components, blocks=records,
                        seconds=seconds, timing='Scoring wall time, potentially including other work; not a throughput comparison',
                        batch_record_sha256={p.name:sha256(p) for p in batch_paths})
                    immutable_json(root/REPORTS/f'test-{label}.json', result)
                    result_hashes[label] = sha256(root/REPORTS/f'test-{label}.json')
                    results.append(result)
                    print('Completed held-out learning-rule likelihood: '+label, flush=True)
                unchanged(root, identity, selection, identity_hash, selection_hash)
                # Verify cache payload hashes again before committing the all-condition summary.
                _, _, _, final_coverage = load_test(root, inputs.lexicon, identity['test_metadata'])
                if final_coverage != coverage: raise ValueError('Test payload changed during evaluation')
                if sha256(root/EVALUATION/'identity.json') != evaluation_hash:
                    raise ValueError('Evaluation identity changed during scoring')
                for row in results:
                    label = row['condition']['label']
                    if sha256(root/REPORTS/f'test-{label}.json') != result_hashes[label]:
                        raise ValueError('Completed result changed during scoring')
                    for name, expected in row['batch_record_sha256'].items():
                        if sha256(root/EVALUATION/label/name) != expected:
                            raise ValueError('Cached batch changed during scoring')
                result = dict(evaluation_identity_sha256=evaluation_hash,
                    **summarize(results, policy), result_sha256=result_hashes)
                immutable_json(root/SUMMARY, result)
                return result
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    score_study(Path.cwd())
