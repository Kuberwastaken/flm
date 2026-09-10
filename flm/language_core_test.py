"""Gate and score the declared language computation comparisons after all fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .language_core_study import (GRAPH, LEXICON, REPORTS, conditions, read_json,
                                  run_binding, verify_complete, verify_identity)
from .language_core_train import restore, verify_exposure
from .language_test import paired_interval
from .language_topology_test import validate_score
from .language_train import evaluate
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache


def freeze_selection(root):
    identity_path = root / REPORTS / 'identity.json'
    identity = read_json(identity_path)
    verify_identity(root, identity)
    # Check the whole inventory before constructing models or decoding test text.
    for condition in conditions():
        if not (root / condition['output'] / 'complete.json').exists():
            raise ValueError('Language computation run incomplete: ' + condition['label'])
    lexicon = Lexicon(root / LEXICON)
    selected = [verify_complete(root, row, identity, lexicon) for row in conditions()]
    test = 'data/processed/wikitext2-bpe/test.npz'
    frozen = dict(study_identity_sha256=sha256(identity_path), runs=selected,
                  test_cache_sha256=identity['inputs'][test], tokenizer_sha256=lexicon.sha256,
                  selection='Lowest validation BPB at updates 500..6000; earliest exact tie',
                  prior_test_access=identity['prior_test_access'])
    path = root / REPORTS / 'selection.json'
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError('Different language computation selection already frozen')
    else:
        write_json(path, frozen)
    return frozen, identity


def summarize(results):
    lookup = {row['label']: row for row in results}
    if len(lookup) != len(results) or set(lookup) != {row['label'] for row in conditions()}:
        raise ValueError('All eight unique language computation conditions are required')
    primary = []
    independent_memory = []
    for seed in (42, 43):
        full = lookup[f'full-s{seed}']['score']['documents']
        for control in ('fixed_dynamics', 'no_lateral', 'no_temporal_state'):
            primary.append(dict(first='full', second=control, training_seed=seed,
                                **paired_interval(full, lookup[f'{control}-s{seed}']['score']['documents'])))
        independent_memory.append(dict(first='no_lateral', second='no_temporal_state', training_seed=seed,
            **paired_interval(lookup[f'no_lateral-s{seed}']['score']['documents'],
                              lookup[f'no_temporal_state-s{seed}']['score']['documents'])))
    means = [dict(control=control, mean_difference_bpb=float(np.mean(
             [row['difference_bpb'] for row in primary if row['second'] == control])))
             for control in ('fixed_dynamics', 'no_lateral', 'no_temporal_state')]
    return dict(primary_contrasts=primary, primary_means=means, independent_unit_memory_contrasts=independent_memory,
                independent_unit_memory_mean_difference_bpb=float(np.mean([r['difference_bpb'] for r in independent_memory])),
                sign='First minus second test BPB; negative favors the first model',
                uncertainty='Intervals condition on each trained pair. Contrasts share references and articles; two initializations do not establish training uncertainty.',
                scope='Measured subset only; no topology-by-trainability factorial or equivalence test')


def score_study(root):
    frozen, identity = freeze_selection(root)
    selection_hash = sha256(root / REPORTS / 'selection.json')
    torch.set_num_threads(identity['training_protocol']['threads'])
    lexicon = Lexicon(root / LEXICON)
    documents = read_cache(root / 'data/processed/wikitext2-bpe/test.npz')
    results = []
    for selected in frozen['runs']:
        verify_identity(root, identity)
        if sha256(root / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed before scoring')
        bound = dict(**selected, selection_sha256=selection_hash, test_cache_sha256=frozen['test_cache_sha256'])
        path = root / REPORTS / ('test-' + selected['label'] + '.json')
        if path.exists():
            result = read_json(path)
            if any(result.get(key) != value for key, value in bound.items()):
                raise ValueError('Cached computation score identity changed')
        elif selected['reference']:
            previous_path = root / f'reports/wikitext2/test-flm-s{selected["seed"]}.json'
            previous = read_json(previous_path)
            for key in ('checkpoint_sha256', 'test_cache_sha256', 'checkpoint_step', 'seed'):
                if previous[key] != bound[key]:
                    raise ValueError('Existing full-model test score identity changed')
            if previous['parameters'] != bound['trainable_parameters']:
                raise ValueError('Existing full-model parameter count changed')
            result = dict(**bound, score=previous['score'], reused_score_sha256=sha256(previous_path))
        else:
            model, saved = restore(root / selected['checkpoint'], root / GRAPH, lexicon, run_binding(root, selected, identity))
            verify_exposure(saved, identity)
            if saved['_file_sha256'] != selected['checkpoint_sha256'] or saved['step'] != selected['checkpoint_step']:
                raise ValueError('Checkpoint changed after computation selection freeze')
            score = evaluate(model, documents, lexicon)
            score['mechanism'] = ('Reset both states for every token, including within chunks' if selected['control'] == 'no_temporal_state'
                                  else 'Carry native fast/slow state within each article; reset per article')
            result = dict(**bound, score=score)
        validate_score(result['score'], documents, lexicon)
        verify_identity(root, identity)
        if not path.exists():
            write_json(path, result)
        results.append(result)
        print(f'{selected["label"]}: {result["score"]["bits_per_byte"]:.6f} test bits/byte', flush=True)
    report = dict(study=identity['study'], study_identity_sha256=frozen['study_identity_sha256'],
                  selection_sha256=selection_hash, completed_utc=datetime.now(timezone.utc).isoformat(),
                  runs=results, **summarize(results))
    write_json(root / REPORTS / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'score'))
    args = parser.parse_args()
    root = Path('.').resolve()
    if args.action == 'freeze':
        freeze_selection(root)
    else:
        score_study(root)


if __name__ == '__main__':
    main()
