"""Read final computation-control results only after the full study gate passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flm.language_core_study import (LEXICON, REPORTS, conditions, read_json,
                                    verify_complete, verify_identity)
from flm.language_core_test import summarize
from flm.language_topology_test import validate_score
from flm.provenance import sha256
from flm.tokenizer import Lexicon, read_cache


def verified_report(root):
    """Check checkpoint selection before decoding test data; perform no writes.

    This revalidates supplied scores against their frozen identities and
    recomputes the declared comparisons. It does not rerun likelihood inference.
    """
    root = Path(root).resolve()
    folder = root / REPORTS
    if not (folder / 'summary.json').is_file() or not (folder / 'selection.json').is_file():
        raise ValueError('Final computation scores and frozen selections are not available')
    identity = read_json(folder / 'identity.json')
    verify_identity(root, identity)
    for condition in conditions():
        if not (root / condition['output'] / 'complete.json').is_file():
            raise ValueError('Registered computation condition incomplete: ' + condition['label'])
    lexicon = Lexicon(root / LEXICON)
    selected = [verify_complete(root, condition, identity, lexicon) for condition in conditions()]
    selection = read_json(folder / 'selection.json')
    if selected != selection['runs']:
        raise ValueError('Selected computation checkpoints changed')
    identity_hash = sha256(folder / 'identity.json')
    selection_hash = sha256(folder / 'selection.json')
    if selection['study_identity_sha256'] != identity_hash or selection['tokenizer_sha256'] != lexicon.sha256:
        raise ValueError('Frozen selection belongs to a different study or tokenizer')
    report = read_json(folder / 'summary.json')
    if (report['study_identity_sha256'] != identity_hash or report['selection_sha256'] != selection_hash or
            report['study'] != identity['study']):
        raise ValueError('Final computation report belongs to a different study or selection')
    test_relative = 'data/processed/wikitext2-bpe/test.npz'
    test_path = root / test_relative
    test_hash = sha256(test_path)
    if test_hash != selection['test_cache_sha256'] or test_hash != identity['inputs'][test_relative]:
        raise ValueError('Frozen computation test cache changed')
    # No test token decoding until every real run, checkpoint and identity passed.
    documents = read_cache(test_path)
    expected = {row['label']: row for row in selected}
    if len(report['runs']) != 8 or {run['label'] for run in report['runs']} != set(expected):
        raise ValueError('All eight unique computation conditions are required')
    for run in report['runs']:
        bound = dict(**expected[run['label']], selection_sha256=selection_hash, test_cache_sha256=test_hash)
        if any(run.get(key) != value for key, value in bound.items()):
            raise ValueError('Final computation score identity changed: ' + run['label'])
        if run != read_json(folder / ('test-' + run['label'] + '.json')):
            raise ValueError('Summary differs from its individual score: ' + run['label'])
        validate_score(run['score'], documents, lexicon)
        if run['reference']:
            previous_path = root / f'reports/wikitext2/test-flm-s{run["seed"]}.json'
            previous = read_json(previous_path)
            if run.get('reused_score_sha256') != sha256(previous_path) or run['score'] != previous['score']:
                raise ValueError('Published full-model reference score changed')
        else:
            state_protocol = ('Reset both states for every token, including within chunks' if run['control'] == 'no_temporal_state'
                              else 'Carry native fast/slow state within each article; reset per article')
            if run['score'].get('mechanism') != state_protocol:
                raise ValueError('Scored computation mechanism declaration changed')
    for key, value in summarize(report['runs']).items():
        if report[key] != value:
            raise ValueError('Recomputed computation contrasts differ: ' + key)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = verified_report(args.root)
    print(json.dumps(dict(study_identity_sha256=report['study_identity_sha256'],
                          selection_sha256=report['selection_sha256'], verified_models=len(report['runs']),
                          primary_means=report['primary_means'],
                          independent_unit_memory_mean_difference_bpb=report['independent_unit_memory_mean_difference_bpb'],
                          scope='Verified supplied scores and checkpoint identities; not independent test inference'), indent=2))


if __name__ == '__main__':
    main()
