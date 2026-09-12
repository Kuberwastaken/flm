"""Apply the dated allocation rule to completed records, without model imports.

This is a reporting consumer, not a replacement for the frozen scorer's integrity
checks. It never loads checkpoints, raw corpus text, or partial held-out results.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = 'reports/selection-language/'
RULE = DIRECTORY + 'interpretation-v1.json'
RULE_SHA256 = 'b1ce71b64c29d8b8d14ac803b2db6f912d934712cfc712a4f82af34dec5df364'
IDENTITY = DIRECTORY + 'study-identity.json'
SUMMARY = DIRECTORY + 'test-summary.json'
SELECTION = DIRECTORY + 'study-selection.json'
EVALUATION = 'runs/selection-language-evaluation-v1/identity.json'
DOCUMENT = 'docs/ANATOMICAL-PRIOR-DECISION.md'
SELECTORS = ('candidate', 'contact_ranked', 'uniform_s201', 'uniform_s203', 'uniform_s207',
             'stratified_s201', 'stratified_s203', 'stratified_s207')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Undefined/nonfinite required estimate')
    return value


def comparisons(group):
    """Names and membership from the frozen 42-contrast design; no scorer import."""
    graph = lambda selector, wiring='measured': f'{group}/{selector}/{wiring}'
    rows = []
    for family, selectors in (('ranked', ('contact_ranked',)),
                             ('uniform', SELECTORS[2:5]), ('stratified', SELECTORS[5:8])):
        rows.append(('selection', 'family', 'candidate-minus-'+family,
                     [graph('candidate')], [graph(s) for s in selectors]))
        for selector in selectors:
            rows.append(('selection', 'individual', 'candidate-minus-'+selector,
                         [graph('candidate')], [graph(selector)]))
    for selector in SELECTORS:
        nulls = [graph(selector, f'null{s}') for s in (101, 103, 107)]
        rows.append(('topology', 'family', selector+'-measured-minus-rewires', [graph(selector)], nulls))
        for null in nulls:
            rows.append(('topology', 'individual', selector+'-measured-minus-'+null.split('/')[-1],
                         [graph(selector)], [null]))
    return rows


def apply_rule(rule, rows):
    """Pure gate calculation. Caller must establish complete-record provenance."""
    expected = {}
    for group in rule['groups']:
        for seed in rule['training_seeds']:
            for subset in rule['analysis_subsets']:
                for kind, level, name, first, second in comparisons(group):
                    expected[group, seed, subset, kind, level, name] = (first, second)
    actual = {}
    for row in rows:
        key = tuple(row[k] for k in ('group', 'training_seed', 'subset', 'kind', 'level', 'name'))
        require(key not in actual and key in expected, 'Duplicate or unexpected declared contrast')
        require((row['first'], row['second']) == expected[key], 'Contrast membership changed')
        estimate = row['estimate']
        require(isinstance(estimate, dict) and row['reason'] is None, 'Undefined declared contrast')
        for field in ('difference_bpb', 'lower_95', 'upper_95'):
            finite(estimate[field])
        require(estimate['lower_95'] <= estimate['upper_95'], 'Reversed interval')
        require(estimate['replicates'] == 10000 and estimate['seed'] == 31415, 'Interval policy changed')
        actual[key] = row
    require(set(actual) == set(expected), 'Incomplete original contrast inventory')
    limits = rule['all_required']
    analyses = []
    for subset in rule['analysis_subsets']:
        cells = []
        for group in rule['groups']:
            for seed in rule['training_seeds']:
                key = (group, seed, subset, 'topology')
                estimate = actual[(*key, 'family', rule['primary_contrast'])]['estimate']
                singles = [actual[(*key, 'individual', f'candidate-measured-minus-null{s}')]['estimate']['difference_bpb']
                           for s in rule['rewiring_seeds']]
                require(math.isclose(estimate['difference_bpb'], math.fsum(singles)/3,
                                     rel_tol=0, abs_tol=1e-10), 'Family and individual point arithmetic disagree')
                cells.append(dict(group=group, training_seed=seed, **estimate,
                                  individual_differences=dict(zip(map(str, rule['rewiring_seeds']), singles))))
        mean = math.fsum(c['difference_bpb'] for c in cells)/len(cells)
        checks = dict(
            margin=mean <= limits['equal_four_cell_mean_bpb_at_most'],
            every_cell_negative=all(c['difference_bpb'] < limits['each_four_cell_mean_bpb_strictly_below'] for c in cells),
            every_interval_upper_negative=all(c['upper_95'] < limits['each_four_cell_conditional_95pct_interval_upper_strictly_below'] for c in cells),
            every_individual_negative=all(v < limits['each_twelve_individual_contrast_bpb_strictly_below']
                                          for c in cells for v in c['individual_differences'].values()))
        analyses.append(dict(subset=subset, equal_four_cell_mean_bpb=mean, checks=checks, cells=cells))
    decision = 'pass' if all(all(a['checks'].values()) for a in analyses) else 'fail'
    return dict(decision=decision, analyses=analyses, action=rule[decision+'_action'])


def validate_mac_identity(parent, native):
    """The amendment changes runtime and native initialization, never the design."""
    expected = copy.deepcopy(parent)
    require(len(native['conditions']) == len(parent['conditions']) == 128, 'Incomplete Mac matrix')
    for old, new in zip(expected['conditions'], native['conditions']):
        for key in ('initial_state_sha256', 'initial_parameters_sha256', 'initial_nonedge_parameters_sha256'):
            value = new['base_binding'][key]
            require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                    'Invalid native initialization hash')
            old['base_binding'][key] = value
    expected['torch'] = '2.8.0'
    expected['execution_amendment'] = native['execution_amendment']
    require(native == expected, 'Mac study changed more than the declared runtime/initialization amendment')


def report(root, *, mac=False):
    root = Path(root).resolve()
    directory = DIRECTORY+'mac-v1/' if mac else DIRECTORY
    identity_name = directory+'study-identity.json'
    selection_name = directory+'study-selection.json'
    summary_name = directory+'test-summary.json'
    evaluation_name = 'runs/selection-language-mac-evaluation-v1/identity.json' if mac else EVALUATION
    bindings = {}

    def path(name):
        resolved = (root / name).resolve()
        require(resolved.is_relative_to(root), 'Record path escapes repository')
        return resolved

    def read(name, expected=None):
        payload = path(name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        require(expected is None or digest == expected, 'Checksum mismatch: '+name)
        bindings[name] = digest
        return json.loads(payload)

    output = dict(format='flm-selection-allocation-report-v1', decision='pending',
                  input_sha256=bindings, reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Consumes completed scorer records and verifies their hash links, inventories and point arithmetic. '
                        'Does not rerun inference, bootstrap intervals, checkpoint validation or corpus integrity checks; '
                        'those remain the frozen scorer responsibility. No general biological inference follows.')
    try:
        rule = read(RULE, RULE_SHA256)
        parent = read(IDENTITY, rule['original_identity_sha256'])
        identity = parent
        require(hashlib.sha256(path(DOCUMENT).read_bytes()).hexdigest() == rule['decision_document_sha256'],
                'Dated decision document changed')
        bindings[DOCUMENT] = rule['decision_document_sha256']
        if mac:
            if not path(identity_name).is_file():
                return dict(output, missing=[identity_name], action='Wait for the complete native initialization record; no scientific decision.')
            identity = read(identity_name)
            validate_mac_identity(parent, identity)
            amendment = identity['execution_amendment']
            require(amendment['parent_study_identity_sha256'] == bindings[IDENTITY], 'Mac parent lineage changed')
            profile = read(directory+'runtime-profile.json', amendment['profile_sha256'])
            read(directory+'qualification-probe.json', amendment['qualification_probe_sha256'])
            for name, expected in [('docs/SELECTION-MAC-EXECUTION.md', amendment['document_sha256']),
                                   ('scripts/selection_mac_runtime.py', amendment['adapter_sha256'])]:
                require(hashlib.sha256(path(name).read_bytes()).hexdigest() == expected, 'Mac amendment source changed: '+name)
                bindings[name] = expected
            require(profile['parent_study_identity_sha256'] == bindings[IDENTITY]
                    and profile['amendment_sha256'] == amendment['document_sha256']
                    and profile['qualification_probe_sha256'] == amendment['qualification_probe_sha256']
                    and profile['environment']['adapter_sha256'] == amendment['adapter_sha256']
                    and profile['environment']['torch'] == identity['torch'], 'Mac runtime profile linkage changed')
            output['execution_lineage'] = 'Complete fresh Mac cohort under a disclosed runtime/initialization amendment; original allocation thresholds retained.'
        require(rule['groups'] == ['KCg-d-L-t5', 'KCg-d-R-t5'] and rule['training_seeds'] == [42, 43]
                and rule['rewiring_seeds'] == [101, 103, 107]
                and rule['analysis_subsets'] == ['official', 'overlap_filtered'], 'Rule inventory changed')
        conditions = [r['condition'] for r in identity['conditions']]
        labels = [r['label'] for r in conditions]
        require(len(conditions) == rule['all_required']['complete_original_inventory'] == 128
                and len(set(labels)) == 128, 'Original inventory is not 128 unique conditions')
        # Never open partial evaluation outputs while training/selection is ongoing.
        missing = [name for name in (selection_name, summary_name) if not path(name).is_file()]
        if missing:
            return dict(output, missing=missing, action='Wait for the unchanged complete train/select/test queue; no scientific decision.')
        selection = read(selection_name)
        summary = read(summary_name)
        evaluation = read(evaluation_name, summary['evaluation_identity_sha256'])
        require(selection['study_identity_sha256'] == bindings[identity_name]
                and evaluation['study_identity_sha256'] == bindings[identity_name]
                and evaluation['selection_sha256'] == bindings[selection_name], 'Study/selection hash linkage changed')
        require(evaluation['policy'] == identity['evaluation_policy']
                and evaluation['source_sha256'] == identity['sources']
                and evaluation['test_metadata'] == identity['test_metadata'], 'Evaluation declaration changed')
        require([r['condition'] for r in selection['conditions']] == conditions
                and [r['condition'] for r in summary['runs']] == conditions
                and set(summary['result_sha256']) == set(labels), 'Incomplete selected/result inventory')
        for name, expected in identity['sources'].items():
            require(hashlib.sha256(path('flm/'+name).read_bytes()).hexdigest() == expected,
                    'Frozen numerical source changed: '+name)
        for selected, row in zip(selection['conditions'], summary['runs']):
            label = selected['condition']['label']
            result = read(directory+'test/'+label+'/result.json', summary['result_sha256'][label])
            require({k: v for k, v in result.items() if k != 'blocks'} == row, 'Summary/result mismatch: '+label)
            require(row['identity'] == dict(evaluation_identity_sha256=bindings[evaluation_name], selected=selected)
                    and row['checkpoint_step'] == selected['checkpoint_step'], 'Selected checkpoint linkage changed')
        by_run = {(r['condition']['graph'], r['condition']['seed']): r for r in summary['runs']}
        # Verify every declared point difference against the completed pooled loss
        # summaries. The scorer already validates block alignment and denominators.
        for row in summary['paired_comparisons']:
            def loss(graph):
                value = by_run[graph, row['training_seed']]['components']['all'][row['subset']]
                require(value is not None, 'Undefined pooled analysis')
                return finite(value['bits_per_byte'])
            point = math.fsum(loss(g) for g in row['first'])/len(row['first']) - math.fsum(loss(g) for g in row['second'])/len(row['second'])
            require(isinstance(row['estimate'], dict) and
                    math.isclose(finite(row['estimate']['difference_bpb']), point, rel_tol=0, abs_tol=1e-10),
                    'Contrast and pooled loss arithmetic disagree')
        # Retain all original source aggregates and across-seed summaries, even
        # though the allocation rule only consumes pooled candidate contrasts.
        aggregates = {}
        for row in summary['aggregates']:
            key = (row['graph'], row['component'], row['subset'])
            require(key not in aggregates, 'Duplicate source aggregate')
            aggregates[key] = row
        expected_aggregates = set()
        for graph in {c['graph'] for c in conditions}:
            for component in by_run[graph, 42]['components']:
                for subset in rule['analysis_subsets']:
                    key = graph, component, subset
                    expected_aggregates.add(key)
                    values = [by_run[graph, seed]['components'][component][subset] for seed in (42, 43)]
                    values = [v['bits_per_byte'] for v in values if v is not None]
                    require(aggregates[key]['seeds'] == [42, 43] and aggregates[key]['bits_per_byte'] == values,
                            'Source aggregate values changed')
        require(set(aggregates) == expected_aggregates, 'Source aggregate inventory changed')
        paired_seeds = {}
        for row in summary['paired_comparisons']:
            key = tuple(row[k] for k in ('group', 'subset', 'kind', 'level', 'name'))
            paired_seeds.setdefault(key, {})[row['training_seed']] = row['estimate']['difference_bpb']
        seen = set()
        for row in summary['paired_seed_summary']:
            key = tuple(row[k] for k in ('group', 'subset', 'kind', 'level', 'name'))
            require(key not in seen and row['seeds'] == [42, 43]
                    and row['difference_bpb'] == [paired_seeds[key][s] for s in (42, 43)],
                    'Across-seed summary changed')
            seen.add(key)
        require(seen == set(paired_seeds), 'Incomplete across-seed summary inventory')
        output.update(apply_rule(rule, summary['paired_comparisons']))
        output['complete_conditions'] = len(conditions)
        output['declared_contrasts'] = len(summary['paired_comparisons'])
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        output.update(decision='invalid', reason=str(error),
                      action='Preserve the record; follow frozen recovery rules. No scientific failure or continuation pass.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mac-study', action='store_true', help='Consume the complete fresh Mac cohort and verify its original-study lineage')
    parser.add_argument('--output', type=Path, help='Optional new report path; refuses to overwrite different bytes')
    args = parser.parse_args()
    result = report(args.root, mac=args.mac_study)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            require(args.output.read_bytes() == payload.encode(), 'Output already exists with different contents')
        else:
            with args.output.open('x', encoding='utf8', newline='\n') as stream:
                stream.write(payload)
    print(payload, end='')
    return 2 if result['decision'] == 'invalid' else 0


if __name__ == '__main__':
    raise SystemExit(main())
