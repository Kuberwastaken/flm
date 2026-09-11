"""Check complete BabyLM likelihood artifacts and export byte-weighted tables.

Uses saved measurements and text-free block metadata only. No model, corpus
token payload or bootstrap is executed. Reported interval endpoints are retained
with explicit provenance; their resampling is not independently reproduced here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

RUNS = tuple((scale, seed, variant) for scale in ('10m', '100m')
             for seed in (42, 43) for variant in ('flm', 'gru', 'transformer'))
COMPONENTS = ('bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard')
SUBSETS = ('official', 'overlap_filtered')
EVALUATOR_SOURCES = {'babylm_test.py', 'corpus_evaluation.py', 'language_train.py', 'model.py', 'baselines.py'}
REPORTS = Path('reports/babylm')
CACHE = Path('data/processed/babylm-2026-bpe/test')


def require(condition, message):
    if not condition: raise ValueError(message)


def key(row): return row['scale'], row['seed'], row['variant']
def digest(data): return hashlib.sha256(data).hexdigest()


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            'Arithmetic mismatch: ' + label)


def indexed(rows, fields, label):
    result = {tuple(row[field] for field in fields): row for row in rows}
    require(len(result) == len(rows), 'Duplicate ' + label)
    return result


def pooled(blocks):
    if not blocks: return None
    nll = math.fsum(row['nll'] for row in blocks)
    size = sum(row['bytes'] for row in blocks)
    tokens = sum(row['tokens'] for row in blocks)
    return dict(blocks=len(blocks), nll=nll, tokens=tokens, bytes=size,
                bits_per_byte=nll / size / math.log(2), token_perplexity=math.exp(nll / tokens))


def compare_score(actual, expected):
    if expected is None:
        require(actual is None, 'Empty overlap-filtered component must remain null')
        return
    require(isinstance(actual, dict) and set(actual) == set(expected), 'Score field inventory changed')
    for name, value in expected.items():
        if name in ('blocks', 'tokens', 'bytes'):
            require(type(actual[name]) is int and actual[name] == value, 'Score denominator changed')
        else: close(actual[name], value, name)


def derive(selection, results, summary, metadata, manifest):
    """Recompute arithmetic independently of the inference/summary implementation."""
    selected = indexed(selection['runs'], ('scale', 'seed', 'variant'), 'selection')
    runs = indexed(results, ('scale', 'seed', 'variant'), 'run')
    require(set(selected) == set(runs) == set(RUNS), 'Require all twelve registered runs')
    require(len(metadata) == manifest['blocks'] and len({r['id'] for r in metadata}) == len(metadata),
            'Prepared block inventory changed')
    require({r['component'] for r in metadata} == set(COMPONENTS), 'Require all six corpus components')
    expected_blocks = {}
    for row in metadata:
        require(type(row['overlap_filtered_eligible']) is bool, 'Nonboolean overlap flag')
        require(all(type(row[n]) is int for n in ('token_start', 'token_end', 'utf8_bytes')), 'Noninteger prepared count')
        tokens = row['token_end'] - row['token_start'] - 2
        require(tokens > 0 and row['utf8_bytes'] > 0, 'Invalid prepared block counts')
        expected_blocks[row['id']] = dict(component=row['component'], bytes=row['utf8_bytes'],
            tokens=tokens, overlap_filtered_eligible=row['overlap_filtered_eligible'])
    require(sum(r['bytes'] for r in expected_blocks.values()) == manifest['utf8_bytes']
            and sum(r['tokens'] for r in expected_blocks.values()) == manifest['text_tokens'],
            'Prepared corpus totals changed')
    rows = []; recomputed = {}
    for identity in RUNS:
        result = runs[identity]; chosen = selected[identity]
        require(all(result.get(name) == value for name, value in chosen.items()), 'Result selection changed')
        require(all(result['identity'].get(name) == value for name, value in chosen.items()), 'Run identity changed')
        require(result['identity']['test_manifest_sha256'] == selection['manifest_sha256']['test'],
                'Run test manifest changed')
        require(type(result['parameters']) is int and result['parameters'] > 0, 'Invalid parameter count')
        exposure = chosen['final_exposure']
        require(type(chosen['checkpoint_step']) is int and chosen['checkpoint_step'] in range(500, 12001, 500),
                'Invalid selected checkpoint update')
        require(exposure['presented_tokens'] == 18432000
                and all(type(exposure[n]) is int and exposure[n] > 0 for n in ('presented_tokens', 'presented_bytes', 'scored_bytes'))
                and exposure['scored_bytes'] <= exposure['presented_bytes'], 'Invalid final training exposure')
        blocks = indexed(result['blocks'], ('document',), 'scored block')
        require(set(blocks) == {(name,) for name in expected_blocks}, 'Scored block coverage changed')
        for (name,), block in blocks.items():
            expected = expected_blocks[name]
            require(all(block[field] == value and type(block[field]) is type(value)
                        for field, value in expected.items()), 'Scored block metadata changed')
            require(type(block['nll']) in (int, float) and math.isfinite(block['nll']) and block['nll'] >= 0,
                    'Invalid block likelihood')
            close(block['bits_per_byte'], block['nll'] / block['bytes'] / math.log(2), 'block BPB')
        require(set(result['components']) == {'all', *COMPONENTS}, 'Component inventory changed')
        for component in ('all', *COMPONENTS):
            group = [r for r in result['blocks'] if component == 'all' or r['component'] == component]
            excluded = [r for r in group if not r['overlap_filtered_eligible']]
            saved = result['components'][component]
            require(saved['excluded_blocks'] == len(excluded)
                    and saved['excluded_bytes'] == sum(r['bytes'] for r in excluded), 'Overlap exclusions changed')
            for subset in SUBSETS:
                score = pooled(group if subset == 'official' else [r for r in group if r['overlap_filtered_eligible']])
                compare_score(saved[subset], score)
                recomputed[(*identity, component, subset)] = score
                rows.append(dict(scale=identity[0], seed=identity[1], variant=identity[2],
                    component=component, subset=subset, available=score is not None,
                    **(score or {name: None for name in ('blocks', 'nll', 'tokens', 'bytes', 'bits_per_byte', 'token_perplexity')}),
                    excluded_blocks=len(excluded), excluded_bytes=sum(r['bytes'] for r in excluded),
                    parameters=result['parameters'], checkpoint_step=chosen['checkpoint_step'],
                    checkpoint_sha256=chosen['checkpoint_sha256'],
                    final_presented_tokens=exposure['presented_tokens'], final_presented_bytes=exposure['presented_bytes'],
                    final_scored_bytes=exposure['scored_bytes']))
    # Require summary run records to preserve every original field except blocks.
    summary_runs = indexed(summary['runs'], ('scale', 'seed', 'variant'), 'summary run')
    require(set(summary_runs) == set(runs), 'Summary run inventory changed')
    for identity, result in runs.items():
        require(summary_runs[identity] == {k: v for k, v in result.items() if k != 'blocks'},
                'Summary run differs from its measured artifact')
    aggregates = indexed(summary['aggregates'], ('scale', 'subset', 'component', 'variant'), 'aggregate')
    means = []; expected_aggregates = set()
    for scale in ('10m', '100m'):
        for subset in SUBSETS:
            for component in ('all', *COMPONENTS):
                for variant in ('flm', 'gru', 'transformer'):
                    scores = [recomputed[(scale, seed, variant, component, subset)] for seed in (42, 43)]
                    if scores[0] is None: continue
                    identity = scale, subset, component, variant; expected_aggregates.add(identity)
                    require(identity in aggregates, 'Missing seed aggregate')
                    saved = aggregates[identity]; values = [r['bits_per_byte'] for r in scores]
                    require(saved['seeds'] == [42, 43] and len(saved['bits_per_byte']) == 2, 'Seed inventory changed')
                    for actual, value in zip(saved['bits_per_byte'], values): close(actual, value, 'seed BPB')
                    mean = statistics.mean(values); sd = statistics.stdev(values)
                    close(saved['mean_bpb'], mean, 'seed mean'); close(saved['seed_standard_deviation'], sd, 'seed SD')
                    means.append(dict(scale=scale, subset=subset, component=component, variant=variant,
                        seed_42_bpb=values[0], seed_43_bpb=values[1], mean_bpb=mean,
                        seed_standard_deviation=sd, scored_bytes_per_seed=scores[0]['bytes']))
    require(set(aggregates) == expected_aggregates, 'Unexpected seed aggregates')
    pairs = indexed(summary['paired_comparisons'], ('scale', 'subset', 'training_seed', 'first', 'second'), 'comparison')
    expected_pairs = {(s, sub, seed, 'flm', other) for s in ('10m', '100m') for sub in SUBSETS
                      for seed in (42, 43) for other in ('gru', 'transformer')}
    require(set(pairs) == expected_pairs, 'Comparison inventory changed')
    comparisons = []
    for identity in sorted(expected_pairs):
        scale, subset, seed, first, second = identity; saved = pairs[identity]
        a = recomputed[(scale, seed, first, 'all', subset)]
        b = recomputed[(scale, seed, second, 'all', subset)]
        require(a is not None and b is not None, 'Empty whole-mixture comparison')
        close(saved['difference_bpb'], a['bits_per_byte'] - b['bits_per_byte'], 'paired difference')
        require(saved['replicates'] == 10000 and saved['seed'] == 31415
                and saved['unit'] == 'paired blocks, resampled within each source component', 'Interval method changed')
        require(all(type(saved[n]) in (int, float) and math.isfinite(saved[n]) for n in ('lower_95', 'upper_95'))
                and saved['lower_95'] <= saved['upper_95'], 'Invalid interval endpoints')
        comparisons.append(dict(saved, interval_verification='Copied from bound evaluator summary; resampling not rerun by this exporter'))
    return dict(scores=rows, seed_aggregates=means, paired_comparisons=comparisons,
        interpretation=['Lower bits per UTF-8 byte is better; pooled scores weight bytes, not components equally.',
            'Official and overlap-filtered analyses remain separate; an empty filtered component has no score.',
            'Seed SD describes two trained runs, not a confidence interval. Block intervals are conditional and not independent-document intervals.',
            'Equal updates expose different fractions of the 10M/100M corpora. These scores do not establish chatbot or biological capability.'])


def collect(root):
    root = Path(root)
    paths = [REPORTS/'selection.json', REPORTS/'summary.json'] + [
        REPORTS/f'test-{scale}-{variant}-s{seed}.json' for scale, seed, variant in RUNS]
    # Gate before opening even the test metadata; never make a partial scoreboard.
    require(all((root/p).is_file() for p in paths), 'Complete BabyLM evaluation artifacts are required')
    paths += [CACHE/'manifest.json', CACHE/'blocks.jsonl']
    raw = {p: (root/p).read_bytes() for p in paths}
    selection = json.loads(raw[paths[0]]); summary = json.loads(raw[paths[1]])
    manifest = json.loads(raw[CACHE/'manifest.json'])
    require(digest(raw[CACHE/'manifest.json']) == selection['manifest_sha256']['test'], 'Test manifest hash changed')
    require(digest(raw[CACHE/'blocks.jsonl']) == manifest['files']['blocks.jsonl'], 'Prepared metadata hash changed')
    metadata = [json.loads(line) for line in raw[CACHE/'blocks.jsonl'].splitlines()]
    results = [json.loads(raw[p]) for p in paths[2:14]]
    source_hashes = {}
    for result in results:
        require(result['identity']['selection_sha256'] == digest(raw[paths[0]]), 'Selection hash changed')
        require(set(result['identity']['evaluator_source_sha256']) == EVALUATOR_SOURCES, 'Evaluator source inventory changed')
        for name, expected in result['identity']['evaluator_source_sha256'].items():
            require(Path(name).name == name and name.endswith('.py'), 'Invalid evaluator source path')
            p = Path('flm')/name; source = (root/p).read_bytes()
            require(digest(source) == expected, 'Evaluator source changed: ' + name)
            source_hashes[p.as_posix()] = expected
    require(set(source_hashes) == {'flm/'+n for n in EVALUATOR_SOURCES},
            'Evaluator source inventory changed')
    report = derive(selection, results, summary, metadata, manifest)
    require(all((root/p).read_bytes() == data for p, data in raw.items()), 'An input changed during export')
    require(all(digest((root/p).read_bytes()) == value for p, value in source_hashes.items()), 'A source changed during export')
    report.update(input_sha256={p.as_posix(): digest(data) for p, data in raw.items()},
        evaluator_source_sha256=source_hashes, exporter_sha256=digest(Path(__file__).read_bytes()),
        scope='Arithmetic and artifact consistency check. No weight restoration, inference, token-payload read, or bootstrap replication.')
    return report


def export(root, output):
    output = Path(output)
    require(not output.exists(), 'Preserve existing result tables; choose a fresh destination')
    report = collect(root)
    files = {'tables.json': (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()}
    for name in ('scores', 'seed_aggregates', 'paired_comparisons'):
        stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=list(report[name][0]))
        writer.writeheader(); writer.writerows(report[name]); files[name+'.csv'] = stream.getvalue().encode()
    files['manifest.json'] = (json.dumps({n: dict(bytes=len(d), sha256=digest(d)) for n, d in files.items()}, indent=2)+'\n').encode()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items(): (output/name).write_bytes(data)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = export(args.root, args.output)
    print(f"Exported {len(result['scores'])} run/component/subset rows with exact denominators.")
