"""Whole-inventory held-out likelihoods for frozen neuron-selection groups."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .babylm_test import scored_blocks
from .corpus_evaluation import aggregate, component_summary, paired_block_interval
from .language_learning_test import load_test, validate_scores, immutable_json
from .language_learning_train import declaration, optimizer_for, restore, training_lease
from .provenance import sha256
from .selection_language_study import (IDENTITY, REPORTS, SELECTION, STUDY, SELECTORS, matrix,
    freeze_selection, prepared_run, read, unchanged as unchanged_study, verified_context)

EVALUATION = Path('runs/selection-language-evaluation-v1')
SUMMARY = REPORTS/'test-summary.json'
POLICY = dict(batch_size=8, chunk_size=96, threads=4, bootstrap_replicates=10000, bootstrap_seed=31415,
    subsets=['official','overlap_filtered'], boundary_targets_scored=False, evaluation_warmup=0,
    state='Reset at each block; carry across chunks within that block',
    selection_comparisons='Candidate minus ranked, mean of three uniform selections, and mean of three stratified selections; also retain all seven individual contrasts',
    topology_comparisons='Measured minus mean of three rewires for every selector; also retain all 24 individual contrasts',
    averaging='Arithmetic mean of per-block negative log likelihoods, not a probability ensemble',
    uncertainty='Paired blocks within source, conditional on each training seed and the selected graphs/checkpoints; unadjusted descriptive intervals. Do not count rewires or node selections as independent training replications.')


def comparison_plan(group):
    """Eleven family contrasts and 31 individual contrasts per group/seed/subset."""
    def label(selector, suffix='measured'): return f'{group}/{selector}/{suffix}'
    result = []
    for family, selectors in (('ranked',('contact_ranked',)),
            ('uniform',('uniform_s201','uniform_s203','uniform_s207')),
            ('stratified',('stratified_s201','stratified_s203','stratified_s207'))):
        result.append(dict(kind='selection', level='family', name='candidate-minus-'+family,
            first=[label('candidate')], second=[label(s) for s in selectors]))
        for selector in selectors:
            result.append(dict(kind='selection', level='individual', name='candidate-minus-'+selector,
                first=[label('candidate')], second=[label(selector)]))
    for selector in SELECTORS:
        nulls = [label(selector,f'null{seed}') for seed in (101,103,107)]
        result.append(dict(kind='topology', level='family', name=selector+'-measured-minus-rewires',
            first=[label(selector)], second=nulls))
        for null in nulls:
            result.append(dict(kind='topology', level='individual', name=selector+'-measured-minus-'+null.split('/')[-1],
                first=[label(selector)], second=[null]))
    return result


def average_blocks(runs):
    if not runs: raise ValueError('Require a nonempty comparison family')
    first = runs[0]; aggregate(first)
    for records in runs[1:]:
        aggregate(records)
        if len(records) != len(first) or any(any(a[key] != b[key] for key in
                ('document','tokens','bytes','component','overlap_filtered_eligible')) for a,b in zip(first,records)):
            raise ValueError('Comparison family has unmatched blocks or eligibility')
    output = []
    for index, base in enumerate(first):
        nll = math.fsum(records[index]['nll'] for records in runs)/len(runs)
        output.append(dict(base, nll=nll, bits_per_byte=nll/base['bytes']/math.log(2)))
    return output


def summarize(results, identity, policy):
    expected = [r['condition'] for r in identity['conditions']]
    if [r['condition'] for r in results] != expected or not results:
        raise ValueError('Require every registered selection result in frozen order')
    # Cross-condition denominators, eligibility and ordering must agree, including
    # for contrasts not directly selected by the comparison plan.
    for row in results:
        average_blocks([results[0]['blocks'],row['blocks']])
        if component_summary(row['blocks']) != row['components']:
            raise ValueError('Result component arithmetic changed')
    aggregate_rows = []; comparisons = []; point_differences = {}
    by_graph_seed = {(r['condition']['graph'],r['condition']['seed']):r for r in results}
    for subset in policy['subsets']:
        for graph in sorted({r['condition']['graph'] for r in results}):
            rows = [by_graph_seed[graph,seed] for seed in (42,43)]
            for component in sorted(rows[0]['components']):
                values = [r['components'][component][subset]['bits_per_byte'] for r in rows
                          if r['components'][component][subset] is not None]
                if values and len(values) != 2: raise ValueError('Seeds have unmatched component coverage')
                aggregate_rows.append(dict(graph=graph,component=component,subset=subset,seeds=[42,43],
                    bits_per_byte=values,mean_bpb=float(np.mean(values)) if values else None,
                    seed_standard_deviation=float(np.std(values,ddof=1)) if values else None))
        for group in sorted(identity['request']['groups']):
            for seed in (42,43):
                for comparison in comparison_plan(group):
                    def side(labels):
                        records = average_blocks([by_graph_seed[label,seed]['blocks'] for label in labels])
                        return records if subset=='official' else [r for r in records if r['overlap_filtered_eligible']]
                    first = side(comparison['first']); second = side(comparison['second'])
                    if bool(first) != bool(second): raise ValueError('Comparison subset coverage differs')
                    estimate = paired_block_interval(first,second,policy['bootstrap_replicates'],policy['bootstrap_seed']) if first else None
                    comparisons.append(dict(group=group,training_seed=seed,subset=subset,**comparison,estimate=estimate,
                        reason=None if estimate else 'No eligible blocks in frozen filtered subset'))
                    key = (group,subset,comparison['kind'],comparison['level'],comparison['name'])
                    point_differences.setdefault(key,[]).append(estimate['difference_bpb'] if estimate else None)
    across_seeds = []
    for (group,subset,kind,level,name),values in point_differences.items():
        if len(values)!=2 or any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError('Incomplete paired training-seed comparison')
        present = values[0] is not None
        across_seeds.append(dict(group=group,subset=subset,kind=kind,level=level,name=name,seeds=[42,43],
            difference_bpb=values,mean_difference_bpb=float(np.mean(values)) if present else None,
            seed_standard_deviation=float(np.std(values,ddof=1)) if present else None))
    return dict(aggregates=aggregate_rows,paired_comparisons=comparisons,paired_seed_summary=across_seeds,
        runs=[{k:v for k,v in r.items() if k!='blocks'} for r in results],
        caution='Negative first-minus-second BPB favors the first model/family. Selection methods differ in edge and parameter counts; rewires within a subset preserve allocation. Family means average losses across fixed graph samples, not probabilities. All individual outcomes remain visible. Intervals are descriptive and unadjusted, conditional on checkpoints and artificial blocks with unknown dependence. Two training seeds and three graph samples do not establish general biological or training robustness. No animal behavior or conversation ability is measured.')


def unchanged(root, identity, selection, identity_hash, selection_hash):
    unchanged_study(root,identity,identity_hash)
    if sha256(root/SELECTION) != selection_hash: raise ValueError('Frozen selection changed during evaluation')
    for row in selection['conditions']:
        folder = root/STUDY/row['condition']['label']
        for path, expected in ((root/row['checkpoint'],row['checkpoint_sha256']),
                (folder/'complete.json',row['complete_sha256']),
                (folder/'validation-selection.json',row['validation_selection_sha256'])):
            if sha256(path) != expected: raise ValueError('Selected run changed during evaluation')


def restore_selected(root,catalog,corpus,registered,selected,identity_hash):
    model,binding,settings = prepared_run(catalog,corpus,registered,identity_hash)
    declared = declaration(model,corpus.documents,corpus.lexicon,settings,'bptt',binding)
    path = root/selected['checkpoint']
    if sha256(path) != selected['checkpoint_sha256']: raise ValueError('Selected payload checksum changed')
    saved,_,_ = restore(path,model,optimizer_for(model,settings),declared,corpus.documents,corpus.lexicon.lengths,settings)
    if saved['step'] != selected['checkpoint_step']: raise ValueError('Selected update changed')
    return model


def score_study(root):
    root = Path(root); previous_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            selection = freeze_selection(root)
            with training_lease(root/STUDY), training_lease(root/EVALUATION):
                identity,catalog,corpus,_ = verified_context(root)
                identity_hash=sha256(root/IDENTITY); selection_hash=sha256(root/SELECTION)
                if selection != read(root/SELECTION) or selection['study_identity_sha256'] != identity_hash:
                    raise ValueError('Selection no longer belongs to the frozen study')
                if [r['condition'] for r in selection['conditions']] != matrix(catalog,identity['request']['groups']):
                    raise ValueError('Selected inventory differs from complete groups')
                policy=identity['evaluation_policy']
                if policy != POLICY: raise ValueError('Frozen selection evaluation policy changed')
                unchanged(root,identity,selection,identity_hash,selection_hash)
                torch.set_num_threads(policy['threads'])
                documents,metadata,_,coverage=load_test(root,corpus.lexicon,identity['test_metadata'])
                common=dict(study_identity_sha256=identity_hash,selection_sha256=selection_hash,
                    test_metadata=identity['test_metadata'],coverage=coverage,policy=policy,
                    source_sha256=identity['sources'],torch=str(torch.__version__),numpy=str(np.__version__))
                immutable_json(root/EVALUATION/'identity.json',common); evaluation_hash=sha256(root/EVALUATION/'identity.json')
                registered={r['condition']['label']:r for r in identity['conditions']}; results=[]; hashes={}
                for selected in selection['conditions']:
                    unchanged(root,identity,selection,identity_hash,selection_hash)
                    label=selected['condition']['label']
                    model=restore_selected(root,catalog,corpus,registered[label],selected,identity_hash)
                    run_identity=dict(evaluation_identity_sha256=evaluation_hash,selected=selected)
                    destination=root/EVALUATION/label
                    records,seconds=scored_blocks(model,documents,metadata,corpus.lexicon,destination,
                        run_identity,policy['batch_size'],policy['chunk_size'])
                    if not math.isfinite(seconds) or seconds<0: raise ValueError('Invalid scoring time')
                    components=validate_scores(records,coverage,metadata)
                    batches=sorted(destination.glob('batch-*.json'))
                    if [p.name for p in batches] != [f'batch-{i:05d}.json' for i in range(math.ceil(len(documents)/policy['batch_size']))]:
                        raise ValueError('Unexpected evaluation batch inventory')
                    if any(not math.isfinite(read(p)['seconds']) or read(p)['seconds']<0 for p in batches):
                        raise ValueError('Invalid cached batch time')
                    result=dict(condition=selected['condition'],identity=run_identity,checkpoint_step=selected['checkpoint_step'],
                        graph=registered[label]['base_binding']['graph'],components=components,blocks=records,seconds=seconds,
                        timing='Scoring wall time only; not a throughput comparison',batch_record_sha256={p.name:sha256(p) for p in batches})
                    report=root/REPORTS/'test'/label/'result.json'
                    immutable_json(report,result); hashes[label]=sha256(report); results.append(result)
                    print('Completed held-out selection likelihood: '+label,flush=True)
                unchanged(root,identity,selection,identity_hash,selection_hash)
                if load_test(root,corpus.lexicon,identity['test_metadata'])[3] != coverage:
                    raise ValueError('Test payload changed during evaluation')
                if verified_context(root)[0] != identity: raise ValueError('Study input context changed during evaluation')
                if sha256(root/EVALUATION/'identity.json') != evaluation_hash: raise ValueError('Evaluation identity changed')
                for row in results:
                    label=row['condition']['label']
                    if sha256(root/REPORTS/'test'/label/'result.json') != hashes[label]: raise ValueError('Completed result changed')
                    for name,expected in row['batch_record_sha256'].items():
                        if sha256(root/EVALUATION/label/name) != expected: raise ValueError('Cached batch changed')
                result=dict(evaluation_identity_sha256=evaluation_hash,**summarize(results,identity,policy),result_sha256=hashes)
                immutable_json(root/SUMMARY,result)
                return result
    finally: torch.set_num_threads(previous_threads)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    score_study(Path.cwd())
