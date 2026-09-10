"""Freeze all twelve BabyLM runs, then score the complete held-out corpus."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
from .corpus_cache import read_mmap
from .corpus_evaluation import evaluate_batch, aggregate, component_summary, paired_block_interval
from .language_train import restore
from .provenance import sha256, write_json
from .tokenizer import Lexicon

SCALES = ('10m', '100m')
SEEDS = (42, 43)
VARIANTS = ('flm', 'gru', 'transformer')
TOKENIZER = Path('data/tokenizers/babylm-2026-4096/tokenizer.json')
GRAPH = Path('data/graphs/central-1024/graph.npz')
CACHE = Path('data/processed/babylm-2026-bpe')
OUTPUT = Path('reports/babylm')


def read_json(path): return json.loads(path.read_text(encoding='utf8'))


def freeze_selection(root=Path('.')):
    """Check completeness first, then every declared identity; never read losses."""
    folders = [(scale, seed, variant, Path(f'runs/babylm-{scale}/{variant}-s{seed}'))
               for scale in SCALES for seed in SEEDS for variant in VARIANTS]
    for _, _, _, folder in folders:
        if not all((root / folder / name).exists() for name in ('complete.json', 'best.pt')):
            raise ValueError(f'Registered BabyLM run incomplete: {folder}')
    tokenizer_hash = sha256(root / TOKENIZER); graph_hash = sha256(root / GRAPH)
    card_path = TOKENIZER.with_name('tokenization-card.json'); card = read_json(root / card_path)
    if card['tokenizer_sha256'] != tokenizer_hash: raise ValueError('Tokenization card changed')
    manifest_hashes = {}
    for partition in ('train-10m', 'train-100m', 'validation', 'test'):
        path = root / CACHE / partition
        _, _, manifest = read_mmap(path, tokenizer_hash)
        if manifest != card['partitions'][partition]: raise ValueError('Prepared corpus differs from its frozen card')
        manifest_hashes[partition] = sha256(path / 'manifest.json')
    panel_hash = sha256(root / TOKENIZER.with_name('validation-panel.json'))
    protocol_hash = sha256(root / 'docs/BABYLM-PROTOCOL.md')
    protocols = {}; studies = {}
    for scale in SCALES:
        study_path = root / f'runs/babylm-{scale}/study.json'
        study = read_json(study_path)
        expected = dict(dataset=f'BabyLM 2026 {scale}', steps=12000, seeds=list(SEEDS), variants=list(VARIANTS),
            protocol_sha256=protocol_hash, tokenizer_sha256=tokenizer_hash,
            train_manifest_sha256=manifest_hashes[f'train-{scale}'],
            validation_manifest_sha256=manifest_hashes['validation'], validation_panel_sha256=panel_hash)
        if study != expected: raise ValueError('Registered BabyLM declaration changed')
        studies[scale] = sha256(study_path)
        protocols[scale] = dict(steps=12000, batch=16, sequence=96, warmup=16, learning_rate=.002,
            final_learning_rate=.0002, lr_warmup_updates=100, weight_decay=.01, eval_tokens=49152,
            threads=4, tokenizer_sha256=tokenizer_hash, train_cache_sha256=manifest_hashes[f'train-{scale}'],
            validation_cache_sha256=manifest_hashes['validation'], graph_sha256=graph_hash,
            validation_panel_sha256=panel_hash, evaluation_unit='block', cache_identity='SHA-256 of verified mmap manifest')
    runs = []
    for scale, seed, variant, folder in folders:
        complete = read_json(root / folder / 'complete.json'); run = read_json(root / folder / 'run.json')
        if complete['steps'] != 12000 or complete['protocol'] != protocols[scale] or run['protocol'] != protocols[scale]:
            raise ValueError(f'Completed run protocol changed: {folder}')
        digest = sha256(root / folder / 'best.pt')
        if complete['best_checkpoint_sha256'] != digest or not math.isfinite(complete['best_validation_bpb']):
            raise ValueError(f'Invalid selected checkpoint: {folder}')
        if run['seed'] != seed or run['parameter_card']['config']['variant'] != variant or run['test_set_used_for_training']:
            raise ValueError(f'Wrong registered run identity: {folder}')
        runs.append(dict(scale=scale, seed=seed, variant=variant, checkpoint=(folder / 'best.pt').as_posix(),
            checkpoint_sha256=digest, selection_validation_bpb=complete['best_validation_bpb'],
            training_source_commit=run['source_commit']))
    return dict(dataset='BabyLM 2026 English', runs=runs, protocols=protocols, study_sha256=studies,
        tokenizer_sha256=tokenizer_hash, graph_sha256=graph_hash, manifest_sha256=manifest_hashes,
        tokenization_card_sha256=sha256(root / card_path), training_protocol_sha256=protocol_hash,
        evaluation_protocol_sha256=sha256(root / 'docs/BABYLM-EVALUATION.md'),
        prompts_sha256=sha256(root / 'data/prompts/babylm-original.json'),
        selection='Lowest fixed-panel validation bits/byte during each registered 12000-update run; all twelve frozen before test likelihoods')


def restore_selected(selected, selection, lexicon):
    model, saved = restore(Path(selected['checkpoint']), GRAPH, lexicon)
    if saved['_file_sha256'] != selected['checkpoint_sha256'] or saved['run']['protocol'] != selection['protocols'][selected['scale']]:
        raise ValueError('Checkpoint changed after selection freeze')
    if saved['run']['seed'] != selected['seed'] or saved['config']['variant'] != selected['variant']:
        raise ValueError('Checkpoint belongs to another run')
    if saved['best'] != selected['selection_validation_bpb'] or not 500 <= saved['step'] <= 12000 or saved['step'] % 500:
        raise ValueError('Checkpoint does not represent the declared validation selection')
    return model, saved


def scored_blocks(model, documents, metadata, lexicon, destination, identity, batch_size=8, chunk_size=96):
    """Atomic per-batch recovery with exact text counts, never partially scored blocks."""
    if batch_size < 1 or chunk_size < 1: raise ValueError('Require positive batch and chunk sizes')
    lookup = {r['id']: r for r in metadata}; doc_lookup = dict(documents)
    if len(lookup) != len(metadata) or len(doc_lookup) != len(documents) or set(lookup) != set(doc_lookup):
        raise ValueError('Block inventory differs from the cache')
    declaration = dict(identity=identity, batch_size=batch_size, chunk_size=chunk_size,
        block_order='ascending target length, then stable ID')
    marker = destination / 'identity.json'
    if marker.exists() and read_json(marker) != declaration: raise ValueError('Evaluation resume identity changed')
    write_json(marker, declaration)
    ordered = sorted(documents, key=lambda item: (len(item[1]), item[0])); records = []; seconds = 0.
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]; path = destination / f'batch-{start // batch_size:05d}.json'
        if path.exists():
            measured = read_json(path)
            if measured.get('identity') != declaration: raise ValueError('Cached batch belongs to a different evaluation')
        else:
            measured = dict(evaluate_batch(model, batch, lexicon, chunk_size), identity=declaration)
            write_json(path, measured)
        if [r['document'] for r in measured['documents']] != [identity for identity, _ in batch]:
            raise ValueError('Cached batch block order changed')
        aggregate(measured['documents'])
        for record, (_, document) in zip(measured['documents'], batch):
            meta = lookup[record['document']]
            if record['bytes'] != meta['utf8_bytes'] or record['tokens'] != len(document) - 2:
                raise ValueError('Scored text counts do not cover the complete prepared block')
            records.append(dict(record, component=meta['component'],
                overlap_filtered_eligible=meta['overlap_filtered_eligible']))
        seconds += measured['seconds']
        print(json.dumps(dict(event='test_progress', blocks=len(records), total_blocks=len(ordered),
                              scored_tokens=sum(r['tokens'] for r in records))), flush=True)
    by_id = {r['document']: r for r in records}
    return [by_id[identity] for identity, _ in documents], seconds


def summarize(results):
    aggregates = []; comparisons = []
    for scale in SCALES:
        for subset in ('official', 'overlap_filtered'):
            for component in results[0]['components']:
                for variant in VARIANTS:
                    values = [r['components'][component][subset]['bits_per_byte'] for r in results
                              if r['scale'] == scale and r['variant'] == variant and r['components'][component][subset] is not None]
                    if len(values) == len(SEEDS):
                        aggregates.append(dict(scale=scale, subset=subset, component=component, variant=variant,
                            seeds=list(SEEDS), bits_per_byte=values, mean_bpb=float(np.mean(values)),
                            seed_standard_deviation=float(np.std(values, ddof=1))))
            for seed in SEEDS:
                chosen = {r['variant']: r for r in results if r['scale'] == scale and r['seed'] == seed}
                blocks = {v: [r for r in chosen[v]['blocks'] if subset == 'official' or r['overlap_filtered_eligible']] for v in VARIANTS}
                for variant in ('gru', 'transformer'):
                    comparisons.append(dict(scale=scale, subset=subset, training_seed=seed, first='flm', second=variant,
                        **paired_block_interval(blocks['flm'], blocks[variant])))
    return dict(dataset='BabyLM 2026 English', aggregates=aggregates, paired_comparisons=comparisons,
        runs=[{k: v for k, v in r.items() if k != 'blocks'} for r in results],
        caution='Equal updates sample different fractions of the two training corpora. Two seeds provide limited training uncertainty. Artificial blocks have unknown document dependence. Component perplexity does not establish chatbot ability.')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1: p.error('Positive threads required')
    selection = freeze_selection(); frozen = OUTPUT / 'selection.json'
    if frozen.exists() and read_json(frozen) != selection: raise ValueError('Another BabyLM selection is already frozen')
    write_json(frozen, selection); torch.set_num_threads(a.threads)
    lexicon = Lexicon(TOKENIZER); documents, metadata, manifest = read_mmap(CACHE / 'test', lexicon.sha256)
    code = {name: sha256(Path(__file__).with_name(name)) for name in
            ('babylm_test.py', 'corpus_evaluation.py', 'language_train.py', 'model.py', 'baselines.py')}
    common = dict(selection_sha256=sha256(frozen), test_manifest_sha256=selection['manifest_sha256']['test'],
        evaluator_source_sha256=code, threads=a.threads, dtype='float32', torch_version=str(torch.__version__))
    results = []
    for selected in selection['runs']:
        model, saved = restore_selected(selected, selection, lexicon)
        identity = dict(**common, **selected)
        records, seconds = scored_blocks(model, documents, metadata, lexicon,
            Path('runs/babylm-evaluation') / selected['scale'] / f'{selected["variant"]}-s{selected["seed"]}', identity)
        total = aggregate(records)
        if total['tokens'] != manifest['text_tokens'] or total['bytes'] != manifest['utf8_bytes']:
            raise ValueError('Evaluation failed complete-test coverage')
        result = dict(**selected, identity=identity, checkpoint_step=saved['step'],
            parameters=model.parameter_card()['trainable_parameters'], components=component_summary(records),
            blocks=records, seconds=seconds, timing='Scoring wall time may overlap other work; not an isolated speed comparison',
            evaluator_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        write_json(OUTPUT / f'test-{selected["scale"]}-{selected["variant"]}-s{selected["seed"]}.json', result)
        results.append(result)
        print(json.dumps(dict(event='test_complete', scale=selected['scale'], variant=selected['variant'], seed=selected['seed'], bits_per_byte=total['bits_per_byte'])), flush=True)
    write_json(OUTPUT / 'summary.json', summarize(results))


if __name__ == '__main__': main()
