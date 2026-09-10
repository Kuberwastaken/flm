"""Publish a dated validation snapshot without loading weights or test losses."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from .babylm_test import SCALES, SEEDS, VARIANTS, TOKENIZER, CACHE, read_json
from .provenance import sha256, write_json


def collect(root=Path('.')):
    tokenizer_hash = sha256(root / TOKENIZER)
    panel_path = TOKENIZER.with_name('validation-panel.json'); panel = read_json(root / panel_path)
    metadata = {row['id']: row for row in map(json.loads,
        (root / CACHE / 'validation/blocks.jsonl').read_text(encoding='utf8').splitlines())}
    token_card = read_json(root / TOKENIZER.with_name('tokenization-card.json'))
    records = []
    for scale in SCALES:
        for seed in SEEDS:
            for variant in VARIANTS:
                folder = root / f'runs/babylm-{scale}/{variant}-s{seed}'
                record = dict(scale=scale, seed=seed, variant=variant, complete=False, parameters=None, validation=[])
                if (folder / 'run.json').exists():
                    run = read_json(folder / 'run.json')
                    if run['protocol']['tokenizer_sha256'] != tokenizer_hash or run['protocol']['validation_panel_sha256'] != sha256(root / panel_path):
                        raise ValueError('Validation run does not use the registered tokenizer and panel')
                    record.update(parameters=run['parameter_card']['trainable_parameters'], source_commit=run['source_commit'], protocol=run['protocol'])
                    for path in [folder / 'initial-validation.json'] + sorted(folder.glob('validation-*.json')):
                        if not path.exists(): continue
                        data = read_json(path); step = 0 if path.name.startswith('initial') else int(path.stem.split('-')[-1])
                        if [r['document'] for r in data['documents']] != panel['ids'] or data['tokenizer_sha256'] != tokenizer_hash:
                            raise ValueError('Validation point belongs to a different panel')
                        components = {}
                        for component in sorted({metadata[i]['component'] for i in panel['ids']}):
                            chosen = [r for r in data['documents'] if metadata[r['document']]['component'] == component]
                            nll = math.fsum(r['nll'] for r in chosen); tokens = sum(r['tokens'] for r in chosen); size = sum(r['bytes'] for r in chosen)
                            components[component] = dict(bits_per_byte=nll / size / math.log(2), token_perplexity=math.exp(nll / tokens), bytes=size, tokens=tokens)
                        components['all'] = {key: data[key] for key in ('bits_per_byte', 'token_perplexity', 'bytes', 'tokens')}
                        record['validation'].append(dict(step=step, presented_tokens=step * 1536, components=components, artifact_sha256=sha256(path)))
                    if (folder / 'complete.json').exists():
                        done = read_json(folder / 'complete.json')
                        if done['steps'] != 12000 or done['protocol'] != run['protocol'] or done['best_checkpoint_sha256'] != sha256(folder / 'best.pt'):
                            raise ValueError('Completed validation snapshot has changed')
                        record['complete'] = True
                records.append(record)
    return dict(dataset='BabyLM 2026 English', snapshot_utc=datetime.now(timezone.utc).isoformat(),
        registered_updates=12000, registered_runs=12, runs=records, tokenizer_sha256=tokenizer_hash,
        validation_panel_sha256=sha256(root / panel_path),
        training_text_tokens={scale: token_card['partitions'][f'train-{scale}']['text_tokens'] for scale in SCALES},
        measurement='Validation only, fixed 48-block panel; choose an update present for all three models. This snapshot is not a process monitor or a test result.',
        exposure='18,432,000 presented tokens per completed run, sampled with replacement; the 100M-word corpus receives proportionally less coverage.',
        planned_continuations=288)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('public/research/babylm-validation.json'))
    args = parser.parse_args(); report = collect(); write_json(args.output, report)
    print(f'Published {sum(r["complete"] for r in report["runs"])} completed and {sum(bool(r["validation"]) and not r["complete"] for r in report["runs"])} partial runs.')


if __name__ == '__main__': main()
