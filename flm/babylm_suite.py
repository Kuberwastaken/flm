"""Run the registered BabyLM compact comparisons serially, with checkpoint resume."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scales', nargs='+', choices=('10m', '100m'), default=['10m', '100m'])
    p.add_argument('--seeds', nargs='+', type=int, choices=(42, 43), default=[42, 43])
    p.add_argument('--variants', nargs='+', choices=('flm', 'gru', 'transformer'), default=['flm', 'gru', 'transformer'])
    a = p.parse_args()
    # Preserve the isolated WikiText timing study before beginning the next training queue.
    if not Path('reports/wikitext2/runtime.json').exists():
        raise ValueError('Finish WikiText training and its isolated runtime benchmark first')
    root = Path('data/processed/babylm-2026-bpe'); tokenizer = Path('data/tokenizers/babylm-2026-4096')
    protocol_path = Path('docs/BABYLM-PROTOCOL.md')
    for scale in a.scales:
        output = Path(f'runs/babylm-{scale}'); declaration = output / 'study.json'
        study = dict(dataset=f'BabyLM 2026 {scale}', steps=12000, seeds=[42, 43], variants=['flm', 'gru', 'transformer'],
            protocol_sha256=sha256(protocol_path), tokenizer_sha256=sha256(tokenizer / 'tokenizer.json'),
            train_manifest_sha256=sha256(root / f'train-{scale}' / 'manifest.json'),
            validation_manifest_sha256=sha256(root / 'validation' / 'manifest.json'),
            validation_panel_sha256=sha256(tokenizer / 'validation-panel.json'))
        if declaration.exists() and json.loads(declaration.read_text(encoding='utf8')) != study:
            raise ValueError('The registered BabyLM study changed; declare a separate experiment')
        write_json(declaration, study)
        for seed in a.seeds:
            for variant in a.variants:
                folder = output / f'{variant}-s{seed}'; complete = folder / 'complete.json'
                if complete.exists():
                    done = json.loads(complete.read_text(encoding='utf8'))
                    if done['steps'] != 12000 or done['best_checkpoint_sha256'] != sha256(folder / 'best.pt'):
                        raise ValueError('Completed run identity is invalid')
                    continue
                command = [sys.executable, '-X', 'utf8', '-m', 'flm.language_train', '--steps', '12000',
                    '--variant', variant, '--seed', str(seed), '--data', str(root), '--unit', 'block',
                    '--training-cache', str(root / f'train-{scale}'), '--validation-cache', str(root / 'validation'),
                    '--validation-panel', str(tokenizer / 'validation-panel.json'),
                    '--tokenizer', str(tokenizer / 'tokenizer.json'), '--output', str(folder)]
                if (folder / 'run.json').exists():
                    if not (folder / 'last.pt').exists():
                        raise ValueError(f'Run exists without a recoverable checkpoint: {folder}')
                    command += ['--resume', str(folder / 'last.pt')]
                print(json.dumps(dict(event='start', scale=scale, seed=seed, variant=variant, command=command)), flush=True)
                subprocess.run(command, check=True)


if __name__ == '__main__': main()
