"""Serially launch the registered lexical runs, resuming only explicit checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .provenance import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    p.add_argument('--variants', nargs='+', choices=['flm', 'gru', 'transformer', 'rewired', 'no_recurrence', 'no_slow'], default=['flm', 'gru', 'transformer'])
    p.add_argument('--output', type=Path, default=Path('runs/wikitext2'))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--after', type=Path, help='Wait for a preceding run completion manifest before starting this queue')
    a = p.parse_args()
    if a.after:
        print(f'Waiting for completion manifest: {a.after}', flush=True)
        while not a.after.exists(): time.sleep(20)
    for seed in a.seeds:
        for variant in a.variants:
            output = a.output / f'{variant}-s{seed}'
            complete, best, last = output / 'complete.json', output / 'best.pt', output / 'last.pt'
            if complete.exists():
                record = json.loads(complete.read_text(encoding='utf8'))
                if record['steps'] != 6000 or record['best_checkpoint_sha256'] != sha256(best):
                    raise ValueError(f'Inconsistent completion artifact: {output}')
                print(f'Already complete: {variant} seed {seed}', flush=True); continue
            command = [sys.executable, '-X', 'utf8', '-m', 'flm.language_train', '--variant', variant,
                '--seed', str(seed), '--steps', '6000', '--threads', str(a.threads), '--output', str(output)]
            if last.exists(): command += ['--resume', str(last)]
            print(f'Starting {variant} seed {seed}', flush=True)
            subprocess.run(command, check=True)
    print('Requested training runs completed. This queue never evaluates the test split.', flush=True)


if __name__ == '__main__': main()
