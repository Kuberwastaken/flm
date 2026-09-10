"""Publish the completed fixed-panel grammar comparison and all scored identities."""
from pathlib import Path
import json
import sys
import zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flm.provenance import sha256, write_json

source = ROOT / 'reports/wikitext2/blimp'; output = ROOT / 'public/research'; records = []
for variant in ('flm', 'gru', 'transformer'):
    for seed in (42, 43):
        path = source / f'{variant}-s{seed}.json'; value = json.loads(path.read_text(encoding='utf8'))
        if value['identity']['variant'] != variant or value['identity']['seed'] != seed or len(value['pairs']) != 6700:
            raise ValueError('Incomplete or mismatched grammar comparison')
        records.append({k: v for k, v in value.items() if k != 'pairs'} | {'artifact_sha256': sha256(path)})
if len({r['identity']['panel_sha256'] for r in records}) != 1: raise ValueError('Panels differ')
aggregates = []
for variant in ('flm', 'gru', 'transformer'):
    values = [r['macro_paradigm_accuracy'] for r in records if r['identity']['variant'] == variant]
    aggregates.append(dict(variant=variant, accuracies=values, mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1))))
archive = output / 'grammar-pair-scores.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(source.glob('*.json')): z.write(path, path.name)
summary = dict(dataset='WikiText-2-trained models on a fixed BLiMP panel', pairs=6700, paradigms=67,
    seeds=[42,43], source='https://github.com/alexwarstadt/blimp', aggregates=aggregates, runs=records,
    pair_scores_sha256=sha256(archive), panel_sha256=records[0]['identity']['panel_sha256'],
    caveat='100 pairs per paradigm, not the full 67,000-pair benchmark. Full-sentence likelihood, reset per sentence, no length normalization, strict preference. This panel did not select checkpoints or training settings.')
write_json(output / 'grammar-results.json', summary)
print('Published the fixed-panel grammar comparison and every pair score.')
