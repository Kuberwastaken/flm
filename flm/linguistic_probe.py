"""Score a frozen BLiMP panel using reset, full-sentence causal likelihoods."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
from .language_train import restore
from .provenance import sha256, write_json
from .tokenizer import Lexicon


@torch.no_grad()
def sentence_nll(model, sentences, lexicon, batch_size=32):
    if batch_size < 1 or not sentences: raise ValueError('Positive batch and nonempty sentences required')
    encoded = [[0] + lexicon.encode(text) for text in sentences]
    if any(len(tokens) < 2 for tokens in encoded): raise ValueError('Cannot score an empty sentence')
    order = sorted(range(len(encoded)), key=lambda i:len(encoded[i])); scores = [None] * len(encoded)
    was_training = model.training; model.eval()
    for start in range(0, len(order), batch_size):
        chosen = order[start:start + batch_size]; width = max(len(encoded[i]) for i in chosen)
        values = torch.zeros((len(chosen), width), dtype=torch.long)
        mask = torch.zeros((len(chosen), width - 1), dtype=torch.bool)
        for j, index in enumerate(chosen):
            values[j, :len(encoded[index])] = torch.tensor(encoded[index])
            mask[j, :len(encoded[index]) - 1] = True
        logits, _ = model(values[:, :-1], None)
        losses = F.cross_entropy(logits.flatten(0, 1), values[:, 1:].reshape(-1), reduction='none').reshape(mask.shape)
        totals = (losses * mask).sum(1)
        if not torch.isfinite(totals).all(): raise ValueError('Nonfinite sentence likelihood')
        for j, index in enumerate(chosen): scores[index] = float(totals[j])
    model.train(was_training); return scores


def panel_records(card):
    pairs = []
    for source in card['files']:
        path = Path(source['path'])
        if sha256(path) != source['sha256']: raise ValueError('BLiMP source changed')
        lookup = {str(r['pairID']): r for r in map(json.loads, path.read_text(encoding='utf8').splitlines())}
        for identity in source['selected_pair_ids']: pairs.append(lookup[str(identity)])
    if len(pairs) != card['selected_pairs']: raise ValueError('Incomplete diagnostic panel')
    return pairs


def aggregate(records):
    groups = {}
    for record in records:
        for kind, key in [('paradigm', record['paradigm']), ('category', record['category'])]:
            groups.setdefault((kind, key), []).append(record)
    result = [dict(kind=kind, name=key, pairs=len(rows), correct=sum(r['correct'] for r in rows),
        ties=sum(r['good_nll'] == r['bad_nll'] for r in rows), accuracy=sum(r['correct'] for r in rows) / len(rows))
        for (kind, key), rows in sorted(groups.items())]
    return dict(groups=result, macro_paradigm_accuracy=float(np.mean([r['accuracy'] for r in result if r['kind'] == 'paradigm'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, default=Path('reports/wikitext2/selection.json'))
    p.add_argument('--tokenizer', type=Path, default=Path('data/tokenizers/wikitext2-4096/tokenizer.json'))
    p.add_argument('--output', type=Path, default=Path('reports/wikitext2/blimp'))
    a = p.parse_args(); torch.set_num_threads(2); lexicon = Lexicon(a.tokenizer)
    panel_path = Path('data/cards/blimp-panel.json'); card = json.loads(panel_path.read_text(encoding='utf8'))
    if card['protocol_sha256'] != sha256(Path('docs/SYNTAX-PROTOCOL.md')): raise ValueError('Diagnostic protocol changed')
    selection = json.loads(a.selection.read_text(encoding='utf8')); pairs = panel_records(card)
    sentences = [r[key] for r in pairs for key in ('sentence_good', 'sentence_bad')]
    for run in selection['runs']:
        identity = dict(checkpoint_sha256=run['checkpoint_sha256'], tokenizer_sha256=lexicon.sha256,
            selection_sha256=sha256(a.selection), panel_sha256=sha256(panel_path), variant=run['variant'], seed=run['seed'])
        destination = a.output / f"{run['variant']}-s{run['seed']}.json"
        if destination.exists():
            previous = json.loads(destination.read_text(encoding='utf8'))
            if previous['identity'] != identity: raise ValueError('Existing diagnostic has different inputs')
            continue
        model, saved = restore(Path(run['checkpoint']), Path('data/graphs/central-1024/graph.npz'), lexicon)
        if saved['_file_sha256'] != run['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
        scores = sentence_nll(model, sentences, lexicon); records = []
        for index, pair in enumerate(pairs):
            good, bad = scores[2 * index:2 * index + 2]
            records.append(dict(paradigm=pair['UID'], pair_id=pair['pairID'], category=pair['linguistics_term'],
                good_nll=good, bad_nll=bad, margin_nats=bad - good, correct=good < bad))
        result = dict(identity=identity, **aggregate(records), pairs=records,
            evaluator_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            protocol='Full text-token likelihood from BOS; sentence reset; EOS excluded; no length normalization; strict preference; padding excluded.',
            limits='Frozen 100-pair subset per paradigm, not the complete BLiMP benchmark. No model selection or training uses these scores.')
        write_json(destination, result)
        print(json.dumps(dict(variant=run['variant'], seed=run['seed'], accuracy=result['macro_paradigm_accuracy'])), flush=True)


if __name__ == '__main__': main()
