"""Short real-data CPU integration checks; never a capacity or quality benchmark."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from flm.colab_capacity import CapacityFLM, load_graph, sized_config, train
from flm.colab_data import prepare, load_prepared
from flm.colab_reference import TransformerReference, reference_config
from flm.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for kind in ('fineweb', 'oasst'):
        if not (a.work / kind / 'data-card.json').exists():
            prepare(ROOT, a.work / kind, kind=kind, max_documents=150, max_train_tokens=100000, length=32)
    graph = load_graph(ROOT / 'data/graphs/central-1024/graph.npz')
    results = {}
    for arch, phase, steps in [('flm', 'fineweb', 6), ('flm', 'oasst', 4), ('transformer', 'fineweb', 3)]:
        data = a.work / phase
        examples, validation, card = load_prepared(data)
        if phase == 'fineweb':
            model = (CapacityFLM(graph, sized_config(graph, card['vocabulary'], .6), chunk=8)
                     if arch == 'flm' else TransformerReference(reference_config(card['vocabulary'], .6)))
        # The chat smoke continues the prior FLM weights; optimizer starts fresh.
        identity = dict(purpose='CPU integration smoke, not a language benchmark', data_card_sha256=sha256(data/'data-card.json'))
        if phase == 'oasst':
            identity['initial_checkpoint_sha256'] = sha256(a.work / 'flm-fineweb/last.pt')
        report = train(model, examples, validation, output=a.work / f'{arch}-{phase}', identity=identity,
                       pad=card['pad'], seconds=120, max_steps=steps, batch_size=1, accumulation=1, length=32,
                       lr=5e-5 if phase == 'oasst' else 3e-4)
        results[f'{arch}-{phase}'] = {k: report[k] for k in ('parameters', 'environment', 'initial_dev', 'final_dev',
                                                         'total_steps', 'total_scored_tokens', 'elapsed_seconds', 'gpu_fit_verified')}
    write_json(a.report, dict(scope='CPU plumbing checks on actual pinned corpora; no capacity or quality evidence',
               results=results, data_cards={k: json.loads((a.work/k/'data-card.json').read_text()) for k in ('fineweb', 'oasst')},
               source_sha256={name: sha256(ROOT/'flm'/name) for name in ('colab_capacity.py', 'colab_data.py', 'colab_reference.py')}))


if __name__ == '__main__':
    main()
