"""Audit six language sources for later SCAN transfer; no fitting or corpus access.

Compare reconstructed initial tensors and restored synthetic-token inference
with the recorded historical constructor and baseline implementation. This does
not establish multithreaded backward parity or benchmark performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import torch

from flm.inference import state_hash
from flm.language_train import construct, restore
from flm.provenance import sha256, write_json
from flm.tokenizer import Lexicon


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def historical(root, commit, filename):
    source = subprocess.check_output(['git', 'show', f'{commit}:{filename}'], cwd=root)
    name = 'flm._scan_source_' + Path(filename).stem
    module = types.ModuleType(name); module.__package__ = 'flm'
    previous = sys.modules.get(name); sys.modules[name] = module
    try:
        exec(compile(source, f'{commit}:{filename}', 'exec', dont_inherit=True), module.__dict__)
    finally:
        if previous is None: sys.modules.pop(name, None)
        else: sys.modules[name] = previous
    return module, hashlib.sha256(source).hexdigest()


def audit(root):
    root = Path(root); torch.set_num_threads(1)
    selection_path = root / 'reports/wikitext2/selection.json'
    selection = read_json(selection_path)
    expected = {(v, s) for v in ('flm', 'gru', 'transformer') for s in (42, 43)}
    if len(selection['runs']) != 6 or {(r['variant'], r['seed']) for r in selection['runs']} != expected:
        raise ValueError('Require all six original language sources')
    graph = root / 'data/graphs/central-1024/graph.npz'
    tokenizer = root / 'data/tokenizers/wikitext2-4096/tokenizer.json'
    lexicon = Lexicon(tokenizer); results = []
    if (selection['protocol']['steps'] != 6000 or selection['protocol']['graph_sha256'] != sha256(graph)
            or selection['protocol']['tokenizer_sha256'] != lexicon.sha256):
        raise ValueError('Language-source protocol or input identity changed')
    for row in selection['runs']:
        path = root / row['checkpoint'].replace('\\', '/')
        trained, saved = restore(path, graph, lexicon)
        complete = read_json(path.with_name('complete.json')); run = saved['run']
        if (saved['_file_sha256'] != row['checkpoint_sha256'] or run['protocol'] != selection['protocol']
                or run['seed'] != row['seed'] or saved['config']['variant'] != row['variant']
                or saved['best'] != row['selection_validation_bpb'] or run['test_set_used_for_training'] is not False
                or complete['steps'] != 6000 or complete['best_checkpoint_sha256'] != row['checkpoint_sha256']
                or str(torch.__version__) != run['python_torch']):
            raise ValueError('Original language selection, completion or software changed')
        sources = {}
        for name in ('flm/model.py', 'flm/tokenizer.py'):
            original = subprocess.check_output(['git', 'show', f"{run['source_commit']}:{name}"], cwd=root)
            digest = hashlib.sha256(original).hexdigest()
            if sha256(root / name) != digest: raise ValueError('Historical dependency changed: ' + name)
            sources[name] = digest
        old, sources['flm/language_train.py'] = historical(root, run['source_commit'], 'flm/language_train.py')
        base, sources['flm/baselines.py'] = historical(root, run['source_commit'], 'flm/baselines.py')
        old.GRU = base.GRU; old.Transformer = base.Transformer; old.BaselineConfig = base.BaselineConfig
        original = old.construct(row['variant'], graph, lexicon.vocabulary, row['seed'])
        initial = construct(row['variant'], graph, lexicon.vocabulary, row['seed'])
        initial_hash = state_hash(initial)
        if state_hash(original) != initial_hash:
            raise ValueError('Reconstructed original initialization changed')
        if initial.parameter_card() != trained.parameter_card() or trained.parameter_card() != run['parameter_card']:
            raise ValueError('Original and selected architectures differ')
        initial_buffers = dict(initial.named_buffers())
        if any(not torch.equal(value, initial_buffers[name]) for name, value in trained.named_buffers()):
            raise ValueError('Language source changed a fixed graph or pooling buffer')
        original.load_state_dict(saved['model']); original.eval(); trained.eval()
        before = state_hash(trained)
        inputs = torch.arange(222, dtype=torch.int64).reshape(2, 111) + 2
        states = [None, None]; gaps = []
        with torch.no_grad():
            for chunk in (inputs[:, :96], inputs[:, 96:]):
                a, states[0] = original(chunk, states[0]); b, states[1] = trained(chunk, states[1])
                gaps.append(float((a - b).abs().max()))
                if not torch.equal(a, b): raise ValueError('Historical synthetic-token inference differs')
        if before != state_hash(trained) or before != state_hash(original):
            raise ValueError('Probe mutated source weights')
        results.append(dict(variant=row['variant'], seed=row['seed'], source_commit=run['source_commit'],
            checkpoint_sha256=row['checkpoint_sha256'], checkpoint_step=saved['step'],
            initial_state_sha256=initial_hash, pretrained_state_sha256=before,
            historical_source_sha256=sources, initial_tensors_exact=True,
            chunk_maximum_logit_gaps=gaps, probe_tensors_unchanged=True))
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        scope='Original language sources for a later SCAN comparison; no SCAN training or evaluation.',
        source_sha256={name:sha256(root/name) for name in ('scripts/scan_source_audit.py',
            'flm/language_train.py','flm/model.py','flm/baselines.py','flm/tokenizer.py','flm/inference.py')},
        language_selection_sha256=sha256(selection_path), graph_sha256=sha256(graph), tokenizer_sha256=lexicon.sha256,
        torch=str(torch.__version__), threads=1, dtype='float32', sources=results,
        probe='Two synthetic ID rows, 111 inputs each, in consecutive chunks of 96 and 15; no corpus text.',
        no_corpus_opened=True, no_fitting=True,
        limitations=['Exact observed initialization and forward probes do not prove all-input or backward parity.',
            'Current attention caches copy their bounded slices; historical caches retained views.',
            'This is source preparation, not a transfer result or an official downstream study identity.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/scan-runtime/source-preflight.json'))
    args = parser.parse_args(); result = audit(Path.cwd()); write_json(args.output, result)
    print(f"Verified {len(result['sources'])} language sources; no fitting or corpus access")
