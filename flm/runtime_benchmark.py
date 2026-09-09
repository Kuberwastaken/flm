"""Measure the implemented CPU inference path after training jobs have stopped."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .language_test import freeze_selection
from .language_train import restore
from .provenance import write_json
from .tokenizer import Lexicon


def state_storage(state):
    """Count both logical tensor bytes and unique allocated tensor backing bytes."""
    seen = set(); logical = allocated = 0
    def visit(value):
        nonlocal logical, allocated
        if isinstance(value, torch.Tensor):
            logical += value.numel() * value.element_size()
            storage = value.untyped_storage(); key = (str(value.device), storage.data_ptr())
            if key not in seen: seen.add(key); allocated += storage.nbytes()
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
    visit(state)
    return dict(logical_tensor_bytes=logical, allocated_tensor_bytes=allocated,
        scope='Persistent sequence tensors only; excludes Python bookkeeping, model weights, temporary activations and adapters')


@torch.no_grad()
def measure(model, prefix, continuation, repeats=5):
    model.eval(); _, state = model(prefix)
    for token in continuation[:8]: _, state = model(torch.tensor([[token]]), state)
    prefill, decode = [], []; gc.collect()
    for _ in range(repeats):
        started = time.perf_counter(); _, state = model(prefix); prefill.append(time.perf_counter() - started)
        started = time.perf_counter()
        for token in continuation: _, state = model(torch.tensor([[token]]), state)
        decode.append(time.perf_counter() - started)
    return dict(prefill_tokens=prefix.shape[1], decode_tokens=len(continuation), repeats=repeats,
        prefill_seconds=prefill, decode_seconds=decode,
        median_prefill_tokens_per_second=prefix.shape[1] / float(np.median(prefill)),
        median_decode_tokens_per_second=len(continuation) / float(np.median(decode)), state=state_storage(state))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('runs/wikitext2'))
    p.add_argument('--output', type=Path, default=Path('reports/wikitext2/runtime.json'))
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(); selection = freeze_selection(a.runs); torch.set_num_threads(a.threads)
    lexicon = Lexicon(Path('data/tokenizers/wikitext2-4096/tokenizer.json'))
    stimulus = ('A small bird lives in a garden. Each morning it leaves the tree and returns to the same branch. ' * 30)
    tokens = [0] + lexicon.encode(stimulus); prefix = torch.tensor([tokens[:128]]); continuation = tokens[128:256]
    results = []
    for run in selection['runs']:
        if run['seed'] != 42: continue
        model, saved = restore(Path(run['checkpoint']), Path('data/graphs/central-1024/graph.npz'), lexicon)
        result = dict(variant=run['variant'], checkpoint_sha256=saved['_file_sha256'],
            parameters=model.parameter_card()['trainable_parameters'], **measure(model, prefix, continuation))
        results.append(result); print(json.dumps(result), flush=True)
    write_json(a.output, dict(platform=platform.platform(), torch=str(torch.__version__), threads=a.threads,
        batch=1, dtype='float32', device='CPU', tokenizer_sha256=lexicon.sha256, results=results,
        protocol='Call only after all training processes have exited. One warmup and five repeated identical original-text trials; tokenization excluded; forced identical token inputs across models.',
        caveat='This measures the implemented eager PyTorch path, including FLM recurrent-weight preparation per forward call. It is not a hardware-independent FLOP, peak-memory or energy comparison. Normal OS background activity can affect timings.'))


if __name__ == '__main__': main()
