"""Resume a serial, budget-matched comparison suite without touching test data."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import torch
from torch.nn import functional as F
from .baselines import GRU, Transformer
from .graph import rewire
from .model import FLM, Config, load_graph
from .provenance import sha256, write_json
from .train import Sampler, evaluate, read_documents, save_checkpoint

VARIANTS = ['flm', 'rewired', 'no_recurrence', 'no_slow', 'gru', 'transformer']


def construct(variant, graph_path, seed):
    torch.manual_seed(seed)
    if variant == 'gru': return GRU()
    if variant == 'transformer': return Transformer()
    graph = load_graph(graph_path)
    if variant == 'rewired':
        graph['row'], graph['col'] = rewire(graph['row'], graph['col'], graph['source_sign'], seed + 1000)
    return FLM(graph, Config(neurons=len(graph['body_ids']), pools=int(graph['pool'].max()) + 1, variant=variant))


def restore(path, graph_path):
    payload = path.read_bytes()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    saved['_file_sha256'] = hashlib.sha256(payload).hexdigest()
    variant = saved['config']['variant']
    # Saved graph buffers contain the actual rewired topology; no need to rewire again.
    model = construct('flm' if variant == 'rewired' else variant, graph_path, saved['run']['seed'])
    model.config.variant = variant
    model.load_state_dict(saved['model'])
    if saved['run']['graph_sha256'] != sha256(graph_path): raise ValueError('Graph checksum mismatch')
    return model, saved


def run_one(args, variant, seed, documents, validation):
    output = args.output / f'{variant}-s{seed}'; output.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=seed, variant=variant, steps=args.steps, batch=16, sequence=96, warmup=16,
                    learning_rate=.002, weight_decay=.01, eval_bytes=32768,
                    train_sha256=sha256(args.data / 'train.jsonl'),
                    validation_sha256=sha256(args.data / 'validation.jsonl'), graph_sha256=sha256(args.graph))
    last = output / 'last.pt'
    sampler = Sampler(documents, seed, 96); start, best = 0, float('inf')
    if last.exists():
        model, saved = restore(last, args.graph)
        if saved['run']['comparison_protocol'] != protocol: raise ValueError(f'Changed protocol for {output}; use a new output directory')
        start, best = saved['step'], saved['best']
        if start >= args.steps and (output / 'complete.json').exists():
            print(f'Skipping completed {variant} seed {seed}', flush=True); return
        sampler.rng.bit_generator.state = saved['sampler_rng']; torch.set_rng_state(saved['torch_rng'])
        run = saved['run']
    else:
        model = construct(variant, args.graph, seed)
        run = dict(seed=seed, graph_sha256=protocol['graph_sha256'], train_sha256=protocol['train_sha256'],
            validation_sha256=protocol['validation_sha256'], comparison_protocol=protocol,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            python_torch=str(torch.__version__), parameter_card=model.parameter_card(),
            test_set_used_for_training=False, training=dict(steps=args.steps, batch=16, sequence=96, warmup_tokens=16,
                learning_rate=.002, weight_decay=.01, device='cpu', threads=args.threads))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    if last.exists(): optimizer.load_state_dict(saved['optimizer'])
    write_json(output / 'run.json', run)
    started = time.perf_counter(); loss_sum = 0.0; interval_steps = 0
    print(json.dumps(dict(event='start', variant=variant, seed=seed, resume_step=start, parameters=model.parameter_card()['trainable_parameters'])), flush=True)
    for step in range(start + 1, args.steps + 1):
        model.train(); x, y = sampler.sample(16, 'cpu'); optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x); loss = F.cross_entropy(logits[:, 16:].reshape(-1, 258), y[:, 16:].reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite gradient')
        optimizer.step(); loss_sum += float(loss.detach()); interval_steps += 1
        if step % 100 == 0:
            record = dict(variant=variant, seed=seed, step=step, train_loss=loss_sum / interval_steps,
                          presented_tokens=step * 16 * 96, attempt_seconds=time.perf_counter() - started)
            with (output / 'history.jsonl').open('a', encoding='utf8') as f: f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True); loss_sum = 0.; interval_steps = 0
        if step % 500 == 0 or step == args.steps:
            measured = evaluate(model, validation, 96, 32768)
            write_json(output / f'validation-{step:06d}.json', measured)
            improved = measured['bits_per_byte'] < best; best = min(best, measured['bits_per_byte'])
            save_checkpoint(last, model, optimizer, sampler, step, best, run)
            if improved: save_checkpoint(output / 'best.pt', model, optimizer, sampler, step, best, run)
            print(json.dumps(dict(variant=variant, seed=seed, step=step, validation_bpb=measured['bits_per_byte'])), flush=True)
    write_json(output / 'complete.json', dict(variant=variant, seed=seed, steps=args.steps, best_validation_bpb=best,
                                             best_checkpoint_sha256=sha256(output / 'best.pt'), protocol=protocol))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/ami'))
    parser.add_argument('--graph', type=Path, default=Path('data/graphs/central-1024/graph.npz'))
    parser.add_argument('--output', type=Path, default=Path('runs/comparison'))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--after', type=Path, help='Wait for a preceding run completion artifact before using the CPU')
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1: parser.error('Positive steps and threads required')
    if args.after:
        print(f'Waiting for {args.after}', flush=True)
        while not args.after.exists(): time.sleep(20)
    torch.set_num_threads(args.threads)
    documents = read_documents(args.data / 'train.jsonl'); validation = read_documents(args.data / 'validation.jsonl')
    for seed in args.seeds:
        for variant in args.variants: run_one(args, variant, seed, documents, validation)
    print('All requested matched training runs are complete. Test data have not been read.', flush=True)


if __name__ == '__main__': main()
