"""Train and score the shared-tokenizer language study with exact exposure accounting."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time
import torch
from torch.nn import functional as F
from .baselines import GRU, Transformer, BaselineConfig
from .graph import rewire
from .model import FLM, Config, load_graph
from .provenance import sha256, write_json
from .tokenizer import Lexicon, read_cache
from .train import Sampler, save_checkpoint


def construct(variant, graph_path, vocabulary, seed):
    torch.manual_seed(seed)
    if variant in ('gru', 'transformer'):
        config = BaselineConfig(variant, embedding=96, hidden=200, width=108, heads=6,
                                layers=2, window=96, vocabulary=vocabulary, tied_readout=True)
        model = GRU(config) if variant == 'gru' else Transformer(config)
        torch.nn.init.normal_(model.embedding.weight, std=.2)
        torch.nn.init.normal_(model.readout.weight, std=.025); torch.nn.init.zeros_(model.readout.bias)
    else:
        graph = load_graph(graph_path)
        if variant == 'rewired': graph['row'], graph['col'] = rewire(graph['row'], graph['col'], graph['source_sign'], seed + 1000)
        model = FLM(graph, Config(neurons=len(graph['body_ids']), pools=int(graph['pool'].max()) + 1,
            embedding=96, variant=variant, vocabulary=vocabulary, tied_readout=True))
    return model


def restore(path, graph_path, lexicon):
    payload = path.read_bytes()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    if checkpoint['run']['tokenizer_sha256'] != lexicon.sha256 or checkpoint['run']['graph_sha256'] != sha256(graph_path):
        raise ValueError('Checkpoint tokenizer/graph mismatch')
    variant = checkpoint['config']['variant']
    model = construct('flm' if variant == 'rewired' else variant, graph_path, lexicon.vocabulary, checkpoint['run']['seed'])
    model.config.variant = variant
    if asdict(model.config) != checkpoint['config']: raise ValueError('Checkpoint configuration mismatch')
    model.load_state_dict(checkpoint['model']); checkpoint['_file_sha256'] = hashlib.sha256(payload).hexdigest()
    return model, checkpoint


@torch.no_grad()
def evaluate(model, documents, lexicon, token_limit=None, chunk_size=96):
    was_training = model.training; model.eval(); lengths = torch.from_numpy(lexicon.lengths)
    started = time.perf_counter(); records = []; total_nll, total_tokens, total_bytes = 0., 0, 0
    for i, (identity, document) in enumerate(documents):
        if token_limit:
            quota = token_limit // len(documents) + int(i < token_limit % len(documents))
            document = document[:quota + 1]
        state = None; nll, tokens, byte_count = 0., 0, 0
        for start in range(0, len(document) - 1, chunk_size):
            values = torch.from_numpy(document[start:start + chunk_size + 1].copy()).unsqueeze(0)
            logits, state = model(values[:, :-1], state); targets = values[:, 1:].reshape(-1)
            losses = F.cross_entropy(logits.reshape(-1, lexicon.vocabulary), targets, reduction='none'); mask = targets >= 2
            nll += float(losses[mask].sum()); tokens += int(mask.sum()); byte_count += int(lengths[targets].sum())
        records.append(dict(document=identity, nll=nll, tokens=tokens, bytes=byte_count, bits_per_byte=nll / max(1, byte_count) / math.log(2)))
        total_nll += nll; total_tokens += tokens; total_bytes += byte_count
    model.train(was_training)
    return dict(bits_per_byte=total_nll / total_bytes / math.log(2), token_perplexity=math.exp(total_nll / total_tokens),
        nll=total_nll, tokens=total_tokens, bytes=total_bytes, documents=records, seconds=time.perf_counter() - started,
        tokenizer_sha256=lexicon.sha256, subset='fixed per-article token prefixes' if token_limit else 'entire supplied split',
        protocol='Reset per article; carry native state within article; score IDs >=2; normalize by exact decoded UTF-8 byte lengths; boundary tokens excluded')


def train(args):
    torch.set_num_threads(args.threads); lexicon = Lexicon(args.tokenizer)
    documents = read_cache(args.data / 'train.npz'); validation = read_cache(args.data / 'validation.npz')
    protocol = dict(steps=args.steps, batch=16, sequence=96, warmup=16, learning_rate=.002, final_learning_rate=.0002,
        lr_warmup_updates=100, weight_decay=.01, eval_tokens=32768, threads=args.threads,
        tokenizer_sha256=lexicon.sha256, train_cache_sha256=sha256(args.data / 'train.npz'),
        validation_cache_sha256=sha256(args.data / 'validation.npz'), graph_sha256=sha256(args.graph))
    sampler = Sampler(documents, args.seed, 96); start, best = 0, float('inf'); presented_bytes = 0; scored_bytes = 0
    if args.resume:
        model, saved = restore(args.resume, args.graph, lexicon)
        if saved['run']['protocol'] != protocol or model.config.variant != args.variant or saved['run']['seed'] != args.seed:
            raise ValueError('Resume protocol mismatch; keep the registered run settings or use a new experiment')
        sampler.rng.bit_generator.state = saved['sampler_rng']; torch.set_rng_state(saved['torch_rng'])
        start, best, run = saved['step'], saved['best'], saved['run']
        presented_bytes = run.get('exposure', {}).get('presented_bytes', 0); scored_bytes = run.get('exposure', {}).get('scored_bytes', 0)
    else:
        if (args.output / 'run.json').exists(): raise ValueError('Run exists; resume its checkpoint or choose a new output directory')
        model = construct(args.variant, args.graph, lexicon.vocabulary, args.seed)
        run = dict(seed=args.seed, protocol=protocol, graph_sha256=protocol['graph_sha256'], tokenizer_sha256=lexicon.sha256,
            data=str(args.data), tokenizer=str(args.tokenizer), model_version='flm-lexical-v2',
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            python_torch=str(torch.__version__), parameter_card=model.parameter_card(), test_set_used_for_training=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    if args.resume: optimizer.load_state_dict(saved['optimizer'])
    args.output.mkdir(parents=True, exist_ok=True); write_json(args.output / 'run.json', run)
    if not args.resume:
        initial = evaluate(model, validation, lexicon, protocol['eval_tokens'])
        write_json(args.output / 'initial-validation.json', initial)
        print(json.dumps(dict(event='initial', variant=args.variant, seed=args.seed, parameters=model.parameter_card()['trainable_parameters'], bits_per_byte=initial['bits_per_byte'])), flush=True)
    started = time.perf_counter(); interval = started; loss_sum = 0.; interval_steps = 0
    for step in range(start + 1, args.steps + 1):
        progress = max(0., (step - 100) / max(1, args.steps - 100))
        rate = (.0002 + .0018 * .5 * (1 + math.cos(math.pi * min(progress, 1.)))) * min(step / 100, 1.)
        for group in optimizer.param_groups: group['lr'] = rate
        model.train(); x, y = sampler.sample(16, 'cpu'); optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x); loss = F.cross_entropy(logits[:, 16:].reshape(-1, lexicon.vocabulary), y[:, 16:].reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(gradient): raise FloatingPointError(f'Nonfinite gradient at {step}')
        optimizer.step(); loss_sum += float(loss.detach()); interval_steps += 1
        presented_bytes += int(lexicon.lengths[x.numpy()].sum()); scored_bytes += int(lexicon.lengths[y[:, 16:].numpy()].sum())
        if step % 100 == 0:
            record = dict(step=step, train_loss=loss_sum / interval_steps, tokens_per_second=interval_steps * 1536 / (time.perf_counter() - interval),
                learning_rate=rate, presented_tokens=step * 1536, presented_bytes=presented_bytes, scored_bytes=scored_bytes,
                gradient_norm=float(gradient), attempt_seconds=time.perf_counter() - started)
            with (args.output / 'history.jsonl').open('a', encoding='utf8') as handle: handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True); interval = time.perf_counter(); interval_steps = 0; loss_sum = 0.
        if step % 500 == 0 or step == args.steps:
            measured = evaluate(model, validation, lexicon, protocol['eval_tokens'])
            write_json(args.output / f'validation-{step:06d}.json', measured)
            improved = measured['bits_per_byte'] < best; best = min(best, measured['bits_per_byte'])
            run['exposure'] = dict(presented_tokens=step * 1536, presented_bytes=presented_bytes, scored_bytes=scored_bytes)
            save_checkpoint(args.output / 'last.pt', model, optimizer, sampler, step, best, run)
            if improved: save_checkpoint(args.output / 'best.pt', model, optimizer, sampler, step, best, run)
            if step % 1000 == 0: save_checkpoint(args.output / f'checkpoint-{step:06d}.pt', model, optimizer, sampler, step, best, run)
            print(json.dumps(dict(step=step, validation_bpb=measured['bits_per_byte'], token_perplexity=measured['token_perplexity'], best=best)), flush=True)
            interval = time.perf_counter()
    write_json(args.output / 'complete.json', dict(steps=args.steps, best_validation_bpb=best, best_checkpoint_sha256=sha256(args.output / 'best.pt'), protocol=protocol))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['flm', 'gru', 'transformer', 'rewired', 'no_recurrence', 'no_slow'], default='flm')
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--data', type=Path, default=Path('data/processed/wikitext2-bpe'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tokenizers/wikitext2-4096/tokenizer.json'))
    parser.add_argument('--graph', type=Path, default=Path('data/graphs/central-1024/graph.npz'))
    parser.add_argument('--output', type=Path, default=Path('runs/wikitext2/flm-s42'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1: parser.error('Positive steps and threads required')
    train(args)


if __name__ == '__main__': main()
