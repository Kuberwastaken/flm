"""Load the released comparison checkpoints and generate local continuations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import torch

from .language_train import restore
from .provenance import sha256
from .tokenizer import Lexicon

RUNTIME_FILES = ('flm/__init__.py', 'flm/inference.py', 'flm/language_train.py',
                 'flm/model.py', 'flm/baselines.py', 'flm/graph.py', 'flm/train.py',
                 'flm/tokenizer.py', 'flm/corpus_cache.py', 'flm/provenance.py')


def state_hash(model):
    """Hash all learned tensors and fixed buffers with explicit field boundaries."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        header = json.dumps([name, list(array.shape), array.dtype.str]).encode()
        payload = array.tobytes()
        for part in (header, payload):
            digest.update(len(part).to_bytes(8, 'little')); digest.update(part)
    return digest.hexdigest()


def bundle_path(root, relative):
    root = Path(root)
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Invalid bundle path')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or path == PurePosixPath('.'):
        raise ValueError('Bundle path leaves its directory')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Bundle path leaves its directory')
    return resolved


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'bundle.json').read_text(encoding='utf8'))
    if manifest.get('format') != 'flm-wikitext-inference-v1':
        raise ValueError('Unsupported inference bundle')
    if manifest.get('runtime_sources') != list(RUNTIME_FILES):
        raise ValueError('Inference runtime inventory changed')
    for name, record in manifest['files'].items():
        path = bundle_path(root, name)
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Bundle file failed integrity verification: {name}')
    runtime_root = Path(__file__).resolve().parents[1]
    for name in RUNTIME_FILES:
        if sha256(runtime_root / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Run with the matching bundled runtime: {name}')
    expected = {f'{variant}-s{seed}' for seed in (42, 43) for variant in ('flm', 'gru', 'transformer')}
    if {row['id'] for row in manifest['models']} != expected or len(manifest['models']) != 6:
        raise ValueError('The complete six-model comparison is required')
    for name in (manifest['graph'], manifest['tokenizer'], *(r['file'] for r in manifest['models'])):
        if name not in manifest['files']:
            raise ValueError('A required model input is absent from the manifest')
    return manifest


def load_model(root, identity):
    root = Path(root).resolve(); manifest = verify_bundle(root)
    record = next((row for row in manifest['models'] if row['id'] == identity), None)
    if record is None:
        raise ValueError('Unknown comparison model')
    lexicon = Lexicon(bundle_path(root, manifest['tokenizer']))
    model, saved = restore(bundle_path(root, record['file']), bundle_path(root, manifest['graph']), lexicon)
    if saved.get('format') != 'flm-inference-checkpoint-v1' or 'optimizer' in saved:
        raise ValueError('Expected a model-only inference checkpoint')
    if (saved['source_checkpoint_sha256'] != record['source_checkpoint_sha256'] or
            saved['step'] != record['checkpoint_step'] or saved['run']['seed'] != record['training_seed'] or
            model.config.variant != record['variant'] or state_hash(model) != record['state_sha256']):
        raise ValueError('Inference model does not match its selected-checkpoint identity')
    model.eval()
    return model, lexicon, record


@torch.no_grad()
def generate(model, lexicon, prompt, *, seed=17, temperature=.8, top_k=40, maximum_tokens=128):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Temperature must be finite and nonnegative')
    if (type(top_k) is not int or type(maximum_tokens) is not int or
            not 1 <= top_k <= lexicon.vocabulary or not 1 <= maximum_tokens <= 1024):
        raise ValueError('Use a valid top-k and between 1 and 1024 output tokens')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Sampling seed must be between 0 and 2^63 - 1')
    model.eval(); generator = torch.Generator().manual_seed(seed)
    prefix = [0] + lexicon.encode(prompt); state = None
    # Carry each architecture's own state across bounded prefill chunks.
    for start in range(0, len(prefix), 96):
        logits, state = model(torch.tensor([prefix[start:start + 96]]), state)
    allowed = torch.tensor([False, True] + [all(b in (9, 10) or b >= 32 for b in piece)
                                           for piece in lexicon.pieces[2:]])
    generated = []
    for _ in range(maximum_tokens):
        scores = logits[0, -1].clone(); scores[~allowed] = -torch.inf
        if temperature == 0:
            chosen = int(scores.argmax())
        else:
            values, indices = scores.topk(top_k)
            chosen = int(indices[torch.multinomial(torch.softmax(values / temperature, dim=0), 1, generator=generator)])
        if chosen == 1:
            break
        generated.append(chosen)
        logits, state = model(torch.tensor([[chosen]]), state)
    payload = b''.join(lexicon.pieces[token] for token in generated)
    return dict(prompt=prompt, continuation=payload.decode('utf8', errors='replace'),
                tokens=generated, bytes=len(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('.'))
    parser.add_argument('--model', choices=('flm', 'gru', 'transformer'), default='flm')
    parser.add_argument('--training-seed', type=int, choices=(42, 43), default=42)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--tokens', type=int, default=128)
    args = parser.parse_args(); torch.set_num_threads(1)
    model, lexicon, record = load_model(args.bundle, f'{args.model}-s{args.training_seed}')
    result = generate(model, lexicon, args.prompt, seed=args.sampling_seed,
                      temperature=args.temperature, top_k=args.top_k, maximum_tokens=args.tokens)
    print(json.dumps(dict(model=record, sampling=dict(seed=args.sampling_seed,
                     temperature=args.temperature, top_k=args.top_k, maximum_tokens=args.tokens,
                     threads=1, engine='PyTorch CPU multinomial; temperature zero uses argmax'),
                     **result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
