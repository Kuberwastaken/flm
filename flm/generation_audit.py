"""Reproducible unedited continuations and descriptive token-repetition measures."""
from __future__ import annotations
import hashlib
import math
import torch

SETTINGS = dict(seeds=[17, 29], temperature=.8, top_k=40, maximum_tokens=256,
    engine='PyTorch CPU multinomial', stop='EOS or maximum new-token count',
    display_filter='Exclude BOS and tokens containing ASCII control bytes except tab/newline; no filter in likelihood scoring')


def repetition(tokens):
    def fraction(order):
        grams = [tuple(tokens[i:i + order]) for i in range(max(0, len(tokens) - order + 1))]
        return len(set(grams)) / len(grams) if grams else None
    longest = current = 0; previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current); previous = token
    four = fraction(4)
    return dict(distinct_token_bigram_fraction=fraction(2),
        repeated_token_fourgram_fraction=None if four is None else 1 - four,
        longest_identical_token_run=longest)


@torch.no_grad()
def generate(model, lexicon, prompt, seed, settings=SETTINGS):
    if settings['maximum_tokens'] < 1 or settings['top_k'] < 1 or not math.isfinite(settings['temperature']) or settings['temperature'] <= 0:
        raise ValueError('Require positive generation settings')
    was_training = model.training; model.eval()
    generator = torch.Generator().manual_seed(seed)
    allowed = torch.tensor([False, True] + [all(b in (9, 10) or b >= 32 for b in piece) for piece in lexicon.pieces[2:]])
    generated = []; stop = 'maximum_tokens'
    try:
        logits, state = model(torch.tensor([[0] + lexicon.encode(prompt)], dtype=torch.int64))
        for position in range(settings['maximum_tokens']):
            scores = logits[0, -1].clone()
            if not torch.isfinite(scores[allowed]).all(): raise FloatingPointError('Nonfinite generation logits')
            scores[~allowed] = -torch.inf
            values, ids = scores.topk(min(settings['top_k'], int(allowed.sum())))
            index = torch.multinomial(torch.softmax(values / settings['temperature'], dim=0), 1, generator=generator)
            chosen = int(ids[index])
            if chosen == 1: stop = 'eos'; break
            generated.append(chosen)
            if position + 1 < settings['maximum_tokens']:
                logits, state = model(torch.tensor([[chosen]], dtype=torch.int64), state)
    finally:
        model.train(was_training)
    payload = b''.join(lexicon.pieces[token] for token in generated)
    try: text = payload.decode('utf8', errors='strict'); invalid_utf8 = False
    except UnicodeDecodeError: text = payload.decode('utf8', errors='replace'); invalid_utf8 = True
    return dict(prompt=prompt, sampling_seed=seed, continuation=text, tokens=generated, generated_tokens=len(generated),
        bytes=len(payload), generated_bytes_sha256=hashlib.sha256(payload).hexdigest(),
        termination=stop, invalid_utf8=invalid_utf8, repetition=repetition(generated))
