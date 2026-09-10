"""Complete independent-block likelihoods with bounded-memory batching."""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate_batch(model, documents, lexicon, chunk_size=96):
    """Right-pad exhausted rows; retain native state only within the same block.

    All rows begin together at position zero. A row is never reused for another
    block, so a Transformer's shared position offset is valid for every live row.
    Padding follows all real inputs and cannot affect an earlier causal target.
    """
    if not documents or chunk_size < 1:
        raise ValueError('Require nonempty blocks and a positive chunk size')
    identities = [identity for identity, _ in documents]
    if len(set(identities)) != len(identities) or any(len(doc) < 2 for _, doc in documents):
        raise ValueError('Require unique block IDs and at least two tokens per block')
    was_training = model.training; model.eval(); started = time.perf_counter()
    batch = len(documents); sizes = np.array([len(doc) - 1 for _, doc in documents])
    nll = torch.zeros(batch, dtype=torch.float64)
    tokens = torch.zeros(batch, dtype=torch.int64); byte_count = tokens.clone()
    lengths = torch.as_tensor(lexicon.lengths, dtype=torch.int64)
    state = None
    try:
        for start in range(0, int(sizes.max()), chunk_size):
            width = min(chunk_size, int(sizes.max()) - start)
            inputs = torch.zeros(batch, width, dtype=torch.int64)
            targets = torch.zeros_like(inputs); valid = torch.zeros_like(inputs, dtype=torch.bool)
            for i, (_, document) in enumerate(documents):
                count = min(width, max(0, int(sizes[i]) - start))
                if count:
                    values = torch.from_numpy(document[start:start + count + 1].astype(np.int64, copy=True))
                    if int(values.min()) < 0 or int(values.max()) >= lexicon.vocabulary:
                        raise ValueError('Token ID outside the declared vocabulary')
                    inputs[i, :count] = values[:-1]; targets[i, :count] = values[1:]
                    valid[i, :count] = values[1:] >= 2
            logits, state = model(inputs, state)
            losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape(batch, width)
            if not torch.isfinite(losses[valid]).all(): raise FloatingPointError('Nonfinite scored likelihood')
            # Double reduction avoids repeatedly rounding long-block sums to float32.
            nll += losses.masked_fill(~valid, 0).double().sum(1)
            tokens += valid.sum(1); byte_count += lengths[targets].masked_fill(~valid, 0).sum(1)
    finally:
        model.train(was_training)
    records = [dict(document=identity, nll=float(nll[i]), tokens=int(tokens[i]), bytes=int(byte_count[i]))
               for i, identity in enumerate(identities)]
    for record in records:
        if record['bytes'] <= 0 or record['tokens'] <= 0: raise ValueError('Block has no scored text')
        record['bits_per_byte'] = record['nll'] / record['bytes'] / math.log(2)
    return dict(documents=records, seconds=time.perf_counter() - started)


def aggregate(records):
    if not records or len({r['document'] for r in records}) != len(records):
        raise ValueError('Require nonempty unique scored blocks')
    if any(r['bytes'] <= 0 or r['tokens'] <= 0 or not math.isfinite(r['nll']) or r['nll'] < 0 for r in records):
        raise ValueError('Invalid scored block')
    nll = math.fsum(r['nll'] for r in records)
    tokens = sum(r['tokens'] for r in records); size = sum(r['bytes'] for r in records)
    return dict(blocks=len(records), nll=nll, tokens=tokens, bytes=size,
                bits_per_byte=nll / size / math.log(2), token_perplexity=math.exp(nll / tokens))


def component_summary(records):
    """Official and fixed overlap-filtered scores, including exclusions by source."""
    output = {}
    for component in ['all'] + sorted({r['component'] for r in records}):
        chosen = records if component == 'all' else [r for r in records if r['component'] == component]
        retained = [r for r in chosen if r['overlap_filtered_eligible']]
        output[component] = dict(official=aggregate(chosen),
            overlap_filtered=aggregate(retained) if retained else None,
            excluded_blocks=len(chosen) - len(retained),
            excluded_bytes=sum(r['bytes'] for r in chosen if not r['overlap_filtered_eligible']))
    return output


def paired_block_interval(first, second, replicates=10000, seed=31415):
    """Paired resampling within source component, in bounded batches of draws."""
    if replicates < 1: raise ValueError('Require positive bootstrap replicates')
    a = {r['document']: r for r in first}; b = {r['document']: r for r in second}
    if not a or len(a) != len(first) or len(b) != len(second) or set(a) != set(b):
        raise ValueError('Paired scores require the same unique blocks')
    for identity in a:
        if any(a[identity][key] != b[identity][key] for key in ('component', 'bytes', 'tokens')):
            raise ValueError('Paired block identity/count mismatch')
    aggregate(first); aggregate(second)
    groups = []
    for component in sorted({r['component'] for r in first}):
        ids = sorted(i for i in a if a[i]['component'] == component)
        groups.append((np.array([a[i]['nll'] - b[i]['nll'] for i in ids]), np.array([a[i]['bytes'] for i in ids])))
    rng = np.random.default_rng(seed); values = np.empty(replicates)
    for start in range(0, replicates, 64):
        count = min(64, replicates - start); numerator = np.zeros(count); denominator = np.zeros(count)
        for delta, sizes in groups:
            draws = rng.integers(0, len(delta), (count, len(delta)))
            numerator += delta[draws].sum(1); denominator += sizes[draws].sum(1)
        values[start:start + count] = numerator / denominator / math.log(2)
    return dict(difference_bpb=math.fsum(a[i]['nll'] - b[i]['nll'] for i in a) / sum(r['bytes'] for r in first) / math.log(2),
        lower_95=float(np.quantile(values, .025)), upper_95=float(np.quantile(values, .975)),
        replicates=replicates, seed=seed, unit='paired blocks, resampled within each source component',
        caution='Conditional on these checkpoints and component block counts; adjacent artificial blocks may share an unknown document.')
