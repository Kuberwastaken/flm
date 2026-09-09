"""An interpolated byte n-gram reference; fit train, select smoothing on validation."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
from .provenance import sha256, write_json
from .train import read_documents


class NGram:
    def __init__(self, context=4):
        self.context = context; self.tables = [defaultdict(Counter) for _ in range(context + 1)]
        self.totals = []

    def fit(self, documents):
        for _, document in documents:
            tokens = [int(x) for x in document]
            for i in range(1, len(tokens)):
                target = tokens[i]
                for length in range(min(self.context, i) + 1):
                    self.tables[length][tuple(tokens[i - length:i])][target] += 1
        self.totals = [{key: sum(value.values()) for key, value in table.items()} for table in self.tables]
        return self

    def probability(self, target, history, smoothing=1.):
        base = self.tables[0][()]; probability = (base.get(target, 0) + .1) / (self.totals[0][()] + 258 * .1)
        for length in range(1, min(self.context, len(history)) + 1):
            key = tuple(history[-length:]); counts = self.tables[length].get(key)
            if counts:
                strength = smoothing * len(counts)
                probability = (counts.get(target, 0) + strength * probability) / (self.totals[length][key] + strength)
        return probability

    def evaluate(self, documents, smoothing=1., byte_limit=None):
        nll, size, scores = 0., 0, []; started = time.perf_counter()
        limit = max(96, byte_limit // len(documents)) if byte_limit else None
        for identity, document in documents:
            tokens = [int(x) for x in document[:limit + 1]] if limit else [int(x) for x in document]
            doc_nll, doc_size = 0., 0
            for i in range(1, len(tokens)):
                if tokens[i] >= 256: continue
                doc_nll -= math.log(self.probability(tokens[i], tokens[max(0, i - self.context):i], smoothing)); doc_size += 1
            nll += doc_nll; size += doc_size
            scores.append(dict(document=identity, nll=doc_nll, bytes=doc_size, bits_per_byte=doc_nll / max(1, doc_size) / math.log(2)))
        return dict(bits_per_byte=nll / size / math.log(2), bytes=size, documents=scores,
                    seconds=time.perf_counter() - started, subset='fixed document prefixes' if byte_limit else 'entire supplied split')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/ami'))
    parser.add_argument('--output', type=Path, default=Path('runs/ngram'))
    parser.add_argument('--test', action='store_true', help='Read the held-out test split only after the experiment protocol is fixed')
    args = parser.parse_args(); started = time.perf_counter()
    model = NGram(4).fit(read_documents(args.data / 'train.jsonl'))
    validation = read_documents(args.data / 'validation.jsonl')
    candidates = {str(value): model.evaluate(validation, value, 32768) for value in (.25, 1., 4.)}
    selected = min(candidates, key=lambda key: candidates[key]['bits_per_byte'])
    result = dict(context_bytes=4, selection='lowest validation prefix bits/byte', smoothing=float(selected),
        smoothing_rule='interpolate each context with its suffix; strength = smoothing * distinct continuation count',
        train_sha256=sha256(args.data / 'train.jsonl'), validation_sha256=sha256(args.data / 'validation.jsonl'),
        validation=candidates, observed_contexts=[len(table) for table in model.tables],
        observed_counts=sum(len(counter) for table in model.tables for counter in table.values()),
        fit_and_validation_seconds=time.perf_counter() - started, parameter_matched=False)
    if args.test:
        result['test'] = model.evaluate(read_documents(args.data / 'test.jsonl'), float(selected))
        result['test_sha256'] = sha256(args.data / 'test.jsonl')
    write_json(args.output / 'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('validation', 'test')}, indent=2))


if __name__ == '__main__': main()
