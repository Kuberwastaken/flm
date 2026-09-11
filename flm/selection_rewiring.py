"""Resumable generation of untrained degree/sign-matched selection controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .model import load_graph
from .provenance import sha256, write_json
from .scan_train import training_lease
from .wiring_controls import GRAPH_SEEDS, rewire_graph, validate_control


def prepare_one(graph_path, destination, seed, *, swaps_per_edge=10, max_proposals_per_edge=100):
    binding = dict(source_graph_sha256=sha256(graph_path), seed=int(seed), swaps_per_edge=swaps_per_edge,
        max_proposals_per_edge=max_proposals_per_edge,
        generator_source_sha256=sha256(Path('flm/wiring_controls.py')),
        runner_source_sha256=sha256(Path(__file__)))
    original = load_graph(graph_path)
    receipt_path = destination/'receipt.json'
    target = destination/'graph.npz'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        if receipt['binding'] != binding:
            raise ValueError('Existing rewiring identity changed; use a separate output directory')
        if receipt['status'] == 'failed':
            return receipt
        if receipt['status'] != 'complete' or sha256(target) != receipt['graph_sha256']:
            raise ValueError('Existing rewiring receipt or graph changed')
        validate_control(original, load_graph(target))
        if receipt['report']['accepted_swaps'] != swaps_per_edge*len(original['row']):
            raise ValueError('Incomplete accepted-swap receipt')
        return receipt
    destination.mkdir(parents=True, exist_ok=True)
    try:
        graph, report = rewire_graph(original, seed, swaps_per_edge=swaps_per_edge,
                                     max_proposals_per_edge=max_proposals_per_edge)
    except ValueError as error:
        receipt = dict(binding=binding, status='failed', reason=str(error),
                       verified_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        return receipt
    # A payload without a receipt is an interrupted, uncommitted attempt and
    # can be recreated deterministically. Only the receipt commits a case.
    temporary = target.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **graph)
    temporary.replace(target)
    restored = load_graph(target)
    validate_control(original, restored)
    if any(not np.array_equal(value,restored[name],equal_nan=True) for name,value in graph.items()):
        raise ValueError('Rewired graph roundtrip failed')
    receipt = dict(binding=binding,status='complete',graph_sha256=sha256(target),report=report,
                   verified_utc=datetime.now(timezone.utc).isoformat())
    write_json(receipt_path,receipt)
    return receipt


def run(graph_directory, destination, report_path):
    manifest_path = graph_directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['export_source_sha256'] != sha256(Path('flm/selection_graphs.py')):
        raise ValueError('Graph exporter source changed')
    jobs = []
    for name, graph in manifest['graphs'].items():
        parts = name.split('/')
        if len(parts) != 2 or any(not part or not all(c.isalnum() or c in '-_' for c in part) for part in parts):
            raise ValueError('Unsafe selection name')
        path = graph_directory/graph['path']
        if not path.resolve().is_relative_to(graph_directory.resolve()) or sha256(path) != graph['graph_sha256']:
            raise ValueError('Graph manifest path or identity changed')
        jobs.extend((name,path,seed) for seed in GRAPH_SEEDS)
    def snapshot(records):
        return dict(verified_utc=datetime.now(timezone.utc).isoformat(),
            graph_manifest_sha256=sha256(manifest_path), runner_source_sha256=sha256(Path(__file__)),
            generator_source_sha256=sha256(Path('flm/wiring_controls.py')), planned_graphs=len(jobs),
            graph_seeds=list(GRAPH_SEEDS), swaps_per_edge=10, max_proposals_per_edge=100,
            complete=sum(r['status'] == 'complete' for r in records.values()),
            failed=sum(r['status'] == 'failed' for r in records.values()), records=records,
            status='complete' if len(records) == len(jobs) else 'partial',
            limitation='Graph preparation only. Finite swap chains do not establish uniform sampling or mixing. No language fits, performance scores or final training matrix.')
    with training_lease(destination):
        records = {}
        for name,path,seed in jobs:
            print(json.dumps(dict(starting=name,graph_seed=seed,completed=len(records),planned=len(jobs))),flush=True)
            receipt = prepare_one(path,destination/name/f'null{seed}',seed)
            records[name+f'/null{seed}'] = receipt
            write_json(report_path,snapshot(records))
            print(json.dumps(dict(selection=name,graph_seed=seed,status=receipt['status'],completed=len(records),planned=len(jobs))),flush=True)
        return snapshot(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs',type=Path,default=Path('work/selection-graphs-v1'))
    parser.add_argument('--output',type=Path,default=Path('work/selection-rewiring-v1'))
    parser.add_argument('--report',type=Path,default=Path('work/selection-rewiring-v1/progress.json'))
    args = parser.parse_args()
    report = run(args.graphs,args.output,args.report)
    print(json.dumps(dict(complete=report['complete'],failed=report['failed'],planned=report['planned_graphs'])),flush=True)


if __name__ == '__main__': main()
