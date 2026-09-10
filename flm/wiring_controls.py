"""Generate explicitly artificial, signed degree-matched wiring controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .model import load_graph
from .provenance import sha256, write_json

GRAPH_SEEDS = (101, 103, 107)


def topology_stats(graph):
    row, col = graph['row'], graph['col']; n = len(graph['body_ids'])
    edges = set(zip(row.tolist(), col.tolist()))
    off_diagonal = [(i, j) for i, j in edges if i != j]
    adjacency = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    count, labels = connected_components(adjacency, directed=True, connection='strong')
    return dict(neurons=n, edges=len(edges), self_edges=int(np.sum(row == col)),
        reciprocal_off_diagonal_fraction=sum((j, i) in edges for i, j in off_diagonal) / max(1, len(off_diagonal)),
        strong_components=int(count), strong_component_sizes=sorted(np.bincount(labels).tolist(), reverse=True))


def validate_control(original, control):
    n = len(original['body_ids']); row, col = original['row'], original['col']; changed = control['col']
    if len(set(zip(row.tolist(), changed.tolist()))) != len(row): raise ValueError('Duplicate null edge')
    for name, value in original.items():
        if name != 'col' and not np.array_equal(value, control[name], equal_nan=True):
            raise ValueError(f'Control changed fixed array {name}')
    if np.any(changed < 0) or np.any(changed >= n): raise ValueError('Invalid null endpoint')
    if not np.array_equal(np.bincount(col, minlength=n), np.bincount(changed, minlength=n)):
        raise ValueError('Null out-degree changed')
    signs = original['source_sign']
    if not np.array_equal(signs[col], signs[changed]): raise ValueError('Null input sign changed')
    if not np.array_equal(row == col, row == changed): raise ValueError('Null self-edges changed')
    if not np.array_equal(np.sign(original['weight']), signs[changed]): raise ValueError('Weight/sign mismatch')
    return dict(in_degree=True, out_degree=True, incoming_sign_counts=True,
        incoming_signed_weight_multiset=True, self_edges_and_weights=True,
        node_identities_positions_and_pools=True)


def rewire_graph(graph, seed, swaps_per_edge=10, max_proposals_per_edge=100):
    if swaps_per_edge < 1 or max_proposals_per_edge < swaps_per_edge:
        raise ValueError('Invalid swap budget')
    result = {name: value.copy() for name, value in graph.items()}
    row, col, signs = result['row'], result['col'], result['source_sign']
    if len(row) != len(col) or not len(row): raise ValueError('Empty or malformed edge arrays')
    validate_control(graph, result)
    movable = np.flatnonzero(row != col)
    if len(movable) < 2: raise ValueError('Insufficient non-self edges to rewire')
    edges = set(zip(row.tolist(), col.tolist())); original_edges = edges.copy()
    rng = np.random.default_rng(seed); accepted = 0; target = swaps_per_edge * len(row)
    for attempt in range(1, max_proposals_per_edge * len(row) + 1):
        a, b = movable[rng.integers(len(movable), size=2)]
        i, k, j, l = int(row[a]), int(row[b]), int(col[a]), int(col[b])
        if a == b or i == k or j == l or signs[j] != signs[l] or i == l or k == j: continue
        if (i, l) in edges or (k, j) in edges: continue
        edges.remove((i, j)); edges.remove((k, l)); edges.add((i, l)); edges.add((k, j))
        col[a], col[b] = l, j; accepted += 1
        if accepted == target: break
    if accepted != target: raise ValueError(f'Insufficient accepted swaps: {accepted}/{target}')
    preserved = validate_control(graph, result)
    if edges == original_edges: raise ValueError('Control failed to change topology')
    report = dict(seed=int(seed), accepted_swaps=accepted, proposed_swaps=attempt,
        acceptance_fraction=accepted / attempt, preserved=preserved,
        original_edge_overlap_fraction=len(edges & original_edges) / len(edges),
        unchanged_endpoint_slots_fraction=float(np.mean(graph['col'] == col)),
        original=topology_stats(graph), randomized=topology_stats(result),
        limitation='Finite swap chain; uniform sampling and mixing are not established. Outgoing weighted strength and higher motifs are not fixed.')
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/graphs/central-256/graph.npz'))
    parser.add_argument('--seeds', nargs='+', type=int, choices=GRAPH_SEEDS, default=list(GRAPH_SEEDS))
    args = parser.parse_args(); graph = load_graph(args.source)
    for seed in args.seeds:
        directory = args.source.parent.with_name(args.source.parent.name + f'-null{seed}')
        randomized, report = rewire_graph(graph, seed)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'graph.npz', **randomized)
        card = dict(name=f'Artificial signed degree control, seed {seed}', graph_sha256=sha256(directory / 'graph.npz'),
            source_graph=str(args.source.as_posix()), source_graph_sha256=sha256(args.source),
            graph_type='Artificial rewiring of retained MaleCNS node identities; not measured synapses',
            magnitude_meaning='Weights and contacts are transferred input magnitudes at their original postsynaptic slots, not measurements of the new edge',
            orientation='row postsynaptic, column presynaptic', license='CC-BY-4.0',
            protocol_sha256=sha256(Path('docs/WIRING-LEARNING-PROTOCOL.md')),
            implementation_sha256=sha256(Path(__file__)), **report)
        write_json(directory / 'graph-card.json', card)
        print(json.dumps(dict(path=str(directory), **report)), flush=True)


if __name__ == '__main__': main()
