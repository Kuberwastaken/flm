"""Audit the four frozen language graphs without reading model losses or rewiring."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import numpy as np

from .model import load_graph
from .provenance import sha256, write_json
from .wiring_controls import GRAPH_SEEDS, topology_stats, validate_control


def node_statistics(graph):
    n = len(graph['body_ids'])
    row, col = graph['row'], graph['col']
    weight = np.asarray(graph['weight'], dtype=np.float64)
    return {
        'in_degree': np.bincount(row, minlength=n),
        'out_degree': np.bincount(col, minlength=n),
        'positive_in_degree': np.bincount(row[weight > 0], minlength=n),
        'negative_in_degree': np.bincount(row[weight < 0], minlength=n),
        'incoming_absolute_weight': np.bincount(row, weights=np.abs(weight), minlength=n),
        'outgoing_absolute_weight': np.bincount(col, weights=np.abs(weight), minlength=n),
    }


def pair_audit(measured, control):
    preserved = validate_control(measured, control)
    before, after = node_statistics(measured), node_statistics(control)
    fixed = ('in_degree', 'out_degree', 'positive_in_degree', 'negative_in_degree', 'incoming_absolute_weight')
    maxima = {name: float(np.max(np.abs(before[name] - after[name]))) for name in fixed}
    if any(maxima.values()):
        raise ValueError('A declared node statistic changed')
    difference = after['outgoing_absolute_weight'] - before['outgoing_absolute_weight']
    return dict(preserved=preserved, maximum_absolute_node_differences=maxima,
                outgoing_absolute_weight_difference=dict(
                    changed_nodes=int(np.count_nonzero(difference)),
                    mean_absolute=float(np.mean(np.abs(difference))),
                    maximum_absolute=float(np.max(np.abs(difference)))))


def overlap_matrix(graphs):
    edges = [set(zip(g['row'].tolist(), g['col'].tolist())) for g in graphs]
    if not all(edges) or len({len(e) for e in edges}) != 1:
        raise ValueError('Overlap comparison requires nonempty equal edge counts')
    return [[len(a & b) / len(a) for b in edges] for a in edges]


def collect(root):
    identity_path = root / 'reports/language-topology/identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    # The audit has no dependency on validation/test text, losses or checkpoints.
    for relative, digest in identity['sources'].items():
        if sha256(root / relative) != digest:
            raise ValueError(f'Frozen numerical source changed: {relative}')
    protocol = root / 'docs/LANGUAGE-TOPOLOGY-PROTOCOL.md'
    if sha256(protocol) != identity['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    graphs, rows, paths = [], [], [identity_path, protocol]
    for seed in (None, *GRAPH_SEEDS):
        name = 'central-1024' + (f'-null{seed}' if seed else '')
        relative = f'data/graphs/{name}/graph.npz'
        path = root / relative
        if sha256(path) != identity['inputs'][relative]:
            raise ValueError(f'Frozen graph changed: {name}')
        graph = load_graph(path)
        stats = topology_stats(graph)
        if stats['neurons'] != 1024 or stats['edges'] != 76130:
            raise ValueError('Unexpected language graph dimensions')
        card_path = path.with_name('graph-card.json')
        card = json.loads(card_path.read_text(encoding='utf8'))
        if card['graph_sha256'] != sha256(path):
            raise ValueError('Graph card does not identify its graph')
        nodes = node_statistics(graph)
        degree_profile = dict(zero_in_degree=int(np.sum(nodes['in_degree'] == 0)),
                              zero_out_degree=int(np.sum(nodes['out_degree'] == 0)),
                              isolated_nodes=int(np.sum((nodes['in_degree'] == 0) & (nodes['out_degree'] == 0))))
        item = dict(label=f'Rewired {seed}' if seed else 'Measured', graph_seed=seed,
                    graph_sha256=sha256(path), statistics=stats, degree_profile=degree_profile)
        if seed:
            if sha256(card_path) != identity['inputs'][card_path.relative_to(root).as_posix()]:
                raise ValueError('Frozen null card changed')
            if card['randomized'] != stats or card['original'] != rows[0]['statistics']:
                raise ValueError('Recorded graph statistics do not reproduce')
            audit = pair_audit(graphs[0], graph)
            if audit['preserved'] != card['preserved']:
                raise ValueError('Recorded invariant checks do not reproduce')
            item.update(audit=audit, accepted_swaps=card['accepted_swaps'],
                        proposed_swaps=card['proposed_swaps'])
        graphs.append(graph); rows.append(item); paths.extend([path, card_path])
    overlap = overlap_matrix(graphs)
    for index, item in enumerate(rows):
        item['measured_edge_overlap_fraction'] = overlap[0][index]
    return dict(schema_version=1, study_identity_sha256=sha256(identity_path),
                generator_sha256=sha256(Path(__file__)), graphs=rows,
                pairwise_edge_overlap_fraction=overlap,
                orientation='row postsynaptic, column presynaptic; unchanged node order',
                interpretation='Structural audit only; no language outcomes or claim of null-chain mixing.'), graphs, paths


def main():
    root = Path('.').resolve()
    report, graphs, paths = collect(root)
    target = root / 'public/research/language-topology-structure.json'
    write_json(target, report)
    write_json(root / 'reports/language-topology/structure.json', report)
    csv_path = root / 'public/research/figures/language-topology-nodes.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(node_statistics(graphs[0]))
    with csv_path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('graph', 'node_index', 'body_id', 'source_sign', 'pool', *columns))
        for item, graph in zip(report['graphs'], graphs):
            nodes = node_statistics(graph)
            for index in range(len(graph['body_ids'])):
                writer.writerow((item['label'], index, int(graph['body_ids'][index]),
                                 int(graph['source_sign'][index]), int(graph['pool'][index]),
                                 *(nodes[key][index].item() for key in columns)))
    paths.extend([target, csv_path, root / 'docs/LANGUAGE-STRUCTURE.md', root / 'LICENSE', root / 'licenses/CC-BY-4.0.txt',
                  root / 'licenses/DATA-ATTRIBUTION.md', Path(__file__).resolve()])
    manifest = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    archive = root / 'public/research/language-topology-graphs.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(root).as_posix())
        bundle.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
        bundle.writestr('README.txt',
            'FLM language topology: the four exact training graphs, before learned gains.\n'
            'NPZ arrays are readable with numpy.load(path, allow_pickle=False).\n'
            'Rows receive signals; columns send signals. Node indices are zero based.\n'
            'Only the measured graph records retained anatomical edges. Null contacts\n'
            'and weights are transferred incoming magnitudes, not measured new synapses.\n'
            'Node CSV includes all 4096 graph/node records. Outgoing weighted strength\n'
            'is intentionally not fixed. This bundle contains no corpus or model losses.\n'
            'The identity record binds the original experiment, including local cache\n'
            'hashes. Full training reproduction requires those separately acquired data.\n'
            'In the FLM source checkout, run python -m flm.language_structure to audit.\n'
            'Graph data: CC BY 4.0; source attribution and license included. Code: MIT.\n')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Graph archive failed CRC verification')
        import hashlib
        for name, digest in manifest.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError(f'Graph archive hash mismatch: {name}')
    print(json.dumps(dict(graphs=len(graphs), node_rows=4096, archive_bytes=archive.stat().st_size,
                          overlap=report['pairwise_edge_overlap_fraction']), indent=2))


if __name__ == '__main__':
    main()
