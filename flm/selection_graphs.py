"""Export untrained selector inventories to the existing FLM graph format."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import csr_matrix
import torch

from .model import Config, FLM, load_graph
from .provenance import GRAPH_BASE, sha256, write_json


def induced_graph(matrix, metadata, ids, signs, selected, pools=128):
    selected = np.asarray(selected)
    n = len(ids)
    if (matrix.shape != (n, n) or len(metadata) != n or signs.shape != (n,)
            or selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or len(selected) == 0 or np.any(selected < 0) or np.any(selected >= n)
            or len(np.unique(selected)) != len(selected) or type(pools) is not int
            or not 1 <= pools <= len(selected) or not np.isin(signs, [-1,0,1]).all()):
        raise ValueError('Invalid induced graph inputs')
    selected = np.array(sorted(selected.tolist(), key=lambda i: (metadata[i][1], int(ids[i]))), dtype=np.int32)
    compact = matrix[selected][:,selected].tocsr()
    compact.sort_indices()
    if (not compact.has_canonical_format or not np.issubdtype(compact.data.dtype, np.integer)
            or np.any(compact.data <= 0) or np.any(compact.data > np.iinfo(np.uint32).max)):
        raise ValueError('Expected unique positive integer source contacts')
    row = np.repeat(np.arange(len(selected), dtype=np.int32), np.diff(compact.indptr))
    col = compact.indices.astype(np.int32)
    contacts = compact.data.astype(np.uint32)
    # The historical exporter used float32 intermediate contact counts.
    if not np.array_equal(contacts, contacts.astype(np.float32).astype(np.uint32)):
        raise ValueError('Contacts exceed exact compatibility with historical float32 exporter')
    source_sign = signs[selected].astype(np.int8)
    active = source_sign[col] != 0
    row, col, contacts = row[active], col[active], contacts[active]
    raw = np.log1p(contacts.astype(np.float32))
    denominator = np.bincount(row, weights=raw, minlength=len(selected)).astype(np.float32)
    weight = raw / np.maximum(denominator[row], 1e-12) * source_sign[col]
    pool = np.minimum(np.arange(len(selected))*pools//len(selected), pools-1).astype(np.int32)
    positions = np.asarray([metadata[i][6] or [float('nan')]*3 for i in selected], dtype=np.float32)
    return dict(row=row, col=col, weight=weight.astype(np.float32), contacts=contacts,
        source_sign=source_sign, body_ids=ids[selected], source_indices=selected,
        positions=positions, pool=pool)


def verify_model(graph, config):
    # Three synthetic token IDs check the real forward path, not language quality.
    # No training, corpus, benchmark timing or saved initialization is involved.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(42)
        model = FLM(graph, config).eval()
        logits, state = model(torch.tensor([[17,29,43]], dtype=torch.long))
        if logits.shape != (1,3,config.vocabulary) or not torch.isfinite(logits).all():
            raise ValueError('Exported graph failed FLM forward check')
        if any(x.shape != (1,config.neurons) or not torch.isfinite(x).all() for x in state):
            raise ValueError('Exported graph failed recurrent state check')
        card = model.parameter_card()
    card['sequence_state_bytes_batch1_float32'] = 2*config.neurons*4
    card['forward_fixture'] = dict(seed=42, token_ids=[17,29,43], finite_logits_and_states=True,
                                   interpretation='Synthetic interface check, not trained generation or measured throughput')
    return card


def export(source, selection_directory, destination, archive_path):
    report_path = selection_directory/'summary.json'
    selection_report = json.loads(report_path.read_text(encoding='utf8'))
    identities_path = selection_directory/'control-body-ids.json'
    if sha256(identities_path) != selection_report['control_body_ids_sha256']:
        raise ValueError('Selection membership file changed')
    if sha256(Path('flm/selection_controls.py')) != selection_report['audit_source_sha256']:
        raise ValueError('Selection rule source changed')
    hashes = {name: sha256(source/name) for name in selection_report['runtime_source_sha256']}
    if hashes != selection_report['runtime_source_sha256']:
        raise ValueError('Acquired source changed')
    metadata = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    load = lambda name: np.load(source/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    offsets, sources, counts, signs, ids = (load(name) for name in ('offsets','sources','counts','fast_sign','body_ids'))
    if len(metadata) != len(ids) or not np.array_equal(ids, np.array([r[0] for r in metadata])):
        raise ValueError('Source metadata identities disagree')
    matrix = csr_matrix((counts, sources, offsets), shape=(len(ids),len(ids)), copy=False)
    reference_path = Path('data/graphs/central-1024/graph.npz')
    reference_card = json.loads(reference_path.with_name('graph-card.json').read_text(encoding='utf8'))
    if sha256(reference_path) != reference_card['graph_sha256']:
        raise ValueError('Historical reference graph changed')
    reference = load_graph(reference_path)
    repeated = induced_graph(matrix, metadata, ids, signs, reference['source_indices'])
    if set(reference) != set(repeated) or any(reference[key].dtype != repeated[key].dtype or
            not np.array_equal(reference[key], repeated[key], equal_nan=True) for key in reference):
        raise ValueError('Exporter does not exactly reproduce frozen graph arrays')
    index = {int(body): i for i, body in enumerate(ids)}
    selections = json.loads(identities_path.read_text(encoding='utf8'))
    destination.mkdir(parents=True, exist_ok=True)
    records, contents = {}, {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for name, body_ids in selections.items():
            candidate, label = name.split('/')
            if any(not part or not all(c.isalnum() or c in '-_' for c in part) for part in (candidate,label)):
                raise ValueError('Unsafe graph record name')
            selected = np.array([index[int(body)] for body in body_ids], dtype=np.int64)
            graph = induced_graph(matrix, metadata, ids, signs, selected)
            expected = selection_report['candidates'][candidate][label]
            if (len(graph['body_ids']) != expected['neurons'] or len(graph['row']) != expected['inside_fast_edges']
                    or int(graph['contacts'].sum(dtype=np.uint64)) != expected['inside_fast_contacts']):
                raise ValueError('Export differs from independently counted inventory')
            relative = 'graphs/'+name+'.npz'
            path = destination/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **graph)
            restored = load_graph(path)
            if any(not np.array_equal(graph[key], restored[key], equal_nan=True) for key in graph):
                raise ValueError('Graph array roundtrip failed')
            config = Config(neurons=len(selected), pools=128, embedding=96, vocabulary=4096, tied_readout=True)
            model_card = verify_model(restored, config)
            incoming_degree = np.bincount(graph['row'],minlength=len(selected))
            model_card.update(path=relative, graph_sha256=sha256(path), graph_bytes=path.stat().st_size,
                selection=name, graph_type='Measured induced body-pair subset, untrained',
                zero_fast_sign_nodes=int((graph['source_sign'] == 0).sum()),
                empty_fast_incoming_nodes=int((incoming_degree == 0).sum()),
                single_fast_incoming_nodes=int((incoming_degree == 1).sum()),
                empty_fast_outgoing_nodes=int((np.bincount(graph['col'],minlength=len(selected)) == 0).sum()))
            records[name] = model_card
            contents[relative] = path.read_bytes()
    finally:
        torch.set_num_threads(previous_threads)
    manifest = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        status='Untrained graph exports; no tokenizer, learned weights, language scores or frozen fit matrix',
        export_source_sha256=sha256(Path(__file__)), model_source_sha256=sha256(Path('flm/model.py')),
        source_file_sha256=hashes, selection_summary_sha256=sha256(report_path),
        source_revision=reference_card['source_revision'], runtime_source_url=GRAPH_BASE,
        anatomical_source_url='https://male-cns.janelia.org/download/', source_license='CC-BY-4.0',
        selection_body_ids_sha256=sha256(identities_path), graphs=records,
        compatibility=dict(reference_graph_sha256=sha256(reference_path), arrays_exact=list(reference), dtype_exact=True),
        rules=dict(ordering='Cell type then body ID', pools='128 contiguous balanced pools with the same index rule in every selection',
            weight='log1p raw contacts, float32 absolute incoming normalization, then source sign',
            sign='Current fast-sign rule; zero-sign outgoing edges excluded',
            interface='Same learned input to every node and tied readout over pooled fast/slow state; no role-specific privilege'),
        limitations=['All 64 graphs are exported for inspection, not declared as 64 future language fits.',
            'Biological boundaries and model sign assumptions remain; these are not intact biological circuits.',
            'Contiguous type-based pools are consistent across selections but do not match their composition.',
            'Parameter counts include zero-initialized edge gains and all allocated trainable entries, not equal effective capacity.',
            'Single-input rows normalize to the source sign; their sole edge-gain entries do not control relative incoming magnitude.',
            'Forward fixtures verify compatibility only; no speed, learning or topology advantage is demonstrated.'])
    write_json(destination/'manifest.json', manifest)
    contents['manifest.json'] = (destination/'manifest.json').read_bytes()
    contents['README.txt'] = b'FLM untrained graph preparation archive\n\nContains 64 anatomical selection graphs and parameter/interface checks.\nNo language weights or tokenizer are included. No graph is certified intact.\nLoad an NPZ using numpy.load(path, allow_pickle=False); row is postsynaptic,\ncol is presynaptic, source_indices map to the acquired source, and body_ids\npreserve source identities. See manifest.json for every graph checksum,\nselection identity, model configuration, assumptions and limitations.\nGraphs derive from MaleCNS, https://male-cns.janelia.org/download/, CC BY 4.0.\nLicense: https://creativecommons.org/licenses/by/4.0/\nThe acquired runtime derivative comes from Xenova/fruit-fly-simulation; its\npinned Hugging Face source URL and revision are preserved in manifest.json.\nFLM original code: Kuber Mehta, MIT.\n'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,9,11,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info,data)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or any(archive.read(name) != data for name,data in contents.items()):
            raise ValueError('Graph archive roundtrip failed')
    release = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive_bytes=archive_path.stat().st_size,
        archive_sha256=sha256(archive_path), manifest_sha256=sha256(destination/'manifest.json'),
        graphs=len(records), all_roundtrips_and_forward_fixtures_passed=True,
        frozen_reference_arrays_exact=list(reference), graph_count_is_not_a_language_fit_declaration=True)
    return manifest, release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/processed/connectome'))
    parser.add_argument('--selections', type=Path, default=Path('reports/selection-controls'))
    parser.add_argument('--output', type=Path, default=Path('work/selection-graphs-v1'))
    parser.add_argument('--archive', type=Path, default=Path('public/research/selection-graphs.zip'))
    args = parser.parse_args()
    manifest, release = export(args.source, args.selections, args.output, args.archive)
    write_json(Path('reports/selection-graphs/manifest.json'),manifest)
    write_json(Path('reports/selection-graphs/export-release.json'),release)
    print(json.dumps(release,indent=2))


if __name__ == '__main__': main()
