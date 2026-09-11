"""Independent NumPy/SciPy audit of a completed untrained rewiring archive.

This module does not import FLM, Torch or the swap generator. It verifies supplied
graphs and receipts, not the swap trajectory, mixing or language performance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SEEDS = (101,103,107)
FIELDS = {'row','col','weight','contacts','source_sign','body_ids','source_indices','positions','pool'}


def read_graph(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def topology(graph):
    row,col = graph['row'],graph['col']; n = len(graph['body_ids'])
    edges = set(zip(row.tolist(),col.tolist()))
    nonself = {(i,j) for i,j in edges if i != j}
    adjacency = csr_matrix((np.ones(len(row),dtype=np.uint8),(row,col)),shape=(n,n))
    components,labels = connected_components(adjacency,directed=True,connection='strong')
    return dict(neurons=n,edges=len(edges),self_edges=int(np.sum(row == col)),
        reciprocal_off_diagonal_fraction=sum((j,i) in edges for i,j in nonself)/max(1,len(nonself)),
        strong_components=int(components),strong_component_sizes=sorted(np.bincount(labels).tolist(),reverse=True))


def audit_pair(original, changed):
    if set(original) != FIELDS or set(changed) != FIELDS:
        raise ValueError('Unexpected graph arrays')
    n = len(original['body_ids']); e = len(original['row'])
    if n < 1 or e < 1:
        raise ValueError('Empty graph')
    for name in FIELDS:
        if original[name].shape != changed[name].shape or original[name].dtype != changed[name].dtype:
            raise ValueError('Array shape or dtype changed: '+name)
        if name != 'col' and not np.array_equal(original[name],changed[name],equal_nan=True):
            raise ValueError('Fixed array changed: '+name)
    for name in ('row','col','contacts','weight'):
        if original[name].shape != (e,): raise ValueError('Malformed edge array')
    for name in ('body_ids','source_sign','source_indices','pool'):
        if original[name].shape != (n,): raise ValueError('Malformed node array')
    if original['positions'].shape != (n,3): raise ValueError('Malformed positions')
    row,col,old = changed['row'],changed['col'],original['col']
    if not all(np.issubdtype(x.dtype,np.integer) for x in (row,col,old)):
        raise ValueError('Noninteger endpoints')
    if np.any(row < 0) or np.any(row >= n) or np.any(col < 0) or np.any(col >= n) or np.any(old < 0) or np.any(old >= n):
        raise ValueError('Invalid endpoint')
    before_edges = set(zip(row.tolist(),old.tolist())); after_edges = set(zip(row.tolist(),col.tolist()))
    if len(before_edges) != e or len(after_edges) != e: raise ValueError('Duplicate edge')
    if before_edges == after_edges: raise ValueError('Topology did not change')
    if not np.array_equal(np.bincount(old,minlength=n),np.bincount(col,minlength=n)):
        raise ValueError('Outgoing degree changed')
    if not np.array_equal(row == old,row == col): raise ValueError('Self edge changed')
    signs = original['source_sign']
    if (not np.isin(signs,[-1,0,1]).all() or not np.array_equal(signs[old],signs[col])
            or not np.array_equal(np.sign(original['weight']),signs[col])):
        raise ValueError('Source sign constraint changed')
    if not np.isfinite(original['weight']).all() or np.any(original['contacts'] <= 0):
        raise ValueError('Invalid source magnitudes')
    return dict(original=topology(original),randomized=topology(changed),
        original_edge_overlap_fraction=len(before_edges & after_edges)/e,
        unchanged_endpoint_slots_fraction=float(np.mean(old == col)),
        preserved=dict(in_degree=True,out_degree=True,incoming_sign_counts=True,
            incoming_signed_weight_multiset=True,self_edges_and_weights=True,node_identities_positions_and_pools=True))


def check_receipt(receipt, source_sha256, seed, original, changed=None):
    binding = receipt['binding']
    if (binding['source_graph_sha256'] != source_sha256 or binding['seed'] != seed
            or binding['swaps_per_edge'] != 10 or binding['max_proposals_per_edge'] != 100):
        raise ValueError('Rewiring binding differs from declared case')
    if receipt['status'] == 'failed':
        if changed is not None or not isinstance(receipt.get('reason'),str) or not receipt['reason']:
            raise ValueError('Malformed failure record')
        return dict(status='failed',reason=receipt['reason'],failure_not_reexecuted=True)
    if receipt['status'] != 'complete' or changed is None:
        raise ValueError('Unfinished rewiring case')
    measured = audit_pair(original,changed); report = receipt['report']; e = len(original['row'])
    if report['accepted_swaps'] != 10*e or not 10*e <= report['proposed_swaps'] <= 100*e:
        raise ValueError('Invalid declared swap budget')
    if report['seed'] != seed or report['acceptance_fraction'] != report['accepted_swaps']/report['proposed_swaps']:
        raise ValueError('Invalid seed or acceptance arithmetic')
    for field,value in measured.items():
        if report[field] != value: raise ValueError('Reported structural diagnostic disagrees: '+field)
    return dict(status='complete',**measured)


def audit_archive(path, expected_sha256=None):
    digest = lambda data: hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest(path.read_bytes()) != expected_sha256:
        raise ValueError('Archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None: raise ValueError('Archive CRC failed')
        manifest = json.loads(archive.read('rewiring-manifest.json'))
        source_bytes = archive.read('source-manifest.json')
        if digest(source_bytes) != manifest['source_manifest_sha256']:
            raise ValueError('Source manifest checksum changed')
        source = json.loads(source_bytes)
        expected = {name+f'/null{seed}' for name in source['graphs'] for seed in SEEDS}
        if set(manifest['records']) != expected:
            raise ValueError('Incomplete or extra control inventory')
        originals = {}
        expected_files = {'README.txt','source-manifest.json','rewiring-manifest.json'}
        for name,record in source['graphs'].items():
            filename = 'original/'+record['path']; expected_files.add(filename)
            data = archive.read(filename)
            if digest(data) != record['graph_sha256']: raise ValueError('Original graph checksum changed')
            originals[name] = read_graph(data)
        results = {}
        for name,receipt in manifest['records'].items():
            selection,seed_label = name.rsplit('/',1); seed = int(seed_label.removeprefix('null'))
            if (receipt['binding']['generator_source_sha256'] != manifest['generator_source_sha256']
                    or receipt['binding']['runner_source_sha256'] != manifest['runner_source_sha256']):
                raise ValueError('Mixed generation sources')
            changed = None
            if receipt['status'] == 'complete':
                filename = 'rewired/'+name+'.npz'; expected_files.add(filename)
                data = archive.read(filename)
                if digest(data) != receipt['graph_sha256']: raise ValueError('Rewired graph checksum changed')
                changed = read_graph(data)
            results[name] = check_receipt(receipt,source['graphs'][selection]['graph_sha256'],seed,originals[selection],changed)
        if set(archive.namelist()) != expected_files or len(archive.namelist()) != len(expected_files):
            raise ValueError('Unexpected or duplicate archive entries')
    complete = sum(r['status'] == 'complete' for r in results.values())
    failed = sum(r['status'] == 'failed' for r in results.values())
    if manifest['complete'] != complete or manifest['failed'] != failed or manifest['planned'] != len(expected):
        raise ValueError('Manifest completion totals disagree')
    return dict(original_graphs=len(originals),planned=len(expected),complete=complete,failed=failed,
        arrays_degrees_signs_weights_self_edges_and_diagnostics_verified=True,
        failures_not_reexecuted=failed,
        scope='Supplied untrained graph and receipt audit; no replay of swap trajectories, mixing claim or language evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive,args.sha256),indent=2))


if __name__ == '__main__': main()
