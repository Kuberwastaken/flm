"""Measure the frozen language subset's selection and severed graph boundary."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from .provenance import sha256, write_json


def measure(offsets, sources, counts, eligible, selected, signs, *, chunk_rows=4096):
    """Count raw contacts exactly, without constructing a dense or full-copy graph."""
    n = len(eligible)
    if (n == 0 or any(np.ndim(value) != 1 for value in (offsets, sources, counts, eligible, selected, signs))
            or eligible.dtype != np.bool_ or selected.dtype != np.bool_
            or not all(np.issubdtype(value.dtype, np.integer) for value in (offsets, sources, counts))
            or len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != len(sources)
            or len(sources) != len(counts) or np.any(offsets[1:] < offsets[:-1])
            or len(selected) != n or len(signs) != n or chunk_rows < 1
            or not np.isin(signs, [-1, 0, 1]).all() or np.any(counts <= 0)):
        raise ValueError('Inconsistent graph audit inputs')
    groups = {name:dict(connections=0, contacts=0) for name in
              ('inside', 'outside_to_selected', 'selected_to_outside', 'outside', 'inside_active', 'inside_zero_sign')}
    fields = ('central_in_contacts','central_out_contacts','full_in_contacts','full_out_contacts',
              'inside_in_contacts','inside_out_contacts','boundary_in_contacts','boundary_out_contacts',
              'active_in_contacts','active_out_contacts','full_in_connections','full_out_connections',
              'boundary_in_connections','boundary_out_connections')
    node = {name:np.zeros(n,dtype=np.uint64) for name in fields}
    retained = []

    def add(field, index, weight=None):
        values = np.bincount(index, weights=weight, minlength=n)
        node[field] += values.astype(np.uint64)

    maximum_count = 0
    for start in range(0,n,chunk_rows):
        end = min(n,start+chunk_rows); a,b = int(offsets[start]),int(offsets[end])
        pre = np.asarray(sources[a:b],dtype=np.int64)
        post = np.repeat(np.arange(start,end,dtype=np.int64), np.diff(offsets[start:end+1]).astype(np.int64))
        contact = np.asarray(counts[a:b],dtype=np.uint64)
        if len(contact) and (np.any(pre < 0) or np.any(pre >= n) or np.any(contact == 0)):
            raise ValueError('Invalid source indices or zero contacts')
        maximum_count = max(maximum_count,int(contact.max(initial=0)))
        central = eligible[post] & eligible[pre]
        inside = selected[post] & selected[pre]
        incoming = selected[post] & ~selected[pre]
        outgoing = ~selected[post] & selected[pre]
        active = inside & (signs[pre] != 0)
        masks = dict(inside=inside,outside_to_selected=incoming,selected_to_outside=outgoing,
                     outside=~(selected[post]|selected[pre]),inside_active=active,inside_zero_sign=inside & ~active)
        for name,mask in masks.items():
            groups[name]['connections'] += int(mask.sum())
            groups[name]['contacts'] += int(contact[mask].sum(dtype=np.uint64))
        add('central_in_contacts',post[central],contact[central]); add('central_out_contacts',pre[central],contact[central])
        add('full_in_contacts',post,contact); add('full_out_contacts',pre,contact)
        add('full_in_connections',post); add('full_out_connections',pre)
        for mask, prefix in ((inside,'inside'),(incoming,'boundary'),(active,'active')):
            add(prefix+'_in_contacts',post[mask],contact[mask])
        for mask, prefix in ((inside,'inside'),(outgoing,'boundary'),(active,'active')):
            add(prefix+'_out_contacts',pre[mask],contact[mask])
        add('boundary_in_connections',post[incoming]); add('boundary_out_connections',pre[outgoing])
        retained.append((post[active],pre[active],contact[active]))
    total = sum(groups[name]['contacts'] for name in ('inside','outside_to_selected','selected_to_outside','outside'))
    if total >= 2**53:
        raise ValueError('Contact totals exceed the exact integer range of bincount weights')
    for key in ('contacts','connections'):
        if groups['inside'][key] != groups['inside_active'][key]+groups['inside_zero_sign'][key]:
            raise ValueError('Sign exclusion accounting failed')
    arrays = [np.concatenate([part[i] for part in retained]) for i in range(3)]
    return groups,node,arrays,maximum_count


def audit(root, output):
    source = root/'data/processed/connectome'; graph_path = root/'data/graphs/central-1024/graph.npz'
    card_path = graph_path.with_name('graph-card.json')
    card = json.loads(card_path.read_text(encoding='utf8'))
    if sha256(graph_path) != card['graph_sha256']:
        raise ValueError('The selected graph differs from its source card')
    for name,expected in card['source_file_sha256'].items():
        if sha256(source/name) != expected:
            raise ValueError('Acquired source file changed: '+name)
    metadata = json.loads(gzip.decompress((source/'neurons.json.gz').read_bytes()))
    load = lambda name:np.load(source/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    offsets,sources,counts,signs,ids = (load(name) for name in ('offsets','sources','counts','fast_sign','body_ids'))
    n=len(metadata); eligible=np.array([row[2]=='cb_intrinsic' for row in metadata])
    with np.load(graph_path,allow_pickle=False) as z:
        graph={name:z[name].copy() for name in z.files}
    selected_indices=graph['source_indices']; selected=np.zeros(n,dtype=bool);selected[selected_indices]=True
    if not np.array_equal(ids[selected_indices],graph['body_ids']) or not eligible[selected].all():
        raise ValueError('Selected source identities/classes changed')
    groups,node,retained,maximum_count=measure(offsets,sources,counts,eligible,selected,signs)
    strength=node['central_in_contacts']+node['central_out_contacts']
    # The original selector sums float32 nonnegative integer contact counts.
    # This bound proves those intermediate sums equal the exact integer audit.
    if maximum_count >= 2**24 or strength.max() >= 2**24:
        raise ValueError('Exact integer ranking is not proven equivalent to the float32 selector')
    population=np.flatnonzero(eligible)
    ordering=np.lexsort((ids[population],-strength[population].astype(np.int64)))
    ranked=population[ordering]
    replay=np.array(sorted(ranked[:len(selected_indices)],key=lambda i:(metadata[i][1],int(ids[i]))))
    if not np.array_equal(replay,selected_indices):
        raise ValueError('Anatomy-only selection replay differs from the frozen graph')
    ranks=np.zeros(n,dtype=np.int64);ranks[ranked]=np.arange(1,len(ranked)+1)
    local=np.full(n,-1,dtype=np.int64);local[selected_indices]=np.arange(len(selected_indices))
    row,col,contact=local[retained[0]],local[retained[1]],retained[2]
    order=np.lexsort((col,row));row,col,contact=row[order],col[order],contact[order]
    for name,actual in [('row',row),('col',col),('contacts',contact)]:
        if not np.array_equal(actual,graph[name]):
            raise ValueError('Retained source edges differ from the trained graph: '+name)
    raw=np.log1p(contact.astype(np.float32))
    denominator=np.bincount(row,weights=raw,minlength=len(selected_indices)).astype(np.float32)
    weights=raw/np.maximum(denominator[row],1e-12)*signs[selected_indices][col]
    if not np.array_equal(weights.astype(np.float32),graph['weight']):
        raise ValueError('Recomputed initial normalized weights differ from the graph')
    output.mkdir(parents=True,exist_ok=True)
    node_csv=output/'nodes.csv'
    with node_csv.open('w',encoding='utf8',newline='') as handle:
        names=['body_id','cell_type','source_class','side','transmitter_label','fast_sign','selection_rank','central_strength']+list(node)
        writer=csv.DictWriter(handle,fieldnames=names,lineterminator='\n');writer.writeheader()
        for index in selected_indices:
            writer.writerow(dict(body_id=int(ids[index]),cell_type=metadata[index][1],source_class=metadata[index][2],
                side=metadata[index][3],transmitter_label=metadata[index][4],fast_sign=int(signs[index]),
                selection_rank=int(ranks[index]),central_strength=int(strength[index]),
                **{name:int(values[index]) for name,values in node.items()}))
    full_types=Counter(row[1] for row in metadata)
    eligible_types=Counter(metadata[i][1] for i in population)
    retained_types=Counter(metadata[i][1] for i in selected_indices)
    types_csv=output/'cell-types.csv'
    with types_csv.open('w',encoding='utf8',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n');writer.writerow(['cell_type','source_neurons','eligible_neurons','selected_neurons'])
        for name in sorted(full_types):writer.writerow([name,full_types[name],eligible_types[name],retained_types[name]])
    named_examples={}
    for prefix in ('KC','MBON','EPG','PFN','PFL','FB','ER','LHCENT'):
        named_examples[prefix]={name:sum(meta[1].startswith(prefix) for meta in rows) for name,rows in
            [('source',metadata),('eligible',[metadata[i] for i in population]),('selected',[metadata[i] for i in selected_indices])]}
    inside=groups['inside']['contacts'];incoming=groups['outside_to_selected']['contacts'];outgoing=groups['selected_to_outside']['contacts']
    boundary=dict(incoming_cut_fraction=incoming/(inside+incoming),outgoing_cut_fraction=outgoing/(inside+outgoing),
                  incident_contacts_counting_internal_once=inside+incoming+outgoing,
                  retained_fast_fraction_of_incident_contacts=groups['inside_active']['contacts']/(inside+incoming+outgoing),
                  denominator_note='Incoming and outgoing fractions each count internal contacts once; their denominators differ. Incident-union totals count an internal contact once, not twice.')
    history=subprocess.check_output(['git','log','--reverse','--format=%H %aI %s','--','data/graphs/central-1024/graph.npz'],cwd=root,text=True).splitlines()
    report=dict(study='Selection and boundary audit of the frozen 1024-neuron language graph',
        graph_sha256=sha256(graph_path),graph_card_sha256=sha256(card_path),source_revision=card['source_revision'],
        sources=card['source_file_sha256'],auditor_sha256=sha256(Path(__file__)),
        population=dict(acquired=n,eligible=int(eligible.sum()),selected=len(selected_indices),
            fraction_of_acquired=len(selected_indices)/n,fraction_of_eligible=len(selected_indices)/int(eligible.sum()),
            selected_cell_type_labels=len(retained_types),selected_nonempty_cell_type_labels=len(set(retained_types)-{''}),
            selected_side_counts=dict(Counter(metadata[i][3] for i in selected_indices)),
            selected_source_signs={str(key):value for key,value in sorted(Counter(int(signs[i]) for i in selected_indices).items())}),
        selection=dict(rule='Top incoming-plus-outgoing raw contact strength within the cb_intrinsic population; body ID breaks ties; canonical order is cell type then body ID.',
            exact_rank_replay=True,exact_retained_edges_contacts_and_initial_weights=True,
            maximum_central_strength=int(strength.max()),last_selected_strength=int(strength[ranked[len(selected_indices)-1]]),
            first_excluded_strength=int(strength[ranked[len(selected_indices)]]),
            graph_git_history=history,interpretation='Anatomical eligibility plus a connectivity ranking and computational size limit; not a functionally complete circuit selection.'),
        connections=groups,boundary=boundary,cell_type_prefix_examples=named_examples,
        tables={path.name:dict(bytes=path.stat().st_size,sha256=sha256(path)) for path in (node_csv,types_csv)},
        limitations=['No boundary input or feedback from excluded neurons is simulated.',
            'Raw contact fractions do not estimate lost functional current or preserved computational capacity.',
            'Cell-type prefix examples use the published labels literally; they do not establish complete circuit membership.',
            'Zero-sign source outputs are removed after node selection; their loss is distinct from the subset boundary.',
            'Graph history and source replay are evidence of the recorded procedure, not proof about unrecorded design decisions.',
            'This audit is independent of language outcomes and does not establish any anatomical language advantage.'])
    write_json(output/'summary.json',report)
    print(json.dumps({key:report[key] for key in ('population','connections','boundary','cell_type_prefix_examples')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reports/subset-audit'))
    args=parser.parse_args();audit(Path('.').resolve(),args.output)


if __name__=='__main__':main()
