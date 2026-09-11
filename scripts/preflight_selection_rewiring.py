"""Audit a dated snapshot without modifying or resuming the live graph queue."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from scripts.audit_selection_rewiring import read_graph,check_receipt,SEEDS
from scripts.package_selection_rewiring import inventory_gate


def digest(data): return hashlib.sha256(data).hexdigest()


def preflight(graph_directory,rewiring_directory):
    source_bytes=(graph_directory/'manifest.json').read_bytes(); source=json.loads(source_bytes)
    progress_bytes=(rewiring_directory/'progress.json').read_bytes(); progress=json.loads(progress_bytes)
    if progress['graph_manifest_sha256'] != digest(source_bytes): raise ValueError('Source manifest identity changed')
    source_hashes={name:digest(Path(name).read_bytes()) for name in (
        'flm/selection_rewiring.py','flm/wiring_controls.py','scripts/audit_selection_rewiring.py',
        'scripts/package_selection_rewiring.py','scripts/preflight_selection_rewiring.py')}
    if (progress['runner_source_sha256'] != source_hashes['flm/selection_rewiring.py']
            or progress['generator_source_sha256'] != source_hashes['flm/wiring_controls.py']):
        raise ValueError('Generation sources changed')
    expected={name+f'/null{seed}' for name in source['graphs'] for seed in SEEDS}
    if not set(progress['records']).issubset(expected) or progress['planned_graphs'] != len(expected):
        raise ValueError('Unexpected queue inventory')
    complete=sum(r['status']=='complete' for r in progress['records'].values())
    failed=sum(r['status']=='failed' for r in progress['records'].values())
    if (complete+failed != len(progress['records']) or complete != progress['complete'] or failed != progress['failed']):
        raise ValueError('Snapshot counters disagree')
    if len(progress['records']) < len(expected):
        try: inventory_gate(source,progress)
        except ValueError as error: gate=dict(final_archive_allowed=False,reason=str(error))
        else: raise AssertionError('Incomplete queue unexpectedly passed publication gate')
    else:
        inventory_gate(source,progress); gate=dict(final_archive_allowed=True)
    originals={}; results={}
    for name,receipt in progress['records'].items():
        selection,seed_label=name.rsplit('/',1); seed=int(seed_label.removeprefix('null'))
        entry=source['graphs'][selection]
        if selection not in originals:
            path=graph_directory/entry['path']
            if not path.resolve().is_relative_to(graph_directory.resolve()): raise ValueError('Unsafe source path')
            data=path.read_bytes()
            if digest(data) != entry['graph_sha256']: raise ValueError('Source graph checksum changed')
            originals[selection]=read_graph(data)
        case=rewiring_directory/name
        if not case.resolve().is_relative_to(rewiring_directory.resolve()): raise ValueError('Unsafe case path')
        receipt_bytes=(case/'receipt.json').read_bytes()
        if json.loads(receipt_bytes) != receipt: raise ValueError('Committed receipt changed since snapshot')
        if (receipt['binding']['generator_source_sha256'] != progress['generator_source_sha256']
                or receipt['binding']['runner_source_sha256'] != progress['runner_source_sha256']):
            raise ValueError('Case generation source differs from queue')
        changed=None
        if receipt['status']=='complete':
            data=(case/'graph.npz').read_bytes()
            if digest(data) != receipt['graph_sha256']: raise ValueError('Rewired graph checksum changed')
            changed=read_graph(data)
        checked=check_receipt(receipt,entry['graph_sha256'],seed,originals[selection],changed)
        results[name]=dict(receipt_sha256=digest(receipt_bytes),graph_sha256=receipt.get('graph_sha256'),
            source_graph_sha256=entry['graph_sha256'],**checked)
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),snapshot_utc=progress['verified_utc'],
        progress_snapshot_sha256=digest(progress_bytes),source_manifest_sha256=digest(source_bytes),
        source_sha256=source_hashes,planned=len(expected),complete=complete,failed=failed,
        publication_gate=gate,checked=results,
        limitation='Dated terminal-case snapshot, not a live counter or language result. No swap-trajectory replay or mixing claim. Generation continues independently.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs',type=Path,default=Path('work/selection-graphs-v1'))
    parser.add_argument('--rewiring',type=Path,default=Path('work/selection-rewiring-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Keep dated audit records immutable; select a new output path')
    result=preflight(args.graphs,args.rewiring)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'checked'},indent=2))


if __name__=='__main__': main()
