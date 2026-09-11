"""Package only a complete terminal inventory of untrained structural controls."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

from flm.provenance import sha256,write_json
from flm.scan_train import training_lease
from scripts.audit_selection_rewiring import SEEDS,audit_archive


def inventory_gate(source,progress):
    expected = {name+f'/null{seed}' for name in source['graphs'] for seed in SEEDS}
    if (progress.get('status') != 'complete' or set(progress.get('records',{})) != expected
            or progress.get('planned_graphs') != len(expected)):
        raise ValueError('Rewiring inventory is incomplete; no final archive may be published')
    complete = sum(r.get('status') == 'complete' for r in progress['records'].values())
    failed = sum(r.get('status') == 'failed' for r in progress['records'].values())
    if complete+failed != len(expected) or progress.get('complete') != complete or progress.get('failed') != failed:
        raise ValueError('Terminal inventory counters disagree')
    return expected


def package(graph_directory,rewiring_directory,archive_path,report_directory):
    source_path = graph_directory/'manifest.json'; source_bytes = source_path.read_bytes(); source = json.loads(source_bytes)
    source_release = json.loads(Path('reports/selection-graphs/export-release.json').read_text(encoding='utf8'))
    if sha256(source_path) != source_release['manifest_sha256'] or len(source['graphs']) != source_release['graphs']:
        raise ValueError('Graph export manifest differs from verified release')
    progress_path = rewiring_directory/'progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf8'))
    inventory_gate(source,progress)
    if (progress['graph_manifest_sha256'] != sha256(source_path)
            or progress['runner_source_sha256'] != sha256(Path('flm/selection_rewiring.py'))
            or progress['generator_source_sha256'] != sha256(Path('flm/wiring_controls.py'))):
        raise ValueError('Graph preparation source identities changed')
    # This OS lock prevents publication while a generation/resume writer is live.
    with training_lease(rewiring_directory):
        if json.loads(progress_path.read_text(encoding='utf8')) != progress:
            raise ValueError('Progress changed while acquiring finalization lock')
        manifest = dict(verified_utc=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha256(source_path),
            generator_source_sha256=progress['generator_source_sha256'],runner_source_sha256=progress['runner_source_sha256'],
            planned=progress['planned_graphs'],complete=progress['complete'],failed=progress['failed'],records=progress['records'],
            graph_seeds=list(SEEDS),swaps_per_edge=10,max_proposals_per_edge=100,
            status='Terminal structural preparation; no language fits or final training matrix',
            limitations=['Transferred contacts and weights are original incoming-slot magnitudes, not measured new synapses.',
                         'Finite swap chains do not establish uniform sampling or sufficient mixing.',
                         'Failures are retained and not independently rerun by the arithmetic audit.',
                         'No graph-control generation result measures language learning or biological function.'])
        contents = {'source-manifest.json':source_bytes,
                    'rewiring-manifest.json':(json.dumps(manifest,indent=2)+'\n').encode('utf8')}
        for name,record in source['graphs'].items():
            path = graph_directory/record['path']
            if not path.resolve().is_relative_to(graph_directory.resolve()) or sha256(path) != record['graph_sha256']:
                raise ValueError('Original graph identity changed')
            contents['original/'+record['path']] = path.read_bytes()
        for name,receipt in progress['records'].items():
            case = rewiring_directory/name
            if not case.resolve().is_relative_to(rewiring_directory.resolve()): raise ValueError('Unsafe case path')
            if json.loads((case/'receipt.json').read_text(encoding='utf8')) != receipt:
                raise ValueError('Case receipt differs from final progress record')
            if receipt['status'] == 'complete':
                path = case/'graph.npz'
                if sha256(path) != receipt['graph_sha256']: raise ValueError('Rewired graph identity changed')
                contents['rewired/'+name+'.npz'] = path.read_bytes()
        contents['README.txt'] = b'FLM untrained anatomical selections and artificial rewiring controls\n\nSee source-manifest.json and rewiring-manifest.json for all identities,\nconstraints, seeds, diagnostics and retained failures. No trained weights\nor tokenizer are included. Rewired edges are artificial: transferred\ncontacts are input-slot magnitudes, not measurements of those body pairs.\nMaleCNS anatomical source: https://male-cns.janelia.org/download/\nLicense: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\nPinned runtime derivative attribution is in source-manifest.json.\nFLM original code: Kuber Mehta, MIT.\n'
        archive_path.parent.mkdir(parents=True,exist_ok=True)
        temporary = archive_path.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name,data in sorted(contents.items()):
                info = zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o644 << 16; archive.writestr(info,data)
        # A failed audit leaves the published path untouched.
        checked = audit_archive(temporary)
        temporary.replace(archive_path)
        report = dict(verified_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha256(archive_path),
            archive_bytes=archive_path.stat().st_size,packager_source_sha256=sha256(Path(__file__)),
            independent_auditor_source_sha256=sha256(Path('scripts/audit_selection_rewiring.py')),
            rewiring_manifest_sha256=hashlib.sha256(contents['rewiring-manifest.json']).hexdigest(),**checked)
        write_json(report_directory/'manifest.json',manifest)
        write_json(report_directory/'release.json',report)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs',type=Path,default=Path('work/selection-graphs-v1'))
    parser.add_argument('--rewiring',type=Path,default=Path('work/selection-rewiring-v1'))
    parser.add_argument('--archive',type=Path,default=Path('public/research/selection-rewiring.zip'))
    parser.add_argument('--report',type=Path,default=Path('reports/selection-rewiring'))
    args=parser.parse_args()
    print(json.dumps(package(args.graphs,args.rewiring,args.archive,args.report),indent=2))


if __name__ == '__main__': main()
