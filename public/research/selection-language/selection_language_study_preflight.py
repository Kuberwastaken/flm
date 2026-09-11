"""Audit available complete selection groups and pending gates without model execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from flm.provenance import sha256
from flm.selection_language import load_catalog
from flm.selection_language_study import IDENTITY, PILOT, PROTOCOL, SELECTION, SOURCES, matrix
from flm.selection_pilot import prerequisites


def preflight(root):
    catalog = load_catalog(root); rows = []
    for group in sorted({r['candidate'] for r in catalog.entries.values()}):
        conditions = matrix(catalog, [group])
        entries = [catalog.entries[r['graph']] for r in conditions]
        rows.append(dict(candidate=group, possible_fits=len(conditions), graph_definitions=len({r['graph'] for r in conditions}),
            selectors=sorted({r['selector'] for r in entries}), training_seeds=sorted({r['seed'] for r in conditions}),
            neurons=entries[0]['config']['neurons'], allocated_parameters_min=min(r['trainable_parameters'] for r in entries),
            allocated_parameters_max=max(r['trainable_parameters'] for r in entries), selected_for_training=False))
    try: prerequisites(root)
    except ValueError as error: gate = dict(ready=False, reason=str(error))
    else: gate = dict(ready=True, process_handles_still_require_inspection=True)
    pending = {name:(root/path).exists() for name,path in
        (('measured_cost_pilot',PILOT),('official_protocol',PROTOCOL),('frozen_identity',IDENTITY),('frozen_selection',SELECTION))}
    if any(pending.values()): raise ValueError('This preparation record is only for the unregistered, pre-pilot stage')
    sources = ['flm/'+name for name in SOURCES]
    sources += ['scripts/selection_language_study_preflight.py','tests/test_selection_language_study.py']
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), catalog_binding=catalog.binding,
        available_groups=rows, possible_fit_count=sum(r['possible_fits'] for r in rows),
        source_sha256={name:sha256(root/name) for name in sources}, priority_gate=gate, official_files_present=pending,
        coordinator_implemented=True, official_held_out_scorer_implemented=False,
        model_initializations=0, model_forward_calls=0, gradient_updates=0, corpus_payloads_opened=False,
        scope='Catalog and coordinator readiness only. No group selected, budget frozen, cost measured or language model fitted. Synthetic coordinator tests are separate from this record.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated preparation records')
    result = preflight(Path.cwd()); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps({key:result[key] for key in ('possible_fit_count','priority_gate','official_files_present','scope')},indent=2))
