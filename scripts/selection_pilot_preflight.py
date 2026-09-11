"""Check full-size gradient compatibility with tiny synthetic token fixtures."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from flm.model import Config,load_graph
from flm.provenance import sha256
from flm.selection_pilot import Pilot,measure,ordered_selections,prepare_inputs,prerequisites


def preflight(root):
    graph_root=root/'work/selection-graphs-v1'; manifest_path=graph_root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    release=json.loads((root/'reports/selection-graphs/export-release.json').read_text(encoding='utf8'))
    if (sha256(manifest_path) != release['manifest_sha256'] or len(manifest['graphs']) != 64
            or manifest['model_source_sha256'] != sha256(root/'flm/model.py')):
        raise ValueError('Released graphs or model source changed')
    # This read verifies corpus availability and source bytes. These actual text
    # tokens are not passed to the synthetic gradient fixtures below.
    documents,lexicon,binding=prepare_inputs(root)
    blocks=len(documents); cache_tokens=sum(len(tokens) for _,tokens in documents)
    if any(tokens.min() < 0 or tokens.max() >= lexicon.vocabulary for _,tokens in documents):
        raise ValueError('Training cache contains invalid token IDs')
    del documents
    synthetic=[('synthetic-token-cycle',np.tile(np.array([17,29,43,59,71],dtype=np.int32),4))]
    pilot=Pilot(warmup_updates=1,measured_updates=1,batch=1,sequence=3,context_warmup=1,threads=1,seed=42)
    results=[]
    for name in ordered_selections(manifest):
        entry=manifest['graphs'][name]; path=graph_root/entry['path']
        if not path.resolve().is_relative_to(graph_root.resolve()) or sha256(path) != entry['graph_sha256']:
            raise ValueError('Graph identity changed')
        result=measure(load_graph(path),Config(**entry['config']),synthetic,lexicon.lengths,pilot)
        if result['parameter_card']['trainable_parameters'] != entry['trainable_parameters']:
            raise ValueError('Instantiated parameter allocation changed')
        if results and result['sampled_windows_sha256'] != results[0]['sampled_fixture_windows_sha256']:
            raise ValueError('Synthetic windows differ between selections')
        # Deliberately omit all fixture timing fields and all parameter states.
        results.append(dict(selection=name,graph_sha256=entry['graph_sha256'],
            neurons=entry['config']['neurons'],edges=entry['edges'],
            trainable_parameters=result['parameter_card']['trainable_parameters'],
            graph_unchanged=result['graph_unchanged'],disposable_parameters_updated=result['disposable_parameters_updated'],
            sampled_fixture_windows_sha256=result['sampled_windows_sha256']))
    try: prerequisites(root)
    except ValueError as error: gate=dict(ready=False,reason=str(error))
    else: gate=dict(ready=True,process_handles_still_require_inspection=True)
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),graphs_verified=len(results),
        graph_manifest_sha256=sha256(manifest_path),training_input_binding=binding,
        training_blocks_verified=blocks,training_cache_tokens_including_boundaries=cache_tokens,
        fixture=dict(synthetic_token_cycle=[17,29,43,59,71],updates_per_graph=2,batch=1,sequence=3,
            context_warmup=1,threads=1,initialization_and_sampling_seed=42),
        source_sha256={name:sha256(root/name) for name in (
            'flm/selection_pilot.py','flm/model.py','flm/train.py','flm/corpus_cache.py','flm/tokenizer.py',
            'flm/scan_train.py','flm/inference.py','scripts/selection_pilot_preflight.py','tests/test_selection_pilot.py')},
        official_timing_prerequisites=gate,official_timing_run_started=False,conditions=results,
        scope='Full-size graph gradient compatibility on tiny synthetic fixtures. Training cache verified separately; no corpus fit, throughput estimate, validation/test access or saved model.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated preflight records; choose a new output')
    result=preflight(Path.cwd()); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('conditions','source_sha256')},indent=2))
