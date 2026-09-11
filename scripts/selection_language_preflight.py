"""Verify the complete selection/rewiring language interface without corpus fitting."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import torch

from flm.language_learning_inputs import load_corpus
from flm.provenance import sha256
from flm.selection_language import load_catalog,prepare_case
from flm.selection_pilot import prerequisites
from flm.language_learning_train import Settings,replay


def preflight(root):
    corpus=load_corpus(root); catalog=load_catalog(root)
    previous_threads=torch.get_num_threads(); results=[]; matched={}; common={}; windows={}
    try:
        torch.set_num_threads(1)
        for seed in (42,43):
            settings=Settings(steps=15,batch=16,sequence=96,warmup=16,learning_rate=.002,final_learning_rate=.002,
                lr_warmup_updates=0,weight_decay=.01,gradient_clip=1.,checkpoint_interval=15,seed=seed,threads=4,score_boundaries=True)
            _,exposure,digest=replay(corpus.documents,corpus.lexicon.lengths,settings,15)
            windows[str(seed)]=dict(sampled_windows_sha256=digest.hexdigest(),exposure=exposure,
                scope='Sampling only at the pilot dimensions; no forward call, gradient update or timing')
        for index,(label,entry) in enumerate(catalog.entries.items(),1):
            initial={}
            for seed in (42,43):
                model,binding=prepare_case(catalog,corpus,label,seed)
                initial[str(seed)]={key:binding[key] for key in ('initial_state_sha256','initial_parameters_sha256','initial_nonedge_parameters_sha256')}
                matched.setdefault((entry['selection'],seed),set()).add(binding['initial_parameters_sha256'])
                common.setdefault((entry['candidate'],seed),set()).add(binding['initial_nonedge_parameters_sha256'])
            results.append(dict(label=label,selection=entry['selection'],kind=entry['kind'],graph_seed=entry['graph_seed'],
                graph_sha256=entry['graph_sha256'],neurons=entry['config']['neurons'],edges=entry['edges'],
                trainable_parameters=entry['trainable_parameters'],initialization=initial))
            if index%32==0: print(f'Prepared {index}/256 graph definitions for both training seeds',flush=True)
    finally: torch.set_num_threads(previous_threads)
    if len(results)!=256 or any(len(values)!=1 for values in matched.values()) or any(len(values)!=1 for values in common.values()):
        raise ValueError('Incomplete or mismatched graph initialization inventory')
    try: prerequisites(root)
    except ValueError as error: gate=dict(ready=False,reason=str(error))
    else: gate=dict(ready=True,process_handles_still_require_inspection=True)
    sources=('flm/selection_language.py','flm/language_learning_inputs.py','flm/language_learning_train.py',
        'flm/model.py','flm/train.py','flm/wiring_controls.py','flm/inference.py','flm/provenance.py',
        'flm/selection_pilot.py','flm/corpus_cache.py','flm/tokenizer.py','flm/babylm.py',
        'flm/language_eligibility.py','flm/embedding_eligibility.py','flm/local_learning.py',
        'scripts/selection_language_preflight.py','tests/test_selection_language.py','tests/test_language_learning_inputs.py')
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),catalog_binding=catalog.binding,
        corpus_binding=corpus.binding,graph_definitions=256,training_seed_initializations=512,
        matched_learned_parameters_across_measured_and_rewired=True,
        matched_nonedge_parameters_across_same_size_selectors=True,training_windows=windows,graphs=results,
        source_sha256={name:sha256(root/name) for name in sources},timing_prerequisites=gate,
        model_forward_calls=0,gradient_updates=0,validation_or_test_payloads_opened=False,
        official_training_matrix_frozen=False,official_training_budget_selected=False,official_corpus_fit_started=False,
        scope='Full acquired training-source and structural-archive verification, model initialization and sampler replay. This prepares 512 possible conditions; it does not commit to fitting that matrix or report learning, timing, language quality or biological behavior.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated preparation records')
    result=preflight(Path.cwd()); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps({key:result[key] for key in ('graph_definitions','training_seed_initializations','timing_prerequisites','scope')},indent=2))
