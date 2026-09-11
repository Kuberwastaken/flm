"""Verify actual training inputs and separate tiny synthetic update compatibility."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import torch

from flm.inference import state_hash
from flm.language_learning_inputs import conditions,fingerprint,load_inputs,prepare_condition
from flm.language_learning_train import Settings,declaration,optimizer_for,update
from flm.local_learning import CORE_PARAMETERS
from flm.provenance import sha256


def preflight(root):
    inputs=load_inputs(root); original=fingerprint(inputs.graph)
    results=[]; old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for condition in conditions():
                model,binding=prepare_condition(inputs,condition)
                settings=Settings(steps=2,batch=1,sequence=3,warmup=1,learning_rate=.002,
                    final_learning_rate=.0002,lr_warmup_updates=0,weight_decay=.01,
                    gradient_clip=1.,checkpoint_interval=1,seed=condition['seed'],threads=1)
                # Declaration validation reads the real training inventory but
                # launches no fit. These dimensions are software fixtures only.
                declared=declaration(model,inputs.documents,inputs.lexicon,settings,condition['method'],
                    dict(binding,purpose='Input/declaration compatibility only; no official training budget'))
                initial=state_hash(model)
                core={name:p.detach().clone() for name,p in model.named_parameters() if name in CORE_PARAMETERS}
                optimizer=optimizer_for(model,settings)
                x=torch.tensor([[17,29,43]]); y=torch.tensor([[29,43,59]])
                for step in (1,2): update(model,optimizer,x,y,settings,condition['method'],step)
                if state_hash(model)==initial: raise ValueError('Synthetic optimizer did not change parameters')
                fixed_unchanged=all(torch.equal(dict(model.named_parameters())[name],value) for name,value in core.items())
                if condition['method']=='fixed_core' and not fixed_unchanged:
                    raise ValueError('Fixed core changed during synthetic update')
                if fingerprint(inputs.graph) != original: raise ValueError('Input graph was mutated')
                results.append(dict(condition=condition,initial_state_sha256=initial,
                    ordered_training_documents_sha256=declared['ordered_training_documents_sha256'],
                    training_documents=declared['training_documents'],
                    eligible_training_documents_at_fixture_sequence=declared['eligible_training_documents'],
                    trainable_parameters=declared['trainable_parameters'],
                    optimizer_parameter_names=declared['optimizer_parameter_names'],
                    frozen_parameter_names=declared['frozen_parameter_names'],
                    real_training_declaration_accepted=True,synthetic_updates_finite=True,
                    disposable_parameters_changed=True,
                    frozen_core_preserved=fixed_unchanged if condition['method']=='fixed_core' else None))
    finally: torch.set_num_threads(old_threads)
    for seed in (42,43):
        if len({r['initial_state_sha256'] for r in results if r['condition']['seed']==seed}) != 1:
            raise ValueError('Learning rules did not start from identical tensors')
    if len({r['ordered_training_documents_sha256'] for r in results}) != 1:
        raise ValueError('Learning-rule training inventories differ')
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),input_binding=inputs.binding,
        prepared_conditions=len(results),conditions=results,
        synthetic_fixture=dict(updates=2,batch=1,input_token_ids=[17,29,43],target_token_ids=[29,43,59],
            context_warmup=1,score_boundaries=False,threads=1),
        source_sha256={name:sha256(root/name) for name in (
            'flm/language_learning_inputs.py','flm/language_learning_train.py','flm/language_eligibility.py',
            'flm/embedding_eligibility.py','flm/local_learning.py','flm/model.py','flm/train.py',
            'flm/inference.py','flm/corpus_cache.py','flm/tokenizer.py','flm/babylm.py',
            'scripts/language_learning_input_preflight.py','tests/test_language_learning_inputs.py')},
        official_corpus_fit_started=False,checkpoint_written=False,validation_or_test_payloads_opened=False,
        full_size_timing_run_started=False,training_protocol_frozen=False,
        scope='Actual training-source and declaration audit plus separate synthetic updates. No language performance, corpus fit, timing benchmark or official evaluation gate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated preflights; choose a new output')
    result=preflight(Path.cwd()); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(verified_utc=result['verified_utc'],coverage=result['input_binding']['coverage'],
        prepared_conditions=result['prepared_conditions'],official_corpus_fit_started=False,
        full_size_timing_run_started=False,scope=result['scope']),indent=2))
