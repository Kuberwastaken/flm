"""Verify full-window pilot declarations and sampled exposure without training."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import torch

from flm.language_learning_inputs import fingerprint, load_inputs, prepare_condition
from flm.language_learning_pilot import Pilot, ordered_conditions, prerequisites
from flm.language_learning_train import declaration, replay
from flm.provenance import sha256


def preflight(root):
    inputs = load_inputs(root); graph_before = fingerprint(inputs.graph)
    rows = []; original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for condition in ordered_conditions():
            model, binding = prepare_condition(inputs, condition)
            settings = Pilot().settings(condition['seed'])
            declared = declaration(model, inputs.documents, inputs.lexicon, settings, condition['method'], binding)
            _, full_exposure, full_digest = replay(inputs.documents, inputs.lexicon.lengths, settings, settings.steps)
            _, warmup_exposure, _ = replay(inputs.documents, inputs.lexicon.lengths, settings, Pilot().warmup_updates)
            measured = {key: full_exposure[key]-warmup_exposure[key] for key in full_exposure}
            if measured['presented_tokens'] != 18432 or measured['scored_tokens'] <= 0:
                raise ValueError('Unexpected full-window pilot exposure')
            if fingerprint(inputs.graph) != graph_before: raise ValueError('Input graph changed')
            rows.append(dict(condition=condition, settings=declared['settings'],
                initial_state_sha256=declared['initial_state_sha256'],
                ordered_training_documents_sha256=declared['ordered_training_documents_sha256'],
                training_documents=declared['training_documents'], eligible_training_documents=declared['eligible_training_documents'],
                trainable_parameters=declared['trainable_parameters'], optimizer_parameter_names=declared['optimizer_parameter_names'],
                frozen_parameter_names=declared['frozen_parameter_names'], full_exposure=full_exposure,
                warmup_exposure=warmup_exposure, measured_exposure=measured,
                sampled_windows_sha256=full_digest.hexdigest(), gradient_updates_run=0))
    finally: torch.set_num_threads(original_threads)
    for seed in (42, 43):
        group = [r for r in rows if r['condition']['seed'] == seed]
        for field in ('initial_state_sha256', 'ordered_training_documents_sha256', 'sampled_windows_sha256'):
            if len({r[field] for r in group}) != 1: raise ValueError('Unmatched full-window preparation')
        if any(r['measured_exposure'] != group[0]['measured_exposure'] for r in group):
            raise ValueError('Unmatched masked target exposure')
    try:
        prerequisites(root); priority = 'completion files accepted; inspect actual process handles before launch'
    except ValueError as error:
        priority = str(error)
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), input_binding=inputs.binding,
        conditions=rows, condition_order_seed=617, prepared_conditions=len(rows), priority_gate=priority,
        source_sha256={name:sha256(root/name) for name in (
            'flm/language_learning_pilot.py', 'flm/language_learning_inputs.py', 'flm/language_learning_train.py',
            'flm/language_eligibility.py', 'flm/embedding_eligibility.py', 'flm/local_learning.py',
            'flm/model.py', 'flm/train.py', 'flm/inference.py', 'flm/corpus_cache.py', 'flm/tokenizer.py',
            'flm/provenance.py', 'scripts/language_learning_pilot_preflight.py', 'tests/test_language_learning_pilot.py')},
        full_window_sampling_verified=True, gradient_updates_run=0, full_size_timing_run_started=False,
        checkpoint_written=False, validation_or_test_payloads_opened=False, training_protocol_frozen=False,
        scope='Actual training-only input, initialization, declaration and sampled-mask verification at pilot dimensions; no forward/backward updates, timing observations, language score or official fit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated preflights; choose a new output')
    result = preflight(Path.cwd()); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(dict(prepared_conditions=result['prepared_conditions'], priority_gate=result['priority_gate'],
        exposures={r['condition']['label']:r['measured_exposure'] for r in result['conditions']},
        gradient_updates_run=0, full_size_timing_run_started=False), indent=2))
