"""Record coordinator readiness and real metadata gates without initializing a study."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from flm.language_learning_inputs import conditions
from flm.language_learning_test import POLICY, SUMMARY
from flm.language_learning_pilot import prerequisites
from flm.language_learning_study import IDENTITY, PILOT, PROTOCOL, SELECTION, SOURCES, test_metadata, verify_pilot
from flm.provenance import sha256
from flm.tokenizer import Lexicon


def preflight(root):
    lexicon = Lexicon(root/'data/tokenizers/babylm-2026-4096/tokenizer.json')
    metadata = test_metadata(root, lexicon)
    gates = {}
    for name, check in (('priority_completion', prerequisites), ('measured_cost_pilot', verify_pilot)):
        try:
            check(root); gates[name] = dict(accepted=True)
        except ValueError as error:
            gates[name] = dict(accepted=False, reason=str(error))
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), conditions=conditions(),
        gates=gates, test_metadata=metadata,
        official_protocol_exists=(root/PROTOCOL).exists(), official_timing_exists=(root/PILOT).exists(),
        official_identity_exists=(root/IDENTITY).exists(), official_selection_exists=(root/SELECTION).exists(),
        source_sha256={**{'flm/'+name:sha256(root/'flm'/name) for name in SOURCES},
            **{name:sha256(root/name) for name in ('scripts/language_learning_study_preflight.py',
                                                 'tests/test_language_learning_study.py', 'tests/test_language_learning_test.py')}},
        model_forward_calls=0, gradient_updates=0, test_payloads_opened=False,
        official_study_initialized=False, official_condition_trained=False,
        all_condition_selection_gate_implemented=True, official_test_scorer_implemented=True,
        evaluation_policy=POLICY, official_test_summary_exists=(root/SUMMARY).exists(),
        scope='Code readiness, prerequisite status and existing test metadata only. No budget chosen, study identity frozen, corpus fit, model likelihood or test payload read. Completion metadata is not a live process lock.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated study preparation records')
    result = preflight(Path.cwd()); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(dict(gates=result['gates'], conditions=len(result['conditions']),
        official_study_initialized=False, test_payloads_opened=False), indent=2))
