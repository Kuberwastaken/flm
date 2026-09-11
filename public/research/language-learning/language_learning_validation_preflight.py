"""Verify validation-panel bytes and denominator identities without model scoring."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from flm.language_learning_validation import load_panel
from flm.provenance import sha256
from flm.tokenizer import Lexicon


def preflight(root):
    lexicon = Lexicon(root/'data/tokenizers/babylm-2026-4096/tokenizer.json')
    panel = load_panel(root, lexicon)
    components = {}
    for row in panel.binding['coverage']['documents']:
        name = row['document'].split('/')[1]
        stats = components.setdefault(name, dict(blocks=0, scored_tokens=0, scored_bytes=0))
        stats['blocks'] += 1; stats['scored_tokens'] += row['tokens']; stats['scored_bytes'] += row['bytes']
    if len(components) != 6 or any(row['blocks'] != 8 for row in components.values()):
        raise ValueError('Expected eight validation blocks for each of six components')
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), panel_binding=panel.binding,
        components=components, source_sha256={name:sha256(root/name) for name in (
            'flm/language_learning_validation.py', 'flm/language_learning_train.py', 'flm/language_train.py',
            'flm/language_learning_inputs.py', 'flm/corpus_cache.py', 'flm/tokenizer.py', 'flm/model.py',
            'flm/train.py', 'flm/inference.py', 'flm/provenance.py',
            'scripts/language_learning_validation_preflight.py', 'tests/test_language_learning_validation.py')},
        model_forward_calls=0, gradient_updates=0, checkpoint_selected=False,
        test_payloads_opened=False, study_identity_frozen=False, all_condition_test_gate_implemented=False,
        scope='Verified existing validation panel and exact scoring denominators; no model likelihoods, official checkpoint selection or test evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve dated preflight records')
    result = preflight(Path.cwd()); args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(dict(components=result['components'], model_forward_calls=0,
        gradient_updates=0, checkpoint_selected=False), indent=2))
