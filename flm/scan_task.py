"""Causal instruction/target formatting and strict action-sequence measurements.

No grammar interpreter is used to construct a model's prediction. Targets here
come from the already audited source records and are for supervised loss only.
"""
from __future__ import annotations

from .scan import ACTIONS


def prompt(command):
    if not isinstance(command, str) or not command or command != ' '.join(command.split()):
        raise ValueError('Expected one canonical nonempty command')
    return f'IN: {command}\nOUT:'


def encode_example(record, lexicon):
    actions = record['actions']
    if not isinstance(actions, (list, tuple)) or not actions or any(action not in ACTIONS for action in actions):
        raise ValueError('Invalid supervised action sequence')
    prefix = prompt(record['command'])
    target = ' ' + ' '.join(actions)
    prefix_ids = lexicon.encode(prefix)
    content_ids = lexicon.encode(prefix + target)
    if content_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError('Tokenizer merged across the instruction/target boundary')
    if lexicon.decode(content_ids) != prefix + target:
        raise ValueError('Tokenization changed an instruction or action byte')
    tokens = [0] + content_ids + [1]
    start = len(prefix_ids)
    # x[i] predicts y[i] == tokens[i+1]. The first output is therefore at
    # position len(prefix_ids), and the final supervised prediction is EOS.
    mask = [False] * start + [True] * (len(tokens) - 1 - start)
    return dict(input=tokens[:-1], target=tokens[1:], loss_mask=mask,
                generation_prefix=[0] + prefix_ids, supervised_tokens=sum(mask),
                target_utf8_bytes=len(target.encode('utf8')), output_start=start)


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, actual in enumerate(reference, 1):
        current = [i]
        for j, predicted in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (actual != predicted)))
        previous = current
    return previous[-1]


def score_actions(reference, prediction, *, terminated):
    if not isinstance(reference, (list, tuple)) or not reference or any(action not in ACTIONS for action in reference):
        raise ValueError('Reference must be a nonempty SCAN action sequence')
    if not isinstance(prediction, (list, tuple)) or any(not isinstance(action, str) for action in prediction):
        raise ValueError('Predictions must retain their complete parsed token list')
    if type(terminated) is not bool:
        raise ValueError('EOS termination must be recorded explicitly')
    valid = bool(prediction) and all(action in ACTIONS for action in prediction)
    same = list(reference) == list(prediction)
    distance = edit_distance(reference, prediction)
    return dict(exact_match=valid and same and terminated, actions_match_without_eos=same,
                valid_action_sequence=valid, terminated=terminated,
                reference_actions=len(reference), predicted_actions=len(prediction),
                edit_distance=distance, edits_per_reference_action=distance/len(reference))
