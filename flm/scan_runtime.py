"""Shared masked training and unconstrained generation for instruction transfer.

This module has no corpus loader, optimizer, training schedule or test selector.
The later registered study must supply audited partitions and frozen conditions.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.nn import functional as F

from .scan_task import decode_action_ids, encode_example, prompt, score_actions


def instruction_batch(records, lexicon, *, context_limit=96, device='cpu'):
    """Right-pad independent complete examples; supervise only actions and EOS.

    Repeated source rows remain repeated examples. Padding uses the existing BOS
    ID, carries no loss, and follows all real tokens so it cannot affect earlier
    causal predictions. Each example starts from a reset model state.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('A nonempty sequence of instruction records is required')
    if type(context_limit) is not int or context_limit < 1:
        raise ValueError('Context limit must be a positive integer')
    examples = [encode_example(row, lexicon, action_format='byte') for row in records]
    lengths = [len(row['input']) for row in examples]
    if max(lengths) > context_limit:
        raise ValueError('A complete instruction example exceeds the shared context limit')
    shape = (len(records), max(lengths))
    inputs = torch.zeros(shape, dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    for index, row in enumerate(examples):
        length = lengths[index]
        inputs[index, :length] = torch.tensor(row['input'], dtype=torch.long, device=device)
        targets[index, :length] = torch.tensor(row['target'], dtype=torch.long, device=device)
        mask[index, :length] = torch.tensor(row['loss_mask'], dtype=torch.bool, device=device)
    if (inputs < 0).any() or (inputs >= lexicon.vocabulary).any() or (targets >= lexicon.vocabulary).any():
        raise ValueError('Instruction IDs leave the shared vocabulary')
    return dict(input=inputs, target=targets, loss_mask=mask, lengths=tuple(lengths),
                examples=len(records), input_tokens=sum(lengths), supervised_tokens=int(mask.sum()),
                vocabulary=lexicon.vocabulary)


def masked_nll(logits, targets, mask):
    """Mean next-token NLL over action/EOS positions, with no prompt/padding loss.

    This weights supervised tokens equally, rather than assigning equal loss
    weight to examples with different target lengths. Gradients can still flow
    through unscored instruction tokens into earlier recurrent/attention states.
    """
    if (logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape or
            targets.dtype != torch.long or mask.dtype != torch.bool or not mask.any()):
        raise ValueError('Matching logits, target IDs and a nonempty boolean loss mask are required')
    selected = logits[mask]
    if not torch.isfinite(selected).all():
        raise FloatingPointError('Nonfinite supervised instruction logits')
    return F.cross_entropy(selected, targets[mask], reduction='mean')


def batch_loss(model, batch):
    if model.config.vocabulary != batch['vocabulary']:
        raise ValueError('Instruction batch and model vocabulary differ')
    if hasattr(model.config, 'window') and max(batch['lengths']) > model.config.window:
        raise ValueError('Instruction batch exceeds the model attention window')
    logits, _ = model(batch['input'], None)
    return masked_nll(logits, batch['target'], batch['loss_mask'])


def greedy_instructions(model, lexicon, commands, *, maximum_tokens=49, batch_size=32, context_limit=96):
    """Generate from command prefixes alone, over the full existing vocabulary.

    Equal-length prefixes share a batch without padding. Results retain caller
    order, duplicate commands, EOS and invalid token IDs. Rows stop independently
    at EOS; finished rows remain in a batch but their later logits are ignored.
    No reference actions or grammar oracle enter this function.
    """
    if (not isinstance(commands, (list, tuple)) or not commands or
            any(type(value) is not int or value < 1 for value in (maximum_tokens, batch_size, context_limit))):
        raise ValueError('Commands and positive integer generation limits are required')
    if model.config.vocabulary != lexicon.vocabulary:
        raise ValueError('Instruction model and tokenizer vocabulary differ')
    prefixes = [[0] + lexicon.encode(prompt(command)) for command in commands]
    limit = min(context_limit, getattr(model.config, 'window', context_limit))
    if any(len(prefix) + maximum_tokens - 1 > limit for prefix in prefixes):
        raise ValueError('Generation prefix and output cap exceed the shared context limit')
    if any(type(token) is not int or not 0 <= token < lexicon.vocabulary for prefix in prefixes for token in prefix):
        raise ValueError('Instruction prefix leaves the shared vocabulary')
    groups = defaultdict(list)
    for index, prefix in enumerate(prefixes):
        groups[len(prefix)].append(index)
    device = next(model.parameters()).device
    output = [None] * len(commands)
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for length in sorted(groups):
                indices = groups[length]
                for start in range(0, len(indices), batch_size):
                    chosen_indices = indices[start:start + batch_size]
                    tokens = torch.tensor([prefixes[index] for index in chosen_indices], dtype=torch.long, device=device)
                    logits, state = model(tokens, None)
                    histories = [[] for _ in chosen_indices]
                    active = torch.ones(len(chosen_indices), dtype=torch.bool, device=device)
                    for step in range(maximum_tokens):
                        scores = logits[:, -1]
                        if scores.shape != (len(chosen_indices), lexicon.vocabulary) or not torch.isfinite(scores[active]).all():
                            raise FloatingPointError('Invalid or nonfinite active generation logits')
                        chosen = scores.argmax(dim=-1)  # No action-vocabulary restriction or output repair.
                        for row, token in enumerate(chosen.tolist()):
                            if bool(active[row]):
                                histories[row].append(token)
                        active &= chosen != 1
                        if not bool(active.any()) or step + 1 == maximum_tokens:
                            break
                        feedback = torch.where(active, chosen, torch.ones_like(chosen))
                        logits, state = model(feedback[:, None], state)
                    for index, generated in zip(chosen_indices, histories):
                        decoded = decode_action_ids(generated, lexicon)
                        output[index] = dict(command=commands[index], prefix_tokens=prefixes[index],
                            tokens=generated, **decoded, stop_reason='eos' if decoded['terminated'] else 'length',
                            maximum_tokens=maximum_tokens)
    finally:
        model.train(was_training)
    return output


def score_outputs(records, outputs, lexicon, *, maximum_tokens=49):
    """Score the entire supplied cohort, including invalid and unfinished traces.

    Raw generated IDs are decoded again; supplied action labels cannot replace
    them. The later study owns split membership, expected cohort identities and
    file provenance. This function does not declare a supplied cohort to be a
    complete official test partition.
    """
    if (not isinstance(records, (list, tuple)) or not records or len(records) != len(outputs) or
            type(maximum_tokens) is not int or maximum_tokens < 1):
        raise ValueError('A complete nonempty supplied cohort and declared output cap are required')
    rows = []
    for index, (record, output) in enumerate(zip(records, outputs)):
        expected_prefix = [0] + lexicon.encode(prompt(record['command']))
        if (output['command'] != record['command'] or output['prefix_tokens'] != expected_prefix or
                output['maximum_tokens'] != maximum_tokens):
            raise ValueError('Generation is bound to a different command or cap')
        tokens = output['tokens']
        decoded = decode_action_ids(tokens, lexicon)
        if any(output.get(key) != value for key, value in decoded.items()):
            raise ValueError('Supplied action trace differs from its raw generated tokens')
        expected_stop = 'eos' if decoded['terminated'] else 'length'
        if (not 1 <= len(tokens) <= maximum_tokens or output['stop_reason'] != expected_stop or
                (not decoded['terminated'] and len(tokens) != maximum_tokens)):
            raise ValueError('Generation ended outside its declared EOS/cap rule')
        metric = score_actions(record['actions'], decoded['actions'], terminated=decoded['terminated'])
        rows.append(dict(row=index, command=record['command'], tokens=list(tokens), actions=decoded['actions'],
            stop_reason=expected_stop, invalid_token_output=any(action.startswith('INVALID_TOKEN:') for action in decoded['actions']),
            **metric))

    def aggregate(chosen):
        count = len(chosen)
        correct = sum(row['exact_match'] for row in chosen)
        edits = sum(row['edit_distance'] for row in chosen)
        reference_actions = sum(row['reference_actions'] for row in chosen)
        invalid = sum(row['invalid_token_output'] for row in chosen)
        invalid_or_empty = sum(not row['valid_action_sequence'] for row in chosen)
        terminated = sum(row['terminated'] for row in chosen)
        return dict(examples=count, exact_matches=correct, exact_match_rate=correct/count,
            total_edit_distance=edits, mean_edit_distance=edits/count,
            total_reference_actions=reference_actions, edits_per_reference_action=edits/reference_actions,
            invalid_token_outputs=invalid, invalid_token_output_rate=invalid/count,
            invalid_or_empty_action_outputs=invalid_or_empty,
            invalid_or_empty_action_output_rate=invalid_or_empty/count,
            eos_terminated=terminated, eos_termination_rate=terminated/count,
            cap_exhausted=count-terminated, cap_exhaustion_rate=(count-terminated)/count)

    lengths = sorted({row['reference_actions'] for row in rows})
    return dict(rows=rows, aggregate=aggregate(rows),
                by_reference_length=[dict(reference_length=length,
                    **aggregate([row for row in rows if row['reference_actions'] == length])) for length in lengths],
                maximum_tokens=maximum_tokens,
                scope='Supplied-cohort sequence scores; official partition coverage must be verified by the study harness')
