import copy
import unittest

from flm.scan import audit_pair, interpret, parse, statistics
from flm.scan_task import action_ids, decode_action_ids, encode_example, score_actions


class ByteLexicon:
    """An independent byte codec makes the loss-boundary check transparent."""
    vocabulary = 258
    def encode(self, text):
        return [byte + 2 for byte in text.encode('utf8')]

    def decode(self, tokens):
        return bytes(token - 2 for token in tokens).decode('utf8')


class ScanDataTests(unittest.TestCase):
    def test_direction_repetition_and_order_have_independent_expected_traces(self):
        self.assertEqual(interpret('jump'), ('I_JUMP',))
        self.assertEqual(interpret('turn opposite right twice'), ('I_TURN_RIGHT',) * 4)
        self.assertEqual(interpret('walk around left'), ('I_TURN_LEFT', 'I_WALK') * 4)
        self.assertEqual(interpret('jump left after walk twice'), ('I_WALK', 'I_WALK', 'I_TURN_LEFT', 'I_JUMP'))
        self.assertEqual(interpret('look thrice and run'), ('I_LOOK', 'I_LOOK', 'I_LOOK', 'I_RUN'))

    def test_invalid_commands_are_not_silently_normalized_or_interpreted(self):
        for command in ('', 'walk  left', 'jump\nleft', 'walk twice thrice', 'turn', 'fly',
                        'walk and run after jump', 'walk backwards', 'turn opposite around left'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                interpret(command)

    def test_source_target_corruption_is_detected_independently_of_checksum(self):
        with self.assertRaisesRegex(ValueError, 'interpretation disagrees'):
            parse(b'IN: walk after jump OUT: I_WALK I_JUMP\n', 'fixture')
        with self.assertRaises(ValueError):
            parse(b'IN: walk OUT: I_FLY\n', 'fixture')

    def test_repeated_primitive_rows_keep_their_weight_and_provenance(self):
        rows = parse(b'IN: jump OUT: I_JUMP\nIN: jump OUT: I_JUMP\n', 'fixture')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['id'], rows[1]['id'])
        self.assertEqual([row['source_line'] for row in rows], [1, 2])
        self.assertEqual(statistics(rows)['repeated_rows'], 1)
        self.assertEqual(statistics(rows)['actions'], 2)

    def test_command_overlap_is_rejected_but_action_equivalence_is_reported(self):
        rows = parse(b'IN: turn around left OUT: I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT\n'
                     b'IN: turn opposite left twice OUT: I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT\n', 'fixture')
        canonical = {row['command']: row['actions'] for row in rows}
        result = audit_pair(rows[:1], rows[1:], canonical, 'simple')
        self.assertEqual(result['shared_action_sequences'], 1)
        self.assertEqual(result['test_rows_with_action_sequence_seen_in_train'], 1)
        with self.assertRaisesRegex(ValueError, 'crosses'):
            audit_pair(rows, rows[1:], canonical, 'simple')

    def test_canonical_target_changes_and_missing_coverage_fail(self):
        rows = parse(b'IN: walk OUT: I_WALK\nIN: jump OUT: I_JUMP\n', 'fixture')
        canonical = {row['command']: row['actions'] for row in rows}
        changed = copy.deepcopy(rows); changed[1]['actions'] = ['I_RUN']
        with self.assertRaisesRegex(ValueError, 'canonical source'):
            audit_pair(changed[:1], changed[1:], canonical, 'simple')
        with self.assertRaisesRegex(ValueError, 'cover'):
            audit_pair(rows[:1], [], canonical, 'simple')


class ScanTaskTests(unittest.TestCase):
    def test_only_target_bytes_and_eos_receive_loss(self):
        example = encode_example({'command': 'walk', 'actions': ['I_WALK']}, ByteLexicon(), action_format='literal')
        scored = [token for token, active in zip(example['target'], example['loss_mask']) if active]
        self.assertEqual(ByteLexicon().decode(scored[:-1]), ' I_WALK')
        self.assertEqual(scored[-1], 1)
        self.assertEqual(len(example['input']), len(example['target']))
        self.assertEqual(len(example['target']), len(example['loss_mask']))
        self.assertEqual(ByteLexicon().decode(example['generation_prefix'][1:]), 'IN: walk\nOUT:')
        self.assertEqual(example['input'][example['output_start']], example['generation_prefix'][-1])

    def test_gold_action_changes_cannot_change_generation_prefix(self):
        first = encode_example({'command': 'walk', 'actions': ['I_WALK']}, ByteLexicon())
        second = encode_example({'command': 'walk', 'actions': ['I_JUMP']}, ByteLexicon())
        self.assertEqual(first['generation_prefix'], second['generation_prefix'])
        self.assertNotEqual(first['target'], second['target'])

    def test_byte_actions_are_one_token_each_and_roundtrip_exactly(self):
        reference = ['I_WALK', 'I_WALK', 'I_TURN_LEFT', 'I_JUMP']
        example = encode_example({'command': 'jump left after walk twice', 'actions': reference}, ByteLexicon())
        scored = [token for token, mask in zip(example['target'], example['loss_mask']) if mask]
        self.assertEqual(example['supervised_tokens'], len(reference) + 1)
        self.assertEqual(decode_action_ids(scored, ByteLexicon()), {'actions':reference, 'terminated':True})
        self.assertEqual(ByteLexicon().decode(scored[:-1]), 'WW<J')

    def test_invalid_generated_tokens_and_post_eos_text_are_not_repaired(self):
        result = decode_action_ids([0, 1], ByteLexicon())
        self.assertEqual(result['actions'], ['INVALID_TOKEN:0'])
        self.assertFalse(score_actions(['I_WALK'], result['actions'], terminated=result['terminated'])['exact_match'])
        self.assertEqual(decode_action_ids([], ByteLexicon()), {'actions':[], 'terminated':False})
        with self.assertRaises(ValueError):
            decode_action_ids([1, action_ids(ByteLexicon())['I_WALK']], ByteLexicon())

    def test_terminal_marker_is_required_for_exact_success(self):
        self.assertTrue(score_actions(['I_WALK'], ['I_WALK'], terminated=True)['exact_match'])
        unfinished = score_actions(['I_WALK'], ['I_WALK'], terminated=False)
        self.assertFalse(unfinished['exact_match'])
        self.assertTrue(unfinished['actions_match_without_eos'])

    def test_invalid_extra_missing_and_reordered_actions_remain_errors(self):
        for predicted in ([], ['I_WALK', 'I_WALK'], ['I_FLY'], ['I_WALK', 'explanation']):
            result = score_actions(['I_WALK'], predicted, terminated=True)
            self.assertFalse(result['exact_match'])
            self.assertGreater(result['edit_distance'], 0)
        result = score_actions(['I_TURN_LEFT', 'I_WALK'], ['I_WALK', 'I_TURN_LEFT'], terminated=True)
        self.assertEqual(result['edit_distance'], 2)
        self.assertFalse(result['exact_match'])

    def test_empty_or_ambiguous_reference_and_prediction_types_fail(self):
        for reference, prediction in (([], []), (['I_WALK'], 'I_WALK'), (['I_WALK'], [1])):
            with self.assertRaises(ValueError):
                score_actions(reference, prediction, terminated=True)


if __name__ == '__main__':
    unittest.main()
