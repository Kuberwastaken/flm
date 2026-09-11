"""Generation/evaluation integration fixtures; no official SCAN scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from flm import scan_evaluate as evaluate, scan_study as study
from flm.provenance import sha256, write_json
from flm.scan import statistics
from flm.scan_conditions import conditions
from flm.scan_runtime import greedy_instructions, score_outputs
from flm.scan_task import action_ids, decode_action_ids, prompt
from test_scan_runtime import ToySequenceModel, RECORDS, models
from test_scan_train import FixtureLexicon
import test_scan_study as study_fixtures


ROWS = [dict(command='jump', actions=['I_JUMP']), dict(command='look', actions=['I_LOOK']),
        dict(command='run', actions=['I_RUN']), dict(command='walk', actions=['I_WALK']),
        dict(command='jump', actions=['I_JUMP'])]
POLICY = dict(evaluate.POLICY, batch_size=2, maximum_tokens=4, threads=1)


def coverage(rows):
    return dict(ordered_rows_sha256=evaluate.digest(rows), statistics=json.loads(json.dumps(statistics(rows))), fixture=True)


class ScanBatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.lexicon = FixtureLexicon()

    def generate(self, directory=None, model=None, rows=None):
        rows = ROWS if rows is None else rows
        return evaluate.generate_partition(model or ToySequenceModel(), self.lexicon, rows, coverage(rows),
            directory or self.root/'batches', {'fixture':True}, POLICY)

    def test_invalid_empty_and_cap_exhausted_outputs_remain_in_denominator(self):
        result = self.generate()
        aggregate = result['metrics']['aggregate']
        self.assertEqual(aggregate['examples'], 5)
        self.assertEqual(aggregate['exact_matches'], 2)
        self.assertEqual(aggregate['invalid_token_outputs'], 1)
        self.assertEqual(aggregate['invalid_or_empty_action_outputs'], 2)
        self.assertEqual(aggregate['cap_exhausted'], 1)
        self.assertEqual([r['command'] for r in result['metrics']['rows']], [r['command'] for r in ROWS])
        with patch.object(evaluate, 'greedy_instructions', side_effect=AssertionError('cached work repeated')):
            self.assertEqual(self.generate(), result)

    def test_interruption_resumes_only_missing_batches_with_identical_tokens(self):
        reference = self.generate(self.root/'reference')
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError('synthetic generation interruption')
            return greedy_instructions(*args, **kwargs)
        with patch.object(evaluate, 'greedy_instructions', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'synthetic generation interruption'): self.generate()
        with patch.object(evaluate, 'greedy_instructions', wraps=greedy_instructions) as generate:
            resumed = self.generate(); self.assertEqual(generate.call_count, 2)
        self.assertEqual(reference['metrics'], resumed['metrics'])

    def test_wrong_cached_tokens_repaired_actions_cap_or_row_identity_rejected(self):
        self.generate(); path = self.root/'batches/batch-00000.json'; original = study.read(path)
        faults = []
        row = copy.deepcopy(original); row['outputs'][0]['tokens'] = [0,1]; faults.append(row)
        row = copy.deepcopy(original); row['outputs'][0]['actions'] = ['I_WALK']; faults.append(row)
        row = copy.deepcopy(original); row['outputs'][0]['maximum_tokens'] = 3; faults.append(row)
        row = copy.deepcopy(original); row['identity']['start'] = 1; faults.append(row)
        row = copy.deepcopy(original); row['seconds'] = float('nan'); faults.append(row)
        for row in faults:
            write_json(path, row)
            with self.assertRaises(ValueError): self.generate()

    def test_holes_extra_batches_and_missing_identity_fail(self):
        self.generate(); folder = self.root/'batches'; path = folder/'batch-00001.json'
        original = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'noncontiguous'): self.generate()
        path.write_bytes(original); extra = folder/'batch-99999.json'; extra.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'inventory'): self.generate()
        extra.unlink(); (folder/'identity.json').unlink()
        with self.assertRaisesRegex(ValueError, 'without their identity'): self.generate()

    def test_generation_receives_commands_only_and_model_mutation_cannot_be_saved(self):
        model = ToySequenceModel()
        def mutate(model, lexicon, commands, **kwargs):
            self.assertTrue(all(isinstance(command, str) for command in commands))
            generated = greedy_instructions(model, lexicon, commands, **kwargs)
            with torch.no_grad(): model.dummy.add_(1)
            return generated
        with patch.object(evaluate, 'greedy_instructions', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'parameters changed'): self.generate(model=model)
        self.assertFalse((self.root/'batches/batch-00000.json').exists())

    def test_small_real_architectures_resume_exactly_without_new_generation(self):
        for model in models():
            folder = self.root/model.config.variant
            original = self.generate(folder, model, RECORDS)
            with patch.object(evaluate, 'greedy_instructions', side_effect=AssertionError('regenerated')):
                self.assertEqual(self.generate(folder, model, RECORDS), original)

    def test_full_49_token_policy_retains_unterminated_stream(self):
        result = evaluate.generate_partition(ToySequenceModel(), self.lexicon, ROWS, coverage(ROWS),
            self.root/'full-cap', {'fixture':True}, dict(POLICY, maximum_tokens=49))
        walk = result['metrics']['rows'][3]
        self.assertEqual(len(walk['tokens']), 49)
        self.assertFalse(walk['exact_match']); self.assertEqual(walk['stop_reason'], 'length')

    def test_comparison_signs_and_complete_inventory(self):
        results = []; codec = action_ids(self.lexicon)
        for condition in conditions():
            # Only the artificial pretrained FLM returns correct sequences here.
            correct = condition['variant'] == 'flm' and condition['initialization'] == 'wikitext'
            outputs = []
            for row in ROWS:
                tokens = [codec[action] for action in row['actions']]+[1] if correct else [1]
                outputs.append(dict(command=row['command'], prefix_tokens=[0]+self.lexicon.encode(prompt(row['command'])),
                    tokens=tokens, **decode_action_ids(tokens,self.lexicon), stop_reason='eos', maximum_tokens=4))
            results.append(dict(condition=condition, coverage=coverage(ROWS),
                metrics=score_outputs(ROWS,outputs,self.lexicon,maximum_tokens=4)))
        summary = evaluate.summarize(results)
        transfer = next(row for row in summary['paired_comparisons'] if row['first']=='flm-wikitext' and row['kind']=='transfer')
        self.assertEqual(transfer['exact_match_rate_difference'],1)
        self.assertEqual(transfer['mean_edit_distance_difference'],-1)
        self.assertEqual(transfer['first_only_correct'],5); self.assertEqual(transfer['neither_correct'],0)
        for bad in (results[:-1], list(reversed(results)), results[:-1]+[results[0]]):
            with self.assertRaisesRegex(ValueError,'every registered'): evaluate.summarize(bad)
        results[1]['coverage']['ordered_rows_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'unmatched'): evaluate.summarize(results)


class ScanWholeEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Six-update, four-row fixtures inherited as setup only, not official fits.
        cls.policy_patch = patch.object(evaluate, 'POLICY', POLICY)
        cls.study_policy_patch = patch.object(study, 'POLICY', POLICY)
        cls.policy_patch.start(); cls.addClassCleanup(cls.policy_patch.stop)
        cls.study_policy_patch.start(); cls.addClassCleanup(cls.study_policy_patch.stop)
        study_fixtures.ScanStudyTests.setUpClass.__func__(cls)

    def setUp(self):
        study_fixtures.ScanStudyTests.setUp(self)
        def fixture_loader(root, split, frozen):
            selected = study.read(root/study.SELECTION)
            self.assertEqual([r['condition'] for r in selected['conditions']], conditions())
            return copy.deepcopy(RECORDS), coverage(RECORDS)
        self.loader = patch.object(evaluate, 'load_test_partition', side_effect=fixture_loader)
        self.loader_mock = self.loader.start(); self.addCleanup(self.loader.stop)

    def score(self):
        with patch('builtins.print'): return evaluate.score_study(self.root)

    def test_all_36_actual_fixture_checkpoints_generate_and_resume_with_identical_summary(self):
        torch.manual_seed(912); rng = torch.get_rng_state().clone(); threads = torch.get_num_threads()
        summary = self.score()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng)); self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(len(summary['result_sha256']), 36)
        self.assertEqual(len(summary['aggregates']), 18)
        self.assertEqual(len(summary['paired_comparisons']), 42)
        self.assertEqual(self.loader_mock.call_count, 6)
        with patch.object(evaluate, 'greedy_instructions', side_effect=AssertionError('cached work repeated')):
            self.assertEqual(self.score(), summary)

    def test_incomplete_36th_condition_stops_before_test_loader_and_any_evaluation_artifact(self):
        (self.root/study.STUDY/conditions()[-1]['label']/'complete.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.score()
        self.loader_mock.assert_not_called(); self.assertFalse((self.root/evaluate.EVALUATION).exists())

    def test_failure_preserved_and_resume_keeps_completed_batches(self):
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4: raise RuntimeError('synthetic whole-study interruption')
            return greedy_instructions(*args, **kwargs)
        with patch.object(evaluate, 'greedy_instructions', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'whole-study interruption'): self.score()
        first = self.root/evaluate.EVALUATION/conditions()[0]['label']/'batch-00000.json'
        before = sha256(first)
        failures = list((self.root/evaluate.EVALUATION/'attempts').glob('*/failure.json'))
        self.assertEqual(len(failures), 1); self.assertFalse((self.root/evaluate.SUMMARY).exists())
        result = self.score()
        self.assertEqual(len(result['result_sha256']), 36)
        self.assertEqual(sha256(first), before); self.assertTrue(failures[0].exists())

    def test_changed_sealed_batch_rejected_before_regeneration(self):
        self.score()
        first = self.root/evaluate.EVALUATION/conditions()[0]['label']/'batch-00000.json'
        first.write_bytes(first.read_bytes()+b' ')
        with patch.object(evaluate, 'greedy_instructions', side_effect=AssertionError('regenerated')):
            with self.assertRaisesRegex(ValueError, 'Previously completed'): self.score()

    def test_late_batch_mutation_prevents_whole_study_summary(self):
        original = evaluate.generate_partition; calls = 0
        def mutate_after_last(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs); calls += 1
            if calls == 36:
                first = self.root/evaluate.EVALUATION/conditions()[0]['label']/'batch-00000.json'
                first.write_bytes(first.read_bytes()+b' ')
            return result
        with patch.object(evaluate,'generate_partition',side_effect=mutate_after_last):
            with self.assertRaisesRegex(ValueError,'before summary'): self.score()
        self.assertFalse((self.root/evaluate.SUMMARY).exists())


if __name__ == '__main__': unittest.main()
