import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from flm.inference import state_hash
from flm.scan_conditions import conditions, prepare_condition, source_model
from flm.scan_train import Settings, fit


class ScanConditionTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2)
        self.lexicon = SimpleNamespace(sha256=hashlib.sha256(b'fixture').hexdigest())
        self.rows = [dict(command='jump', actions=['I_JUMP']), dict(command='jump', actions=['I_JUMP'])]
        self.source = dict(initial_state_sha256=state_hash(self.model), tokenizer_sha256=self.lexicon.sha256)
        self.condition = conditions()[0]
        self.settings = Settings(6, 2, .002, .0002, 1, .01, 1., 3, 42, 1)

    def prepare(self):
        with patch('flm.scan_conditions.load_training_partition', return_value=(copy.deepcopy(self.rows), {'fixture': True})), \
                patch('flm.scan_conditions.source_model', return_value=(self.model, self.lexicon, self.source)):
            return prepare_condition(Path('.'), self.condition)

    def test_complete_inventory_has_six_conditions_per_split_and_seed(self):
        rows = conditions()
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({r['label'] for r in rows}), 36)
        for split in ('simple', 'length', 'add_primitive_jump'):
            for seed in (42, 43):
                chosen = [r for r in rows if r['split'] == split and r['seed'] == seed]
                self.assertEqual({(r['variant'], r['initialization']) for r in chosen},
                    {(v, i) for v in ('flm', 'gru', 'transformer') for i in ('initial', 'wikitext')})

    def test_invalid_condition_fails_before_any_input_load(self):
        with patch('flm.scan_conditions.load_training_partition') as loader:
            with self.assertRaisesRegex(ValueError, 'exact declared'):
                prepare_condition(Path('.'), dict(self.condition, label='another/path'))
            loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Unknown instruction source'):
            source_model(Path('missing'), 'flm', 99, 'initial')

    def test_binding_preserves_rows_and_cannot_be_modified_through_return_value(self):
        prepared = self.prepare(); binding = prepared.training_binding(self.settings)
        self.assertEqual(len(prepared.records), 2)
        binding['condition']['seed'] = 99
        self.assertEqual(prepared.binding['condition']['seed'], 42)

    def test_changed_sampler_model_rows_or_tokenizer_refuse_binding(self):
        prepared = self.prepare()
        wrong = Settings(6, 2, .002, .0002, 1, .01, 1., 3, 43, 1)
        with self.assertRaisesRegex(ValueError, 'row sampling'): prepared.training_binding(wrong)
        prepared.records.pop()
        with self.assertRaisesRegex(ValueError, 'changed before fitting'): prepared.training_binding(self.settings)
        prepared.records = copy.deepcopy(self.rows)
        with torch.no_grad(): self.model.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'changed before fitting'): prepared.training_binding(self.settings)
        self.source['initial_state_sha256'] = state_hash(self.model)
        prepared.lexicon = SimpleNamespace(sha256='changed')
        with self.assertRaisesRegex(ValueError, 'changed before fitting'): prepared.training_binding(self.settings)

    def test_incomplete_or_stale_source_audit_rejects_before_checkpoint_restore(self):
        rows = [dict(variant=v, seed=s) for v in ('flm', 'gru', 'transformer') for s in (42, 43)]
        for audit in (dict(sources=rows[:-1]), dict(sources=rows, source_sha256={})):
            with patch('flm.scan_conditions.read_json', side_effect=[audit, dict(runs=rows)]), \
                    patch('flm.scan_conditions.restore') as restore:
                with self.assertRaises(ValueError): source_model(Path('.'), 'flm', 42, 'initial')
                restore.assert_not_called()

    def test_fresh_preparation_resumes_fixture_training_for_all_architectures(self):
        from test_scan_runtime import models, RECORDS
        from test_scan_train import FixtureLexicon, payload
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            for index, variant in enumerate(('flm', 'gru', 'transformer')):
                condition = next(r for r in conditions() if r['split'] == 'simple' and r['seed'] == 42
                                 and r['variant'] == variant and r['initialization'] == 'initial')
                def prepared():
                    model = models()[index]; lexicon = FixtureLexicon()
                    source = dict(initial_state_sha256=state_hash(model), tokenizer_sha256=lexicon.sha256)
                    with patch('flm.scan_conditions.load_training_partition', return_value=(copy.deepcopy(RECORDS), {'fixture': True})), \
                            patch('flm.scan_conditions.source_model', return_value=(model, lexicon, source)):
                        return prepare_condition(Path('.'), condition)
                whole = Path(temporary) / f'whole-{variant}'; resumed = Path(temporary) / f'resumed-{variant}'
                first = prepared()
                fit(first.model, first.records, first.lexicon, self.settings, first.training_binding(self.settings), whole)
                partial = prepared()
                fit(partial.model, partial.records, partial.lexicon, self.settings, partial.training_binding(self.settings), resumed, until=3)
                fresh = prepared()
                fit(fresh.model, fresh.records, fresh.lexicon, self.settings, fresh.training_binding(self.settings), resumed)
                a, b = payload(whole), payload(resumed)
                for name in a['model']: self.assertTrue(torch.equal(a['model'][name], b['model'][name]))
                for name in ('declaration', 'history', 'exposure', 'sampler_rng', 'sampled_row_sha256'):
                    self.assertEqual(a[name], b[name])


if __name__ == '__main__': unittest.main()
