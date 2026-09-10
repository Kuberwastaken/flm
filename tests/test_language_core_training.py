import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm.language_core_controls import CONTROLS, FROZEN_DYNAMICS, construct
from flm.language_core_train import (checkpoint_records, commit_checkpoint, learning_rate,
    optimizer_for, optimizer_names, restore, restore_optimizer, train, update, verify_exposure)
from flm.language_core_study import conditions
from flm.language_core_test import freeze_selection, score_study, summarize
from flm.language_train import construct as original_construct
from flm.provenance import sha256, write_json
from flm.train import Sampler, save_checkpoint
from test_model import fixture


class CoreTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.graph = self.root / 'graph.npz'
        np.savez(self.graph, **fixture())
        self.lexicon = SimpleNamespace(vocabulary=32, sha256='fixture-tokenizer')
        self.protocol = dict(steps=6000, batch=2, sequence=4, warmup=1,
                             learning_rate=.002, final_learning_rate=.0002,
                             lr_warmup_updates=100, weight_decay=.01)
        self.documents = [('a', np.arange(107) % 30 + 2), ('b', np.arange(131)[::-1] % 30 + 2)]

    def setup_run(self, control):
        model = construct(control, self.graph, 32, 42)
        optimizer = optimizer_for(model, self.protocol)
        sampler = Sampler(self.documents, 42, 4)
        binding = dict(study_identity_sha256='fixture-study', label=control + '-s42',
                       control=control, seed=42, protocol=self.protocol,
                       tokenizer_sha256=self.lexicon.sha256, graph_sha256=sha256(self.graph))
        run = dict(binding, parameter_card=model.parameter_card(), optimizer_parameter_names=optimizer_names(model))
        return model, optimizer, sampler, binding, run

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_learning_rates_match_original_for_every_registered_update(self):
        for step in range(1, 6001):
            progress = max(0., (step - 100) / max(1, 6000 - 100))
            expected = (.0002 + .0018 * .5 * (1 + math.cos(math.pi * min(progress, 1.)))) * min(step / 100, 1.)
            self.assertEqual(learning_rate(self.protocol, step), expected)

    def test_full_update_matches_independent_original_loop_and_optimizer_moments(self):
        model, optimizer, sampler, _, _ = self.setup_run('full')
        reference = original_construct('flm', self.graph, 32, 42)
        original_optimizer = torch.optim.AdamW(reference.parameters(), lr=.002, weight_decay=.01)
        for step in range(1, 7):
            x, y = sampler.sample(2, 'cpu')
            original_optimizer.param_groups[0]['lr'] = .002 * step / 100
            original_optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(reference(x)[0][:, 1:].reshape(-1, 32), y[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            original_optimizer.step()
            actual, _ = update(model, optimizer, x, y, self.protocol, step)
            self.assertEqual(actual, float(loss.detach()))
            self.assert_nested_equal(model.state_dict(), reference.state_dict())
            self.assert_nested_equal(optimizer.state_dict(), original_optimizer.state_dict())

    def test_every_mechanism_resumes_exact_parameters_moments_rng_and_next_window(self):
        for control in CONTROLS:
            with self.subTest(control=control):
                model, optimizer, sampler, binding, run = self.setup_run(control)
                for step in range(1, 4):
                    update(model, optimizer, *sampler.sample(2, 'cpu'), self.protocol, step)
                path = self.root / (control + '.pt')
                save_checkpoint(path, model, optimizer, sampler, 3, 4., run)
                for step in range(4, 7):
                    update(model, optimizer, *sampler.sample(2, 'cpu'), self.protocol, step)
                original_rng = torch.get_rng_state()
                restored, saved = restore(path, self.graph, self.lexicon, binding)
                resumed_optimizer = restore_optimizer(restored, saved, self.protocol)
                resumed_sampler = Sampler(self.documents, 42, 4)
                resumed_sampler.rng.bit_generator.state = saved['sampler_rng']
                torch.set_rng_state(saved['torch_rng'])
                for step in range(4, 7):
                    update(restored, resumed_optimizer, *resumed_sampler.sample(2, 'cpu'), self.protocol, step)
                self.assert_nested_equal(model.state_dict(), restored.state_dict())
                self.assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
                self.assert_nested_equal(sampler.rng.bit_generator.state, resumed_sampler.rng.bit_generator.state)
                self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                self.assert_nested_equal(sampler.sample(2, 'cpu'), resumed_sampler.sample(2, 'cpu'))
                if control == 'fixed_dynamics':
                    restored.verify_frozen_dynamics()

    def saved_payload(self, control='fixed_dynamics'):
        model, optimizer, sampler, binding, run = self.setup_run(control)
        for step in range(1, 4):
            update(model, optimizer, *sampler.sample(2, 'cpu'), self.protocol, step)
        path = self.root / 'saved.pt'
        save_checkpoint(path, model, optimizer, sampler, 3, 4., run)
        _, saved = restore(path, self.graph, self.lexicon, binding)
        saved.pop('_file_sha256')
        return model, path, saved, binding

    def test_restore_rejects_each_changed_frozen_tensor_and_graph_pool_buffers(self):
        _, path, saved, binding = self.saved_payload()
        for name in (*FROZEN_DYNAMICS, 'row', 'col', 'base_weight', 'pool_index', 'pool_sizes'):
            with self.subTest(tensor=name):
                corrupt = copy.deepcopy(saved)
                corrupt['model'][name].reshape(-1)[0] += 1
                torch.save(corrupt, path)
                with self.assertRaises(ValueError):
                    restore(path, self.graph, self.lexicon, binding)
        corrupt = copy.deepcopy(saved)
        corrupt['model']['embedding.weight'][0, 0] = float('nan')
        torch.save(corrupt, path)
        with self.assertRaisesRegex(ValueError, 'Invalid checkpoint tensor'):
            restore(path, self.graph, self.lexicon, binding)

    def test_restore_rejects_changed_mechanism_binding_and_optimizer_inventory(self):
        _, path, saved, binding = self.saved_payload()
        for field, value in [('control', 'full'), ('seed', 43), ('study_identity_sha256', 'different')]:
            corrupt = copy.deepcopy(saved)
            corrupt['run'][field] = value
            torch.save(corrupt, path)
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                restore(path, self.graph, self.lexicon, binding)
        corrupt = copy.deepcopy(saved)
        corrupt['config']['control'] = 'no_lateral'
        torch.save(corrupt, path)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            restore(path, self.graph, self.lexicon, binding)
        corrupt = copy.deepcopy(saved)
        corrupt['run']['optimizer_parameter_names'].reverse()
        torch.save(corrupt, path)
        with self.assertRaisesRegex(ValueError, 'parameter order changed'):
            restore(path, self.graph, self.lexicon, binding)

    def test_optimizer_restore_rejects_missing_nonfinite_moments_wrong_steps_and_settings(self):
        model, _, saved, _ = self.saved_payload('no_lateral')
        for fault in ('missing', 'nonfinite', 'step', 'weight_decay', 'indices'):
            with self.subTest(fault=fault):
                corrupt = copy.deepcopy(saved)
                states = corrupt['optimizer']['state']
                key = next(iter(states))
                if fault == 'missing':
                    del states[key]
                elif fault == 'nonfinite':
                    states[key]['exp_avg'].reshape(-1)[0] = float('nan')
                elif fault == 'step':
                    states[key]['step'] += 1
                elif fault == 'weight_decay':
                    corrupt['optimizer']['param_groups'][0]['weight_decay'] = .1
                else:
                    corrupt['optimizer']['param_groups'][0]['params'].reverse()
                with self.assertRaises(ValueError):
                    restore_optimizer(model, corrupt, self.protocol)

    def test_save_transaction_ignores_orphans_rejects_committed_corruption_and_gaps(self):
        model, optimizer, sampler, binding, run = self.setup_run('full')
        directory = self.root / 'run'
        directory.mkdir()
        # Simulate interruption before the final manifest write.
        (directory / 'checkpoint-000002.pt').write_bytes(b'interrupted payload')
        self.assertEqual(checkpoint_records(directory, binding, interval=2, steps=6), [])
        record = commit_checkpoint(directory, model, optimizer, sampler, 2, 4., run, binding,
                                   dict(bits_per_byte=4.), [dict(step=2)])
        self.assertEqual(checkpoint_records(directory, binding, interval=2, steps=6), [record])
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            commit_checkpoint(directory, model, optimizer, sampler, 2, 4., run, binding, dict(bits_per_byte=4.), [])
        history = directory / record['history']
        history.write_text('changed', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            checkpoint_records(directory, binding, interval=2, steps=6)
        history.write_text(json.dumps([dict(step=2)], indent=2) + '\n', encoding='utf8')
        (directory / 'saved-000002.json').rename(directory / 'saved-000004.json')
        with self.assertRaisesRegex(ValueError, 'sequence changed'):
            checkpoint_records(directory, binding, interval=2, steps=6)

    def test_exposure_rejects_rng_byte_token_or_step_drift(self):
        audit = dict(sampler_rng={'fixture': 4}, exposure=dict(presented_tokens=100, presented_bytes=123, scored_bytes=98))
        identity = dict(sampling={'42': {'500': audit}})
        saved = dict(step=500, run=dict(seed=42, exposure=copy.deepcopy(audit['exposure'])), sampler_rng={'fixture': 4})
        verify_exposure(saved, identity)
        for key in audit['exposure']:
            corrupt = copy.deepcopy(saved)
            corrupt['run']['exposure'][key] += 1
            with self.assertRaisesRegex(ValueError, 'exposure or RNG'):
                verify_exposure(corrupt, identity)
        for key, value in [('sampler_rng', {'fixture': 5}), ('step', 499)]:
            corrupt = copy.deepcopy(saved)
            corrupt[key] = value
            with self.assertRaises(ValueError):
                verify_exposure(corrupt, identity)

    def test_training_loop_replays_an_interrupted_save_from_last_committed_checkpoint(self):
        # Exercise the actual loop with a tiny corpus/graph, retaining its real
        # 500-update save cadence. Numerical controls use the full budget elsewhere.
        protocol = dict(self.protocol, steps=1000, threads=1, eval_tokens=32)
        lexicon = SimpleNamespace(vocabulary=32, sha256='fixture-tokenizer', lengths=np.ones(32, dtype=np.int64))
        lexicon.lengths[:2] = 0
        sampler = Sampler(self.documents, 42, 4)
        audits = {}
        presented = scored = 0
        for step in range(1, 1001):
            x, y = sampler.sample(2, 'cpu')
            presented += int(lexicon.lengths[x.numpy()].sum())
            scored += int(lexicon.lengths[y[:, 1:].numpy()].sum())
            if step % 500 == 0:
                audits[str(step)] = dict(sampler_rng=copy.deepcopy(sampler.rng.bit_generator.state),
                    exposure=dict(presented_tokens=step * 8, presented_bytes=presented, scored_bytes=scored))
        identity = dict(training_protocol=protocol, source_commit='fixture', sampling={'42': audits})
        condition = dict(reference=False, control='fixed_dynamics', seed=42, label='fixed_dynamics-s42', output='interrupted')
        _, _, _, binding, _ = self.setup_run('fixed_dynamics')
        binding['protocol'] = protocol
        original_commit = commit_checkpoint

        def interrupt(directory, model, optimizer, sampler, step, best, run, binding, score, history):
            if step == 1000:
                # A partial new payload exists, but its final manifest does not.
                (directory / 'checkpoint-001000.pt').write_bytes(b'partial write')
                raise RuntimeError('simulated interrupted save')
            return original_commit(directory, model, optimizer, sampler, step, best, run, binding, score, history)

        with patch('flm.language_core_study.GRAPH', self.graph), \
             patch('flm.language_core_study.verify_identity'), \
             patch('flm.language_core_study.verify_complete'), \
             patch('flm.language_core_study.run_binding', return_value=binding), \
             patch('flm.tokenizer.Lexicon', return_value=lexicon), \
             patch('flm.tokenizer.read_cache', return_value=self.documents), \
             patch('builtins.print'):
            with patch('flm.language_core_train.commit_checkpoint', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupted save'):
                    train(self.root, condition, identity)
            self.assertEqual(len(checkpoint_records(self.root / 'interrupted', binding, steps=1000)), 1)
            train(self.root, condition, identity)
            condition['output'] = 'uninterrupted'
            train(self.root, condition, identity)
        for step in (500, 1000):
            restored = []
            for directory in ('interrupted', 'uninterrupted'):
                _, saved = restore(self.root / directory / f'checkpoint-{step:06d}.pt', self.graph, lexicon, binding)
                restored.append(saved)
            for key in ('model', 'optimizer', 'sampler_rng', 'torch_rng', 'best', 'run'):
                self.assert_nested_equal(restored[0][key], restored[1][key])
        histories = json.loads((self.root / 'interrupted/history-001000.json').read_text(encoding='utf8'))
        self.assertEqual([row['step'] for row in histories], [600, 700, 800, 900, 1000])
        self.assertTrue(all(row['resumed_from'] == 500 for row in histories))

    def test_incomplete_study_fails_before_decoding_test_cache_or_constructing_lexicon(self):
        with patch('flm.language_core_test.read_json', return_value={}), \
             patch('flm.language_core_test.verify_identity'), \
             patch('flm.language_core_test.read_cache') as cache, \
             patch('flm.language_core_test.Lexicon') as lexicon:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                score_study(self.root)
            cache.assert_not_called()
            lexicon.assert_not_called()

    def test_freeze_requires_all_eight_verified_selections_and_rejects_reselection(self):
        identity_path = self.root / 'reports/language-core/identity.json'
        identity = dict(inputs={'data/processed/wikitext2-bpe/test.npz': 'fixture-cache'}, prior_test_access='exploratory')
        write_json(identity_path, identity)
        for row in conditions():
            write_json(self.root / row['output'] / 'complete.json', {})
        selected = [dict(row, checkpoint_sha256='fixture-' + row['label']) for row in conditions()]
        with patch('flm.language_core_test.verify_identity'), \
             patch('flm.language_core_test.Lexicon', return_value=self.lexicon), \
             patch('flm.language_core_test.verify_complete', side_effect=selected) as verifier:
            frozen, _ = freeze_selection(self.root)
            self.assertEqual(verifier.call_count, 8)
            self.assertEqual(len(frozen['runs']), 8)
        changed = copy.deepcopy(selected)
        changed[-1]['checkpoint_sha256'] = 'different'
        with patch('flm.language_core_test.verify_identity'), \
             patch('flm.language_core_test.Lexicon', return_value=self.lexicon), \
             patch('flm.language_core_test.verify_complete', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                freeze_selection(self.root)

    def test_all_declared_effects_have_correct_direction_and_weighted_denominator(self):
        values = {'full': 2., 'fixed_dynamics': 3., 'no_lateral': 4., 'no_temporal_state': 5.}
        results = []
        for row in conditions():
            bpb = values[row['control']] + (row['seed'] - 42) * .1
            documents = [dict(document=name, bytes=size, nll=bpb * size * math.log(2)) for name, size in [('a', 10), ('b', 90)]]
            results.append(dict(row, score=dict(documents=documents)))
        report = summarize(results)
        self.assertEqual(len(report['primary_contrasts']), 6)
        self.assertEqual(len(report['independent_unit_memory_contrasts']), 2)
        for row in report['primary_contrasts']:
            self.assertAlmostEqual(row['difference_bpb'], 2 - values[row['second']])
        self.assertAlmostEqual(report['independent_unit_memory_mean_difference_bpb'], -1.)
        with self.assertRaisesRegex(ValueError, 'eight unique'):
            summarize(results[:-1])
        with self.assertRaisesRegex(ValueError, 'eight unique'):
            summarize(results + [results[0]])


if __name__ == '__main__':
    unittest.main()
