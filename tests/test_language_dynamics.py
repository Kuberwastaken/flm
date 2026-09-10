import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm.language_dynamics import (CORE_GROUPS, SWAPS, core_hash, intervene, linearization,
    parameter_summary, pulse_directions, pulse_response, response_summary, spectrum)
from flm.model import Config, FLM
from flm.language_dynamics_study import (FOLDER, case_hashes, conditions, summary_record,
    verified_case, verified_results)
from flm.provenance import sha256, write_json
from test_model import fixture


class LanguageDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, zero_edges=False):
        graph = fixture()
        if zero_edges:
            graph['weight'][:] = 0
        torch.manual_seed(71)
        model = FLM(graph, Config(neurons=16, pools=8, embedding=8, vocabulary=32)).double()
        return model

    def test_zero_state_jacobian_matches_autograd_and_full_block_spectrum(self):
        model = self.model()
        with torch.no_grad():
            model.edge_log_gain.normal_(0, .2)
        constants = model.constants(); a, _, beta = linearization(constants)
        zero = torch.zeros(32, dtype=torch.float64, requires_grad=True)
        def transition(value):
            state = value[:16].unsqueeze(0), value[16:].unsqueeze(0)
            updated = model.transition(torch.zeros(1, 16, dtype=torch.float64), state, constants)
            return torch.cat(updated, dim=1).squeeze(0)
        actual = torch.autograd.functional.jacobian(transition, zero)
        expected = torch.cat((torch.cat((a, torch.zeros_like(a)), dim=1),
                              torch.cat((beta[:, None] * a, torch.diag(1 - beta)), dim=1)), dim=0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        metrics, values = spectrum(constants)
        direct = torch.linalg.eigvals(actual).abs().sort().values
        analytic = torch.cat((torch.from_numpy(values).abs(), 1 - beta)).sort().values
        torch.testing.assert_close(analytic, direct, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(metrics['joint_spectral_radius'], float(direct[-1]), places=12)
        self.assertLess(metrics['leading_relative_residual'], 1e-12)

    def test_isolated_units_match_closed_form_fast_and_slow_decay(self):
        model = self.model(zero_edges=True)
        directions = pulse_directions(16, count=2)
        result = pulse_response(model, directions, .4, silence=8, snapshot_times=tuple(range(9)))
        _, alpha, beta, _ = model.constants()
        h0 = alpha.detach() * torch.tanh(torch.from_numpy(directions[0]).double() * .4)
        u, v = 1 - alpha.detach(), 1 - beta.detach()
        for time in range(9):
            h = h0 * u ** time
            slow = beta.detach() * h0 * (v ** (time + 1) - u ** (time + 1)) / (v - u)
            np.testing.assert_allclose(result['snapshots'][time], torch.stack((h, slow)).numpy(), rtol=1e-11, atol=1e-13)

    def test_probe_resets_state_and_preserves_source_tensors_and_rng(self):
        model = self.model(); directions = pulse_directions(16, count=2)
        tensors = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
        first = pulse_response(model, directions, .1, silence=8, snapshot_times=(0, 8))
        pulse_response(model, directions, 10., silence=8, snapshot_times=(0, 8))
        again = pulse_response(model, directions, .1, silence=8, snapshot_times=(0, 8))
        for name in ('curves', 'snapshots', 'normalized_linear_deviation'):
            np.testing.assert_array_equal(first[name], again[name])
        for name, value in tensors.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        constants = model.constants(); state = model.initial_state(2)
        for _ in range(40):
            state = model.transition(torch.zeros(2, 16, dtype=torch.float64), state, constants)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in state))

    def test_interventions_copy_only_declared_core_groups(self):
        initial = self.model(); trained = copy.deepcopy(initial)
        with torch.no_grad():
            for parameter in trained.parameters():
                parameter.add_(.2)
        initial_hash = core_hash(initial); trained_hash = core_hash(trained)
        for mode, groups in SWAPS.items():
            with self.subTest(mode=mode):
                result = intervene(initial, trained, mode)
                for name, value in result.named_parameters():
                    source = trained if name in groups else initial
                    self.assertTrue(torch.equal(value, source.get_parameter(name)))
                self.assertEqual(core_hash(initial), initial_hash)
                self.assertEqual(core_hash(trained), trained_hash)
        self.assertEqual(core_hash(intervene(initial, trained, 'trained')), trained_hash)
        bad = copy.deepcopy(trained); bad.base_weight[0] *= -1
        with self.assertRaisesRegex(ValueError, 'buffer changed'):
            intervene(initial, bad, 'trained')
        with self.assertRaises(ValueError):
            intervene(initial, trained, 'undeclared')

    def test_rademacher_directions_are_deterministic_with_unit_rms(self):
        first = pulse_directions(16)
        np.testing.assert_array_equal(first, pulse_directions(16))
        np.testing.assert_array_equal((first ** 2).mean(axis=1), np.ones(8))
        self.assertEqual(set(np.unique(first)), {-1., 1.})
        self.assertFalse(np.array_equal(first, pulse_directions(16, seed=91012)))
        with self.assertRaises(ValueError):
            pulse_directions(0)

    def test_odd_pulse_symmetry_and_large_input_state_bounds(self):
        model = self.model(); directions = pulse_directions(16, count=2)
        positive = pulse_response(model, directions, 10., silence=16, snapshot_times=(0, 1, 8, 16))
        negative = pulse_response(model, -directions, 10., silence=16, snapshot_times=(0, 1, 8, 16))
        np.testing.assert_allclose(positive['snapshots'], -negative['snapshots'], atol=0, rtol=0)
        np.testing.assert_allclose(positive['curves'], negative['curves'], atol=0, rtol=0)
        self.assertLessEqual(float(np.abs(positive['snapshots']).max()), 1.)

    def test_small_pulse_linearization_error_decreases_quadratically(self):
        model = self.model(); directions = pulse_directions(16, count=2)
        small = pulse_response(model, directions, .001, silence=0, snapshot_times=(0,))
        large = pulse_response(model, directions, .002, silence=0, snapshot_times=(0,))
        ratio = large['normalized_linear_deviation'] / small['normalized_linear_deviation']
        np.testing.assert_allclose(ratio, np.full_like(ratio, 4.), rtol=1e-5)

    def test_settling_descriptor_is_censored_and_requires_no_later_rebound(self):
        values = np.zeros((4, 2, 3))
        values[:, 0, 2] = [1., .001, .2, .001]
        values[:, 1, 2] = [1., .5, .2, .1]
        result = response_summary(values, 1.)
        self.assertEqual(result[0]['observed_one_percent_settling_time'], 3)
        self.assertIsNone(result[1]['observed_one_percent_settling_time'])
        self.assertAlmostEqual(result[0]['normalized_response_energy'], 1 + 1e-6 + .04 + 1e-6)

    def test_invalid_pulse_or_misaligned_snapshot_schedule_rejects(self):
        model = self.model(); directions = pulse_directions(16)
        for amplitude, times in [(0, (0,)), (float('nan'), (0,)), (.1, (8, 0)), (.1, (0, 0)), (.1, (0, 9))]:
            with self.subTest(amplitude=amplitude, times=times):
                with self.assertRaises(ValueError):
                    pulse_response(model, directions, amplitude, silence=8, snapshot_times=times)
        description = parameter_summary(model)
        self.assertGreater(description['bare_slow_leak_half_life']['median'], description['bare_fast_leak_half_life']['median'])

    def test_changed_case_identity_or_array_bytes_reject(self):
        with tempfile.TemporaryDirectory(prefix='flm-dynamics-artifact-') as directory:
            root = Path(directory).resolve(); folder = root / FOLDER; folder.mkdir(parents=True)
            condition = conditions()[0]; name = 'case-' + condition['label'] + '.npz'
            np.savez_compressed(folder / name, fixture=np.ones(2))
            record = dict(condition=condition, study_identity_sha256='a' * 64,
                          arrays=name, arrays_sha256=sha256(folder / name))
            marker = folder / ('case-' + condition['label'] + '.json')
            write_json(marker, record)
            self.assertEqual(verified_case(root, condition, 'a' * 64), record)
            with self.assertRaisesRegex(ValueError, 'case identity'):
                verified_case(root, condition, 'b' * 64)
            (folder / name).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'array payload'):
                verified_case(root, condition, 'a' * 64)

    def test_replay_proof_must_bind_all_current_case_and_summary_bytes(self):
        with tempfile.TemporaryDirectory(prefix='flm-dynamics-replay-') as directory:
            root = Path(directory).resolve(); folder = root / FOLDER
            identity = dict(study='Synthetic replay gate', scope='Fixture only')
            write_json(folder / 'identity.json', identity)
            identity_hash = sha256(folder / 'identity.json'); records = []
            for condition in conditions():
                name = 'case-' + condition['label'] + '.npz'
                np.savez_compressed(folder / name, fixture=np.ones(2))
                record = dict(condition=condition, study_identity_sha256=identity_hash,
                              arrays=name, arrays_sha256=sha256(folder / name), metrics={'fixture': True})
                write_json(folder / ('case-' + condition['label'] + '.json'), record); records.append(record)
            summary = summary_record(identity, identity_hash, records)
            write_json(folder / 'summary.json', summary)
            proof = dict(study_identity_sha256=identity_hash, conditions=7, exact_metric_and_array_replay=True,
                         summary_sha256=sha256(folder / 'summary.json'), case_artifact_sha256=case_hashes(root))
            write_json(folder / 'replay.json', proof)
            # The source/input identity gate is separately enforced by prepare/run;
            # here only the binding of replay evidence to output bytes is isolated.
            with patch('flm.language_dynamics_study.verify_identity'):
                self.assertEqual(verified_results(root), summary)
                changed = copy.deepcopy(records[0]); changed['metrics']['fixture'] = False
                write_json(folder / 'case-initial_shared.json', changed)
                summary['cases'][0] = changed
                write_json(folder / 'summary.json', summary)
                with self.assertRaisesRegex(ValueError, 'replay does not bind'):
                    verified_results(root)


if __name__ == '__main__':
    unittest.main()
