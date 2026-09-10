import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from flm.language_bridge import parameter_hash, sampling_audit
from flm.language_topology import conditions, verify_saved, verify_source_identity, PROTOCOL
from flm.language_train import construct
from flm.provenance import sha256
from flm.wiring_controls import rewire_graph
import test_wiring_controls


class LanguageTopologyTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_design_crosses_graph_and_training_seeds_with_retrained_slow_controls(self):
        rows = conditions()
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row['label'] for row in rows}), 10)
        nulls = [row for row in rows if row['topology'] == 'rewired']
        self.assertEqual({(r['seed'], r['graph_seed']) for r in nulls},
                         {(s, g) for s in (42, 43) for g in (101, 103, 107)})
        self.assertEqual([r['seed'] for r in rows if r['variant'] == 'no_slow'], [42, 43])
        self.assertEqual(sum(r['reference'] for r in rows), 2)

    def test_initial_weights_equal_across_graphs_and_slow_control_is_retrainable(self):
        with tempfile.TemporaryDirectory() as directory:
            original = test_wiring_controls.WiringControlTests().graph(); rewired, _ = rewire_graph(original, 101)
            source = Path(directory) / 'source.npz'; null = Path(directory) / 'null.npz'
            np.savez(source, **original); np.savez(null, **rewired)
            models = [construct(v, p, 270, 42) for v, p in [('flm', source), ('flm', null), ('no_slow', source)]]
            self.assertEqual(len({parameter_hash(model) for model in models}), 1)
            x = torch.arange(24).reshape(2, 12)
            logits = [model(x)[0] for model in models]
            self.assertFalse(torch.equal(logits[0], logits[1]))
            self.assertFalse(torch.equal(logits[0], logits[2]))
            slow = models[2]; before = slow.input.weight.detach().clone()
            optimizer = torch.optim.AdamW(slow.parameters(), lr=.002)
            torch.nn.functional.cross_entropy(logits[2].flatten(0, 1), (x+1).flatten()).backward()
            self.assertIsNone(slow.beta_logit.grad)
            self.assertGreater(float(slow.input.weight.grad.norm()), 0)
            optimizer.step()
            self.assertFalse(torch.equal(before, slow.input.weight))
            self.assertEqual(int(torch.count_nonzero(slow(x)[1][1])), 0)

    def test_resume_rejects_stream_exposure_mechanism_and_graph_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.npz'; np.savez(path, **test_wiring_controls.WiringControlTests().graph())
            model = construct('flm', path, 270, 42)
            audit = dict(sampler_rng={'fixture': 1}, exposure={'presented_tokens': 768000})
            identity = dict(training_protocol={'steps': 6000}, sampling={'42': {'500': audit}})
            saved = dict(step=500, run=dict(protocol=dict(steps=6000, graph_sha256=sha256(path)), seed=42,
                exposure=audit['exposure']), config={'variant': 'flm'}, sampler_rng=audit['sampler_rng'], model=model.state_dict())
            condition = dict(seed=42, variant='flm')
            verify_saved(saved, condition, identity, path)
            bad = copy.deepcopy(saved); bad['sampler_rng']['fixture'] = 2
            with self.assertRaisesRegex(ValueError, 'Sampled text'): verify_saved(bad, condition, identity, path)
            bad = copy.deepcopy(saved); bad['run']['exposure']['presented_tokens'] -= 1
            with self.assertRaisesRegex(ValueError, 'exposure'): verify_saved(bad, condition, identity, path)
            bad = copy.deepcopy(saved); bad['config']['variant'] = 'no_slow'
            with self.assertRaisesRegex(ValueError, 'mechanism'): verify_saved(bad, condition, identity, path)
            bad = copy.deepcopy(saved); bad['model']['col'][0] += 1
            with self.assertRaisesRegex(ValueError, 'graph buffer'): verify_saved(bad, condition, identity, path)

    def test_identity_detects_numerical_source_and_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / PROTOCOL).parent.mkdir(parents=True)
            (root / PROTOCOL).write_text('declared'); (root / 'source.py').write_text('original')
            (root / 'data.bin').write_bytes(b'original')
            identity = dict(protocol_sha256=sha256(root / PROTOCOL), sources={'source.py': sha256(root / 'source.py')},
                inputs={'data.bin': sha256(root / 'data.bin')}, torch=str(torch.__version__))
            verify_source_identity(root, identity)
            (root / 'data.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'input'): verify_source_identity(root, identity)
            (root / 'data.bin').write_bytes(b'original'); (root / 'source.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source'): verify_source_identity(root, identity)

    def test_sampling_audit_binds_reproducible_text_and_exact_byte_exposure(self):
        docs = [('a', np.arange(300) % 200 + 2)]
        lexicon = SimpleNamespace(lengths=np.r_[0, 0, np.arange(1, 201)])
        first = sampling_audit(docs, lexicon, 42, steps=500)
        self.assertEqual(first, sampling_audit(docs, lexicon, 42, steps=500))
        self.assertNotEqual(first['500']['stream_sha256'], sampling_audit(docs, lexicon, 43, steps=500)['500']['stream_sha256'])
        self.assertEqual(first['500']['exposure']['presented_tokens'], 500*1536)
        self.assertGreater(first['500']['exposure']['presented_bytes'], first['500']['exposure']['scored_bytes'])


if __name__ == '__main__': unittest.main()
