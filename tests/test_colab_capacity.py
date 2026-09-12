import copy
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np
import torch
from torch.nn import functional as F

from flm.model import FLM, Config
from flm.colab_capacity import CapacityFLM, EdgeMM, sized_config, train
from flm.colab_data import conversation, oasst_branches, partition, tokenizer_from_repo, windows


def graph():
    return dict(row=np.array([0, 0, 1, 2, 3, 4, 4, 5]), col=np.array([1, 3, 2, 0, 4, 0, 5, 2]),
                weight=np.array([.6, -.4, 1, 1, -1, .7, .3, 1], dtype=np.float32),
                body_ids=np.arange(6), pool=np.array([0, 0, 1, 1, 2, 2]))


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_edge_gradient_against_finite_differences(self):
        g = graph()
        indices = torch.tensor(np.stack((g['row'], g['col'])))
        values = torch.randn(len(g['row']), dtype=torch.double, requires_grad=True)
        state = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v, h: EdgeMM.apply(v, indices, h), (values, state)))

    def test_checkpoint_and_sparse_backends_match_original_loss_and_every_gradient(self):
        torch.manual_seed(7)
        c = Config(neurons=6, pools=3, embedding=8, vocabulary=17, tied_readout=True, backend='dense')
        reference = FLM(graph(), c)
        x = torch.randint(0, 17, (2, 9)); y = torch.randint(0, 17, (2, 9)); y[:, :2] = -100
        expected = F.cross_entropy(reference(x)[0].reshape(-1, 17), y.reshape(-1), reduction='sum')
        expected.backward()
        for backend in ('edge', 'csr'):
            for checkpointed in (False, True):
                with self.subTest(backend=backend, checkpointed=checkpointed):
                    m = CapacityFLM(graph(), c, chunk=3, sparse_backend=backend)
                    m.load_state_dict(reference.state_dict())
                    actual = m.loss_sum(x, y, checkpointed=checkpointed); actual.backward()
                    torch.testing.assert_close(actual, expected)
                    for (name, a), (_, b) in zip(reference.named_parameters(), m.named_parameters()):
                        torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4, msg=name)

    def test_resume_matches_uninterrupted_updates_and_rejects_changed_identity(self):
        c = Config(neurons=6, pools=3, embedding=8, vocabulary=17, tied_readout=True)
        torch.manual_seed(19)
        a = CapacityFLM(graph(), c, chunk=3)
        b = copy.deepcopy(a)
        examples = [dict(tokens=[0, 2, 3, 4, 1], labels=[0, 2, 3, 4, 1]),
                    dict(tokens=[0, 5, 6, 1], labels=[-100, -100, 6, 1])]
        with tempfile.TemporaryDirectory() as folder:
            args = dict(examples=examples, validation=examples, identity={'fixture': True}, pad=16,
                        seconds=120, batch_size=1, accumulation=2, length=5)
            train(a, output=Path(folder)/'a', max_steps=4, **args)
            train(b, output=Path(folder)/'b', max_steps=2, **args)
            report = train(b, output=Path(folder)/'b', max_steps=2, resume=True, **args)
            self.assertEqual(report['total_steps'], 4)
            self.assertFalse(report['gpu_fit_verified'])
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, atol=0, rtol=0)
            args['identity'] = {'fixture': False}
            with self.assertRaisesRegex(ValueError, 'differ'):
                train(b, output=Path(folder)/'b', max_steps=1, resume=True, **args)

    def test_target_count_matches_real_parameter_count(self):
        c = sized_config(graph(), 17, .03)
        m = CapacityFLM(graph(), c)
        n, p, e, v, d = c.neurons, c.pools, len(graph()['row']), c.vocabulary, c.embedding
        self.assertEqual(sum(x.numel() for x in m.parameters()), e+3*n+1+4*p+v+d*(v+n+2*p+1))

    def test_window_targets_are_not_duplicated_or_lost(self):
        rows = list(windows(list(range(11)), list(range(11)), 3))
        self.assertEqual([t for row in rows for t in row['labels'][1:]], list(range(1, 11)))

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Optional transformer dependency')
    def test_random_transformer_control_has_finite_training_gradient(self):
        from flm.colab_reference import TransformerReference, ReferenceConfig
        model = TransformerReference(ReferenceConfig(vocabulary=17, hidden=24, intermediate=32, layers=1, heads=3, kv_heads=1))
        x = torch.randint(0, 17, (2, 6))
        loss = model.loss_sum(x, x)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_chat_mask_learns_content_and_end_only(self):
        tokenizer = tokenizer_from_repo(Path(__file__).resolve().parents[1])
        tokens, labels = conversation(tokenizer, [dict(role='user', content='hello'), dict(role='assistant', content='hi')])
        expected = tokenizer.encode('hi', add_special_tokens=False).ids + [tokenizer.token_to_id('<|end|>')]
        self.assertEqual([v for v in labels if v != -100], expected)
        self.assertEqual(len(tokens), len(labels))
        self.assertNotIn(tokenizer.token_to_id('<|bos|>'), tokenizer.encode('!"hello').ids)
        self.assertEqual(tokenizer.get_vocab_size(), 4101)

    def test_synthetic_parent_is_rejected_and_trees_stay_together(self):
        def row(key, parent, role, synthetic=False):
            return dict(message_id=key, parent_id=parent, role=role, synthetic=synthetic, deleted=False,
                        review_result=True, rank=0, lang='en', text=key, message_tree_id='tree')
        records = [row('a', None, 'prompter'), row('b', 'a', 'assistant'),
                   row('c', 'b', 'prompter'), row('d', 'c', 'assistant')]
        branches = list(oasst_branches(records))
        self.assertEqual(len(branches), 2)
        self.assertEqual(len({partition(r['tree_id']) for r in branches}), 1)
        records[0]['synthetic'] = True
        self.assertEqual(list(oasst_branches(records)), [])


if __name__ == '__main__':
    unittest.main()
