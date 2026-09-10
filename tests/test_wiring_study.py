import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from flm.behavior_study import episode_batch
from flm.local_learning import ChoiceFLM
from flm.provenance import sha256, write_json
from flm.wiring_diagnostics import task_batch, gradient_probe, gradient_metrics
from flm.wiring_study import run
from test_model import fixture


class WiringStudyTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(17)

    def test_context_needs_both_balanced_cues_with_separate_temporal_windows(self):
        sensory,targets,meta=task_batch('context',np.random.default_rng(17),8,4)
        cue,context=np.array(meta['cue']),np.array(meta['context'])
        self.assertEqual(sensory.shape,(8,13,4))
        np.testing.assert_array_equal(np.bincount(2*cue+context,minlength=4),[2]*4)
        np.testing.assert_array_equal(targets,cue^context)
        np.testing.assert_array_equal(sensory[:,0,0],2*cue-1)
        np.testing.assert_array_equal(sensory[:,6,1],2*context-1)
        self.assertTrue(np.all(sensory[:,2:,0]==0))
        self.assertTrue(np.all(sensory[:,:6,1]==0)); self.assertTrue(np.all(sensory[:,8:,1]==0))
        self.assertTrue(np.all(sensory[:,:-1,3]==0)); self.assertTrue(np.all(sensory[:,-1,3]==1))
        for values in (cue,context):
            for value in (0,1): self.assertEqual(int(targets[values==value].sum()),2)
        old,y=episode_batch(np.random.default_rng(17),8,4)
        bridge,z,_=task_batch('cue',np.random.default_rng(17),8,4)
        np.testing.assert_array_equal(old,bridge); np.testing.assert_array_equal(y,z)

    def test_gradient_probe_is_read_only_and_exact_when_omitted_routes_are_absent(self):
        graph=fixture(); n=len(graph['body_ids'])
        graph.update(row=np.arange(n),col=np.arange(n),weight=np.ones(n,dtype=np.float32))
        model=ChoiceFLM(graph); sensory=torch.randn(8,19,4); targets=torch.tensor([0,1]*4)
        before=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
        report=gradient_probe(model,sensory,targets)
        for name,value in model.state_dict().items(): torch.testing.assert_close(value,before[name],atol=0,rtol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        exact=report['comparisons']['eligibility']['combined_core']
        self.assertAlmostEqual(exact['cosine'],1.,places=10)
        self.assertLess(exact['relative_l2_error'],1e-10)
        self.assertGreater(report['comparisons']['instantaneous']['combined_core']['relative_l2_error'],.01)
        zero=gradient_metrics(torch.zeros(3),torch.ones(3))
        self.assertIsNone(zero['cosine']); self.assertIsNone(zero['relative_l2_error']); self.assertIsNone(zero['sign_agreement'])

    def prepare(self,root):
        directory=root/'data/graphs/central-256'; directory.mkdir(parents=True)
        np.savez_compressed(directory/'graph.npz',**fixture())
        write_json(directory/'graph-card.json',dict(graph_sha256=sha256(directory/'graph.npz')))
        (root/'docs').mkdir(); (root/'docs/WIRING-LEARNING-PROTOCOL.md').write_text('Synthetic unit-test fixture only.',encoding='utf8')

    def test_reward_checkpoint_resume_preserves_parameters_stream_and_action_rng(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            roots=[Path(a),Path(b)]
            for root in roots: self.prepare(root)
            first=run('context','measured','reward',17,root=roots[0],stop_after=200)
            run('context','measured','reward',17,root=roots[1],stop_after=100)
            resumed=run('context','measured','reward',17,root=roots[1],stop_after=200)
            self.assertEqual(first['training_stream_sha256'],resumed['training_stream_sha256'])
            records=[torch.load(root/'runs/wiring-learning-v1/context-measured-reward-s17/last.pt',weights_only=True,map_location='cpu') for root in roots]
            for name,value in records[0]['model'].items(): torch.testing.assert_close(value,records[1]['model'][name],atol=0,rtol=0)
            torch.testing.assert_close(records[0]['action_rng'],records[1]['action_rng'],atol=0,rtol=0)
            self.assertEqual(records[0]['sensory_rng'],records[1]['sensory_rng'])
            self.assertEqual(records[0]['baseline'],records[1]['baseline'])
            (roots[1]/'docs/WIRING-LEARNING-PROTOCOL.md').write_text('Changed fixture protocol.',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Resume changes'):
                run('context','measured','reward',17,root=roots[1],stop_after=300)


if __name__=='__main__': unittest.main()
