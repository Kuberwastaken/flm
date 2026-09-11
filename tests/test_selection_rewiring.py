from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flm.selection_rewiring import prepare_one


class SelectionRewiringTests(unittest.TestCase):
    def graph(self,path):
        n = 6
        graph = dict(row=np.arange(n,dtype=np.int32),col=np.roll(np.arange(n,dtype=np.int32),-1),
            weight=np.ones(n,dtype=np.float32),contacts=np.ones(n,dtype=np.uint32),
            source_sign=np.ones(n,dtype=np.int8),body_ids=np.arange(100,100+n,dtype=np.int64),
            source_indices=np.arange(n,dtype=np.int32),positions=np.zeros((n,3),dtype=np.float32),
            pool=np.arange(n,dtype=np.int32)//2)
        np.savez_compressed(path,**graph)

    def test_real_swaps_roundtrip_and_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'original.npz'; self.graph(source)
            receipt = prepare_one(source,root/'result',101)
            self.assertEqual(receipt['status'],'complete')
            self.assertEqual(receipt['report']['accepted_swaps'],60)
            with patch('flm.selection_rewiring.rewire_graph',side_effect=AssertionError('Must reuse verified receipt')):
                self.assertEqual(prepare_one(source,root/'result',101),receipt)
            with self.assertRaises(ValueError): prepare_one(source,root/'result',103)
            with (root/'result/graph.npz').open('ab') as handle: handle.write(b'changed')
            with self.assertRaises(ValueError): prepare_one(source,root/'result',101)

    def test_orphan_payload_recreated_and_failures_remain_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'original.npz'; self.graph(source)
            output = root/'result'; output.mkdir(); (output/'graph.npz').write_bytes(b'orphan')
            receipt = prepare_one(source,output,101)
            self.assertEqual(receipt['status'],'complete')
            failed = root/'failed'
            with patch('flm.selection_rewiring.rewire_graph',side_effect=ValueError('fixture infeasible swap budget')):
                first = prepare_one(source,failed,101)
            self.assertEqual(first['status'],'failed')
            self.assertIn('fixture infeasible',first['reason'])
            with patch('flm.selection_rewiring.rewire_graph',side_effect=AssertionError('Failure must not be silently rerun')):
                self.assertEqual(prepare_one(source,failed,101),first)


if __name__ == '__main__': unittest.main()
