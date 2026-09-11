import unittest

import numpy as np
from scipy.sparse import csr_matrix
import torch

from flm.model import Config, FLM
from flm.selection_graphs import induced_graph, verify_model


class SelectionGraphTests(unittest.TestCase):
    def fixture(self):
        ids = np.array([40,10,30,20],dtype=np.uint64)
        metadata = [[int(body), name, 'cb_intrinsic','L','label',1, None if i == 1 else [i,2,3]]
                    for i,(body,name) in enumerate(zip(ids,['z','a','b','c']))]
        signs = np.array([1,0,-1,1],dtype=np.int8)
        matrix = csr_matrix((np.array([2,3,5,7,11,13],dtype=np.uint32),
                             ([0,0,2,2,3,3],[0,1,0,3,2,3])),shape=(4,4))
        return matrix, metadata, ids, signs

    def test_canonical_identity_direction_sign_filter_and_balanced_pools(self):
        matrix, metadata, ids, signs = self.fixture()
        graph = induced_graph(matrix,metadata,ids,signs,np.array([0,1,2]),pools=2)
        np.testing.assert_array_equal(graph['body_ids'],[10,30,40])
        np.testing.assert_array_equal(graph['source_indices'],[1,2,0])
        np.testing.assert_array_equal(graph['row'],[1,2])
        np.testing.assert_array_equal(graph['col'],[2,2])
        np.testing.assert_array_equal(graph['contacts'],[5,2])
        np.testing.assert_array_equal(graph['weight'],[1,1])
        np.testing.assert_array_equal(graph['pool'],[0,0,1])
        self.assertTrue(np.isnan(graph['positions'][0]).all())
        changed_order = induced_graph(matrix,metadata,ids,signs,np.array([2,0,1]),pools=2)
        for key in graph: np.testing.assert_array_equal(graph[key],changed_order[key])

    def test_signed_incoming_normalization_and_real_forward_rng_preservation(self):
        matrix, metadata, ids, signs = self.fixture()
        graph = induced_graph(matrix,metadata,ids,signs,np.arange(4),pools=2)
        for post in np.unique(graph['row']):
            selected = graph['row'] == post
            self.assertAlmostEqual(float(np.abs(graph['weight'][selected]).sum()),1.,places=6)
        np.testing.assert_array_equal(np.sign(graph['weight']),graph['source_sign'][graph['col']])
        state = torch.random.get_rng_state().clone()
        original = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            card = verify_model(graph,Config(neurons=4,pools=2,embedding=8,vocabulary=64,tied_readout=True))
            self.assertEqual(card['sequence_state_bytes_batch1_float32'],32)
            self.assertTrue(card['forward_fixture']['finite_logits_and_states'])
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        finally: torch.set_num_threads(original)

    def test_invalid_indices_counts_pools_and_contact_precision_rejected(self):
        matrix, metadata, ids, signs = self.fixture()
        for selected in (np.array([0,0]),np.array([-1]),np.array([4]),np.array([],dtype=int)):
            with self.assertRaises(ValueError): induced_graph(matrix,metadata,ids,signs,selected,pools=1)
        with self.assertRaises(ValueError): induced_graph(matrix,metadata,ids,signs,np.arange(4),pools=5)
        for count in (-1,0,2**24+1):
            changed = matrix.astype(np.int64); changed.data[0] = count
            with self.assertRaises(ValueError): induced_graph(changed,metadata,ids,signs,np.arange(4),pools=2)

    def test_single_incoming_edge_gain_is_removed_by_actual_model_normalization(self):
        matrix, metadata, ids, signs = self.fixture()
        graph = induced_graph(matrix,metadata,ids,signs,np.arange(4),pools=2)
        singleton_rows = np.flatnonzero(np.bincount(graph['row'],minlength=4) == 1)
        self.assertGreater(len(singleton_rows),0)
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                model = FLM(graph,Config(neurons=4,pools=2,embedding=8,vocabulary=64,tied_readout=True))
                before = model.constants()[0]
                model.edge_log_gain.copy_(torch.linspace(-3,3,len(graph['row'])))
                after = model.constants()[0]
                self.assertTrue(torch.equal(before[singleton_rows],after[singleton_rows]))
                self.assertFalse(torch.equal(before,after))  # Multi-input rows can change.
        finally: torch.set_num_threads(original_threads)


if __name__ == '__main__': unittest.main()
