import unittest
import numpy as np
from flm.wiring_controls import rewire_graph, validate_control


class WiringControlTests(unittest.TestCase):
    def graph(self):
        n = 16
        row = np.repeat(np.arange(n), 4).astype(np.int32)
        col = np.array([(i + d) % n for i in range(n) for d in (0, 1, 3, 6)], dtype=np.int32)
        signs = np.where(np.arange(n) % 2, -1, 1).astype(np.int8)
        return dict(row=row, col=col, weight=np.tile(np.array([.1,.2,.3,.4]), n)*signs[col],
            source_sign=signs, body_ids=np.arange(n), pool=np.arange(n)//2,
            positions=np.full((n,3),np.nan), contacts=np.tile(np.array([1,2,3,4]),n))

    def test_preserves_signed_degrees_weights_self_edges_and_provenance(self):
        graph=self.graph(); before={k:v.copy() for k,v in graph.items()}
        control,report=rewire_graph(graph,101)
        self.assertTrue(all(validate_control(graph,control).values()))
        self.assertLess(report['original_edge_overlap_fraction'],1)
        self.assertEqual(report['accepted_swaps'],10*len(graph['row']))
        self.assertEqual(report['original']['self_edges'],report['randomized']['self_edges'])
        for k in graph: np.testing.assert_array_equal(graph[k],before[k])
        for i in range(len(graph['body_ids'])):
            np.testing.assert_array_equal(np.sort(graph['weight'][graph['row']==i]),np.sort(control['weight'][control['row']==i]))
        same,_=rewire_graph(graph,101)
        np.testing.assert_array_equal(control['col'],same['col'])
        other,_=rewire_graph(graph,103)
        self.assertFalse(np.array_equal(control['col'],other['col']))

    def test_rejects_invariant_changes_and_impossible_controls(self):
        graph=self.graph(); control,_=rewire_graph(graph,101)
        control['weight'][0]*=2
        with self.assertRaisesRegex(ValueError,'fixed array'): validate_control(graph,control)
        only_self={k:v.copy() for k,v in graph.items()}
        keep=graph['row']==graph['col']
        for key in ('row','col','weight','contacts'): only_self[key]=only_self[key][keep]
        with self.assertRaisesRegex(ValueError,'non-self'): rewire_graph(only_self,101)


if __name__=='__main__': unittest.main()
