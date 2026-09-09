import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from flm.model import Config, FLM
from flm.provenance import sha256
from flm.train import Sampler, evaluate, restored, save_checkpoint
from test_model import fixture


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_sampler_never_crosses_documents(self):
        docs = [("a", np.full(40, 11)), ("b", np.full(40, 22))]
        sampler = Sampler(docs, 42, 8)
        x, y = sampler.sample(64, "cpu")
        self.assertTrue(torch.all(x == y))
        self.assertTrue(torch.all(x == x[:, :1]))

    def test_checkpoint_restores_predictions_optimizer_and_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); graph = root / "graph.npz"
            np.savez(graph, **fixture())
            model = FLM(fixture(), Config(neurons=16, pools=8, embedding=8))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            sampler = Sampler([("a", np.arange(100))], 6, 8)
            x, y = sampler.sample(2, "cpu")
            loss = model(x)[0].square().mean(); loss.backward(); optimizer.step()
            run = dict(graph_sha256=sha256(graph), python_torch=torch.__version__)
            save_checkpoint(root / "model.pt", model, optimizer, sampler, 1, 8.0, run)
            loaded, checkpoint = restored(root / "model.pt", graph)
            torch.testing.assert_close(model(x)[0], loaded(x)[0])
            second = Sampler([("a", np.arange(100))], 100, 8)
            second.rng.bit_generator.state = checkpoint["sampler_rng"]
            torch.testing.assert_close(sampler.sample(3, "cpu")[0], second.sample(3, "cpu")[0])
            resumed_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.01)
            resumed_optimizer.load_state_dict(checkpoint["optimizer"])
            for m, opt in [(model, optimizer), (loaded, resumed_optimizer)]:
                opt.zero_grad(); m(x)[0].square().mean().backward(); opt.step()
            torch.testing.assert_close(model(x)[0], loaded(x)[0])

    def test_evaluation_is_invariant_to_document_order(self):
        model = FLM(fixture(), Config(neurons=16, pools=8, embedding=8))
        docs = [("a", np.array([256, 97, 98, 257])), ("b", np.array([256, 100, 101, 102, 257]))]
        a = evaluate(model, docs, sequence=2)
        b = evaluate(model, list(reversed(docs)), sequence=2)
        self.assertAlmostEqual(a["bits_per_byte"], b["bits_per_byte"], places=6)
        self.assertEqual(a["bytes"], 5)


if __name__ == "__main__":
    unittest.main()
