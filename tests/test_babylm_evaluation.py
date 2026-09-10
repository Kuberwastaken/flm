import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from flm.babylm_test import freeze_selection, scored_blocks, read_json, TOKENIZER, GRAPH, CACHE
from flm.provenance import sha256, write_json


class Uniform(torch.nn.Module):
    def forward(self, x, state=None): return torch.zeros(*x.shape, 5), None


class BabyLMEvaluationTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_incomplete_study_fails_before_any_corpus_or_checkpoint_read(self):
        with tempfile.TemporaryDirectory() as folder, patch('flm.babylm_test.read_mmap') as read:
            with self.assertRaisesRegex(ValueError, 'incomplete'): freeze_selection(Path(folder))
            read.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_batch_resume_reuses_only_matching_complete_verified_blocks(self):
        lexicon = SimpleNamespace(vocabulary=5, lengths=np.array([0, 0, 1, 2, 4]))
        docs = [('a', np.array([0, 2, 3, 4, 1], dtype=np.uint16)), ('b', np.array([0, 4, 1], dtype=np.uint16))]
        metadata = [dict(id=i, component='fixture', utf8_bytes=n, overlap_filtered_eligible=True) for i, n in [('a', 7), ('b', 4)]]
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            destination = Path(folder); identity = dict(checkpoint_sha256='fixture')
            a, seconds = scored_blocks(Uniform(), docs, metadata, lexicon, destination, identity, 1, 2)
            with patch('flm.babylm_test.evaluate_batch', side_effect=AssertionError('Must reuse completed scores')):
                b, other_seconds = scored_blocks(Uniform(), docs, metadata, lexicon, destination, identity, 1, 2)
            self.assertEqual(a, b); self.assertEqual(seconds, other_seconds)
            self.assertEqual([r['document'] for r in a], ['a', 'b'])
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                scored_blocks(Uniform(), docs, metadata, lexicon, destination, dict(checkpoint_sha256='changed'), 1, 2)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                scored_blocks(Uniform(), docs, metadata, lexicon, destination, identity, 2, 2)
            path = destination / 'batch-00000.json'; record = read_json(path); record['documents'][0]['bytes'] += 1
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, 'complete prepared block'):
                scored_blocks(Uniform(), docs, metadata, lexicon, destination, identity, 1, 2)

    def test_selection_rejects_altered_training_budget_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in (TOKENIZER, GRAPH, Path('docs/BABYLM-PROTOCOL.md'), Path('docs/BABYLM-EVALUATION.md'),
                         Path('data/prompts/babylm-original.json'), TOKENIZER.with_name('validation-panel.json')):
                (root / path).parent.mkdir(parents=True, exist_ok=True); (root / path).write_bytes(b'fixture')
            token_hash = sha256(root / TOKENIZER); parts = {}; hashes = {}
            for name in ('train-10m', 'train-100m', 'validation', 'test'):
                parts[name] = dict(fixture=name)
                path = root / CACHE / name / 'manifest.json'; write_json(path, parts[name]); hashes[name] = sha256(path)
            write_json(root / TOKENIZER.with_name('tokenization-card.json'), dict(tokenizer_sha256=token_hash, partitions=parts))
            for scale in ('10m', '100m'):
                panel_hash = sha256(root / TOKENIZER.with_name('validation-panel.json'))
                protocol = dict(steps=12000, batch=16, sequence=96, warmup=16, learning_rate=.002,
                    final_learning_rate=.0002, lr_warmup_updates=100, weight_decay=.01, eval_tokens=49152, threads=4,
                    tokenizer_sha256=token_hash, train_cache_sha256=hashes[f'train-{scale}'],
                    validation_cache_sha256=hashes['validation'], graph_sha256=sha256(root / GRAPH),
                    validation_panel_sha256=panel_hash, evaluation_unit='block', cache_identity='SHA-256 of verified mmap manifest')
                write_json(root / f'runs/babylm-{scale}/study.json', dict(dataset=f'BabyLM 2026 {scale}', steps=12000,
                    seeds=[42, 43], variants=['flm', 'gru', 'transformer'], protocol_sha256=sha256(root / 'docs/BABYLM-PROTOCOL.md'),
                    tokenizer_sha256=token_hash, train_manifest_sha256=hashes[f'train-{scale}'],
                    validation_manifest_sha256=hashes['validation'], validation_panel_sha256=panel_hash))
                for seed in (42, 43):
                    for variant in ('flm', 'gru', 'transformer'):
                        dest = root / f'runs/babylm-{scale}/{variant}-s{seed}'; dest.mkdir()
                        (dest / 'best.pt').write_bytes(f'{scale}-{seed}-{variant}'.encode())
                        write_json(dest / 'complete.json', dict(steps=12000, protocol=protocol, best_validation_bpb=2., best_checkpoint_sha256=sha256(dest / 'best.pt')))
                        write_json(dest / 'run.json', dict(seed=seed, protocol=protocol, source_commit='fixture',
                            test_set_used_for_training=False, parameter_card=dict(config=dict(variant=variant))))
            with patch('flm.babylm_test.read_mmap', side_effect=lambda path, _: ([], [], read_json(path / 'manifest.json'))):
                result = freeze_selection(root); self.assertEqual(len(result['runs']), 12)
                path = root / 'runs/babylm-100m/transformer-s43/complete.json'; complete = read_json(path)
                write_json(path, dict(complete, steps=10000))
                with self.assertRaisesRegex(ValueError, 'protocol changed'): freeze_selection(root)
                write_json(path, complete); path.with_name('best.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Invalid selected checkpoint'): freeze_selection(root)


if __name__ == '__main__': unittest.main()
