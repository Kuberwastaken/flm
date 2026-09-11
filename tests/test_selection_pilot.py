import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from flm.model import Config
from flm.selection_pilot import Pilot,graph_digest,measure,ordered_selections,prepare_inputs,prerequisites,run
from flm.train import Sampler


def fixture():
    row=np.arange(6,dtype=np.int32); col=(row+1)%6
    graph=dict(row=row,col=col,weight=np.ones(6,dtype=np.float32),contacts=np.ones(6,dtype=np.uint32),
        source_sign=np.ones(6,dtype=np.int8),body_ids=np.arange(100,106,dtype=np.int64),
        source_indices=row.copy(),positions=np.zeros((6,3),dtype=np.float32),pool=row//2)
    config=Config(neurons=6,pools=3,embedding=4,vocabulary=12,tied_readout=True)
    docs=[('first',np.arange(24,dtype=np.int32)%12),('second',np.arange(35,dtype=np.int32)%12)]
    lengths=np.array([0,0,1,1,1,2,2,2,3,3,3,4],dtype=np.int64)
    return graph,config,docs,lengths


class SelectionPilotTests(unittest.TestCase):
    def test_real_updates_preserve_graph_global_rng_threads_and_match_sampled_exposure(self):
        graph,config,docs,lengths=fixture(); before=graph_digest(graph)
        rng=torch.get_rng_state().clone(); threads=torch.get_num_threads()
        pilot=Pilot(1,2,2,5,1,1,42)
        result=measure(graph,config,docs,lengths,pilot)
        self.assertEqual(graph_digest(graph),before)
        self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertEqual(torch.get_num_threads(),threads)
        self.assertTrue(result['disposable_parameters_updated'])
        self.assertEqual(result['dtype'],'torch.float32'); self.assertFalse(result['checkpoint_written'])
        self.assertFalse(result['language_scores_reported'])
        sampler=Sampler(docs,42,5); digest=hashlib.sha256(); expected=[]
        for _ in range(3):
            x,y=sampler.sample(2,'cpu')
            digest.update(x.numpy().astype('<i8').tobytes()); digest.update(y.numpy().astype('<i8').tobytes())
            expected.append((int(lengths[x.numpy()].sum()),int(lengths[y[:,1:].numpy()].sum())))
        self.assertEqual(result['sampled_windows_sha256'],digest.hexdigest())
        self.assertEqual(result['measured_input_tokens'],20); self.assertEqual(result['measured_supervised_targets'],16)
        self.assertEqual(result['measured_input_bytes'],sum(r[0] for r in expected[1:]))
        self.assertEqual(result['measured_supervised_bytes'],sum(r[1] for r in expected[1:]))
        self.assertEqual([r['warmup'] for r in result['observations']],[True,False,False])
        changed=copy.deepcopy(graph); changed['col']=(changed['col']+1)%6
        second=measure(changed,config,docs,lengths,pilot)
        self.assertEqual(second['sampled_windows_sha256'],result['sampled_windows_sha256'])
        self.assertNotEqual(second['graph_arrays_sha256'],result['graph_arrays_sha256'])

    def test_invalid_dimensions_lengths_tokens_and_nonfinite_loss_fail_cleanly(self):
        graph,config,docs,lengths=fixture(); pilot=Pilot(1,1,2,5,1,1,42)
        for bad in (replace(pilot,batch=0),replace(pilot,context_warmup=5),replace(pilot,seed=-1),replace(pilot,threads=True)):
            with self.assertRaises(ValueError): measure(graph,config,docs,lengths,bad)
        with self.assertRaisesRegex(ValueError,'byte lengths'): measure(graph,config,docs,lengths[:-1],pilot)
        before=graph_digest(graph); rng=torch.get_rng_state().clone(); threads=torch.get_num_threads()
        with patch('flm.selection_pilot.F.cross_entropy',return_value=torch.tensor(float('nan'))):
            with self.assertRaisesRegex(FloatingPointError,'pilot loss'): measure(graph,config,docs,lengths,pilot)
        self.assertEqual(graph_digest(graph),before); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(),threads)
        with self.assertRaisesRegex(ValueError,'invalid token IDs'):
            measure(graph,config,[('bad',np.full(25,99,dtype=np.int32))],lengths,pilot)

    def test_order_is_complete_reproducible_and_independent_of_manifest_key_order(self):
        source=dict(graphs={f'case{i}/candidate':{} for i in range(64)})
        names=ordered_selections(source)
        self.assertEqual(len(names),64); self.assertEqual(set(names),set(source['graphs']))
        self.assertNotEqual(names,sorted(names))
        self.assertEqual(names,ordered_selections(dict(graphs=dict(reversed(list(source['graphs'].items()))))))

    def test_official_run_refuses_before_opening_inputs_or_creating_files(self):
        with tempfile.TemporaryDirectory() as folder,patch('flm.selection_pilot.prepare_inputs') as prepare:
            with self.assertRaisesRegex(ValueError,'Priority BabyLM'): run(Path(folder))
            prepare.assert_not_called(); self.assertEqual(list(Path(folder).iterdir()),[])

    def test_priority_gate_checks_checkpoint_bytes_and_structural_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for scale in ('10m','100m'):
                for variant in ('flm','gru','transformer'):
                    for seed in (42,43):
                        path=root/f'runs/babylm-{scale}/{variant}-s{seed}'
                        path.mkdir(parents=True); (path/'best.pt').write_bytes(b'fixture')
                        (path/'complete.json').write_text(json.dumps(dict(steps=12000,best_checkpoint_sha256=hashlib.sha256(b'fixture').hexdigest())))
            with self.assertRaisesRegex(ValueError,'structural queue'): prerequisites(root)
            archive=root/'public/research/selection-rewiring.zip'; archive.parent.mkdir(parents=True); archive.write_bytes(b'fixture archive')
            release=root/'reports/selection-rewiring/release.json'; release.parent.mkdir(parents=True)
            release.write_text(json.dumps(dict(planned=192,complete=191,failed=1,
                arrays_degrees_signs_weights_self_edges_and_diagnostics_verified=True,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
            prerequisites(root)
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Structural release identity'): prerequisites(root)
            (root/'runs/babylm-10m/flm-s42/best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'completion identity'): prerequisites(root)

    def test_input_preparation_opens_training_cache_only_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tokenizer=root/'data/tokenizers/babylm-2026-4096/tokenizer.json'
            cache=root/'data/processed/babylm-2026-bpe/train-10m'; study=root/'runs/babylm-10m/study.json'
            tokenizer.parent.mkdir(parents=True); tokenizer.write_bytes(b'fixture tokenizer')
            cache.mkdir(parents=True); (cache/'manifest.json').write_bytes(b'fixture manifest')
            study.parent.mkdir(parents=True)
            tokhash=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            study.write_text(json.dumps(dict(dataset='BabyLM 2026 10m',tokenizer_sha256=tokhash,
                train_manifest_sha256=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest())))
            with patch('flm.selection_pilot.Lexicon') as lexicon,patch('flm.selection_pilot.read_mmap') as read:
                lexicon.return_value.sha256=tokhash; lexicon.return_value.vocabulary=4096
                read.return_value=([],[],dict(files={'fixture':'digest'}))
                _,_,binding=prepare_inputs(root)
                read.assert_called_once_with(cache,tokhash)
                self.assertFalse(binding['validation_or_test_opened'])
                tokenizer.write_bytes(b'changed'); read.reset_mock()
                with self.assertRaisesRegex(ValueError,'input identity changed'): prepare_inputs(root)
                read.assert_not_called()

    def test_attempt_failure_is_retained_and_complete_inventory_is_not_overwritten(self):
        repository=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); graph,config,docs,lengths=fixture()
            graph_root=root/'work/selection-graphs-v1'; graph_root.mkdir(parents=True)
            np.savez_compressed(graph_root/'fixture.npz',**graph)
            for name in ('selection_pilot.py','model.py','train.py','corpus_cache.py','tokenizer.py','scan_train.py','inference.py'):
                path=root/'flm'/name; path.parent.mkdir(exist_ok=True); path.write_bytes((repository/'flm'/name).read_bytes())
            from dataclasses import asdict
            entry=dict(path='fixture.npz',graph_sha256=hashlib.sha256((graph_root/'fixture.npz').read_bytes()).hexdigest(),
                config=asdict(config),trainable_parameters=123)
            manifest=dict(graphs={f'case{i}/candidate':entry for i in range(64)},
                model_source_sha256=hashlib.sha256((root/'flm/model.py').read_bytes()).hexdigest())
            (graph_root/'manifest.json').write_text(json.dumps(manifest))
            release=root/'reports/selection-graphs/export-release.json'; release.parent.mkdir(parents=True)
            release.write_text(json.dumps(dict(manifest_sha256=hashlib.sha256((graph_root/'manifest.json').read_bytes()).hexdigest())))
            measured=dict(parameter_card=dict(trainable_parameters=123),sampled_windows_sha256='fixture')
            with patch('flm.selection_pilot.prerequisites'),patch('flm.selection_pilot.prepare_inputs',
                    return_value=(docs,SimpleNamespace(lengths=lengths),dict(fixture=True))) as prepare, \
                    patch('flm.selection_pilot.measure',side_effect=[copy.deepcopy(measured),ValueError('Synthetic failure')]):
                with self.assertRaisesRegex(ValueError,'Synthetic failure'): run(root)
            attempts=[p for p in (root/'runs/selection-timing-pilot').iterdir() if p.is_dir()]
            self.assertEqual(len(attempts),1)
            self.assertEqual(json.loads((attempts[0]/'failure.json').read_text())['completed'],1)
            self.assertFalse((root/'reports/selection-pilot/timing.json').exists())
            with patch('flm.selection_pilot.prerequisites'),patch('flm.selection_pilot.prepare_inputs',
                    return_value=(docs,SimpleNamespace(lengths=lengths),dict(fixture=True))) as prepare, \
                    patch('flm.selection_pilot.measure',side_effect=lambda *args:copy.deepcopy(measured)), \
                    patch('builtins.print'):
                destination=run(root)
                result=json.loads(destination.read_text()); self.assertEqual(len(result['conditions']),64)
                self.assertEqual(len({r['selection'] for r in result['conditions']}),64)
                self.assertFalse(result['training_matrix_frozen'])
                prepare.reset_mock()
                with self.assertRaisesRegex(ValueError,'already recorded'): run(root)
                prepare.assert_not_called()
            self.assertTrue((attempts[0]/'failure.json').exists())
            self.assertEqual(list(root.rglob('*.pt')),[])


if __name__=='__main__': unittest.main()
