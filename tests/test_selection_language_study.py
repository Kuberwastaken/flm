"""Synthetic complete groups; no official selection fit, timing or protocol."""
from contextlib import ExitStack
from dataclasses import asdict, replace
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from flm.language_learning_validation import Panel, document_identity
from flm.provenance import sha256
from flm.selection_language import catalog_from_bytes, digest, load_case_graph, fit_case
from flm.selection_language_study import (IDENTITY, PILOT, PROTOCOL, SELECTORS, SELECTION, SOURCES,
    STUDY, initialize, matrix, train_study, freeze_selection, verified_context, verify_pilot)
from flm.selection_pilot import Pilot, SOURCES as PILOT_SOURCES, measure
from flm.model import Config
from test_selection_language import archive_fixture, SETTINGS

TINY = Pilot(1,2,2,5,1,1,42)
FIT = replace(SETTINGS, steps=2, checkpoint_interval=1)
GROUP = 'fixture-L-t5'


def complete_fixture():
    data, corpus = archive_fixture()
    with zipfile.ZipFile(io.BytesIO(data)) as archive: files = {n:archive.read(n) for n in archive.namelist()}
    source = json.loads(files['source-manifest.json']); rewiring = json.loads(files['rewiring-manifest.json'])
    for selector in SELECTORS[2:]:
        old = GROUP+'/candidate'; name = GROUP+'/'+selector
        row = dict(source['graphs'][old], selection=name, path=f'graphs/{name}.npz')
        source['graphs'][name] = row; files['original/'+row['path']] = files[f'original/graphs/{old}.npz']
        for seed in (101,103,107):
            old_label = old+f'/null{seed}'; label = name+f'/null{seed}'
            rewiring['records'][label] = copy.deepcopy(rewiring['records'][old_label])
            files[f'rewired/{label}.npz'] = files[f'rewired/{old_label}.npz']
    files['source-manifest.json'] = json.dumps(source).encode()
    rewiring.update(source_manifest_sha256=digest(files['source-manifest.json']), complete=24, planned=24)
    files['rewiring-manifest.json'] = json.dumps(rewiring).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, payload in files.items(): archive.writestr(name, payload)
    data = buffer.getvalue()
    corpus.binding.update(partition='train-10m', cache_manifest_sha256='b'*64, cache_files_sha256={'fixture':'c'*64})
    return catalog_from_bytes(data, digest(data), 8), corpus


class SelectionStudyTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads(); torch.set_num_threads(1); self.addCleanup(torch.set_num_threads, previous)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        source = Path(__file__).resolve().parents[1]/'flm'; (self.root/'flm').mkdir()
        for name in SOURCES: shutil.copyfile(source/name, self.root/'flm'/name)
        self.catalog, self.corpus = complete_fixture()
        docs = [('validation-fixture', np.array([0,2,3,4,5,6,7,1], dtype=np.int32))]
        self.panel = Panel(docs, dict(partition='validation', coverage=document_identity(docs, self.corpus.lexicon)))
        self.write(PROTOCOL, 'Artificial fixture protocol; not a chosen official study.', raw=True)
        self.write(Path('runs/babylm-10m/study.json'), dict(fixture=True))
        manifest = dict(files={'ids.bin':'test payload deliberately absent'})
        self.write(Path('data/processed/babylm-2026-bpe/test/manifest.json'), manifest)
        self.write(Path('data/tokenizers/babylm-2026-4096/tokenization-card.json'),
            dict(tokenizer_sha256=self.corpus.lexicon.sha256, partitions=dict(test=manifest)))
        binding = {k:self.corpus.binding[k] for k in ('partition','tokenizer_sha256','cache_manifest_sha256','cache_files_sha256')}
        binding.update(study_sha256=sha256(self.root/'runs/babylm-10m/study.json'), validation_or_test_opened=False)
        self.pilot = dict(binding=binding, conditions=[])
        for label, entry in self.catalog.entries.items():
            if entry['graph_seed'] is not None: continue
            graph, _ = load_case_graph(self.catalog, label)
            row = measure(graph, Config(**entry['config']), self.corpus.documents, self.corpus.lexicon.lengths, TINY)
            row.update(selection=entry['selection'], graph_sha256=entry['graph_sha256'])
            self.pilot['conditions'].append(row)
        self.write(PILOT, self.pilot)
        self.request = dict(groups=[GROUP], settings=asdict(FIT), rationale='Synthetic complete-control fixture only',
            cost_allowance_multiplier=2., wall_time_budget_seconds=100000.)

    def write(self, relative, value, raw=False):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value), encoding='utf8')

    def patched(self):
        stack = ExitStack()
        for name, value in (('load_catalog',self.catalog), ('load_corpus',self.corpus),
                            ('load_panel',self.panel), ('verify_pilot',self.pilot), ('Pilot',TINY)):
            stack.enter_context(patch('flm.selection_language_study.'+name, return_value=value))
        stack.enter_context(patch('flm.selection_language_study.prerequisites'))
        stack.enter_context(patch('builtins.print'))
        return stack

    def test_matrix_requires_whole_known_groups_with_all_controls(self):
        rows = matrix(self.catalog, [GROUP]); self.assertEqual(len(rows),64)
        self.assertEqual(len({r['label'] for r in rows}),64)
        for groups in ([],[GROUP,GROUP],['missing'],[GROUP+'/candidate']):
            with self.assertRaises(ValueError): matrix(self.catalog, groups)
        data, _ = archive_fixture(); incomplete = catalog_from_bytes(data,digest(data),2)
        with self.assertRaisesRegex(ValueError,'complete selector'): matrix(incomplete,[GROUP])

    def test_priority_and_missing_cost_gates_do_not_load_training_inputs(self):
        with patch('flm.selection_language_study.load_corpus') as loader:
            with self.assertRaisesRegex(ValueError,'Priority BabyLM'): initialize(self.root,self.request)
            with patch('flm.selection_language_study.prerequisites'), patch('flm.selection_language_study.load_catalog',return_value=self.catalog):
                (self.root/PILOT).unlink()
                with self.assertRaisesRegex(ValueError,'Measure the full-window'): initialize(self.root,self.request)
            loader.assert_not_called()
        self.assertFalse((self.root/IDENTITY).exists())

    def test_initialization_is_immutable_and_requires_affordable_complete_group(self):
        with self.patched():
            bad = copy.deepcopy(self.request); bad['wall_time_budget_seconds'] = 1e-12
            with self.assertRaisesRegex(ValueError,'exceeds'): initialize(self.root,bad)
            identity = initialize(self.root,self.request)
            self.assertEqual(len(identity['conditions']),64)
            self.assertEqual(sum(r['cost_is_proxy_for_untimed_rewire'] for r in identity['conditions']),48)
            self.assertFalse(identity['test_metadata']['payloads_opened'])
            self.assertEqual(initialize(self.root,self.request),identity)
            changed = copy.deepcopy(self.request); changed['rationale'] += ' changed'
            with self.assertRaisesRegex(ValueError,'different selection'): initialize(self.root,changed)
            altered = copy.deepcopy(identity); altered['conditions'].pop(); self.write(IDENTITY,altered)
            with self.assertRaisesRegex(ValueError,'identity changed'): verified_context(self.root)

    def test_existing_unselected_run_prevents_retroactive_registration(self):
        (self.root/STUDY/'unselected-group').mkdir(parents=True)
        with self.patched():
            with self.assertRaisesRegex(ValueError,'run files already exist'): initialize(self.root,self.request)
        self.assertFalse((self.root/IDENTITY).exists())

    def test_full_tiny_group_resumes_interruption_then_selects_all_64_without_test_data(self):
        from flm import selection_language_study as coordinator
        with self.patched():
            initialize(self.root,self.request); real = fit_case; calls = []
            def interrupted(*args,**kwargs):
                calls.append(args[2])
                if len(calls)==3:
                    real(*args,**kwargs,until=1)
                    raise RuntimeError('fixture interruption')
                return real(*args,**kwargs)
            with patch.object(coordinator,'fit_case',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'): train_study(self.root)
            with patch.object(coordinator,'load_panel') as loader:
                with self.assertRaisesRegex(ValueError,'run incomplete'): freeze_selection(self.root)
                loader.assert_not_called()
            train_study(self.root); result = freeze_selection(self.root)
            self.assertEqual(len(result['conditions']),64); self.assertFalse(result['test_payloads_opened'])
            self.assertEqual(freeze_selection(self.root),result)
            for seed in (42,43):
                rows = [r for r in result['conditions'] if r['condition']['seed']==seed]
                self.assertTrue(all(r['final_exposure']==rows[0]['final_exposure'] for r in rows))
                self.assertEqual(rows[0]['final_exposure']['presented_tokens'],20)
            self.assertFalse((self.root/'data/processed/babylm-2026-bpe/test/ids.bin').exists())
            (self.root/result['conditions'][-1]['checkpoint']).write_bytes(b'changed')
            with self.assertRaises(ValueError): freeze_selection(self.root)

    def test_changed_cost_window_source_protocol_or_metadata_cannot_resume(self):
        with self.patched():
            initialize(self.root,self.request)
            for path in (PROTOCOL,Path('flm/selection_language.py')):
                original = (self.root/path).read_bytes(); (self.root/path).write_bytes(original+b'\nchanged')
                with self.assertRaisesRegex(ValueError,'identity changed'): verified_context(self.root)
                (self.root/path).write_bytes(original)
            self.pilot['conditions'][0]['sampled_windows_sha256'] = 'f'*64
            with self.assertRaisesRegex(ValueError,'windows do not replay'): verified_context(self.root)

    def test_cost_gate_rejects_bad_inventory_source_settings_and_arithmetic(self):
        # Fabricated records test the gate schema, not observed official timing.
        baseline = copy.deepcopy(self.pilot)
        baseline.update(source_sha256={n:sha256(self.root/'flm'/n) for n in PILOT_SOURCES},
            graph_manifest_sha256=self.catalog.binding['source_manifest_sha256'],graph_order_seed=519,
            benchmark_budget_selected=False,training_matrix_frozen=False,torch=str(torch.__version__),numpy=str(np.__version__))
        for row in baseline['conditions']:
            row.update(pilot=asdict(Pilot()),training_settings=asdict(Pilot().training_settings()),
                sampled_windows_sha256='a'*64,measured_seconds=12.,measured_input_tokens=18432,
                measured_supervised_targets=15360,measured_input_bytes=54000,measured_supervised_bytes=45000)
            row['observations']=[dict(step=i,warmup=i<=3,seconds=1.,input_tokens=1536,supervised_targets=1280,
                input_bytes=4500,supervised_bytes=3750) for i in range(1,16)]
        ordered = sorted(baseline['conditions'],key=lambda r:r['selection'])
        order = np.random.Generator(np.random.PCG64(519)).permutation(len(ordered))
        baseline['conditions'] = [ordered[int(i)] for i in order]
        self.write(PILOT,baseline); verify_pilot(self.root,self.catalog)
        for mutate in (lambda r:r['conditions'].pop(), lambda r:r['source_sha256'].clear(),
                lambda r:r['conditions'][0]['training_settings'].update(sequence=3),
                lambda r:r['conditions'][0].update(measured_seconds=999),
                lambda r:r['conditions'][0].update(graph_sha256='f'*64),
                lambda r:r['conditions'].reverse()):
            bad = copy.deepcopy(baseline); mutate(bad); self.write(PILOT,bad)
            with self.assertRaises(ValueError): verify_pilot(self.root,self.catalog)


if __name__=='__main__': unittest.main()
