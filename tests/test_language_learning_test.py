"""Real tiny fits and held-out fixtures, never acquired BabyLM test payloads."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import flm.language_learning_test as evaluator
from flm.language_learning_study import IDENTITY, SELECTION, initialize, train_study, freeze_selection, read
from flm.provenance import sha256
import test_language_learning_study as study_fixture
from test_language_learning_study import SETTINGS
from test_language_learning_train import DOCUMENTS, Lexicon


class LearningTestEvaluationTests(unittest.TestCase):
    setUp = study_fixture.LearningStudyTests.setUp
    write = study_fixture.LearningStudyTests.write
    prepare = study_fixture.LearningStudyTests.prepare
    patched = study_fixture.LearningStudyTests.patched

    def setup_test_cache(self):
        documents = [(f'test/{component}/{i}', document.copy())
            for i, (component, document) in enumerate(zip(('one','one','two','two'),
                (DOCUMENTS[0][1], DOCUMENTS[1][1], DOCUMENTS[1][1], DOCUMENTS[0][1])))]
        destination = self.root/evaluator.CACHE; destination.mkdir(parents=True, exist_ok=True)
        offsets = np.array([0]+list(np.cumsum([len(doc) for _,doc in documents])), dtype=np.int64)
        np.save(destination/'offsets.npy', offsets)
        np.concatenate([doc for _,doc in documents]).astype('<u2').tofile(destination/'tokens.u16')
        metadata = [dict(id=name, token_start=int(offsets[i]), token_end=int(offsets[i+1]),
            component=name.split('/')[1], utf8_bytes=int(Lexicon.lengths[doc].sum()),
            overlap_filtered_eligible=i<2) for i,(name,doc) in enumerate(documents)]
        (destination/'blocks.jsonl').write_text('\n'.join(json.dumps(row) for row in metadata)+'\n', encoding='utf8')
        components = {}
        for component in ('one','two'):
            rows = [(name,doc) for name,doc in documents if name.split('/')[1] == component]
            components[component] = dict(blocks=len(rows), text_tokens=sum(len(doc)-2 for _,doc in rows),
                                        utf8_bytes=sum(int(Lexicon.lengths[doc].sum()) for _,doc in rows))
        manifest = dict(format='flm-mmap-tokens-v1', dtype='<u2', identity=dict(tokenizer_sha256=Lexicon.sha256),
            blocks=4, text_tokens=sum(len(doc)-2 for _,doc in documents),
            utf8_bytes=sum(row['utf8_bytes'] for row in metadata), components=components,
            files={name:sha256(destination/name) for name in ('tokens.u16','offsets.npy','blocks.jsonl')})
        self.write(evaluator.CACHE/'manifest.json', manifest)
        self.write(Path('data/tokenizers/babylm-2026-4096/tokenization-card.json'),
                   dict(tokenizer_sha256=Lexicon.sha256, partitions=dict(test=manifest)))
        return documents

    def evaluation_context(self):
        stack = self.patched()
        policy = dict(evaluator.POLICY, batch_size=2, chunk_size=3, threads=1, bootstrap_replicates=64)
        stack.enter_context(patch.object(evaluator, 'POLICY', policy))
        return stack

    def build_study(self):
        documents = self.setup_test_cache()
        initialize(self.root, SETTINGS)
        train_study(self.root)
        return documents

    def test_missing_or_incomplete_inventory_never_opens_test(self):
        with patch.object(evaluator, 'load_test') as loader:
            with self.assertRaisesRegex(ValueError, 'No frozen'): evaluator.score_study(self.root)
            loader.assert_not_called()
            with self.evaluation_context():
                self.setup_test_cache(); initialize(self.root, SETTINGS)
                with self.assertRaisesRegex(ValueError, 'incomplete'): evaluator.score_study(self.root)
                loader.assert_not_called()
        self.assertFalse((self.root/evaluator.SUMMARY).exists())

    def test_all_eight_real_tiny_fits_match_whole_block_oracle_and_report_all_pairs(self):
        with self.evaluation_context():
            documents = self.build_study()
            rng = torch.get_rng_state().clone(); threads = torch.get_num_threads()
            result = evaluator.score_study(self.root)
            self.assertTrue(torch.equal(rng, torch.get_rng_state())); self.assertEqual(threads, torch.get_num_threads())
            self.assertEqual(len(result['runs']),8); self.assertEqual(len(result['paired_comparisons']),24)
            self.assertEqual(len(result['aggregates']),24)
            selection = read(self.root/SELECTION); identity = read(self.root/IDENTITY)
            registered = {r['condition']['label']:r for r in identity['conditions']}
            for selected in selection['conditions']:
                label = selected['condition']['label']
                model = evaluator.restore_selected(self.root,self.inputs,registered[label],selected)
                records = read(self.root/evaluator.REPORTS/f'test-{label}.json')['blocks']
                for (_,tokens),record in zip(documents,records):
                    with torch.no_grad():
                        logits,_ = model(torch.tensor(tokens[:-1]).unsqueeze(0),None)
                        targets = torch.tensor(tokens[1:]); keep = targets>=2
                        oracle = float(F.cross_entropy(logits[0][keep],targets[keep],reduction='sum'))
                    self.assertAlmostEqual(oracle,record['nll'],places=10)
            filtered = [r for r in result['aggregates'] if r['component']=='two' and r['subset']=='overlap_filtered']
            self.assertTrue(all(r['mean_bpb'] is None for r in filtered))
            self.assertTrue((self.root/evaluator.SUMMARY).is_file())

    def test_cached_resume_checks_records_without_recomputing_likelihood(self):
        with self.evaluation_context():
            self.build_study(); result = evaluator.score_study(self.root)
            with patch('flm.babylm_test.evaluate_batch',side_effect=AssertionError('Cached likelihood must not rerun')):
                self.assertEqual(evaluator.score_study(self.root),result)
            batch = next((self.root/evaluator.EVALUATION/'bptt-s42').glob('batch-*.json'))
            data = read(batch); data['documents'][0]['bits_per_byte'] += .2
            batch.write_text(json.dumps(data),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'arithmetic changed'): evaluator.score_study(self.root)

    def test_last_selected_checkpoint_tampering_refuses_before_any_test_access(self):
        with self.evaluation_context():
            self.build_study(); selection=freeze_selection(self.root)
            (self.root/selection['conditions'][-1]['checkpoint']).write_bytes(b'changed')
            with patch.object(evaluator,'load_test') as loader:
                with self.assertRaises(ValueError): evaluator.score_study(self.root)
                loader.assert_not_called()
        self.assertFalse((self.root/evaluator.EVALUATION).exists())

    def test_corrupt_test_payload_refuses_before_scoring(self):
        with self.evaluation_context():
            self.build_study()
            path=self.root/evaluator.CACHE/'tokens.u16'; path.write_bytes(path.read_bytes()+b'\0\0')
            with patch('flm.babylm_test.evaluate_batch') as scorer:
                with self.assertRaisesRegex(ValueError,'Cache file changed'): evaluator.score_study(self.root)
                scorer.assert_not_called()
        self.assertFalse((self.root/evaluator.SUMMARY).exists())

    def test_changed_policy_or_evaluator_source_refuses_before_test_access(self):
        with self.evaluation_context():
            self.build_study()
            with patch.object(evaluator,'load_test') as loader:
                with patch.object(evaluator,'POLICY',dict(evaluator.POLICY,chunk_size=7)):
                    with self.assertRaisesRegex(ValueError,'identity changed'): evaluator.score_study(self.root)
                source=self.root/'flm/language_learning_test.py'; source.write_bytes(source.read_bytes()+b'\n# changed')
                with self.assertRaisesRegex(ValueError,'identity changed'): evaluator.score_study(self.root)
                loader.assert_not_called()

    def test_scoring_failure_preserves_rng_and_threads_and_resumes_saved_batches(self):
        with self.evaluation_context():
            self.build_study()
            from flm.corpus_evaluation import evaluate_batch
            calls=[]
            def fail_second(*args,**kwargs):
                calls.append(1)
                if len(calls)==2: raise RuntimeError('injected scoring interruption')
                return evaluate_batch(*args,**kwargs)
            rng=torch.get_rng_state().clone(); threads=torch.get_num_threads()
            with patch('flm.babylm_test.evaluate_batch',side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError,'injected'): evaluator.score_study(self.root)
            self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertEqual(threads,torch.get_num_threads())
            self.assertFalse((self.root/evaluator.SUMMARY).exists())
            self.assertEqual(len(list((self.root/evaluator.EVALUATION/'bptt-s42').glob('batch-*.json'))),1)
            with patch('flm.babylm_test.evaluate_batch',wraps=evaluate_batch) as scoring:
                evaluator.score_study(self.root)
                self.assertEqual(scoring.call_count,15)

    def test_complete_result_mutation_and_missing_summary_condition_are_rejected(self):
        with self.evaluation_context():
            self.build_study(); evaluator.score_study(self.root)
            path=self.root/evaluator.REPORTS/'test-bptt-s42.json'
            data=read(path); data['seconds']+=1; path.write_text(json.dumps(data),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'completed evaluation record changed'): evaluator.score_study(self.root)
        with self.assertRaisesRegex(ValueError,'every registered'): evaluator.summarize([],evaluator.POLICY)


if __name__=='__main__': unittest.main()
