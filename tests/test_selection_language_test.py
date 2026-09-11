"""Synthetic selection evaluation, including complete-group fit and likelihood oracles."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import flm.selection_language_test as evaluator
from flm.corpus_evaluation import component_summary, evaluate_batch
from flm.provenance import sha256
from flm.selection_language_study import IDENTITY, SELECTION, STUDY, initialize, train_study, read
import test_selection_language_study as fixture


class SelectionEvaluationTests(unittest.TestCase):
    setUp = fixture.SelectionStudyTests.setUp
    write = fixture.SelectionStudyTests.write
    patched = fixture.SelectionStudyTests.patched

    def context(self):
        stack=self.patched()
        stack.enter_context(patch.object(evaluator,'POLICY',dict(evaluator.POLICY,batch_size=2,chunk_size=3,threads=1,bootstrap_replicates=32)))
        return stack

    def cache(self):
        documents=[(f'test/{component}/{i}',np.array([0,2,3,4,5,6,7,8,1],dtype=np.int32))
            for i,component in enumerate(('one','one','two','two'))]
        cache=Path('data/processed/babylm-2026-bpe/test'); destination=self.root/cache
        offsets=np.array([0]+list(np.cumsum([len(doc) for _,doc in documents])),dtype=np.int64)
        np.save(destination/'offsets.npy',offsets)
        np.concatenate([doc for _,doc in documents]).astype('<u2').tofile(destination/'tokens.u16')
        metadata=[dict(id=name,token_start=int(offsets[i]),token_end=int(offsets[i+1]),component=name.split('/')[1],
            utf8_bytes=int(self.corpus.lexicon.lengths[doc].sum()),overlap_filtered_eligible=i<2)
            for i,(name,doc) in enumerate(documents)]
        (destination/'blocks.jsonl').write_text('\n'.join(json.dumps(r) for r in metadata)+'\n',encoding='utf8')
        manifest=dict(format='flm-mmap-tokens-v1',dtype='<u2',identity=dict(tokenizer_sha256=self.corpus.lexicon.sha256),
            blocks=4,text_tokens=28,utf8_bytes=sum(r['utf8_bytes'] for r in metadata),
            components={name:dict(blocks=2,text_tokens=14,utf8_bytes=sum(r['utf8_bytes'] for r in metadata if r['component']==name))
                for name in ('one','two')},files={name:sha256(destination/name) for name in ('tokens.u16','offsets.npy','blocks.jsonl')})
        self.write(cache/'manifest.json',manifest)
        self.write(Path('data/tokenizers/babylm-2026-4096/tokenization-card.json'),
            dict(tokenizer_sha256=self.corpus.lexicon.sha256,partitions=dict(test=manifest)))
        return documents

    def test_comparison_inventory_keeps_family_and_every_individual_contrast(self):
        rows=evaluator.comparison_plan(fixture.GROUP)
        self.assertEqual(len(rows),42)
        self.assertEqual(sum(r['level']=='family' for r in rows),11)
        self.assertEqual(sum(r['level']=='individual' for r in rows),31)
        self.assertEqual(sum(r['kind']=='topology' for r in rows),32)
        self.assertEqual(len({(r['level'],r['name']) for r in rows}),42)

    def test_family_averages_losses_without_averaging_denominators_or_probabilities(self):
        first=[dict(document='one',component='a',tokens=2,bytes=3,nll=2.,bits_per_byte=2/3/math.log(2),overlap_filtered_eligible=True)]
        second=[dict(first[0],nll=8.,bits_per_byte=8/3/math.log(2))]
        before=copy.deepcopy(first); result=evaluator.average_blocks([first,second])
        self.assertEqual(result[0]['nll'],5.); self.assertEqual(result[0]['bytes'],3); self.assertEqual(first,before)
        probability_ensemble=-math.log((math.exp(-2)+math.exp(-8))/2)
        self.assertNotAlmostEqual(result[0]['nll'],probability_ensemble)
        for key,value in (('bytes',4),('overlap_filtered_eligible',False),('document','other')):
            bad=[dict(second[0],**{key:value})]
            with self.assertRaisesRegex(ValueError,'unmatched'): evaluator.average_blocks([first,bad])

    def test_missing_and_incomplete_study_never_opens_test_payloads(self):
        with patch.object(evaluator,'load_test') as loader:
            with self.assertRaisesRegex(ValueError,'No frozen'): evaluator.score_study(self.root)
            with self.context():
                initialize(self.root,self.request)
                with self.assertRaisesRegex(ValueError,'incomplete'): evaluator.score_study(self.root)
            loader.assert_not_called()
        self.assertFalse((self.root/evaluator.SUMMARY).exists())

    def test_synthetic_summary_requires_all_runs_and_uses_paired_training_seeds(self):
        with self.context():
            identity=initialize(self.root,self.request)
            results=[]
            for row in identity['conditions']:
                condition=row['condition']; graph=condition['graph']
                # Known fixed deltas distinguish the three random graphs and two fitted seeds.
                offset=(condition['seed']-42)*.2
                if '/uniform_s' in graph: offset+=int(graph.split('/')[1][-3:])-200
                if '/null' in graph: offset+=.5
                blocks=[dict(document=f'test/a/{i}',component='a',tokens=3,bytes=4,nll=3.+offset,
                    bits_per_byte=(3.+offset)/4/math.log(2),overlap_filtered_eligible=False) for i in (0,1)]
                results.append(dict(condition=condition,blocks=blocks,components=component_summary(blocks)))
            summary=evaluator.summarize(results,identity,evaluator.POLICY)
            self.assertEqual(len(summary['paired_comparisons']),168)
            self.assertEqual(len(summary['paired_seed_summary']),84)
            row=next(r for r in summary['paired_seed_summary'] if r['name']=='candidate-minus-uniform' and r['level']=='family' and r['subset']=='official')
            expected=-(1+3+7)/3/4/math.log(2)
            self.assertAlmostEqual(row['mean_difference_bpb'],expected)
            self.assertAlmostEqual(row['seed_standard_deviation'],0.,places=12)
            self.assertTrue(all(r['estimate'] is None for r in summary['paired_comparisons'] if r['subset']=='overlap_filtered'))
            with self.assertRaisesRegex(ValueError,'every registered'): evaluator.summarize(results[:-1],identity,evaluator.POLICY)
            bad=copy.deepcopy(results); bad[-1]['blocks'][0]['overlap_filtered_eligible']=True
            with self.assertRaisesRegex(ValueError,'unmatched'): evaluator.summarize(bad,identity,evaluator.POLICY)

    def test_full_group_resumable_scoring_matches_whole_block_oracle_and_rejects_mutations(self):
        with self.context():
            documents=self.cache(); initialize(self.root,self.request); train_study(self.root)
            calls=[]
            def fail_second(*args,**kwargs):
                calls.append(1)
                if len(calls)==2: raise RuntimeError('fixture scoring interruption')
                return evaluate_batch(*args,**kwargs)
            rng=torch.get_rng_state().clone(); threads=torch.get_num_threads()
            with patch('flm.babylm_test.evaluate_batch',side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError,'interruption'): evaluator.score_study(self.root)
            self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertEqual(threads,torch.get_num_threads())
            self.assertFalse((self.root/evaluator.SUMMARY).exists())
            with patch('flm.babylm_test.evaluate_batch',wraps=evaluate_batch) as scorer:
                result=evaluator.score_study(self.root); self.assertEqual(scorer.call_count,127)
            self.assertEqual(len(result['runs']),64); self.assertEqual(len(result['paired_comparisons']),168)
            identity=read(self.root/IDENTITY); selected=read(self.root/SELECTION)
            registered={r['condition']['label']:r for r in identity['conditions']}
            for row in selected['conditions']:
                label=row['condition']['label']
                model=evaluator.restore_selected(self.root,self.catalog,self.corpus,registered[label],row,sha256(self.root/IDENTITY))
                records=read(self.root/evaluator.REPORTS/'test'/label/'result.json')['blocks']
                for (_,tokens),record in zip(documents,records):
                    with torch.no_grad():
                        logits,_=model(torch.tensor(tokens[:-1]).unsqueeze(0),None)
                        targets=torch.tensor(tokens[1:],dtype=torch.int64); keep=targets>=2
                        oracle=float(F.cross_entropy(logits[0][keep],targets[keep],reduction='none').double().sum())
                    self.assertAlmostEqual(oracle,record['nll'],places=10)
            with patch('flm.babylm_test.evaluate_batch',side_effect=AssertionError('No cached likelihood recomputation')):
                self.assertEqual(evaluator.score_study(self.root),result)
            # A changed policy or source and the very last checkpoint fail before test access.
            with patch.object(evaluator,'load_test') as loader:
                with patch.object(evaluator,'POLICY',dict(evaluator.POLICY,chunk_size=7)):
                    with self.assertRaisesRegex(ValueError,'identity changed'): evaluator.score_study(self.root)
                source=self.root/'flm/selection_language_test.py'; original=source.read_bytes(); source.write_bytes(original+b'\nchanged')
                with self.assertRaisesRegex(ValueError,'identity changed'): evaluator.score_study(self.root)
                source.write_bytes(original)
                checkpoint=self.root/selected['conditions'][-1]['checkpoint']; original=checkpoint.read_bytes(); checkpoint.write_bytes(b'changed')
                with self.assertRaises(ValueError): evaluator.score_study(self.root)
                checkpoint.write_bytes(original); loader.assert_not_called()
            payload=self.root/'data/processed/babylm-2026-bpe/test/tokens.u16'; original=payload.read_bytes(); payload.write_bytes(original+b'\0\0')
            with patch('flm.babylm_test.evaluate_batch') as scorer:
                with self.assertRaisesRegex(ValueError,'Cache file changed'): evaluator.score_study(self.root)
                scorer.assert_not_called()
            payload.write_bytes(original)
            label=selected['conditions'][0]['condition']['label']; batch=self.root/evaluator.EVALUATION/label/'batch-00000.json'
            original=batch.read_bytes(); bad=read(batch); bad['documents'][0]['bits_per_byte']+=.1; self.write(batch,bad)
            with self.assertRaisesRegex(ValueError,'arithmetic changed'): evaluator.score_study(self.root)
            batch.write_bytes(original)
            report=self.root/evaluator.REPORTS/'test'/label/'result.json'; bad=read(report); bad['seconds']+=1; self.write(report,bad)
            with self.assertRaisesRegex(ValueError,'completed evaluation record changed'): evaluator.score_study(self.root)


if __name__=='__main__': unittest.main()
