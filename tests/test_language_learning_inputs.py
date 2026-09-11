import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flm.inference import state_hash
from flm.language_learning_inputs import (TrainingInputs,conditions,fingerprint,load_inputs,
    prepare_condition,verify_training_blocks)
from flm.local_learning import CORE_PARAMETERS


class TinyLexicon:
    sha256='a'*64
    vocabulary=4096
    pieces=[b'',b'',b'a','é'.encode('utf8'),b'\n']+[b'']*4091
    lengths=np.array([len(piece) for piece in pieces],dtype=np.int64)
    def decode(self,ids): return b''.join(self.pieces[int(i)] for i in ids).decode('utf8')


def blocks():
    raw={'first':'aé\n'.encode('utf8'),'second':b'aa\n'}
    sources={name:hashlib.sha256(data).hexdigest() for name,data in raw.items()}
    arrays=[np.array([0,2,3,4,1],dtype=np.int32),np.array([0,2,2,4,1],dtype=np.int32)]
    documents=[]; records=[]
    for (name,data),array in zip(raw.items(),arrays):
        identity=f'train-10m/{name}/{sources[name][:16]}/0-{len(data)}'
        documents.append((identity,array)); records.append(dict(id=identity,component=name,source_sha256=sources[name],
            byte_start=0,byte_end=len(data),utf8_bytes=len(data),text_sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(blocks=2,text_tokens=6,utf8_bytes=7,identity=dict(sources=sources),files={'fixture':'fixture'})
    return documents,records,manifest,TinyLexicon(),raw,sources


def graph(n=6,e=6,pools=3):
    row=np.arange(e,dtype=np.int32)%n; col=(np.arange(e,dtype=np.int32)//n+1)%n
    return dict(row=row,col=col,weight=(1/np.bincount(row,minlength=n)[row]).astype(np.float32),
        contacts=np.ones(e,dtype=np.uint32),source_sign=np.ones(n,dtype=np.int8),
        body_ids=np.arange(100,100+n,dtype=np.int64),source_indices=np.arange(n,dtype=np.int32),
        positions=np.zeros((n,3),dtype=np.float32),pool=(np.arange(n)*pools//n).astype(np.int32))


class LanguageLearningInputTests(unittest.TestCase):
    def test_full_utf8_coverage_and_token_counts(self):
        result=verify_training_blocks(*blocks())
        self.assertEqual((result['blocks'],result['text_tokens'],result['utf8_bytes']),(2,6,7))
        self.assertTrue(result['all_source_bytes_covered_once'])

    def test_source_gaps_duplicates_unknown_components_tokens_and_decoding_rejected(self):
        docs,records,manifest,lexicon,raw,sources=blocks()
        mutations=[('byte_start',1),('source_sha256','0'*64),('component','unknown'),('text_sha256','0'*64)]
        for key,value in mutations:
            changed=copy.deepcopy(records); changed[0][key]=value
            with self.assertRaises(ValueError): verify_training_blocks(docs,changed,manifest,lexicon,raw,sources)
        with self.assertRaisesRegex(ValueError,'inventory'):
            verify_training_blocks([docs[0],docs[0]],records,manifest,lexicon,raw,sources)
        for ids in ([0,2,0,4,1],[0,2,4096,4,1],[0,2,2,4,1],[2,2,3,4,1]):
            changed=copy.deepcopy(docs); changed[0]=(docs[0][0],np.array(ids,dtype=np.int32))
            with self.assertRaises(ValueError): verify_training_blocks(changed,records,manifest,lexicon,raw,sources)
        with self.assertRaisesRegex(ValueError,'source checksum'):
            verify_training_blocks(docs,records,manifest,lexicon,{**raw,'first':b'changed'},sources)
        with self.assertRaisesRegex(ValueError,'declared tokens'):
            verify_training_blocks(docs,records,{**manifest,'text_tokens':7},lexicon,raw,sources)

    def inputs(self):
        original=graph(); lexicon=TinyLexicon()
        return TrainingInputs(original,[],lexicon,dict(graph_arrays_sha256=fingerprint(original),tokenizer_sha256=lexicon.sha256))

    def test_each_seed_has_identical_initial_tensors_across_rules_and_isolated_graph_buffers(self):
        inputs=self.inputs(); original=fingerprint(inputs.graph); rng=torch.get_rng_state().clone()
        hashes={}; before_threads=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for condition in conditions():
                model,binding=prepare_condition(inputs,condition)
                hashes.setdefault(condition['seed'],set()).add(state_hash(model))
                self.assertEqual(binding['initial_state_sha256'],state_hash(model))
                for name,p in model.named_parameters():
                    self.assertEqual(p.requires_grad,condition['method'] != 'fixed_core' or name not in CORE_PARAMETERS)
                model.base_weight[0]=0
                self.assertEqual(fingerprint(inputs.graph),original)
        finally: torch.set_num_threads(before_threads)
        self.assertEqual({seed:len(values) for seed,values in hashes.items()},{42:1,43:1})
        self.assertNotEqual(hashes[42],hashes[43]); self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_invalid_condition_or_mutated_inputs_rejected(self):
        inputs=self.inputs()
        for condition in (dict(method='bptt',seed=44,label='bptt-s44'),dict(method='other',seed=42,label='other-s42')):
            with self.assertRaisesRegex(ValueError,'Unknown'): prepare_condition(inputs,condition)
        inputs.graph['weight'][0]*=2
        with self.assertRaisesRegex(ValueError,'identity changed'): prepare_condition(inputs,conditions()[0])

    def workspace(self,root):
        docs,records,manifest,lexicon,raw,sources=blocks()
        def write(name,value):
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
            return path
        selected=[]; acquired=[]
        revision='c92ab16b4f08858304b0815706065b3354d8fc0a'; repository='BabyLM-community/BabyLM-2026-Strict-Small'
        for name,data in raw.items():
            filename=name+'.train.txt'; relative='data/raw/babylm-2026/train-10m/'+filename
            write(relative,data)
            selected.append(dict(partition='train-10m',component=name,repository=repository,revision=revision,
                filename=filename,bytes=len(data),git_blob_sha1=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),
                lfs_sha256=None,url=f'https://huggingface.co/datasets/{repository}/resolve/{revision}/{filename}'))
            acquired.append(dict(partition='train-10m',component=name,path=relative,utf8_bytes=len(data),sha256=sources[name],source_verified=True))
        # Metadata may mention held-out partitions; their payloads do not exist.
        selected.append(dict(partition='test',component='missing-test'))
        source=write('data/sources/babylm-2026.json',dict(files=selected,licensing='Fixture only'))
        write('data/cards/babylm-2026-acquisition.json',dict(manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),files=acquired))
        tokenizer=write('data/tokenizers/babylm-2026-4096/tokenizer.json',b'fixture tokenizer')
        lexicon.sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        write('data/tokenizers/babylm-2026-4096/tokenizer-card.json',dict(tokenizer_sha256=lexicon.sha256,
            training_sources=sources,fitted_partition='train-10m',validation_or_test_used_for_training=False))
        cache=write('data/processed/babylm-2026-bpe/train-10m/manifest.json',manifest)
        path=root/'data/graphs/central-1024/graph.npz'; path.parent.mkdir(parents=True)
        np.savez_compressed(path,**graph(1024,76130,128)); graph_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        write('data/graphs/central-1024/graph-card.json',dict(graph_sha256=graph_hash))
        return docs,records,manifest,lexicon,hashlib.sha256(cache.read_bytes()).hexdigest(),graph_hash

    def test_loader_verifies_only_training_payloads_and_records_source_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); docs,records,manifest,lexicon,cache_hash,graph_hash=self.workspace(root)
            with patch('flm.language_learning_inputs.COMPONENTS',('first','second')), \
                    patch('flm.language_learning_inputs.TOKENIZER_SHA256',lexicon.sha256), \
                    patch('flm.language_learning_inputs.CACHE_SHA256',cache_hash), \
                    patch('flm.language_learning_inputs.GRAPH_SHA256',graph_hash), \
                    patch('flm.language_learning_inputs.Lexicon',return_value=lexicon), \
                    patch('flm.language_learning_inputs.read_mmap',return_value=(docs,records,manifest)) as read:
                loaded=load_inputs(root)
                read.assert_called_once_with(root/'data/processed/babylm-2026-bpe/train-10m',lexicon.sha256)
                self.assertEqual(loaded.binding['coverage']['utf8_bytes'],7)
                self.assertFalse(loaded.binding['validation_or_test_payloads_opened'])
                self.assertFalse(loaded.binding['trained_weights_loaded'])
                self.assertEqual(len(loaded.graph['body_ids']),1024)

    def test_changed_raw_bytes_and_cache_rejected_before_token_payload_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); docs,records,manifest,lexicon,cache_hash,graph_hash=self.workspace(root)
            raw=root/'data/raw/babylm-2026/train-10m/first.train.txt'; original=raw.read_bytes()
            with patch('flm.language_learning_inputs.COMPONENTS',('first','second')), \
                    patch('flm.language_learning_inputs.TOKENIZER_SHA256',lexicon.sha256), \
                    patch('flm.language_learning_inputs.CACHE_SHA256',cache_hash), \
                    patch('flm.language_learning_inputs.read_mmap') as read:
                raw.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'raw training bytes'): load_inputs(root)
                raw.write_bytes(original)
                (root/'data/processed/babylm-2026-bpe/train-10m/manifest.json').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'training cache identity'): load_inputs(root)
                read.assert_not_called()


if __name__=='__main__': unittest.main()
