"""Artificial graph archives and text windows; no official selected-graph fits."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from flm.language_learning_train import Settings
from flm.model import FLM
from flm.selection_language import (MODEL_SHA256, catalog_from_bytes, digest, fit_case,
    load_case_graph, parameter_hash, prepare_case)
from flm.train import Sampler
from test_selection_pilot import fixture

SETTINGS=Settings(steps=4,batch=2,sequence=5,warmup=1,learning_rate=.002,final_learning_rate=.002,
    lr_warmup_updates=0,weight_decay=.01,gradient_clip=1.,checkpoint_interval=2,seed=42,threads=1,score_boundaries=True)


def archive_fixture():
    base,config,documents,lengths=fixture(); base['positions'][0]=np.nan; files={'README.txt':b'Synthetic fixture only'}
    entries={}; rewires={}
    for selector,multiplicity in (('candidate',1),('contact_ranked',2)):
        name='fixture-L-t5/'+selector
        graph={k:v.copy() for k,v in base.items()}
        graph['row']=np.repeat(np.arange(6,dtype=np.int32),multiplicity)
        graph['col']=(graph['row']+np.tile(np.arange(1,multiplicity+1),6))%6
        graph['weight']=np.full(6*multiplicity,1/multiplicity,dtype=np.float32)
        graph['contacts']=np.ones(6*multiplicity,dtype=np.uint32)
        def encode(g):
            buffer=io.BytesIO(); np.savez_compressed(buffer,**g); return buffer.getvalue()
        data=encode(graph); path=f'graphs/{name}.npz'; files['original/'+path]=data
        with torch.random.fork_rng(devices=[]):
            model=FLM(graph,config)
        groups={key:p.numel() for key,p in model.named_parameters()}
        entries[name]=dict(selection=name,path=path,graph_sha256=digest(data),config=vars(config),edges=len(graph['row']),
                           trainable_parameters=sum(groups.values()),parameter_groups=groups)
        for offset,seed in enumerate((101,103,107),1):
            control={k:v.copy() for k,v in graph.items()}; control['col']=(graph['col']+offset)%6
            data=encode(control); label=name+f'/null{seed}'; files[f'rewired/{label}.npz']=data
            rewires[label]=dict(status='complete',binding=dict(source_graph_sha256=entries[name]['graph_sha256'],seed=seed),graph_sha256=digest(data))
    source=json.dumps(dict(graphs=entries,model_source_sha256=MODEL_SHA256)).encode()
    files['source-manifest.json']=source
    files['rewiring-manifest.json']=json.dumps(dict(source_manifest_sha256=digest(source),complete=6,failed=0,planned=6,
        graph_seeds=[101,103,107],records=rewires)).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        for name,data in files.items(): archive.writestr(name,data)
    lexicon=SimpleNamespace(vocabulary=config.vocabulary,lengths=lengths,sha256='a'*64)
    corpus=SimpleNamespace(documents=documents,lexicon=lexicon,binding=dict(tokenizer_sha256=lexicon.sha256,fixture=True))
    return buffer.getvalue(),corpus


class SelectionLanguageTests(unittest.TestCase):
    def setUp(self):
        threads=torch.get_num_threads(); torch.set_num_threads(1); self.addCleanup(torch.set_num_threads,threads)
        self.data,self.corpus=archive_fixture()
        self.catalog=catalog_from_bytes(self.data,digest(self.data),original_count=2)
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); self.root=Path(temporary.name)

    def test_all_graphs_and_seeds_preserve_initial_parameter_matching_and_rng(self):
        rng=torch.get_rng_state().clone(); learned={}; nonedge={}
        # Acquired missing anatomical coordinates are retained; they are not model inputs.
        self.assertTrue(np.isnan(load_case_graph(self.catalog,next(iter(self.catalog.entries)))[0]['positions'][0]).all())
        for label,entry in self.catalog.entries.items():
            for seed in (42,43):
                model,binding=prepare_case(self.catalog,self.corpus,label,seed)
                learned.setdefault((entry['selection'],seed),set()).add(binding['initial_parameters_sha256'])
                nonedge.setdefault(seed,set()).add(binding['initial_nonedge_parameters_sha256'])
                model.base_weight[0]=99
                self.assertNotEqual(load_case_graph(self.catalog,label)[0]['weight'][0],99)
        self.assertTrue(all(len(v)==1 for v in learned.values()))
        self.assertEqual({seed:len(v) for seed,v in nonedge.items()},{42:1,43:1})
        self.assertNotEqual(nonedge[42],nonedge[43]); self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_archive_checksum_missing_control_and_changed_graph_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'checksum'): catalog_from_bytes(self.data+b'changed',digest(self.data),2)
        with zipfile.ZipFile(io.BytesIO(self.data)) as archive: files={name:archive.read(name) for name in archive.namelist()}
        for kind in ('missing','payload','extra'):
            changed=copy.deepcopy(files)
            if kind=='missing':
                record=json.loads(changed['rewiring-manifest.json']); record['records'].pop(next(iter(record['records'])))
                changed['rewiring-manifest.json']=json.dumps(record).encode()
            elif kind=='payload': changed['original/graphs/fixture-L-t5/candidate.npz']=b'changed'
            else: changed['unregistered.npz']=b'extra'
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w') as archive:
                for name,data in changed.items(): archive.writestr(name,data)
            data=buffer.getvalue()
            with self.assertRaises(ValueError): catalog_from_bytes(data,digest(data),2)

    def test_unknown_seed_graph_tokenizer_and_old_graph_binding_are_rejected(self):
        label=next(iter(self.catalog.entries))
        for seed in (44,True):
            with self.assertRaises(ValueError): prepare_case(self.catalog,self.corpus,label,seed)
        with self.assertRaisesRegex(ValueError,'Unknown'): prepare_case(self.catalog,self.corpus,'missing',42)
        changed=copy.deepcopy(self.corpus); changed.binding['tokenizer_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'tokenizer'): prepare_case(self.catalog,changed,label,42)
        changed=copy.deepcopy(self.corpus); changed.binding['graph_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'graph-independent'): prepare_case(self.catalog,changed,label,42)

    def test_fit_and_resume_equal_and_changed_graph_cannot_resume(self):
        label='fixture-L-t5/candidate/null101'; study=dict(fixture='not an official study')
        full=self.root/'full'; resumed=self.root/'resumed'
        a=fit_case(self.catalog,self.corpus,label,SETTINGS,full,study)
        partial=fit_case(self.catalog,self.corpus,label,SETTINGS,resumed,study,until=2)
        self.assertFalse(partial['complete'])
        b=fit_case(self.catalog,self.corpus,label,SETTINGS,resumed,study)
        # Torch checkpoint container bytes need not be canonical across saves.
        self.assertEqual({k:v for k,v in a.items() if k!='checkpoint_sha256'},
                         {k:v for k,v in b.items() if k!='checkpoint_sha256'})
        pa=torch.load(full/'checkpoint-000004.pt',weights_only=True)
        pb=torch.load(resumed/'checkpoint-000004.pt',weights_only=True)
        for name,tensor in pa['model'].items(): self.assertTrue(torch.equal(tensor,pb['model'][name]))
        self.assertEqual(pa['history'],pb['history']); self.assertEqual(pa['sampled_token_sha256'],pb['sampled_token_sha256'])
        with self.assertRaisesRegex(ValueError,'declaration changed'):
            fit_case(self.catalog,self.corpus,'fixture-L-t5/candidate/measured',SETTINGS,resumed,study)
        with self.assertRaisesRegex(ValueError,'declaration changed'):
            fit_case(self.catalog,self.corpus,label,SETTINGS,resumed,dict(fixture='different'))

    def test_update_matches_pilot_boundary_included_objective(self):
        label='fixture-L-t5/contact_ranked/measured'
        self.corpus.documents=[(name,np.tile(np.array([0,2,1,3],dtype=np.int32),12))
                               for name,_ in self.corpus.documents]
        model,_=prepare_case(self.catalog,self.corpus,label,42)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        sampler=Sampler(self.corpus.documents,42,SETTINGS.sequence)
        boundary_seen=False
        for _ in range(4):
            x,y=sampler.sample(SETTINGS.batch,'cpu'); boundary_seen |= bool((y[:,1:]<2).any())
            optimizer.zero_grad(set_to_none=True); logits,_=model(x)
            loss=F.cross_entropy(logits[:,1:].reshape(-1,self.corpus.lexicon.vocabulary),y[:,1:].reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        self.assertTrue(boundary_seen)
        folder=self.root/'fit'; fit_case(self.catalog,self.corpus,label,SETTINGS,folder,dict(fixture=True))
        saved=torch.load(folder/'checkpoint-000004.pt',weights_only=True)
        for name,tensor in model.state_dict().items():
            torch.testing.assert_close(tensor,saved['model'][name],rtol=1e-6,atol=1e-7)
        with self.assertRaisesRegex(ValueError,'boundary targets'):
            fit_case(self.catalog,self.corpus,label,replace(SETTINGS,score_boundaries=False),self.root/'invalid',dict(fixture=True))
        self.assertFalse((self.root/'invalid').exists())


if __name__=='__main__': unittest.main()
