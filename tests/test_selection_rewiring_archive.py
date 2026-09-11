import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from scripts.audit_selection_rewiring import audit_archive,audit_pair,check_receipt
from scripts.package_selection_rewiring import inventory_gate,package
from scripts.preflight_selection_rewiring import preflight


class RewiringArchiveTests(unittest.TestCase):
    def graphs(self):
        n=6; row=np.arange(n,dtype=np.int32); col=(row+1)%n
        signs=np.array([1,-1,1,-1,1,-1],dtype=np.int8)
        original=dict(row=row,col=col,weight=signs[col].astype(np.float32),contacts=np.ones(n,dtype=np.uint32),
            source_sign=signs,body_ids=np.arange(100,106,dtype=np.int64),source_indices=row.copy(),
            positions=np.zeros((n,3),dtype=np.float32),pool=row//2)
        changed={name:value.copy() for name,value in original.items()}; changed['col']=(row+3)%n
        return original,changed

    def encoded(self,graph):
        handle=io.BytesIO(); np.savez_compressed(handle,**graph); return handle.getvalue()

    def fixture(self):
        original,changed=self.graphs(); data=self.encoded(original); null_data=self.encoded(changed)
        digest=lambda data:hashlib.sha256(data).hexdigest()
        source=dict(graphs={'fixture/candidate':dict(path='graphs/fixture/candidate.npz',graph_sha256=digest(data))})
        source_bytes=json.dumps(source).encode()
        records={}
        for seed in (101,103,107):
            receipt=dict(binding=dict(source_graph_sha256=digest(data),seed=seed,swaps_per_edge=10,max_proposals_per_edge=100,
                generator_source_sha256='a'*64,runner_source_sha256='b'*64),status='complete',graph_sha256=digest(null_data),
                report=dict(seed=seed,accepted_swaps=60,proposed_swaps=100,acceptance_fraction=.6,**audit_pair(original,changed)))
            if seed==103: receipt=dict(binding=receipt['binding'],status='failed',reason='Explicit fixture failure')
            records[f'fixture/candidate/null{seed}']=receipt
        manifest=dict(source_manifest_sha256=digest(source_bytes),generator_source_sha256='a'*64,runner_source_sha256='b'*64,
            planned=3,complete=2,failed=1,records=records)
        files={'source-manifest.json':source_bytes,'README.txt':b'Unit fixture, not generated study controls',
            'original/graphs/fixture/candidate.npz':data,
            'rewired/fixture/candidate/null101.npz':null_data,'rewired/fixture/candidate/null107.npz':null_data}
        return source,manifest,files

    def write_archive(self,path,manifest,files):
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('rewiring-manifest.json',json.dumps(manifest))
            for name,data in files.items(): archive.writestr(name,data)

    def test_independent_topology_has_known_cycles_signs_and_overlap(self):
        original,changed=self.graphs(); result=audit_pair(original,changed)
        self.assertEqual(result['original']['strong_component_sizes'],[6])
        self.assertEqual(result['randomized']['strong_component_sizes'],[2,2,2])
        self.assertEqual(result['original']['reciprocal_off_diagonal_fraction'],0)
        self.assertEqual(result['randomized']['reciprocal_off_diagonal_fraction'],1)
        self.assertEqual(result['original_edge_overlap_fraction'],0)
        self.assertEqual(result['unchanged_endpoint_slots_fraction'],0)
        self.assertTrue(all(result['preserved'].values()))

    def test_changed_fixed_arrays_degree_sign_self_edges_and_topology_rejected(self):
        original,changed=self.graphs()
        for name in ('weight','contacts','positions','body_ids','source_indices','pool','row'):
            damaged={k:v.copy() for k,v in changed.items()}; damaged[name].flat[0]+=1
            with self.assertRaisesRegex(ValueError,'Fixed array'): audit_pair(original,damaged)
        damaged=copy.deepcopy(changed); damaged['col'][0]=4
        with self.assertRaisesRegex(ValueError,'Outgoing degree'): audit_pair(original,damaged)
        damaged=copy.deepcopy(changed); damaged['col'][[0,1]]=damaged['col'][[1,0]]
        with self.assertRaisesRegex(ValueError,'Source sign'): audit_pair(original,damaged)
        damaged=copy.deepcopy(changed); damaged['col'][[0,3]]=damaged['col'][[3,0]]
        with self.assertRaisesRegex(ValueError,'Self edge'): audit_pair(original,damaged)
        with self.assertRaisesRegex(ValueError,'Topology did not change'): audit_pair(original,original)

    def test_full_archive_and_retained_failure_verify_with_no_generator(self):
        source,manifest,files=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.zip'; self.write_archive(path,manifest,files)
            result=audit_archive(path,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual((result['original_graphs'],result['planned'],result['complete'],result['failed']),(1,3,2,1))
            self.assertEqual(result['failures_not_reexecuted'],1)
            with self.assertRaisesRegex(ValueError,'checksum'): audit_archive(path,'0'*64)

    def test_duplicate_body_pair_rejected_before_degree_accounting(self):
        original,changed=self.graphs()
        original['row']=np.repeat(np.arange(4,dtype=np.int32),2)
        original['col']=np.array([1,2,2,3,3,0,0,1],dtype=np.int32)
        original['weight']=np.full(8,.5,dtype=np.float32)
        original['contacts']=np.ones(8,dtype=np.uint32)
        original['source_sign']=np.ones(6,dtype=np.int8)
        changed=copy.deepcopy(original)
        changed['col']=np.array([3,3,3,0,0,1,1,2],dtype=np.int32)
        with self.assertRaisesRegex(ValueError,'Duplicate edge'): audit_pair(original,changed)

    def test_tampered_diagnostics_and_payload_with_updated_checksum_rejected(self):
        source,manifest,files=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.zip'
            damaged=copy.deepcopy(manifest)
            damaged['records']['fixture/candidate/null101']['report']['randomized']['strong_components']=1
            self.write_archive(path,damaged,files)
            with self.assertRaisesRegex(ValueError,'structural diagnostic'): audit_archive(path)
            original,changed=self.graphs(); changed['weight'][0]*=.5; payload=self.encoded(changed)
            damaged=copy.deepcopy(manifest); altered=dict(files)
            damaged['records']['fixture/candidate/null101']['graph_sha256']=hashlib.sha256(payload).hexdigest()
            altered['rewired/fixture/candidate/null101.npz']=payload
            self.write_archive(path,damaged,altered)
            with self.assertRaisesRegex(ValueError,'Fixed array'): audit_archive(path)

    def test_whole_inventory_gate_and_extra_or_missing_archive_cases(self):
        source,manifest,files=self.fixture()
        progress=dict(status='complete',planned_graphs=3,complete=2,failed=1,records=manifest['records'])
        self.assertEqual(len(inventory_gate(source,progress)),3)
        for field,value in [('status','partial'),('planned_graphs',2),('complete',3),('failed',0)]:
            bad=copy.deepcopy(progress); bad[field]=value
            with self.assertRaises(ValueError): inventory_gate(source,bad)
        bad=copy.deepcopy(progress); bad['records'].pop('fixture/candidate/null103')
        with self.assertRaises(ValueError): inventory_gate(source,bad)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.zip'
            damaged=copy.deepcopy(manifest); damaged['records'].pop('fixture/candidate/null103')
            self.write_archive(path,damaged,files)
            with self.assertRaisesRegex(ValueError,'inventory'): audit_archive(path)
            self.write_archive(path,manifest,{**files,'unexpected.txt':b'extra'})
            with self.assertRaisesRegex(ValueError,'Unexpected'): audit_archive(path)

    def test_packager_binds_files_audits_output_and_preserves_release_on_failure(self):
        source,manifest,files=self.fixture()
        root=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            workspace=Path(folder)
            def write(name,data):
                target=workspace/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
            def encoded(value): return json.dumps(value).encode()
            for name in ('flm/selection_rewiring.py','flm/wiring_controls.py','scripts/audit_selection_rewiring.py',
                         'scripts/package_selection_rewiring.py','scripts/preflight_selection_rewiring.py'):
                write(name,(root/name).read_bytes())
            generator=hashlib.sha256((workspace/'flm/wiring_controls.py').read_bytes()).hexdigest()
            runner=hashlib.sha256((workspace/'flm/selection_rewiring.py').read_bytes()).hexdigest()
            source_bytes=encoded(source); source_hash=hashlib.sha256(source_bytes).hexdigest()
            write('graphs/manifest.json',source_bytes)
            write('graphs/graphs/fixture/candidate.npz',files['original/graphs/fixture/candidate.npz'])
            write('reports/selection-graphs/export-release.json',encoded(dict(manifest_sha256=source_hash,graphs=1)))
            for name,receipt in manifest['records'].items():
                receipt['binding'].update(generator_source_sha256=generator,runner_source_sha256=runner)
                write('rewiring/'+name+'/receipt.json',encoded(receipt))
                if receipt['status']=='complete': write('rewiring/'+name+'/graph.npz',files['rewired/'+name+'.npz'])
            progress=dict(verified_utc='2026-09-11T00:00:00Z',status='complete',planned_graphs=3,complete=2,failed=1,records=manifest['records'],
                graph_manifest_sha256=source_hash,generator_source_sha256=generator,runner_source_sha256=runner)
            write('rewiring/progress.json',encoded(progress))
            try:
                os.chdir(workspace)
                archive=workspace/'published/control.zip'
                result=package(Path('graphs'),Path('rewiring'),archive,Path('reports/final'))
                self.assertEqual((result['planned'],result['complete'],result['failed']),(3,2,1))
                self.assertEqual(audit_archive(archive,result['archive_sha256'])['complete'],2)
                self.assertEqual(json.loads(Path('reports/final/release.json').read_text()),result)
                checked=preflight(Path('graphs'),Path('rewiring'))
                self.assertEqual((checked['complete'],checked['failed']),(2,1))
                self.assertTrue(checked['publication_gate']['final_archive_allowed'])
                partial=copy.deepcopy(progress); partial['records'].pop('fixture/candidate/null107')
                partial.update(status='partial',complete=1)
                write('rewiring/progress.json',encoded(partial))
                checked=preflight(Path('graphs'),Path('rewiring'))
                self.assertEqual((checked['complete'],checked['failed']),(1,1))
                self.assertFalse(checked['publication_gate']['final_archive_allowed'])
                write('rewiring/progress.json',encoded(progress))
                released=archive.read_bytes()
                write('rewiring/fixture/candidate/null101/graph.npz',b'tampered')
                with self.assertRaisesRegex(ValueError,'Rewired graph identity'): package(Path('graphs'),Path('rewiring'),archive,Path('reports/final'))
                self.assertEqual(archive.read_bytes(),released)
            finally:
                os.chdir(root)


if __name__ == '__main__': unittest.main()
