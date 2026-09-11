"""Bind released selected graphs to fresh BPTT language-training conditions.

This is a reusable input/fit adapter, not an official study matrix or budget.
There is deliberately no command that starts an unfrozen corpus experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .inference import state_hash
from .language_learning_inputs import fingerprint, load_corpus
from .language_learning_train import configure, fit
from .model import Config, FLM
from .provenance import sha256
from .wiring_controls import validate_control

ARCHIVE = Path('public/research/selection-rewiring.zip')
ARCHIVE_SHA256 = '1b849f068ab1c0bd7e8331e2aa7ddb573ed6173f79e739727b86ac7d176aebd3'
MODEL_SHA256 = '2ee5f175bd32474e2f23683c9b6df421ea483a4ee45e9c2ade41e975985d3055'
GRAPH_SEEDS = (101,103,107)
TRAINING_SEEDS = (42,43)


def digest(data): return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Catalog:
    archive: bytes
    index_json: str
    binding_json: str

    @property
    def entries(self): return json.loads(self.index_json)

    @property
    def binding(self): return json.loads(self.binding_json)


def catalog_from_bytes(data, expected_sha256, original_count=64):
    if digest(data) != expected_sha256: raise ValueError('Selection archive checksum changed')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=archive.namelist()
        if len(names) != len(set(names)): raise ValueError('Duplicate archive member')
        source_bytes=archive.read('source-manifest.json'); rewiring_bytes=archive.read('rewiring-manifest.json')
        source=json.loads(source_bytes); rewiring=json.loads(rewiring_bytes)
        if (len(source['graphs']) != original_count or rewiring['source_manifest_sha256'] != digest(source_bytes)
                or rewiring['complete'] != original_count*3 or rewiring['failed'] != 0
                or rewiring['planned'] != original_count*3 or rewiring['graph_seeds'] != list(GRAPH_SEEDS)):
            raise ValueError('Incomplete structural catalog')
        expected_nulls={name+f'/null{seed}' for name in source['graphs'] for seed in GRAPH_SEEDS}
        if set(rewiring['records']) != expected_nulls: raise ValueError('Missing or extra rewired graph')
        entries={}; expected_names={'README.txt','source-manifest.json','rewiring-manifest.json'}
        for name, original in sorted(source['graphs'].items()):
            parts=name.split('/')
            if len(parts)!=2 or any(not p or not all(c.isalnum() or c in '-_' for c in p) for p in parts):
                raise ValueError('Invalid selection name')
            if original['selection'] != name or original['path'] != f'graphs/{name}.npz':
                raise ValueError('Original graph path or identity changed')
            base=dict(selection=name, candidate=parts[0], selector=parts[1], config=original['config'],
                original_graph_sha256=original['graph_sha256'], edges=original['edges'],
                trainable_parameters=original['trainable_parameters'], parameter_groups=original['parameter_groups'])
            original_path='original/'+original['path']
            for graph_seed in (None,*GRAPH_SEEDS):
                suffix='measured' if graph_seed is None else f'null{graph_seed}'
                label=name+'/'+suffix
                path=original_path if graph_seed is None else f'rewired/{name}/{suffix}.npz'
                if graph_seed is None: graph_hash=original['graph_sha256']
                else:
                    row=rewiring['records'][label]
                    if (row['status'] != 'complete' or row['binding']['source_graph_sha256'] != original['graph_sha256']
                            or row['binding']['seed'] != graph_seed):
                        raise ValueError('Rewired graph binding changed')
                    graph_hash=row['graph_sha256']
                if digest(archive.read(path)) != graph_hash: raise ValueError('Graph payload checksum changed')
                expected_names.add(path)
                entries[label]=dict(base, graph_seed=graph_seed, label=label, path=path,
                    original_path=original_path, graph_sha256=graph_hash,
                    kind='measured induced subset' if graph_seed is None else 'artificial degree/sign-preserving rewire')
        if set(names) != expected_names: raise ValueError('Unexpected archive inventory')
    return Catalog(data,json.dumps(entries,sort_keys=True),json.dumps(dict(archive_sha256=expected_sha256,
        source_manifest_sha256=digest(source_bytes),rewiring_manifest_sha256=digest(rewiring_bytes),
        original_graphs=original_count,rewired_graphs=original_count*3,
        model_source_sha256=source['model_source_sha256']),sort_keys=True))


def load_catalog(root):
    root=Path(root)
    if sha256(root/'flm/model.py') != MODEL_SHA256: raise ValueError('Released model implementation changed')
    catalog=catalog_from_bytes((root/ARCHIVE).read_bytes(),ARCHIVE_SHA256)
    if catalog.binding['model_source_sha256'] != MODEL_SHA256: raise ValueError('Graph/model release mismatch')
    for row in catalog.entries.values():
        config=Config(**row['config'])
        if (config.variant!='flm' or config.backend!='auto' or config.embedding!=96 or config.pools!=128
                or config.vocabulary!=4096 or not config.tied_readout or not 1<=config.neurons<=2048):
            raise ValueError('Released model dimensions changed')
    return catalog


def load_case_graph(catalog,label):
    if label not in catalog.entries: raise ValueError('Unknown selected graph')
    entry=catalog.entries[label]
    def array_file(archive,path):
        data=archive.read(path)
        expected=entry['graph_sha256'] if path==entry['path'] else entry['original_graph_sha256']
        if digest(data)!=expected: raise ValueError('Selected graph payload changed')
        with np.load(io.BytesIO(data),allow_pickle=False) as loaded:
            return {name:loaded[name].copy() for name in loaded.files}
    with zipfile.ZipFile(io.BytesIO(catalog.archive)) as archive:
        graph=array_file(archive,entry['path']); original=array_file(archive,entry['original_path'])
    expected={'row','col','weight','contacts','source_sign','body_ids','source_indices','positions','pool'}
    if set(graph)!=expected or set(original)!=expected: raise ValueError('Invalid graph array inventory')
    n=entry['config']['neurons']; e=entry['edges']
    for item in (original,graph):
        if (len(item['body_ids'])!=n or len(set(item['body_ids'].tolist()))!=n
                or any(item[key].shape!=(e,) for key in ('row','col','weight','contacts'))
                or any(item[key].shape!=(n,) for key in ('source_sign','source_indices','pool'))
                or item['positions'].shape!=(n,3)
                or any(not np.isfinite(value).all() for key,value in item.items() if key!='positions')
                or np.isinf(item['positions']).any()
                or any(not np.issubdtype(item[key].dtype,np.integer) for key in ('row','col','body_ids','source_indices','pool','contacts','source_sign'))
                or np.any(item['row']<0) or np.any(item['row']>=n) or np.any(item['col']<0) or np.any(item['col']>=n)
                or np.any(item['pool']<0) or np.any(item['pool']>=entry['config']['pools'])):
            raise ValueError('Invalid graph dimensions, indices or values')
    validate_control(original,graph)
    if entry['graph_seed'] is not None and np.array_equal(original['col'],graph['col']):
        raise ValueError('Rewired graph did not change endpoints')
    return graph,entry


def parameter_hash(model, *, exclude_edges=False):
    return fingerprint({name:p.detach().cpu().numpy() for name,p in model.named_parameters()
                        if not exclude_edges or name!='edge_log_gain'})


def prepare_case(catalog,corpus,label,seed):
    if type(seed) is not int or seed not in TRAINING_SEEDS: raise ValueError('Choose training seed 42 or 43')
    graph,entry=load_case_graph(catalog,label); config=Config(**entry['config'])
    if corpus.lexicon.vocabulary!=config.vocabulary or corpus.lexicon.sha256!=corpus.binding['tokenizer_sha256']:
        raise ValueError('Selected model and verified corpus tokenizer differ')
    if any(key.startswith('graph_') for key in corpus.binding):
        raise ValueError('Supply the graph-independent verified corpus binding')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model=FLM(graph,config).float()
    configure(model,'bptt')
    groups={name:p.numel() for name,p in model.named_parameters()}
    if groups!=entry['parameter_groups'] or sum(groups.values())!=entry['trainable_parameters']:
        raise ValueError('Selected model parameter allocation differs from release')
    binding=dict(format='flm-selection-language-condition-v1',corpus=corpus.binding,catalog=catalog.binding,
        graph=entry,training_seed=seed,initial_state_sha256=state_hash(model),
        initial_parameters_sha256=parameter_hash(model),initial_nonedge_parameters_sha256=parameter_hash(model,exclude_edges=True),
        graph_arrays_sha256=fingerprint(graph),adapter_source_sha256=sha256(Path(__file__)),
        initialized_from='Fresh random parameters; no language checkpoint or pilot weights',
        limitation='A callable condition is not a registered experiment. Matching neuron count does not match edge count or allocated trainable parameters across selectors.')
    return model,binding


def fit_case(catalog,corpus,label,settings,directory,study_binding,*,until=None):
    """Low-level fixed-BPTT fit/resume; an official coordinator must freeze its matrix first."""
    settings.validate()
    if not settings.score_boundaries: raise ValueError('Selection pilot and runner both include boundary targets after warmup')
    if not isinstance(study_binding,dict) or not study_binding: raise ValueError('Explicit study binding required')
    model,binding=prepare_case(catalog,corpus,label,settings.seed)
    result=fit(model,corpus.documents,corpus.lexicon,settings,'bptt',dict(binding,study=study_binding),directory,until=until)
    return result
