"""Verified training-only inputs for a separate FLM learning-rule comparison.

No fit, budget choice, validation selection or test access is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .babylm import COMPONENTS,REPOSITORIES
from .corpus_cache import read_mmap
from .inference import state_hash
from .language_learning_train import METHODS,configure
from .model import Config,FLM,load_graph
from .provenance import sha256
from .tokenizer import Lexicon

PARTITION='train-10m'
GRAPH_SHA256='a2f35369a5ed4981b8ef75c92df285f7df576a020582f119a1372bfbedbaebee'
TOKENIZER_SHA256='46a2ef1bf8dce00e801c5ac188b13ec5c30c7f1ab7e8184031dca3ee057a7242'
CACHE_SHA256='4d205d6d35c87c38749ca3dc858fe96b28e3d7f4b8d7e2eb782e06909c8f15cd'


def fingerprint(graph):
    result=hashlib.sha256()
    for name,array in sorted(graph.items()):
        result.update(name.encode()); result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode()); result.update(array.tobytes())
    return result.hexdigest()


def verify_training_blocks(documents,records,manifest,lexicon,raw,sources):
    if (not documents or len(documents) != len(records) or len(records) != manifest['blocks']
            or len({identity for identity,_ in documents}) != len(documents)
            or set(raw) != set(sources)):
        raise ValueError('Invalid training block or component inventory')
    for name,data in raw.items():
        if hashlib.sha256(data).hexdigest() != sources[name]: raise ValueError('Training source checksum changed')
    ends={name:0 for name in raw}; tokens=0; byte_count=0
    for (identity,array),record in zip(documents,records):
        component=record['component']; start=record['byte_start']; end=record['byte_end']
        if component not in raw or record['source_sha256'] != sources[component]:
            raise ValueError('Block refers to an unknown training source')
        expected=f'{PARTITION}/{component}/{sources[component][:16]}/{start}-{end}'
        if (identity != expected or record['id'] != identity or start != ends[component]
                or not start < end <= len(raw[component]) or record['utf8_bytes'] != end-start):
            raise ValueError('Training identity or byte coverage changed')
        if (array.ndim != 1 or len(array) < 3 or not np.issubdtype(array.dtype,np.integer)
                or int(array[0]) != 0 or int(array[-1]) != 1
                or np.any(array[1:-1] < 2)
                or int(array.min()) < 0 or int(array.max()) >= lexicon.vocabulary):
            raise ValueError('Invalid training token IDs or boundaries')
        text=raw[component][start:end]
        if (hashlib.sha256(text).hexdigest() != record['text_sha256']
                or lexicon.decode(array).encode('utf8') != text
                or int(lexicon.lengths[array].sum()) != len(text)):
            raise ValueError('Tokenized training block does not reproduce source bytes')
        ends[component]=end; tokens += len(array)-2; byte_count += len(text)
    if (ends != {name:len(data) for name,data in raw.items()} or tokens != manifest['text_tokens']
            or byte_count != manifest['utf8_bytes']):
        raise ValueError('Training blocks do not cover all source bytes or declared tokens')
    return dict(blocks=len(records),text_tokens=tokens,utf8_bytes=byte_count,
        all_source_bytes_covered_once=True,all_blocks_decode_to_source_bytes=True)


@dataclass
class TrainingInputs:
    graph: dict
    documents: list
    lexicon: Lexicon
    binding: dict


def load_inputs(root):
    root=Path(root)
    source_path=root/'data/sources/babylm-2026.json'
    acquisition_path=root/'data/cards/babylm-2026-acquisition.json'
    source=json.loads(source_path.read_text(encoding='utf8'))
    acquired=json.loads(acquisition_path.read_text(encoding='utf8'))
    if acquired['manifest_sha256'] != sha256(source_path): raise ValueError('Acquisition manifest changed')
    selected=[r for r in source['files'] if r['partition']==PARTITION]
    local=[r for r in acquired['files'] if r['partition']==PARTITION]
    if (len(selected) != len(COMPONENTS) or len(local) != len(COMPONENTS)
            or {r['component'] for r in selected} != set(COMPONENTS)
            or {r['component'] for r in local} != set(COMPONENTS)):
        raise ValueError('Incomplete or duplicated training component inventory')
    local={r['component']:r for r in local}; raw={}; hashes={}; provenance=[]
    repository,revision,_=REPOSITORIES[PARTITION]
    for row in selected:
        component=row['component']; filename=component+'.train.txt'
        url=f'https://huggingface.co/datasets/BabyLM-community/{repository}/resolve/{revision}/{filename}'
        entry=local[component]; relative=f'data/raw/babylm-2026/{PARTITION}/{filename}'
        if (row['repository'] != 'BabyLM-community/'+repository or row['revision'] != revision
                or row['filename'] != filename or row['url'] != url
                or entry['path'].replace('\\','/') != relative or not entry['source_verified']):
            raise ValueError('Training source provenance changed')
        data=(root/relative).read_bytes(); digest=hashlib.sha256(data).hexdigest()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if (len(data) != row['bytes'] or len(data) != entry['utf8_bytes'] or digest != entry['sha256']
                or (row['lfs_sha256'] is not None and row['lfs_sha256'] != digest)
                or (row['lfs_sha256'] is None and row['git_blob_sha1'] != blob)):
            raise ValueError('Pinned raw training bytes changed')
        raw[component]=data; hashes[component]=digest; provenance.append(dict(component=component,url=url,sha256=digest,bytes=len(data)))
    tokenizer=root/'data/tokenizers/babylm-2026-4096/tokenizer.json'
    tokenizer_card_path=tokenizer.with_name('tokenizer-card.json')
    tokenizer_card=json.loads(tokenizer_card_path.read_text(encoding='utf8'))
    cache=root/'data/processed/babylm-2026-bpe/train-10m'
    if (sha256(tokenizer) != TOKENIZER_SHA256 or sha256(cache/'manifest.json') != CACHE_SHA256
            or tokenizer_card['tokenizer_sha256'] != TOKENIZER_SHA256
            or tokenizer_card['training_sources'] != hashes or tokenizer_card['fitted_partition'] != PARTITION
            or tokenizer_card['validation_or_test_used_for_training']):
        raise ValueError('Pinned tokenizer or training cache identity changed')
    lexicon=Lexicon(tokenizer); documents,records,manifest=read_mmap(cache,lexicon.sha256)
    if lexicon.vocabulary != 4096 or manifest['identity']['sources'] != hashes:
        raise ValueError('Training cache source inventory changed')
    coverage=verify_training_blocks(documents,records,manifest,lexicon,raw,hashes)
    graph_path=root/'data/graphs/central-1024/graph.npz'; card_path=graph_path.with_name('graph-card.json')
    card=json.loads(card_path.read_text(encoding='utf8'))
    if sha256(graph_path) != GRAPH_SHA256 or card['graph_sha256'] != GRAPH_SHA256:
        raise ValueError('Learning-rule comparison requires the fixed original graph')
    graph=load_graph(graph_path)
    if len(graph['body_ids']) != 1024 or len(graph['row']) != 76130 or int(graph['pool'].max())+1 != 128:
        raise ValueError('Original graph dimensions changed')
    binding=dict(format='flm-language-learning-inputs-v1',dataset='BabyLM 2026 English',partition=PARTITION,
        revision=revision,source_manifest_sha256=sha256(source_path),acquisition_card_sha256=sha256(acquisition_path),
        licensing=source['licensing'],training_sources=provenance,
        tokenizer_sha256=lexicon.sha256,tokenizer_card_sha256=sha256(tokenizer_card_path),
        cache_manifest_sha256=sha256(cache/'manifest.json'),cache_files_sha256=manifest['files'],
        graph_sha256=GRAPH_SHA256,graph_card_sha256=sha256(card_path),graph_arrays_sha256=fingerprint(graph),
        graph_scope='Original ranked 1024-neuron subset; this preparation varies learning rules, not neuron selection',
        loader_sha256=sha256(Path(__file__)),coverage=coverage,
        validation_or_test_payloads_opened=False,trained_weights_loaded=False,
        limitation='Shared inventories contain metadata about other partitions; only train-10m raw text and token payloads are opened. No official fit matrix, budget or evaluation gate is defined here.')
    return TrainingInputs(graph,documents,lexicon,binding)


def conditions():
    return [dict(method=method,seed=seed,label=f'{method}-s{seed}') for seed in (42,43) for method in METHODS]


def prepare_condition(inputs,condition):
    if condition not in conditions(): raise ValueError('Unknown learning-rule preparation condition')
    if (fingerprint(inputs.graph) != inputs.binding['graph_arrays_sha256']
            or inputs.lexicon.sha256 != inputs.binding['tokenizer_sha256']):
        raise ValueError('Loaded graph or tokenizer identity changed')
    # Fresh arrays prevent model buffers from sharing writable storage with inputs.
    graph={name:array.copy() for name,array in inputs.graph.items()}
    config=Config(neurons=len(graph['body_ids']),pools=int(graph['pool'].max())+1,
        embedding=96,vocabulary=inputs.lexicon.vocabulary,tied_readout=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(condition['seed']); model=FLM(graph,config).float()
    configure(model,condition['method'])
    binding=dict(inputs.binding,condition=dict(condition),initial_state_sha256=state_hash(model),
        initialized_from='Fresh random parameters; no language checkpoint or pilot weights')
    return model,binding
