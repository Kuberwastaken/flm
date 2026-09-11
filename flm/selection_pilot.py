"""Disposable train-only cost measurement for the 64 original graph selections.

The official command is gated behind the priority training and structural queues.
Small fixtures may call measure directly; their timings are not study estimates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from .corpus_cache import read_mmap
from .inference import state_hash
from .model import Config,FLM,load_graph
from .provenance import sha256,write_json
from .scan_train import training_lease
from .tokenizer import Lexicon
from .train import Sampler


@dataclass(frozen=True)
class Pilot:
    warmup_updates: int = 3
    measured_updates: int = 12
    batch: int = 16
    sequence: int = 96
    context_warmup: int = 16
    threads: int = 4
    seed: int = 42

    def validate(self):
        if any(type(v) is not int or v < 1 for v in (
                self.warmup_updates,self.measured_updates,self.batch,self.sequence,self.threads)):
            raise ValueError('Positive integer pilot dimensions required')
        if (type(self.context_warmup) is not int or not 0 <= self.context_warmup < self.sequence
                or type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError('Invalid context warmup or pilot seed')


def graph_digest(graph):
    result=hashlib.sha256()
    for name,array in sorted(graph.items()):
        result.update(name.encode()); result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode()); result.update(array.tobytes())
    return result.hexdigest()


def measure(graph,config,documents,lengths,pilot=Pilot()):
    """Measure real FLM gradient updates; return no weights, text or loss values."""
    pilot.validate()
    lengths=np.asarray(lengths)
    if (lengths.shape != (config.vocabulary,) or not np.issubdtype(lengths.dtype,np.integer)
            or np.any(lengths < 0)):
        raise ValueError('Invalid tokenizer byte lengths')
    before=graph_digest(graph); previous_threads=torch.get_num_threads()
    sampler=Sampler(documents,pilot.seed,pilot.sequence)
    sampled=hashlib.sha256(); observations=[]
    try:
        torch.set_num_threads(pilot.threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(pilot.seed)
            model=FLM(graph,config).float().train()
            initial=state_hash(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            for step in range(1,pilot.warmup_updates+pilot.measured_updates+1):
                started=time.perf_counter()
                x,y=sampler.sample(pilot.batch,'cpu')
                if torch.any(x < 0) or torch.any(y < 0) or torch.any(x >= config.vocabulary) or torch.any(y >= config.vocabulary):
                    raise ValueError('Training window contains invalid token IDs')
                optimizer.zero_grad(set_to_none=True)
                logits,_=model(x)
                # Match the existing language trainer: exclude warmup positions,
                # but do not introduce a new boundary-target mask for this pilot.
                loss=F.cross_entropy(logits[:,pilot.context_warmup:].reshape(-1,config.vocabulary),
                                     y[:,pilot.context_warmup:].reshape(-1))
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite pilot loss')
                loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(gradient): raise FloatingPointError('Nonfinite pilot gradient')
                optimizer.step()
                seconds=time.perf_counter()-started
                # Integrity checks and exposure bookkeeping are outside timing.
                tensors=list(model.state_dict().values())
                tensors += [v for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
                if any(not torch.isfinite(v).all() for v in tensors): raise FloatingPointError('Nonfinite pilot update')
                sampled.update(x.numpy().astype('<i8',copy=False).tobytes())
                sampled.update(y.numpy().astype('<i8',copy=False).tobytes())
                observations.append(dict(step=step,warmup=step <= pilot.warmup_updates,seconds=seconds,
                    input_tokens=x.numel(),supervised_targets=y[:,pilot.context_warmup:].numel(),
                    input_bytes=int(lengths[x.numpy()].sum()),
                    supervised_bytes=int(lengths[y[:,pilot.context_warmup:].numpy()].sum())))
            changed=state_hash(model) != initial
            card=model.parameter_card()
    finally:
        torch.set_num_threads(previous_threads)
    if graph_digest(graph) != before: raise ValueError('Pilot mutated the source graph')
    if not changed: raise ValueError('Disposable optimizer did not change the model')
    if any(not np.isfinite(r['seconds']) or r['seconds'] <= 0 for r in observations):
        raise ValueError('Invalid update timing')
    timed=[r for r in observations if not r['warmup']]; elapsed=sum(r['seconds'] for r in timed)
    return dict(pilot=asdict(pilot),parameter_card=card,graph_arrays_sha256=before,
        initial_state_sha256=initial,graph_unchanged=True,disposable_parameters_updated=True,
        sampled_windows_sha256=sampled.hexdigest(),observations=observations,
        measured_seconds=elapsed,median_update_seconds=float(np.median([r['seconds'] for r in timed])),
        measured_input_tokens=sum(r['input_tokens'] for r in timed),
        measured_supervised_targets=sum(r['supervised_targets'] for r in timed),
        measured_input_bytes=sum(r['input_bytes'] for r in timed),
        measured_supervised_bytes=sum(r['supervised_bytes'] for r in timed),
        measured_input_tokens_per_second=sum(r['input_tokens'] for r in timed)/elapsed,
        optimizer=dict(name='AdamW',learning_rate=.002,weight_decay=.01,gradient_clip=1.),
        effective_backend='dense' if config.backend=='dense' or (config.backend=='auto' and config.neurons <= 2048) else 'sparse',
        dtype='torch.float32',checkpoint_written=False,language_scores_reported=False,
        timing_scope='Sampling, token checks, forward/backward, gradient clipping and AdamW. Excludes graph/corpus loading, initialization, integrity bookkeeping, checkpoint I/O and validation.',
        scope='Short disposable timing, not convergence, held-out quality, peak memory or biological performance')


def prepare_inputs(root):
    """Open only the pinned 10M training cache and its train-fitted tokenizer."""
    tokenizer=root/'data/tokenizers/babylm-2026-4096/tokenizer.json'
    cache=root/'data/processed/babylm-2026-bpe/train-10m'
    study_path=root/'runs/babylm-10m/study.json'
    study=json.loads(study_path.read_text(encoding='utf8'))
    if (study['dataset'] != 'BabyLM 2026 10m' or sha256(tokenizer) != study['tokenizer_sha256']
            or sha256(cache/'manifest.json') != study['train_manifest_sha256']):
        raise ValueError('Training-only pilot input identity changed')
    lexicon=Lexicon(tokenizer)
    documents,_,manifest=read_mmap(cache,lexicon.sha256)
    if lexicon.vocabulary != 4096: raise ValueError('Pilot requires the shared 4096-entry tokenizer')
    return documents,lexicon,dict(partition='train-10m',study_sha256=sha256(study_path),
        tokenizer_sha256=lexicon.sha256,cache_manifest_sha256=sha256(cache/'manifest.json'),
        cache_files_sha256=manifest['files'],validation_or_test_opened=False)


def ordered_selections(manifest):
    names=sorted(manifest['graphs'])
    order=np.random.Generator(np.random.PCG64(519)).permutation(len(names))
    return [names[int(index)] for index in order]


def prerequisites(root):
    expected=[root/f'runs/babylm-{scale}/{variant}-s{seed}/complete.json'
              for scale in ('10m','100m') for variant in ('flm','gru','transformer') for seed in (42,43)]
    if any(not path.is_file() for path in expected):
        raise ValueError('Priority BabyLM training must finish before selection timing')
    for path in expected:
        completed=json.loads(path.read_text(encoding='utf8'))
        if completed['steps'] != 12000 or completed['best_checkpoint_sha256'] != sha256(path.with_name('best.pt')):
            raise ValueError('Priority BabyLM completion identity changed')
    release_path=root/'reports/selection-rewiring/release.json'
    if not release_path.is_file():
        raise ValueError('Finish and independently audit the structural queue before selection timing')
    release=json.loads(release_path.read_text(encoding='utf8'))
    if (release['planned'] != 192 or release['complete']+release['failed'] != 192
            or not release['arrays_degrees_signs_weights_self_edges_and_diagnostics_verified']
            or release['archive_sha256'] != sha256(root/'public/research/selection-rewiring.zip')):
        raise ValueError('Structural release identity changed')


def run(root):
    root=Path(root); prerequisites(root)
    destination=root/'reports/selection-pilot/timing.json'
    with training_lease(root/'runs/selection-timing-pilot'):
        if destination.exists(): raise ValueError('Timing pilot already recorded; preserve observed runs')
        graph_root=root/'work/selection-graphs-v1'
        manifest_path=graph_root/'manifest.json'; manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        release=json.loads((root/'reports/selection-graphs/export-release.json').read_text(encoding='utf8'))
        if (sha256(manifest_path) != release['manifest_sha256'] or len(manifest['graphs']) != 64
                or manifest['model_source_sha256'] != sha256(root/'flm/model.py')):
            raise ValueError('Released graph inventory or model implementation changed')
        documents,lexicon,binding=prepare_inputs(root)
        attempt=root/'runs/selection-timing-pilot'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt.mkdir(); results=[]; names=ordered_selections(manifest)
        for name in names:
            try:
                entry=manifest['graphs'][name]; path=graph_root/entry['path']
                if not path.resolve().is_relative_to(graph_root.resolve()) or sha256(path) != entry['graph_sha256']:
                    raise ValueError('Selection graph identity changed')
                measured=measure(load_graph(path),Config(**entry['config']),documents,lexicon.lengths)
                if measured['parameter_card']['trainable_parameters'] != entry['trainable_parameters']:
                    raise ValueError('Instantiated parameter allocation differs from export')
                if results and measured['sampled_windows_sha256'] != results[0]['sampled_windows_sha256']:
                    raise ValueError('Pilot selections did not receive identical windows')
            except Exception as error:
                write_json(attempt/'failure.json',dict(selection=name,completed=len(results),
                    error_type=type(error).__name__,error=str(error),source_sha256=sha256(Path(__file__))))
                raise
            measured.update(selection=name,graph_sha256=entry['graph_sha256'])
            results.append(measured); write_json(attempt/(name.replace('/','--')+'.json'),measured)
            print('Measured disposable selection pilot: '+name,flush=True)
        report=dict(verified_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
            torch=str(torch.__version__),numpy=str(np.__version__),binding=binding,
            source_sha256={name:sha256(root/'flm'/name) for name in
                          ('selection_pilot.py','model.py','train.py','corpus_cache.py','tokenizer.py','scan_train.py','inference.py')},
            graph_manifest_sha256=sha256(manifest_path),graph_order_seed=519,
            attempt=attempt.relative_to(root).as_posix(),conditions=results,
            benchmark_budget_selected=False,training_matrix_frozen=False,
            limitations=['Verify live training/process handles before launch; completion files alone are not process locks.',
                'All 64 original graphs are timed; none of the 192 artificial rewires is timed by this command.',
                'A single shuffled pass does not establish stable throughput or account for long-run thermal effects.',
                'Fixed pilot learning rate measures cost, not the convergence of a future optimizer schedule.',
                'No model weights, optimizer state, raw text, predictions or language loss values are saved.'])
        write_json(destination,report)
    return destination


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(run(Path.cwd()))
