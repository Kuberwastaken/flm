"""Validation-only checkpoint selection for the separate learning-rule runner.

No training budget is chosen here and no test data is opened. A study-level
identity and all-condition test gate must still precede an official experiment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .corpus_cache import read_mmap
from .inference import state_hash
from .language_learning_inputs import TOKENIZER_SHA256
from .language_learning_train import (canonical, configure, declaration, inventory,
                                      optimizer_for, restore, training_lease)
from .language_train import evaluate
from .provenance import sha256, write_json

VALIDATION_SHA256 = 'ea3bd0f2625e78723baf6300dca2ffdae91deebdcb21ae748ffd0cc4ba2ef859'
PANEL_SHA256 = '9a0c030150fee1821246dd54f045fc36c437bab3061e3010740d2942900de3ae'


def document_identity(documents, lexicon):
    if (not documents or len({name for name, _ in documents}) != len(documents)
            or np.asarray(lexicon.lengths).shape != (lexicon.vocabulary,)
            or not np.issubdtype(np.asarray(lexicon.lengths).dtype, np.integer)
            or np.any(np.asarray(lexicon.lengths) < 0)
            or np.any(np.asarray(lexicon.lengths)[:2] != 0)):
        raise ValueError('Invalid validation document or byte-length inventory')
    digest = hashlib.sha256(); rows = []
    for name, tokens in documents:
        if (not isinstance(name, str) or not name or not isinstance(tokens, np.ndarray)
                or tokens.ndim != 1 or len(tokens) < 2 or not np.issubdtype(tokens.dtype, np.integer)
                or tokens.min() < 0 or tokens.max() >= lexicon.vocabulary):
            raise ValueError('Invalid validation token sequence')
        for part in (canonical([name, len(tokens)]), tokens.astype('<i8', copy=False).tobytes()):
            digest.update(len(part).to_bytes(8, 'little')); digest.update(part)
        targets = tokens[1:]; selected = targets >= 2
        count = int(selected.sum()); byte_count = int(np.asarray(lexicon.lengths)[targets[selected]].sum())
        if count <= 0 or byte_count <= 0: raise ValueError('Validation block contains no scored text')
        rows.append(dict(document=name, tokens=count, bytes=byte_count))
    return dict(ordered_documents_sha256=digest.hexdigest(), documents=rows,
        tokens=sum(row['tokens'] for row in rows), bytes=sum(row['bytes'] for row in rows),
        tokenizer_sha256=lexicon.sha256,
        byte_lengths_sha256=hashlib.sha256(np.asarray(lexicon.lengths, dtype='<i8').tobytes()).hexdigest())


@dataclass
class Panel:
    documents: list
    binding: dict

    def verify(self, lexicon):
        if self.binding.get('partition') != 'validation' or self.binding.get('coverage') != document_identity(self.documents, lexicon):
            raise ValueError('Validation panel identity changed')


def load_panel(root, lexicon):
    """Read only the pinned validation cache and fixed score-independent panel."""
    root = Path(root)
    cache = root/'data/processed/babylm-2026-bpe/validation'
    panel_path = root/'data/tokenizers/babylm-2026-4096/validation-panel.json'
    if (lexicon.sha256 != TOKENIZER_SHA256 or sha256(cache/'manifest.json') != VALIDATION_SHA256
            or sha256(panel_path) != PANEL_SHA256):
        raise ValueError('Pinned validation inputs changed')
    specification = json.loads(panel_path.read_text(encoding='utf8'))
    if (specification['cache_manifest_sha256'] != VALIDATION_SHA256
            or specification['target_tokens_per_block'] != 1024 or len(specification['ids']) != 48
            or len(set(specification['ids'])) != 48):
        raise ValueError('Invalid fixed validation panel')
    documents, _, manifest = read_mmap(cache, lexicon.sha256)
    lookup = dict(documents); selected = []
    for name in specification['ids']:
        if not name.startswith('validation/') or name not in lookup:
            raise ValueError('Panel block is outside validation')
        # Match the existing panel's per-block prefix convention exactly.
        tokens = lookup[name][:1025].copy(); tokens.setflags(write=False)
        selected.append((name, tokens))
    panel = Panel(selected, dict(partition='validation', cache_manifest_sha256=VALIDATION_SHA256,
        cache_files_sha256=manifest['files'], panel_sha256=PANEL_SHA256,
        prefix_target_positions_per_block=1024, coverage=document_identity(selected, lexicon),
        selection='Existing score-independent BabyLM panel: eight hashed-ID blocks per component',
        test_payloads_opened=False))
    panel.verify(lexicon)
    return panel


def check_score(score, coverage):
    rows = score['documents']
    if len(rows) != len(coverage['documents']) or score['tokenizer_sha256'] != coverage['tokenizer_sha256']:
        raise ValueError('Validation score coverage changed')
    for row, expected in zip(rows, coverage['documents']):
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('Validation block order or denominators changed')
        if not math.isfinite(row['nll']) or row['nll'] < 0:
            raise ValueError('Invalid validation likelihood')
        if not math.isclose(row['bits_per_byte'], row['nll']/row['bytes']/math.log(2), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Invalid block validation arithmetic')
    if score['tokens'] != coverage['tokens'] or score['bytes'] != coverage['bytes']:
        raise ValueError('Validation aggregate denominators changed')
    total = sum(row['nll'] for row in rows)
    if (not math.isfinite(score['nll']) or not math.isclose(score['nll'], total, rel_tol=1e-10, abs_tol=1e-12)
            or not math.isclose(score['bits_per_byte'], total/coverage['bytes']/math.log(2), rel_tol=1e-10, abs_tol=1e-12)
            or not math.isfinite(score['token_perplexity']) or score['token_perplexity'] < 1
            or not math.isclose(math.log(score['token_perplexity']), total/coverage['tokens'], rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError('Invalid aggregate validation arithmetic')


def earliest_minimum(observations, settings):
    expected = list(range(settings.checkpoint_interval, settings.steps+1, settings.checkpoint_interval))
    if [row['step'] for row in observations] != expected:
        raise ValueError('Every declared checkpoint must be validated exactly once in order')
    for row in observations:
        if not math.isfinite(row['score']['bits_per_byte']) or row['score']['bits_per_byte'] < 0:
            raise ValueError('Invalid checkpoint validation score')
    return min(observations, key=lambda row: (row['score']['bits_per_byte'], row['step']))


def select_checkpoint(initial_model, training_documents, lexicon, settings, method, binding, directory, panel, *, chunk_size=96):
    """Validate every committed checkpoint after a complete supplied training run.

    Returns and saves a per-run selection. It is not an all-condition test gate.
    All training payloads are fully restored/audited before any scoring begins.
    """
    if type(chunk_size) is not int or chunk_size < 1: raise ValueError('Positive integer scoring chunk size required')
    directory = Path(directory); settings.validate()
    if not directory.is_dir() or not (directory/'complete.json').is_file():
        raise ValueError('Training run must complete before checkpoint selection')
    panel.verify(lexicon)
    if {name for name, _ in training_documents} & {name for name, _ in panel.documents}:
        raise ValueError('Training and validation block IDs overlap')
    model = copy.deepcopy(initial_model); configure(model, method)
    initial = state_hash(initial_model); previous_threads = torch.get_num_threads()
    declared = declaration(model, training_documents, lexicon, settings, method, binding)
    source_names = ('language_learning_validation.py', 'language_learning_train.py',
                    'language_train.py', 'language_learning_inputs.py', 'corpus_cache.py',
                    'model.py', 'train.py', 'inference.py', 'provenance.py')
    identity = dict(format='flm-learning-rule-validation-v1', declaration=declared, panel_binding=panel.binding,
        chunk_size=chunk_size, selection='Lowest validation bits/byte; earliest checkpoint on an exact tie',
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in source_names})
    identity_hash = hashlib.sha256(canonical(identity)).hexdigest()
    path = directory/'validation-selection.json'
    with training_lease(directory):
        if json.loads((directory/'declaration.json').read_text(encoding='utf8')) != declared:
            raise ValueError('Training declaration file changed')
        complete = json.loads((directory/'complete.json').read_text(encoding='utf8'))
        records = inventory(directory, declared)
        if (len(records) != settings.steps//settings.checkpoint_interval or complete.get('complete') is not True
                or complete.get('declaration') != declared or complete.get('step') != settings.steps
                or complete.get('checkpoint') != records[-1]['checkpoint']
                or complete.get('checkpoint_sha256') != records[-1]['checkpoint_sha256']):
            raise ValueError('Complete training inventory or endpoint changed')
        checkpoint_hashes = {row['checkpoint']:row['checkpoint_sha256'] for row in records}
        complete_hash = sha256(directory/'complete.json')
        optimizer = optimizer_for(model, settings)
        try:
            torch.set_num_threads(settings.threads)
            with torch.random.fork_rng(devices=[]):
                for record in records:
                    saved, _, _ = restore(directory/record['checkpoint'], model, optimizer, declared,
                                          training_documents, lexicon.lengths, settings)
                    if saved['step'] != record['step']: raise ValueError('Payload and marker steps disagree')
                    if record is records[-1] and saved['exposure'] != complete['exposure']:
                        raise ValueError('Complete training exposure changed')
                if path.exists():
                    result = json.loads(path.read_text(encoding='utf8'))
                    if (result['identity_sha256'] != identity_hash or result['identity'] != identity
                            or result['checkpoint_sha256'] != checkpoint_hashes or result['complete_sha256'] != complete_hash):
                        raise ValueError('Existing validation selection identity changed')
                    observations = result['observations']
                    for row in observations: check_score(row['score'], panel.binding['coverage'])
                    if result['selected'] != earliest_minimum(observations, settings):
                        raise ValueError('Existing validation selection is not the earliest minimum')
                else:
                    observations = []
                    for record in records:
                        restore(directory/record['checkpoint'], model, optimizer, declared,
                                training_documents, lexicon.lengths, settings)
                        score = evaluate(model, panel.documents, lexicon, chunk_size=chunk_size, unit='block')
                        check_score(score, panel.binding['coverage'])
                        observations.append(dict(step=record['step'], checkpoint=record['checkpoint'],
                            checkpoint_sha256=record['checkpoint_sha256'], score=score))
                    result = dict(identity_sha256=identity_hash, identity=identity, complete_sha256=complete_hash,
                        checkpoint_sha256=checkpoint_hashes, observations=observations,
                        selected=earliest_minimum(observations, settings), test_payloads_opened=False,
                        scope='Per-run validation selection only. A frozen study identity and complete all-condition gate remain required before test evaluation.')
                for row, record in zip(observations, records):
                    if any(row[key] != record[key] for key in ('step', 'checkpoint', 'checkpoint_sha256')):
                        raise ValueError('Validation checkpoint record changed')
                panel.verify(lexicon)
                if sha256(directory/'complete.json') != complete_hash or inventory(directory, declared) != records:
                    raise ValueError('Training records changed during selection')
                if not path.exists(): write_json(path, result)
        finally:
            torch.set_num_threads(previous_threads)
            if state_hash(initial_model) != initial: raise ValueError('Validation changed the supplied initial model')
    return result
