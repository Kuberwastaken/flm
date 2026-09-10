"""Prepare matched instruction conditions without choosing a budget or fitting.

Every fit/resume must prepare its original source again and bind its declared
settings. The study-wide declaration and completion/test gate are separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .inference import state_hash
from .language_train import construct, restore
from .provenance import sha256
from .scan import SPLITS
from .scan_inputs import load_training_partition
from .scan_train import canonical
from .tokenizer import Lexicon


VARIANTS = ('flm', 'gru', 'transformer')
SEEDS = (42, 43)
INITIALIZATIONS = ('initial', 'wikitext')
GRAPH = Path('data/graphs/central-1024/graph.npz')
TOKENIZER = Path('data/tokenizers/wikitext2-4096/tokenizer.json')
SELECTION = Path('reports/wikitext2/selection.json')
SOURCE_AUDIT = Path('reports/scan-runtime/source-preflight.json')
SOURCE_FILES = ('scripts/scan_source_audit.py', 'flm/language_train.py', 'flm/model.py',
                'flm/baselines.py', 'flm/tokenizer.py', 'flm/inference.py')


def conditions():
    return [dict(label=f'{split}/{variant}-{initialization}-s{seed}', split=split,
                 variant=variant, seed=seed, initialization=initialization)
            for split in SPLITS for seed in SEEDS for variant in VARIANTS
            for initialization in INITIALIZATIONS]


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def source_model(root, variant, seed, initialization):
    if variant not in VARIANTS or type(seed) is not int or seed not in SEEDS or initialization not in INITIALIZATIONS:
        raise ValueError('Unknown instruction source condition')
    root = Path(root); audit = read_json(root / SOURCE_AUDIT); selection = read_json(root / SELECTION)
    expected = {(v, s) for v in VARIANTS for s in SEEDS}
    for rows in (audit['sources'], selection['runs']):
        if len(rows) != 6 or {(r['variant'], r['seed']) for r in rows} != expected:
            raise ValueError('Require the complete six-source inventory')
    if (set(audit['source_sha256']) != set(SOURCE_FILES)
            or any(sha256(root / name) != audit['source_sha256'][name] for name in SOURCE_FILES)
            or audit['language_selection_sha256'] != sha256(root / SELECTION)
            or audit['graph_sha256'] != sha256(root / GRAPH)
            or audit['tokenizer_sha256'] != sha256(root / TOKENIZER)
            or audit['torch'] != str(torch.__version__)):
        raise ValueError('Verified language source or software changed; rerun source preparation')
    source = next(r for r in audit['sources'] if (r['variant'], r['seed']) == (variant, seed))
    selected = next(r for r in selection['runs'] if (r['variant'], r['seed']) == (variant, seed))
    path = Path(f'runs/wikitext2/{variant}-s{seed}/best.pt')
    if (selected['checkpoint'].replace('\\', '/') != path.as_posix()
            or selected['checkpoint_sha256'] != source['checkpoint_sha256']
            or source['initial_tensors_exact'] is not True
            or source['probe_tensors_unchanged'] is not True
            or source['chunk_maximum_logit_gaps'] != [0., 0.]):
        raise ValueError('Source checkpoint or compatibility record changed')
    lexicon = Lexicon(root / TOKENIZER)
    trained, saved = restore(root / path, root / GRAPH, lexicon)
    if (saved['_file_sha256'] != source['checkpoint_sha256']
            or saved['run']['protocol'] != selection['protocol'] or saved['run']['seed'] != seed
            or saved['config']['variant'] != variant or saved['step'] != source['checkpoint_step']
            or saved['best'] != selected['selection_validation_bpb']
            or saved['run']['source_commit'] != source['source_commit']
            or saved['run']['test_set_used_for_training'] is not False
            or state_hash(trained) != source['pretrained_state_sha256']):
        raise ValueError('Restored language checkpoint differs from the verified source')
    initial = construct(variant, root / GRAPH, lexicon.vocabulary, seed)
    if state_hash(initial) != source['initial_state_sha256']:
        raise ValueError('Reconstructed original initialization changed')
    model = initial if initialization == 'initial' else trained
    return model, lexicon, dict(language_selection_sha256=sha256(root / SELECTION),
        source_audit_sha256=sha256(root / SOURCE_AUDIT), source_checkpoint_sha256=source['checkpoint_sha256'],
        original_source_commit=source['source_commit'], initial_state_sha256=state_hash(model),
        inherits_language_weights=initialization == 'wikitext',
        language_pretraining_updates=source['checkpoint_step'] if initialization == 'wikitext' else 0,
        tokenizer_sha256=lexicon.sha256, graph_sha256=audit['graph_sha256'])


@dataclass
class Prepared:
    model: object
    records: list
    lexicon: object
    binding: dict

    def training_binding(self, settings):
        """Check input immutability immediately before passing to scan_train.fit."""
        settings.validate()
        if settings.sampling_seed != self.binding['condition']['seed']:
            raise ValueError('Matched conditions must use the declared initialization seed for row sampling')
        if (state_hash(self.model) != self.binding['language_source']['initial_state_sha256']
                or hashlib.sha256(canonical(self.records)).hexdigest() != self.binding['ordered_rows_sha256']
                or self.lexicon.sha256 != self.binding['language_source']['tokenizer_sha256']):
            raise ValueError('Prepared model, training rows or tokenizer changed before fitting')
        return json.loads(canonical(self.binding))


def prepare_condition(root, condition):
    if condition not in conditions():
        raise ValueError('Expected an exact declared instruction condition')
    root = Path(root)
    rows, data_binding = load_training_partition(root, condition['split'])
    model, lexicon, source_binding = source_model(root, condition['variant'],
        condition['seed'], condition['initialization'])
    binding = dict(format='flm-scan-condition-v1', condition=dict(condition),
        training_input=data_binding, language_source=source_binding,
        ordered_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
        preparation_source_sha256=sha256(Path(__file__)),
        tokenizer_scope='Shared train-fitted WikiText vocabulary in both initial and pretrained conditions')
    return Prepared(model, rows, lexicon, binding)
