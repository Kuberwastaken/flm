"""Explicit Mac execution amendment; original numerical modules stay unchanged.

Every condition is freshly initialized and fitted on this runtime. Windows
checkpoints are never resumed or pooled into this amended cohort. Only study
paths and the context verifier are adapted; fitting, selection and scoring use
the original functions. The verifier binds this adapter as additional source.
"""
import argparse
import copy
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flm import selection_language_study as coordinator
from flm import selection_language_test as evaluator
from flm.provenance import sha256, write_json

PARENT = Path('reports/selection-language/study-identity.json')
PARENT_SHA256 = 'f9fcffed62db3b9da22e58f1df8dd3f9a47182eed5ff8a5c3dbbe773e75768c3'
REPORTS = Path('reports/selection-language/mac-v1')
STUDY = Path('runs/selection-language-mac-v1')
IDENTITY = REPORTS/'study-identity.json'
PROFILE = REPORTS/'runtime-profile.json'
AMENDMENT = Path('docs/SELECTION-MAC-EXECUTION.md')
PROBE = REPORTS/'qualification-probe.json'
ORIGINAL_UNCHANGED = coordinator.unchanged
PACKAGES = {'torch': '2.8.0', 'numpy': '2.2.6', 'scipy': '1.13.1',
            'tokenizers': '0.22.2', 'requests': '2.32.5'}
INITIAL_FIELDS = {'initial_state_sha256', 'initial_parameters_sha256', 'initial_nonedge_parameters_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def environment():
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Require the declared Apple ARM CPU runtime')
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(packages == PACKAGES, 'Declared runtime package versions changed')
    require(str(torch.__version__) == '2.8.0' and sys.version_info[:2] == (3, 10), 'Runtime build changed')
    return dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                cpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                packages=packages, torch=str(torch.__version__), numpy=str(np.__version__),
                torch_build=torch.__config__.show(), device='cpu', threads_per_fit=4,
                adapter_sha256=sha256(Path(__file__)))


def validate_condition(parent, native):
    require(set(parent) == set(native), 'Condition binding fields changed')
    for key in parent:
        if key not in INITIAL_FIELDS:
            require(parent[key] == native[key], 'Scientific condition changed: '+key)


def context(root):
    root = Path(root)
    require(sha256(root/PARENT) == PARENT_SHA256, 'Original study identity changed')
    parent = read(root/PARENT)
    require(len(parent['conditions']) == 128, 'Require the entire original matrix')
    for name, expected in parent['sources'].items():
        require(sha256(root/'flm'/name) == expected, 'Frozen numerical source changed: '+name)
    require(sha256(root/coordinator.PROTOCOL) == parent['protocol_sha256']
            and sha256(root/coordinator.PILOT) == parent['pilot_sha256'], 'Historical protocol or pilot changed')
    profile = read(root/PROFILE)
    require(profile['environment'] == environment(), 'Frozen Mac runtime profile changed')
    require(profile['parent_study_identity_sha256'] == PARENT_SHA256
            and profile['amendment_sha256'] == sha256(root/AMENDMENT)
            and profile['qualification_probe_sha256'] == sha256(root/PROBE), 'Execution amendment binding changed')
    torch.set_num_threads(4)
    catalog = coordinator.load_catalog(root)
    corpus = coordinator.load_corpus(root)
    panel = coordinator.load_panel(root, corpus.lexicon)
    require(catalog.binding == parent['catalog_binding'] and corpus.binding == parent['corpus_binding']
            and panel.binding == parent['validation_panel_binding'], 'Graph/corpus/panel differs from original study')
    require(coordinator.test_metadata(root, corpus.lexicon) == parent['test_metadata'], 'Test metadata changed')
    require(coordinator.matrix(catalog, parent['request']['groups']) == [r['condition'] for r in parent['conditions']],
            'Original complete condition inventory changed')
    identity = copy.deepcopy(parent)
    learned, nonedge = {}, {}
    for row in identity['conditions']:
        c = row['condition']
        model, binding = coordinator.prepare_case(catalog, corpus, c['graph'], c['seed'])
        validate_condition(row['base_binding'], binding)
        declared = coordinator.declaration(model, corpus.documents, corpus.lexicon,
            coordinator.Settings(**row['settings']), 'bptt', binding)
        require(declared['ordered_training_documents_sha256'] == row['training_documents_sha256']
                and declared['trainable_parameters'] == row['trainable_parameters'], 'Exposure/order or allocation changed')
        row['base_binding'] = binding
        graph = binding['graph']
        learned.setdefault((graph['selection'], c['seed']), set()).add(binding['initial_parameters_sha256'])
        nonedge.setdefault((graph['candidate'], c['seed']), set()).add(binding['initial_nonedge_parameters_sha256'])
    require(all(len(values) == 1 for values in (*learned.values(), *nonedge.values())),
            'Native initial parameters do not match across controls')
    identity['torch'] = str(torch.__version__)
    identity['numpy'] = str(np.__version__)
    identity['execution_amendment'] = dict(parent_study_identity_sha256=PARENT_SHA256,
        document_sha256=sha256(root/AMENDMENT), profile_sha256=sha256(root/PROFILE),
        adapter_sha256=sha256(Path(__file__)), qualification_probe_sha256=sha256(root/PROBE),
        cohort='All 128 fresh CPU fits on the declared Mac runtime; no Windows checkpoints included',
        historical_pilot='Original Windows pilot retained unchanged as historical evidence; its timings and initial tensors are not Mac measurements')
    return identity, catalog, corpus, panel


def verified_context(root):
    expected, catalog, corpus, panel = context(root)
    require(read(Path(root)/IDENTITY) == expected, 'Amended study identity changed')
    return expected, catalog, corpus, panel


def unchanged(root, identity, identity_hash):
    ORIGINAL_UNCHANGED(root, identity, identity_hash)
    amendment = identity['execution_amendment']
    require(sha256(Path(__file__)) == amendment['adapter_sha256']
            and sha256(Path(root)/AMENDMENT) == amendment['document_sha256']
            and sha256(Path(root)/PROFILE) == amendment['profile_sha256'], 'Execution adapter or amendment changed')


def install_adapter():
    """Explicit non-numerical redirects, also used by the original scorer imports."""
    for module in (coordinator, evaluator):
        module.REPORTS = REPORTS
        module.STUDY = STUDY
        module.IDENTITY = IDENTITY
        module.SELECTION = REPORTS/'study-selection.json'
        module.verified_context = verified_context
    coordinator.unchanged = unchanged
    evaluator.unchanged_study = unchanged
    evaluator.EVALUATION = Path('runs/selection-language-mac-evaluation-v1')
    evaluator.SUMMARY = REPORTS/'test-summary.json'


def initialize(root):
    root = Path(root)
    coordinator.prerequisites(root)
    if not (root/PROFILE).exists():
        require(not (root/IDENTITY).exists() and not (root/STUDY).exists(), 'Cannot amend an existing cohort')
        probe = read(root/PROBE)
        parent = read(root/PARENT)
        require(probe['identity_sha256'] == PARENT_SHA256 and probe['conditions_checked'] == 128
                and probe['machine'] == 'arm64' and probe['torch'] == '2.8.0'
                and probe['numpy'] == '2.2.6', 'Missing complete native initialization qualification')
        require([r['condition'] for r in probe['bindings']] == [r['condition'] for r in parent['conditions']]
                and all(set(r['differing_binding_fields']) <= INITIAL_FIELDS for r in probe['bindings']),
                'Qualification changed more than native initial tensors')
        write_json(root/PROFILE, dict(environment=environment(), parent_study_identity_sha256=PARENT_SHA256,
            amendment_sha256=sha256(root/AMENDMENT), qualification_probe_sha256=sha256(root/PROBE)))
    expected, _, _, _ = context(root)
    with coordinator.training_lease(root/STUDY):
        if (root/IDENTITY).exists():
            require(read(root/IDENTITY) == expected, 'Existing native identity changed')
        else:
            require(not any(p.name != 'writer.lock' for p in (root/STUDY).iterdir()), 'Fresh cohort already has run files')
            write_json(root/IDENTITY, expected)
    print('Verified complete Mac matrix:', sha256(root/IDENTITY), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('initialize', 'verify', 'train', 'select', 'test', 'run'))
    args = parser.parse_args()
    install_adapter()
    if args.operation == 'initialize':
        initialize(ROOT)
    elif args.operation == 'verify':
        verified_context(ROOT)
        print('Verified full native identity and unchanged scientific settings', flush=True)
    elif args.operation == 'train':
        coordinator.train_study(ROOT)
    elif args.operation == 'select':
        coordinator.freeze_selection(ROOT)
    elif args.operation == 'test':
        evaluator.score_study(ROOT)
    else:
        initialize(ROOT)
        coordinator.train_study(ROOT)
        coordinator.freeze_selection(ROOT)
        evaluator.score_study(ROOT)


if __name__ == '__main__':
    main()
