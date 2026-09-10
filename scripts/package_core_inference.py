"""Export all eight computation conditions after the completed-study gate."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flm.core_inference import (BUNDLE_FORMAT, CHECKPOINT_FORMAT, RUNTIME_FILES,
                                load_model, parameter_counts, verify_bundle)
from flm.inference import generate, state_hash
from flm.language_core_study import GRAPH, LEXICON, REPORTS, read_json, run_binding
from flm.language_core_train import restore as restore_control
from flm.language_report import PROMPTS, SAMPLING, sample
from flm.language_train import restore as restore_original
from flm.provenance import sha256
from flm.tokenizer import Lexicon
from scripts.language_core_report import verified_report


ARCHIVE = 'public/research/language-core-inference.zip'
RELEASE = 'reports/language-core/inference-release.json'
PUBLIC_RELEASE = 'public/research/language-core-inference-release.json'
ASSETS = (LEXICON.as_posix(), 'data/cards/wikitext2.json',
          'data/tokenizers/wikitext2-4096/tokenizer-card.json',
          'data/tokenizers/wikitext2-4096/tokenization-card.json',
          GRAPH.as_posix(), GRAPH.with_name('graph-card.json').as_posix(),
          'docs/LANGUAGE-CORE-PROTOCOL.md', 'docs/WIKITEXT-PROTOCOL.md',
          'LICENSE', 'licenses/CC-BY-4.0.txt', 'licenses/DATA-ATTRIBUTION.md')
README = b'''# FLM: run the language computation controls

FLM by Kuber Mehta. This archive contains eight selected text-continuation
models on the same 1,024-neuron measured graph subset, at training seeds 42 and
43. Six controls were retrained; the two full-model references are reused.
The complete-study gate precedes export. These models are not instruction tuned.
Generated continuations are unedited model output, not factual answers.

Use a Python environment in this extracted directory:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python -X utf8 -m flm.core_inference --condition full-s42 --prompt "The history of science"
python -X utf8 -m flm.core_inference --condition fixed_dynamics-s42 --prompt "The history of science"
python -X utf8 -m flm.core_inference --condition no_lateral-s42 --prompt "The history of science"
python -X utf8 -m flm.core_inference --condition no_temporal_state-s42 --prompt "The history of science"
```

Replace s42 with s43 for the other training initialization. Full keeps the
original model class. Fixed dynamics freezes the recurrent edges, gain, and
fast/slow decay parameters at initialization; the lexical interface still
learns through backpropagation through time. No lateral removes communication
between units while retaining their fast and slow temporal states. No temporal
state resets both states on EVERY token, including tokens in a longer prompt.
Stored parameters disconnected from the loss are not effective model capacity.
These mechanisms are checked on load, including exact frozen initial tensors.

Keep prompts and sampling settings equal. Defaults: sampling seed 17,
temperature 0.8, top-k 40, at most 128 output tokens, one CPU thread.
--temperature 0 uses argmax; --tokens accepts 1 to 1024. Generation masks the
beginning-of-document token and control-byte pieces other than tab/newline;
the likelihood evaluation does not apply this generation-only mask.

The exporter compares all tensors and graph/pooling buffers, a fixed-prefix
logit/state probe, and all 32 fixed-prompt continuations against the selected
source checkpoints. The release record separately states whether all eight
first-prompt examples passed fresh-archive CLI replay outside the repository.
Library versions or platforms can change numerical and stochastic output.
Hashes describe this supplied release; they do not independently authenticate
the original research. This verification does not rerun held-out likelihoods.

This is an inference release, not a training-resume archive. No optimizer,
training RNG, raw corpus or user conversation is included. Code and weights use
MIT; the imported graph retains CC BY 4.0 attribution. Corpus provenance is in
data/cards/wikitext2.json. Selection, results and protocols are in reports/ and
docs/. The graph is a computational subset, not an intact fly brain.

These are exploratory mechanism comparisons with two reused full references
and shared test articles, not independent new replications of those references.
Conditional article-bootstrap intervals do not establish general equivalence.
A benefit from recurrence would not establish a benefit from anatomical wiring;
the separate topology study addresses that question. The original comparison
archive includes GRU and transformer baselines.
'''


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8')


def file_record(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def package(root):
    root = Path(root).resolve()
    # No source checkpoint loading, sampling, or output writes before this gate.
    report = verified_report(root)
    torch.set_num_threads(1)
    identity_name = (REPORTS / 'identity.json').as_posix()
    selection_name = (REPORTS / 'selection.json').as_posix()
    summary_name = (REPORTS / 'summary.json').as_posix()
    binding_hashes = {name: sha256(root / name) for name in (identity_name, selection_name, summary_name)}
    identity = read_json(root / identity_name)
    selection = read_json(root / selection_name)
    if (root / RELEASE).exists() and read_json(root / RELEASE)['selection_sha256'] != report['selection_sha256']:
        raise ValueError('A different selected study already owns the inference release')
    lexicon = Lexicon(root / LEXICON)
    contents = {}; records = []; originals = {}
    for selected in selection['runs']:
        path, graph = root / selected['checkpoint'], root / selected['graph']
        if selected['reference']:
            original, saved = restore_original(path, graph, lexicon)
        else:
            original, saved = restore_control(path, graph, lexicon, run_binding(root, selected, identity))
        if saved['_file_sha256'] != selected['checkpoint_sha256'] or saved['step'] != selected['checkpoint_step']:
            raise ValueError('A selected source checkpoint changed during export')
        counts = parameter_counts(original)
        if any(selected[key] != value for key, value in counts.items()):
            raise ValueError('Selected source parameter counts changed')
        state_digest = state_hash(original)
        payload = dict(format=CHECKPOINT_FORMAT, control=selected['control'],
            model=original.state_dict(), config=asdict(original.config), step=saved['step'],
            run={key: saved['run'][key] for key in ('seed', 'graph_sha256', 'tokenizer_sha256')},
            source_checkpoint_sha256=saved['_file_sha256'], state_sha256=state_digest)
        buffer = io.BytesIO(); torch.save(payload, buffer)
        name = f'checkpoints/{selected["label"]}.pt'; contents[name] = buffer.getvalue()
        score = next(run['score'] for run in report['runs'] if run['label'] == selected['label'])
        records.append(dict(id=selected['label'], file=name, control=selected['control'],
            training_seed=selected['seed'], reference=selected['reference'], graph=selected['graph'],
            graph_sha256=sha256(graph), source_checkpoint_sha256=saved['_file_sha256'],
            state_sha256=state_digest, checkpoint_step=saved['step'], **counts,
            published_test_bits_per_byte=score['bits_per_byte']))
        originals[selected['label']] = original.eval()
    names = set(RUNTIME_FILES) | set(ASSETS) | {identity_name, selection_name, summary_name}
    contents.update({name: (root / name).read_bytes() for name in sorted(names)})
    contents['requirements-inference.txt'] = b'numpy==2.2.6\nscipy==1.13.1\ntokenizers==0.22.2\n'
    contents['README.md'] = README
    manifest = dict(format=BUNDLE_FORMAT, models=records, runtime_sources=list(RUNTIME_FILES),
        tokenizer=LEXICON.as_posix(), study_identity=identity_name, selection=selection_name,
        study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        python=sys.version, torch=str(torch.__version__),
        files={name: file_record(data) for name, data in sorted(contents.items())})
    passages = []; parity = []
    # Stage privately. A failed parity check cannot create a public archive.
    with tempfile.TemporaryDirectory(prefix='flm-core-export-') as temporary:
        destination = Path(temporary).resolve()
        for name, data in contents.items():
            path = destination / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        (destination / 'bundle.json').write_bytes(encoded(manifest))
        verify_bundle(destination)
        with torch.no_grad():
            probe = torch.tensor([[0] + lexicon.encode(PROMPTS[0])])
            for record in records:
                original = originals[record['id']]
                restored, restored_lexicon, _ = load_model(destination, record['id'])
                if any(not torch.equal(value, restored.state_dict()[name]) for name, value in original.state_dict().items()):
                    raise ValueError('An exported tensor or graph/pooling buffer changed')
                expected, expected_state = original(probe); actual, actual_state = restored(probe)
                if not torch.equal(expected, actual) or any(not torch.equal(a, b) for a, b in zip(expected_state, actual_state)):
                    raise ValueError('Exported logits or recurrent states differ')
                examples = []
                for prompt in PROMPTS:
                    expected = sample(original, lexicon, prompt)
                    actual = generate(restored, restored_lexicon, prompt)
                    if actual != expected:
                        raise ValueError('A fixed-prompt continuation changed during export')
                    examples.append(actual)
                passages.append(dict(condition=record['id'], checkpoint_sha256=record['source_checkpoint_sha256'], passages=examples))
                parity.append(dict(condition=record['id'], tensors_and_buffers_exact=True,
                    state_probe_prompt=PROMPTS[0], state_probe_tokens=probe.shape[1],
                    logits_and_states_exact_at_probe=True, fixed_prompt_continuations_exact=len(examples)))
        samples = dict(selection_sha256=report['selection_sha256'], settings=SAMPLING, models=passages,
            scope='Four pre-existing prompts at all eight selected checkpoints; unedited continuations, not factual answers or held-out scores')
        contents['samples.json'] = encoded(samples)
        manifest['files']['samples.json'] = file_record(contents['samples.json'])
        contents['bundle.json'] = encoded(manifest)
        (destination / 'samples.json').write_bytes(contents['samples.json'])
        (destination / 'bundle.json').write_bytes(contents['bundle.json'])
        verify_bundle(destination)
    if any(sha256(root / name) != value for name, value in binding_hashes.items()):
        raise ValueError('Study records changed during export')
    archive_path = root / ARCHIVE
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary_archive) as archive:
        if archive.testzip() is not None or any(archive.read(name) != data for name, data in contents.items()):
            raise ValueError('Computation inference archive changed its contents')
    temporary_archive.replace(archive_path)
    release = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive=ARCHIVE,
        archive_sha256=sha256(archive_path), bytes=archive_path.stat().st_size, payloads=len(contents),
        selection_sha256=report['selection_sha256'], study_identity_sha256=report['study_identity_sha256'],
        summary_sha256=binding_hashes[summary_name], models=records, parity=parity,
        python=sys.version, torch=str(torch.__version__), exporter_sha256=sha256(Path(__file__)),
        manifest_sha256=hashlib.sha256(contents['bundle.json']).hexdigest(), fresh_archive_cli_verified=False)
    for name in (RELEASE, PUBLIC_RELEASE):
        (root / name).write_bytes(encoded(release))
    (root / 'public/research/language-core-samples.json').write_bytes(encoded(samples))
    return release


if __name__ == '__main__':
    print(json.dumps(package(ROOT), indent=2))
