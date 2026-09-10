"""Export every final topology model; never publish a partial fitted selection."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flm.inference import generate, state_hash
from flm.language_report import PROMPTS, SAMPLING, sample
from flm.language_train import restore
from flm.provenance import sha256
from flm.tokenizer import Lexicon
from flm.topology_inference import RUNTIME_FILES, load_model, verify_bundle
from scripts.language_topology_report import verified_report


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf8')


def package(root):
    torch.set_num_threads(1)
    report = verified_report(root)
    selection_path = root/'reports/language-topology/selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    lexicon_path = 'data/tokenizers/wikitext2-4096/tokenizer.json'
    lexicon = Lexicon(root/lexicon_path)
    destination = root/'output/inference/language-topology'
    if (destination/'bundle.json').exists():
        existing = json.loads((destination/'bundle.json').read_text(encoding='utf8'))
        if existing['selection_sha256'] != report['selection_sha256']:
            raise ValueError('A different selected study already owns the output directory')
    contents = {}; records = []; originals = {}
    for selected in selection['runs']:
        original, saved = restore(root/selected['checkpoint'], root/selected['graph'], lexicon)
        if saved['_file_sha256'] != selected['checkpoint_sha256']:
            raise ValueError('A selected checkpoint changed during export')
        state_digest = state_hash(original)
        payload = dict(format='flm-inference-checkpoint-v1', model=original.state_dict(), config=asdict(original.config),
            step=saved['step'], run={k:saved['run'][k] for k in ('seed', 'graph_sha256', 'tokenizer_sha256')},
            source_checkpoint_sha256=saved['_file_sha256'], state_sha256=state_digest)
        buffer = io.BytesIO(); torch.save(payload, buffer)
        name = f'checkpoints/{selected["label"]}.pt'; contents[name] = buffer.getvalue()
        score = next(run['score'] for run in report['runs'] if run['label'] == selected['label'])
        records.append(dict(id=selected['label'], file=name, variant=selected['variant'],
            training_seed=selected['seed'], graph_seed=selected['graph_seed'], topology=selected['topology'],
            reference=selected['reference'], graph=selected['graph'], graph_sha256=sha256(root/selected['graph']),
            source_checkpoint_sha256=saved['_file_sha256'], state_sha256=state_digest,
            checkpoint_step=saved['step'], parameters=selected['parameters'], published_test_bits_per_byte=score['bits_per_byte']))
        originals[selected['label']] = original.eval()
    names = set(RUNTIME_FILES) | {lexicon_path, 'data/cards/wikitext2.json',
        'data/tokenizers/wikitext2-4096/tokenizer-card.json', 'docs/LANGUAGE-TOPOLOGY-PROTOCOL.md',
        'docs/WIKITEXT-PROTOCOL.md', 'LICENSE', 'licenses/CC-BY-4.0.txt', 'licenses/DATA-ATTRIBUTION.md',
        'reports/language-topology/identity.json', 'reports/language-topology/selection.json',
        'public/research/language-topology-results.json', 'scripts/package_topology_inference.py',
        'flm/language_report.py'}
    for record in records:
        names.add(record['graph']); names.add(str(Path(record['graph']).with_name('graph-card.json')).replace('\\', '/'))
    contents.update({name:(root/name).read_bytes() for name in sorted(names)})
    contents['requirements-inference.txt'] = b'numpy==2.2.6\nscipy==1.13.1\ntokenizers==0.22.2\n'
    contents['README.md'] = b'''# FLM: run the final language wiring controls

FLM by Kuber Mehta. This archive contains the ten final selected models: measured
wiring, three independently rewired graphs, and a retrained no-slow-state control,
each at training seeds 42 and 43. The full study gate precedes export. It uses a
1,024-neuron subset, not a complete brain. These are small text-continuation
models without instruction tuning. Generated text is not a factual answer.

From a Python environment in this extracted directory, install the same CPU
runtime as the original comparison release:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python -X utf8 -m flm.topology_inference --condition measured-s42 --prompt "The history of science"
python -X utf8 -m flm.topology_inference --condition null101-s42 --prompt "The history of science"
python -X utf8 -m flm.topology_inference --condition no-slow-s42 --prompt "The history of science"
```

Other graph identifiers are null103 and null107; replace s42 with s43 for the
second training initialization. Keep the prompt and sampling settings equal.
Defaults: sampling seed 17, temperature 0.8, top-k 40, up to 128 output tokens,
one CPU thread. --temperature 0 selects greedily. --tokens accepts 1 to 1024.
The beginning-of-document token and control-byte pieces other than tab/newline
are masked only during generation, not in the published likelihood scores.

Each checkpoint is bound to its selected graph, tokenizer and slow-state setting.
Changing a file, graph, runtime, or selected condition causes verification to
fail. The command returns the model identity, sampling settings, output token IDs
and decoded text. Repetition and broken continuations are retained. The exporter
checks every tensor, one fixed-prefix logit/state probe per model, and all forty
fixed-prompt continuations exactly in its recorded environment. Different numerical library
versions or platforms can change stochastic output; see the release record for
the environment that was actually checked.

This is an inference release, not a training-resume archive. Optimizer state,
training RNG, raw corpora and user conversations are omitted. Original code and
weights use MIT; imported graphs retain CC BY 4.0 attribution. Corpus source and
license information are in data/cards/wikitext2.json. File hashes in bundle.json
describe this supplied release; they are not independent authentication of the
original research. The two measured references are reused, not new replications.
Test reports and protocols retain the study's limitations and conditional
article-bootstrap interpretation. The older six-model comparison download is
separate and includes the GRU and transformer baselines.
'''
    manifest = dict(format='flm-language-topology-inference-v1', models=records,
        runtime_sources=list(RUNTIME_FILES), tokenizer=lexicon_path,
        study_identity='reports/language-topology/identity.json', selection='reports/language-topology/selection.json',
        study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        python=sys.version, torch=str(torch.__version__),
        files={name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(contents.items())})
    for name, data in contents.items():
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (destination/'bundle.json').write_bytes(encoded(manifest))
    verify_bundle(destination)
    passages = []; parity = []
    with torch.no_grad():
        probe = torch.tensor([[0]+lexicon.encode(PROMPTS[0])])
        for record in records:
            original = originals[record['id']]
            restored, restored_lexicon, _ = load_model(destination, record['id'])
            if any(not torch.equal(value, restored.state_dict()[name]) for name, value in original.state_dict().items()):
                raise ValueError('A released tensor or graph buffer changed')
            expected, expected_state = original(probe); actual, state = restored(probe)
            if not torch.equal(expected, actual) or any(not torch.equal(a, b) for a, b in zip(expected_state, state)):
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
        scope='Four existing original prompts at the final selected checkpoints; unedited examples, not factual answers or a new held-out test')
    contents['samples.json'] = encoded(samples)
    manifest['files']['samples.json'] = dict(bytes=len(contents['samples.json']), sha256=hashlib.sha256(contents['samples.json']).hexdigest())
    contents['bundle.json'] = encoded(manifest)
    (destination/'samples.json').write_bytes(contents['samples.json'])
    (destination/'bundle.json').write_bytes(contents['bundle.json'])
    verify_bundle(destination)
    archive_path = root/'public/research/language-topology-inference.zip'
    temporary = archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or any(archive.read(name) != data for name, data in contents.items()):
            raise ValueError('Topology inference archive changed its contents')
    temporary.replace(archive_path)
    release = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive=archive_path.relative_to(root).as_posix(),
        archive_sha256=sha256(archive_path), bytes=archive_path.stat().st_size, payloads=len(contents),
        selection_sha256=report['selection_sha256'], study_identity_sha256=report['study_identity_sha256'],
        models=records, parity=parity, python=sys.version, torch=str(torch.__version__),
        manifest_sha256=hashlib.sha256(contents['bundle.json']).hexdigest(), fresh_archive_cli_verified=False)
    (root/'reports/language-topology/inference-release.json').write_bytes(encoded(release))
    (root/'public/research/language-topology-inference-release.json').write_bytes(encoded(release))
    (root/'public/research/language-topology-samples.json').write_bytes(encoded(samples))
    return release


if __name__ == '__main__':
    print(json.dumps(package(ROOT), indent=2))
