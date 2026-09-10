"""Release all six already-evaluated WikiText models with exact tensor checks."""
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
from flm.inference import RUNTIME_FILES, state_hash, load_model, generate, verify_bundle
from flm.language_report import PROMPTS, SAMPLING, sample
from flm.language_train import restore
from flm.provenance import sha256, write_json
from flm.tokenizer import Lexicon


def main():
    torch.set_num_threads(1)
    source = ROOT / 'reports/wikitext2/selection.json'
    selection = json.loads(source.read_text(encoding='utf8'))
    expected = {(variant, seed) for seed in (42, 43) for variant in ('flm', 'gru', 'transformer')}
    if {(r['variant'], r['seed']) for r in selection['runs']} != expected or len(selection['runs']) != 6:
        raise ValueError('The complete published six-model selection is required')
    destination = ROOT / 'output/inference/wikitext2'
    destination.mkdir(parents=True, exist_ok=True)
    graph = 'data/graphs/central-1024/graph.npz'
    tokenizer = 'data/tokenizers/wikitext2-4096/tokenizer.json'
    if sha256(ROOT / graph) != selection['protocol']['graph_sha256'] or sha256(ROOT / tokenizer) != selection['protocol']['tokenizer_sha256']:
        raise ValueError('The original graph or tokenizer changed')
    lexicon = Lexicon(ROOT / tokenizer); models = []; contents = {}
    original_models = {}
    for selected in selection['runs']:
        identity = f'{selected["variant"]}-s{selected["seed"]}'
        checkpoint = ROOT / selected['checkpoint'].replace('\\', '/')
        model, saved = restore(checkpoint, ROOT / graph, lexicon)
        test = json.loads((ROOT / f'reports/wikitext2/test-{identity}.json').read_text(encoding='utf8'))
        if (saved['_file_sha256'] != selected['checkpoint_sha256'] or
                test['checkpoint_sha256'] != saved['_file_sha256'] or test['checkpoint_step'] != saved['step'] or
                test['variant'] != model.config.variant or saved['run']['seed'] != selected['seed'] or
                saved['run']['protocol'] != selection['protocol']):
            raise ValueError(f'Published checkpoint identity changed: {identity}')
        digest = state_hash(model)
        payload = dict(format='flm-inference-checkpoint-v1', model=model.state_dict(),
                       config=asdict(model.config), step=saved['step'],
                       run={key: saved['run'][key] for key in ('seed', 'graph_sha256', 'tokenizer_sha256')},
                       source_checkpoint_sha256=saved['_file_sha256'], state_sha256=digest)
        buffer = io.BytesIO(); torch.save(payload, buffer)
        name = f'checkpoints/{identity}.pt'; contents[name] = buffer.getvalue()
        models.append(dict(id=identity, variant=selected['variant'], training_seed=selected['seed'],
                           file=name, checkpoint_step=saved['step'], source_checkpoint_sha256=saved['_file_sha256'],
                           state_sha256=digest, parameters=model.parameter_card()['trainable_parameters'],
                           published_test_bits_per_byte=test['score']['bits_per_byte']))
        original_models[identity] = model
    assets = [*RUNTIME_FILES, graph, tokenizer, 'data/graphs/central-1024/graph-card.json',
              'data/cards/wikitext2.json', 'data/tokenizers/wikitext2-4096/tokenizer-card.json',
              'docs/WIKITEXT-PROTOCOL.md', 'docs/INFERENCE-BUNDLE.md', 'LICENSE',
              'licenses/CC-BY-4.0.txt', 'licenses/DATA-ATTRIBUTION.md',
              'public/research/test-results.json', 'reports/wikitext2/selection.json']
    contents.update({name: (ROOT / name).read_bytes() for name in assets})
    contents['README.md'] = (ROOT / 'docs/INFERENCE-BUNDLE.md').read_bytes()
    contents['requirements-inference.txt'] = b'numpy==2.2.6\nscipy==1.13.1\ntokenizers==0.22.2\n'
    manifest = dict(format='flm-wikitext-inference-v1', graph=graph, tokenizer=tokenizer,
                    runtime_sources=list(RUNTIME_FILES), models=models, selection_sha256=sha256(source),
                    torch='2.8.0 CPU', python='Verified with Python 3.10.11',
                    files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                           for name, data in sorted(contents.items())},
                    scope='Already published WikiText FLM/GRU/transformer comparison; topology controls and BabyLM excluded.',
                    checkpoint_format='Model tensors and fixed buffers only; derived from the selected original checkpoint. No optimizer, training RNG, corpus or user conversation data.')
    for name, data in contents.items():
        path = destination / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    write_json(destination / 'bundle.json', manifest)
    verify_bundle(destination)
    parity = []
    for record in models:
        original = original_models[record['id']]
        restored, restored_lexicon, _ = load_model(destination, record['id'])
        if any(not torch.equal(value, restored.state_dict()[name]) for name, value in original.state_dict().items()):
            raise ValueError('A released tensor or fixed buffer changed')
        previous = json.loads((ROOT / f'public/research/samples-006000-s{record["training_seed"]}.json').read_text(encoding='utf8'))
        published = next(row for row in previous['models'] if row['variant'] == record['variant'])
        if published['step'] != record['checkpoint_step']:
            raise ValueError('Published fixed-prompt checkpoint differs from this release')
        for prompt in PROMPTS:
            expected_passage = next(row for row in published['passages'] if row['prompt'] == prompt)
            actual = generate(restored, restored_lexicon, prompt)
            if actual != expected_passage or actual != sample(original, lexicon, prompt):
                raise ValueError(f'Published sampling replay changed: {record["id"]}: {prompt}')
        parity.append(dict(model=record['id'], all_tensors_and_buffers_exact=True,
                           published_continuations_reproduced=len(PROMPTS), settings=SAMPLING))
    contents['bundle.json'] = (destination / 'bundle.json').read_bytes()
    archive_path = ROOT / 'public/research/wikitext2-inference.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Inference archive failed CRC validation')
        for name, data in contents.items():
            if archive.read(name) != data:
                raise ValueError(f'Inference archive changed {name}')
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive=archive_path.relative_to(ROOT).as_posix(),
                  archive_sha256=sha256(archive_path), bytes=archive_path.stat().st_size,
                  models=models, tensor_and_generation_parity=parity,
                  manifest_sha256=sha256(destination / 'bundle.json'), generator_sha256=sha256(Path(__file__)),
                  file_count=len(contents), source_selection_sha256=sha256(source))
    write_json(ROOT / 'reports/wikitext2/inference-release.json', report)
    write_json(ROOT / 'public/research/inference-release.json', report)
    print(json.dumps(dict(models=len(models), reproduced_continuations=24,
                          bytes=report['bytes'], sha256=report['archive_sha256']), indent=2))


if __name__ == '__main__':
    main()
