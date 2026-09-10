"""Generate every predeclared BabyLM continuation from frozen checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from .babylm_test import freeze_selection, read_json, restore_selected, OUTPUT, TOKENIZER
from .generation_audit import SETTINGS, generate
from .provenance import sha256, write_json
from .tokenizer import Lexicon


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1: p.error('Positive threads required')
    selection = freeze_selection(); frozen = OUTPUT / 'selection.json'
    if not frozen.exists() or read_json(frozen) != selection: raise ValueError('Run the test selection freeze first')
    prompts_path = Path('data/prompts/babylm-original.json'); panel = read_json(prompts_path)['prompts']
    if len(panel) != 12 or len({p['id'] for p in panel}) != len(panel): raise ValueError('Invalid original prompt panel')
    identity = dict(selection_sha256=sha256(frozen), prompts_sha256=sha256(prompts_path), settings=SETTINGS,
        tokenizer_sha256=selection['tokenizer_sha256'], torch_version=str(torch.__version__), threads=a.threads,
        sampler_source_sha256=sha256(Path(__file__).with_name('generation_audit.py')),
        model_source_sha256={name: sha256(Path(__file__).with_name(name)) for name in ('model.py', 'baselines.py', 'language_train.py')})
    lexicon = Lexicon(TOKENIZER); torch.set_num_threads(a.threads); models = []
    for selected in selection['runs']:
        path = OUTPUT / f'samples-{selected["scale"]}-{selected["variant"]}-s{selected["seed"]}.json'
        expected = dict(identity=identity, selected=selected)
        if path.exists():
            record = read_json(path)
            if any(record.get(k) != v for k, v in expected.items()): raise ValueError('Another sampling experiment is already recorded')
            wanted = [(p['id'], seed) for p in panel for seed in SETTINGS['seeds']]
            if [(p['prompt_id'], p['sampling_seed']) for p in record['passages']] != wanted:
                raise ValueError('Saved continuation panel is incomplete or reordered')
        else:
            model, saved = restore_selected(selected, selection, lexicon)
            passages = [dict(prompt_id=p['id'], group=p['group'], **generate(model, lexicon, p['text'], seed))
                        for p in panel for seed in SETTINGS['seeds']]
            record = dict(**expected, checkpoint_step=saved['step'], passages=passages)
            write_json(path, record)
        models.append(record); print(f'Complete: {path}', flush=True)
    write_json(OUTPUT / 'samples.json', dict(identity=identity, models=models,
        note='All 288 fixed-prompt continuations are unedited. Repetition statistics describe token patterns, not truth or semantic quality. Prompt groups do not establish chatbot or domain capability.'))


if __name__ == '__main__': main()
