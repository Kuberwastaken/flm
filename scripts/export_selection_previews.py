"""Release all four measured KC candidates without opening held-out scores."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flm.model import Config, FLM
from flm.selection_language import load_catalog, load_case_graph
from flm.tokenizer import Lexicon

SELECTION = Path('reports/selection-language/mac-v1/study-selection.json')
IDENTITY = SELECTION.with_name('study-identity.json')
TOKENIZER = Path('data/tokenizers/babylm-2026-4096/tokenizer.json')
REPORT = Path('reports/selection-language/mac-v1/browser-preview-v1.json')

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')

def f32(value):
    return value.detach().cpu().numpy().astype('<f4')

def candidates():
    selection = read(ROOT/SELECTION)
    identity = read(ROOT/IDENTITY)
    assert selection['study_identity_sha256'] == sha(ROOT/IDENTITY)
    assert len(selection['conditions']) == 128 and selection['test_payloads_opened'] is False
    expected = {f'KCg-d-{side}-t5/candidate/measured/s{seed}' for side in ('L','R') for seed in (42,43)}
    chosen = [r for r in selection['conditions'] if r['condition']['label'] in expected]
    assert {r['condition']['label'] for r in chosen} == expected
    return chosen, {r['condition']['label']: r for r in identity['conditions']}

def package_id(row):
    side = row['condition']['label'].split('/')[0].split('-')[2].lower()
    return f'flm-kc-{side}-s{row["condition"]["seed"]}'

def fetch(rows):
    connection = read(ROOT/'work/mac-runtime-connection.json')
    for row in rows:
        relative = Path(row['checkpoint'])
        destination = (ROOT/relative).resolve()
        assert destination.is_relative_to(ROOT.resolve()) and relative.as_posix().startswith('runs/selection-language-mac-v1/')
        if destination.exists():
            assert sha(destination) == row['checkpoint_sha256']
            continue
        remote = connection['root'].rstrip('/')+'/'+relative.as_posix()
        data = subprocess.check_output(['tailscale','ssh',connection['user']+'@'+connection['host'], 'cat '+shlex.quote(remote)])
        assert hashlib.sha256(data).hexdigest() == row['checkpoint_sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print('Verified checkpoint transfer: '+package_id(row), flush=True)

def restore(row, registered, catalog):
    path = ROOT/row['checkpoint']
    assert sha(path) == row['checkpoint_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    declared = saved['declaration']
    assert saved['step'] == row['checkpoint_step'] == 3000
    assert declared['settings'] == registered['settings']
    assert declared['binding'] == dict(registered['base_binding'], study=dict(study_identity_sha256=sha(ROOT/IDENTITY)))
    assert declared['ordered_training_documents_sha256'] == registered['training_documents_sha256']
    assert saved['exposure'] == row['final_exposure']
    assert declared['method'] == 'bptt'
    for name, expected in declared['source_sha256'].items():
        assert sha(ROOT/'flm'/name) == expected, name
    graph, _ = load_case_graph(catalog, row['condition']['graph'])
    assert declared['config'] == registered['base_binding']['graph']['config']
    model = FLM(graph, Config(**declared['config']))
    initial = model.state_dict()
    assert initial.keys() == saved['model'].keys()
    for name, value in saved['model'].items():
        assert value.shape == initial[name].shape and value.dtype == initial[name].dtype
        assert torch.isfinite(value).all(), name
    for name, _ in model.named_buffers():
        assert torch.equal(initial[name], saved['model'][name]), name
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    return model, graph, saved

@torch.no_grad()
def reference(model, lexicon):
    tokens = ([0]+lexicon.encode('The little bird returned to the garden. A conversation begins with a question.\n')*20)[:130]
    state = None
    cases = []
    for i, token in enumerate(tokens):
        logits, state = model(torch.tensor([[token]]), state)
        if i in (0,11,96,129):
            cases.append(dict(length=i+1, logits=f32(logits[0,-1]).tolist(), h=f32(state[0][0]).tolist(), slow=f32(state[1][0]).tolist()))
    return dict(tokens=tokens, cases=cases, torch=str(torch.__version__),
        scope='Complete logits and actual fast/slow state from restored checkpoint; fixed diagnostic prose, no held-out input.')

@torch.no_grad()
def export(row, registered, catalog, lexicon, annotations, context):
    model, graph, saved = restore(row, registered, catalog)
    key = package_id(row)
    directory = ROOT/'public/models'/key
    if directory.exists():
        raise ValueError('Preserve existing export: '+key)
    directory.mkdir(parents=True)
    c = model.config
    w, alpha, beta, gain = model.constants()
    rows, cols = model.row.numpy(), model.col.numpy()
    order = np.lexsort((cols, rows))
    values = (w.to_dense() if w.is_sparse else w)[model.row,model.col]*gain
    arrays = dict(embedding=f32(model.embedding.weight), input_weight=f32(model.input.weight), input_bias=f32(model.input.bias),
        offsets=np.r_[0,np.cumsum(np.bincount(rows,minlength=c.neurons))].astype('<u4'), sources=cols[order].astype('<u4'),
        weights=f32(values)[order], alpha=f32(alpha), beta=f32(beta), pool=model.pool_index.numpy().astype('<u4'),
        pool_sizes=f32(model.pool_sizes), norm_weight=f32(model.norm.weight), norm_bias=f32(model.norm.bias),
        projection_weight=f32(model.readout.weight), projection_bias=f32(model.readout.bias), readout_bias=f32(model.output_bias))
    binary = bytearray()
    index = {}
    for name, array in arrays.items():
        index[name] = dict(offset=len(binary), length=array.size, shape=list(array.shape), dtype='uint32' if array.dtype.kind=='u' else 'float32')
        binary.extend(array.tobytes(order='C'))
    (directory/'weights.bin').write_bytes(binary)
    group = registered['base_binding']['graph']['candidate']
    cell_types = [annotations[group,str(int(body))]['type'] for body in graph['body_ids']]
    seed_count = sum(t=='KCg-d' for t in cell_types)
    assert seed_count == (99 if c.neurons==487 else 107)
    anatomy = dict(positions=[p.tolist() if np.isfinite(p).all() else None for p in graph['positions']],
        body_ids=[str(int(x)) for x in graph['body_ids']], cell_types=cell_types, source_sign=graph['source_sign'].tolist(),
        coordinate_units='8 nm voxels', context_positions=context['context_positions'], context_note=context['context_note'])
    write(directory/'anatomy.json', anatomy)
    shutil.copyfile(ROOT/TOKENIZER.with_name('browser-tokenizer.json'), directory/'tokenizer.json')
    label = f'FLM · KC {"left" if c.neurons==487 else "right"} · seed {row["condition"]["seed"]} · Preview'
    details = dict(format='flm-browser-v2', architecture='flm', variant='flm', name=label, model_id=key,
        dataset='BabyLM 2026 English 10M · KC selection study', neurons=c.neurons, pools=c.pools,
        features=c.embedding, embedding=c.embedding, vocabulary=c.vocabulary, bos=0, eos=1, arrays=index,
        checkpoint_step=saved['step'], checkpoint_sha256=row['checkpoint_sha256'],
        weights_sha256=sha(directory/'weights.bin'), weights_bytes=len(binary), anatomy_sha256=sha(directory/'anatomy.json'),
        source_graph_sha256=registered['base_binding']['graph']['graph_sha256'],
        tokenizer_sha256=lexicon.sha256, browser_tokenizer_sha256=sha(directory/'tokenizer.json'),
        trained_parameters=registered['trainable_parameters'], retained_edges=len(rows), source_neurons=166700,
        norm_epsilon=model.norm.eps, selection_validation_bpb=row['validation_bits_per_byte'],
        selection_manifest_sha256=sha(ROOT/SELECTION), study_identity_sha256=sha(ROOT/IDENTITY), condition=row['condition'],
        training=dict(method='bptt', settings=registered['settings'], final_exposure=row['final_exposure'], origin='Fresh native Mac initialization'),
        publication_status='Research preview; complete held-out evaluation pending.', preview=True, instruction_tuned=False,
        literal_kcg_d_neurons=seed_count, publication_guide='/research/selection-previews.md',
        description='Completed 3,000-update measured KC candidate, selected on validation before held-out scoring. Both sides and seeds are released.',
        capability='Base next-token predictor; not instruction tuned. This is a circuit-selection experiment, not a larger chat model.',
        license='MIT implementation; CC BY 4.0 anatomy; BabyLM components retain their source rights. No raw training corpus redistributed.')
    assert read(directory/'tokenizer.json')['tokenizer_sha256'] == lexicon.sha256
    write(directory/'model.json', details)
    write(directory/'parity.json', reference(model, lexicon))
    return dict(path=key, label=label, name='FLM', lexical=True, architecture='flm', study='selection-mac-v1', scale='10m',
        seed=row['condition']['seed'], validation_bpb=row['validation_bits_per_byte'], preview=True,
        manifest_sha256=sha(directory/'model.json'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--reference-output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, registered = candidates()
    if args.fetch:
        fetch(rows)
    catalog = load_catalog(ROOT)
    lexicon = Lexicon(ROOT/TOKENIZER)
    if args.reference_output:
        for row in rows:
            model, _, _ = restore(row, registered[row['condition']['label']], catalog)
            write(args.reference_output/(package_id(row)+'.json'), reference(model, lexicon))
        return
    with (ROOT/'reports/circuit-selection/members.csv').open(encoding='utf8', newline='') as stream:
        annotations = {(r['candidate'],r['body_id']):r for r in csv.DictReader(stream)}
    context = read(ROOT/'public/models/flm-wikitext/anatomy.json')
    packages = {}
    for row in rows:
        key = package_id(row)
        packages[key] = export(row, registered[row['condition']['label']], catalog, lexicon, annotations, context)
        print('Exported '+key, flush=True)
    result = read(ROOT/'web/model-catalog.json')
    assert not (packages.keys() & result['models'].keys())
    result['models'].update(packages)
    result['inputs'][SELECTION.as_posix()] = sha(ROOT/SELECTION)
    result['preview_policy'] = 'All four measured KC candidates, both sides and seeds. Validation-selected; held-out study pending. Excluded from primary automatic-default ranking.'
    write(ROOT/'web/model-catalog.json', result)
    write(ROOT/'public/models/catalog.json', result)
    write(ROOT/REPORT, dict(format='flm-selection-browser-preview-v1', selection_sha256=sha(ROOT/SELECTION),
        study_identity_sha256=sha(ROOT/IDENTITY), exporter_sha256=sha(Path(__file__)),
        anatomy_members_sha256=sha(ROOT/'reports/circuit-selection/members.csv'), packages=packages,
        checkpoint_selection='All four predeclared measured candidates; each checkpoint chosen by frozen validation protocol.',
        unchanged_default=result['default_flm'], heldout_status='Complete study pending; partial test scores not opened by exporter.',
        scope='Checkpoint and graph identity checks plus browser export. Inference parity and browser release review recorded separately.'))

if __name__ == '__main__':
    main()
