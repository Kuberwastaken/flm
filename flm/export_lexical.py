"""Package a tied-embedding lexical FLM without expanding every token into 1,024 drives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .language_train import restore
from .model import FLM, load_graph
from .provenance import sha256, write_json
from .tokenizer import Lexicon


@torch.no_grad()
def export(checkpoint, graph_path, tokenizer_path, output, anatomy_source=None):
    torch.set_num_threads(2)
    lexicon = Lexicon(tokenizer_path)
    model, saved = restore(checkpoint, graph_path, lexicon)
    if not isinstance(model, FLM) or not model.config.tied_readout:
        raise ValueError('The lexical browser package requires a tied-readout FLM')
    model.eval(); graph = load_graph(graph_path)
    n, p, d = model.config.neurons, model.config.pools, model.config.embedding
    w, alpha, beta, gain = model.constants()
    weights = (w.to_dense() if w.is_sparse else w)[model.row, model.col] * gain
    row, col = model.row.numpy(), model.col.numpy(); order = np.lexsort((col, row))
    f32 = lambda x: x.detach().cpu().numpy().astype('<f4')
    arrays = dict(embedding=f32(model.embedding.weight), input_weight=f32(model.input.weight),
        input_bias=f32(model.input.bias), offsets=np.r_[0, np.cumsum(np.bincount(row, minlength=n))].astype('<u4'),
        sources=col[order].astype('<u4'), weights=f32(weights)[order], alpha=f32(alpha), beta=f32(beta),
        pool=model.pool_index.numpy().astype('<u4'), pool_sizes=f32(model.pool_sizes),
        norm_weight=f32(model.norm.weight), norm_bias=f32(model.norm.bias),
        projection_weight=f32(model.readout.weight), projection_bias=f32(model.readout.bias),
        readout_bias=f32(model.output_bias))
    output.mkdir(parents=True, exist_ok=True); binary = bytearray(); index = {}
    for name, array in arrays.items():
        index[name] = dict(offset=len(binary), length=array.size, shape=list(array.shape),
                           dtype='uint32' if array.dtype.kind == 'u' else 'float32')
        binary.extend(array.tobytes(order='C'))
    (output / 'weights.bin').write_bytes(binary)
    card = json.loads(graph_path.with_name('graph-card.json').read_text(encoding='utf8'))
    anatomy = dict(positions=[pos.tolist() if np.isfinite(pos).all() else None for pos in graph['positions']],
        body_ids=[str(x) for x in graph['body_ids']], cell_types=card['cell_types'],
        source_sign=graph['source_sign'].tolist(), coordinate_units='8 nm voxels')
    if anatomy_source:
        positions = np.load(anatomy_source, allow_pickle=False)
        available = positions[np.isfinite(positions).all(axis=1)]
        indices = np.linspace(0, len(available) - 1, min(24000, len(available)), dtype=int)
        anatomy.update(context_positions=available[indices].tolist(),
            context_note='Downsampled anatomical reference only; background points do not display simulated activity')
    write_json(output / 'anatomy.json', anatomy)
    browser_tokenizer = tokenizer_path.with_name('browser-tokenizer.json')
    tokenizer = json.loads(browser_tokenizer.read_text(encoding='utf8'))
    if tokenizer['tokenizer_sha256'] != lexicon.sha256: raise ValueError('Browser tokenizer provenance mismatch')
    shutil.copyfile(browser_tokenizer, output / 'tokenizer.json')
    config = dict(format='flm-browser-v2', name='FLM WikiText', dataset='WikiText-2 raw',
        model_id=f'flm-central-{n}-wikitext2-s{saved["run"]["seed"]}',
        neurons=n, pools=p, features=d, embedding=d, vocabulary=lexicon.vocabulary, bos=0, eos=1,
        arrays=index, variant=model.config.variant, checkpoint_step=saved['step'],
        checkpoint_sha256=saved['_file_sha256'], weights_sha256=sha256(output / 'weights.bin'),
        weights_bytes=len(binary), anatomy_sha256=sha256(output / 'anatomy.json'),
        tokenizer_sha256=lexicon.sha256, browser_tokenizer_sha256=sha256(output / 'tokenizer.json'),
        source_graph_sha256=sha256(graph_path), trained_parameters=saved['run']['parameter_card']['trainable_parameters'],
        retained_edges=len(row), source_neurons=card['source_neurons'], training=saved['run'],
        norm_epsilon=float(model.norm.eps),
        description='A compact fly-wired next-token predictor trained from scratch on WikiText-2 raw.',
        capability='Experimental text completion; not instruction tuned or a reliable question-answering system.',
        license='MIT original code; CC BY 4.0 anatomy; WikiText source attribution and license notes in the dataset card')
    write_json(output / 'model.json', config)
    tokens = [0] + lexicon.encode('The history of science includes many unexpected discoveries.\n')
    logits, state, features = model(torch.tensor([tokens]), return_features=True)
    write_json(output / 'parity.json', dict(tokens=tokens, logits=f32(logits[0, -1]).tolist(),
        h=f32(state[0][0]).tolist(), slow=f32(state[1][0]).tolist(), features=f32(features[0, -1]).tolist()))
    print(json.dumps(dict(model_id=config['model_id'], step=saved['step'], bytes=len(binary))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--graph', type=Path, default=Path('data/graphs/central-1024/graph.npz'))
    p.add_argument('--tokenizer', type=Path, default=Path('data/tokenizers/wikitext2-4096/tokenizer.json'))
    p.add_argument('--output', type=Path, default=Path('public/models/flm-wikitext'))
    p.add_argument('--anatomy-source', type=Path)
    a = p.parse_args(); export(a.checkpoint, a.graph, a.tokenizer, a.output, a.anatomy_source)


if __name__ == '__main__': main()
