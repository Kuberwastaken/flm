"""Plot all four frozen 1024-neuron graphs in the same anatomical node order."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'public/research/figures'


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from flm.language_structure import collect
    report, graphs, _ = collect(ROOT)
    saved = json.loads((ROOT / 'public/research/language-topology-structure.json').read_text(encoding='utf8'))
    if report != saved:
        raise ValueError('Regenerate the structural audit before its figure')
    background = '#faf8f5'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'path', 'figure.facecolor': background})
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, graph, record in zip(axes.flat, graphs, report['graphs']):
        matrix = np.zeros((1024, 1024), dtype=np.int8)
        matrix[graph['row'], graph['col']] = np.sign(graph['weight']).astype(np.int8)
        ax.imshow(matrix, cmap=ListedColormap(['#497569', background, '#a74c20']),
                  norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3), interpolation='nearest', origin='upper')
        ax.set(title=record['label'], xlabel='Presynaptic index', ylabel='Postsynaptic index',
               xticks=[0, 256, 512, 768, 1023], yticks=[0, 256, 512, 768, 1023])
        ax.tick_params(labelsize=9)
    fig.suptitle('Language topology controls · 1,024 neurons, 76,130 edges each', y=.98, fontsize=14)
    fig.text(.5, .022, 'Orange: positive edge    Green: negative edge    Blank: no edge\n'
             'Shared node order; colors show signs, not weight magnitudes or neural activity.',
             ha='center', va='bottom', fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.12, wspace=.25, hspace=.3)
    fig.savefig(FIGURES / 'language-topology-matrices.png', dpi=320)
    fig.savefig(FIGURES / 'language-topology-matrices.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
