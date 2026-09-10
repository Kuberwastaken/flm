"""Generate scientific choice-learning figures from every recorded training seed."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'public/research/figures'
COLORS = dict(bptt='#34322d', reservoir='#497569', eligibility='#a74c20', instantaneous='#77716b', reward='#666277')


def main():
    data = json.loads((ROOT / 'public/research/choice-learning.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'svg.fonttype':'path', 'figure.facecolor':'#faf8f5', 'axes.facecolor':'#faf8f5',
        'axes.labelcolor':'#292820', 'text.color':'#292820', 'xtick.color':'#625e55', 'ytick.color':'#625e55'})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for delay in (8, 48):
        fig, ax = plt.subplots(figsize=(6.8, 3.8), layout='constrained')
        ax.axvspan(300, 600, color='#ddd7cd', alpha=.65)
        for method, label in data['methods'].items():
            rows = [r for r in data['curves'] if r['method'] == method and r['delay'] == delay]
            x = np.array([r['step'] for r in rows]); y = np.array([r['original_accuracy'] for r in rows])
            ax.fill_between(x, y.min(1), y.max(1), color=COLORS[method], alpha=.09, linewidth=0)
            ax.plot(x, y.mean(1), color=COLORS[method], label=label, linewidth=1.8,
                    linestyle='--' if method == 'instantaneous' else '-')
        ax.set(xlim=(0, 900), ylim=(-.03, 1.08), xlabel='Training updates', ylabel='Accuracy under original mapping',
               title=f'Delay {delay} frames' + (' · beyond training delays' if delay == 48 else ' · within training range'))
        ax.set_xticks([0, 300, 600, 900]); ax.set_yticks([0, .25, .5, .75, 1.], ['0%', '25%', '50%', '75%', '100%'])
        ax.text(450, 1.035, 'Reversed training', ha='center', fontsize=8)
        ax.grid(axis='y', color='#d6d0c5', linewidth=.6)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3, frameon=False, fontsize=8)
        fig.savefig(OUTPUT / f'choice-reversal-delay{delay}.svg'); fig.savefig(OUTPUT / f'choice-reversal-delay{delay}.png', dpi=160)
        plt.close(fig)
    with (OUTPUT / 'choice-learning.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['method','training_seed','step','delay_frames','current_mapping','original_accuracy','current_accuracy','current_cross_entropy'])
        for row in data['curves']:
            for i, seed in enumerate(row['seeds']):
                writer.writerow([row['method'], seed, row['step'], row['delay'], row['current_mapping'], row['original_accuracy'][i], row['current_accuracy'][i], row['current_cross_entropy'][i]])


if __name__ == '__main__': main()
