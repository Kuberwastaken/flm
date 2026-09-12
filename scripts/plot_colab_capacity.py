"""Static arithmetic figure; do not mistake planned capacity for measured fit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
card = json.loads((root / 'reports/colab-notebooks/capacity-plan-v1.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'flm-colab-capacity-v1'})
fig, ax = plt.subplots(figsize=(10.5, 4.8))
fig.subplots_adjust(left=.14, right=.94, top=.80, bottom=.28)
colors = ['#94b6d7', '#315d86', '#d69742', '#6f7782']
for i, row in enumerate(card['presets']):
    left = 0
    for j, (label, value) in enumerate(row['parameter_groups'].items()):
        ax.barh(i, value/1e6, left=left, color=colors[j], label=label if i == 0 else None, height=.48)
        left += value/1e6
    ax.text(left+2, i, f'{row["parameters"]/1e6:.2f}M', va='center', fontsize=11)
ax.set_yticks([0, 1], ['150M target\nwidth 728', '300M target\nwidth 1,600'])
ax.invert_yaxis()
ax.set_xlim(0, 340)
ax.set_xlabel('Calculated trainable parameters (millions)')
ax.set_title('Same 166,700 neurons. Larger learned language interfaces.', loc='left', pad=18, weight='bold')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.28), ncol=4, frameon=False, fontsize=9)
fig.text(.14, .035, '24.47M signed edges in each preset. GPU memory fit, throughput and quality remain unmeasured.', fontsize=9, color='#4a4f55')
fig.savefig(root / 'docs/figures/colab-capacity.svg', facecolor='white', metadata={'Date': None})
fig.savefig(root / 'work/colab-capacity.png', facecolor='white', dpi=150)

svg = root / 'docs/figures/colab-capacity.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines()) + '\n', encoding='utf8')
