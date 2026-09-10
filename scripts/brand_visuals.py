"""Create FLM's share image from the actual anatomical coordinates, without synthetic activity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work/brand'; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'public/brand'; OUT.mkdir(parents=True, exist_ok=True)
fonts = {}
for weight in (400, 600):
    source = ROOT / f'node_modules/@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-{weight}-normal.woff'
    font = TTFont(source); font.flavor = None; font.save(WORK / f'plex-{weight}.ttf')
    fonts[weight] = FontProperties(fname=WORK / f'plex-{weight}.ttf')

anatomy_path = ROOT / 'public/models/flm-wikitext/anatomy.json'
anatomy = json.loads(anatomy_path.read_text(encoding='utf8'))
context = np.asarray(anatomy['context_positions'])
context = context[(context[:, 2] >= 9000) & (context[:, 2] <= 58000)]
positions = np.asarray([p if p is not None else [np.nan] * 3 for p in anatomy['positions']])
angle = np.deg2rad(25)
def project(p): return -p[:, 0], -(p[:, 1] * np.sin(angle) + p[:, 2] * np.cos(angle))

plt.rcParams['svg.fonttype'] = 'path'
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor='#faf8f5')
ax = fig.add_axes([.44, .22, .53, .59], frameon=False)
ax.scatter(*project(context), s=1.5, color='#716b60', alpha=.24, linewidths=0)
valid = np.isfinite(positions).all(1)
ax.scatter(*project(positions[valid]), s=4, color='#a74c20', alpha=.9, linewidths=0)
ax.set_aspect('equal'); ax.set_axis_off()

def text(x, y, value, size, weight=400, color='#292820'):
    fig.text(x, y, value, fontsize=size, fontproperties=fonts[weight], color=color, va='top')
text(.055, .86, 'FLM', 102, 600, '#a74c20')
text(.063, .60, 'Fly Language Model', 27)
text(.063, .475, 'Language through\na fly’s wiring.', 24)
text(.063, .295, 'Trained from scratch.\nInspectable in your browser.', 16, color='#716b60')
text(.49, .245, 'MaleCNS anatomy · compact modeled subset', 11, color='#716b60')
fig.add_artist(plt.Line2D([.063, .94], [.14, .14], transform=fig.transFigure, color='#d9d3c8', linewidth=1))
text(.063, .10, 'ChatFLM', 15, 600)
text(.78, .10, 'flm.kuber.studio', 15)
fig.savefig(OUT / 'flm-social.png', dpi=100)
fig.savefig(OUT / 'flm-social.svg'); plt.close(fig)
(OUT / 'provenance.json').write_text(json.dumps(dict(
    anatomy_sha256=hashlib.sha256(anatomy_path.read_bytes()).hexdigest(),
    coordinate_source='MaleCNS; attribution in /licenses/',
    rendering='Rigid 25-degree anatomical projection; gray anatomical context and orange selected neuron positions. No activity is represented.',
    selected_positions=int(valid.sum()), modeled_neurons=len(positions),
    missing_positions=int((~valid).sum()), font='IBM Plex Sans, SIL Open Font License',
    generator='scripts/brand_visuals.py'), indent=2) + '\n', encoding='utf8')
print('Generated 1200 × 630 FLM share image and editable vector source.')
