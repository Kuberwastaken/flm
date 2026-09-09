"""Generate reproducible paper figures from the published numerical artifacts."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'papers/figures'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.labelcolor':'#292820', 'text.color':'#292820',
    'svg.fonttype':'none', 'figure.dpi':150, 'savefig.dpi':300})
COLORS = {'flm':'#a74c20', 'gru':'#497569', 'transformer':'#666277'}
NAMES = {'flm':'FLM', 'gru':'GRU', 'transformer':'Transformer'}

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight'); plt.close(fig)

report = json.loads((ROOT / 'public/research/validation.json').read_text(encoding='utf8'))
runs = [r for r in report['runs'] if r['seed'] == 42 and r['variant'] in NAMES]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
rows = []
for run in runs:
    steps = [p['step'] for p in run['validation']]; losses = [p['bits_per_byte'] for p in run['validation']]
    for ax in axes: ax.plot(steps, losses, color=COLORS[run['variant']], label=NAMES[run['variant']], lw=1.7)
    rows.extend(dict(model=run['variant'], seed=42, step=p['step'], bpb=p['bits_per_byte']) for p in run['validation'])
for ax in axes:
    ax.set(xlabel='Training updates', ylabel='Validation bits / UTF-8 byte'); ax.grid(axis='y', alpha=.2)
axes[0].set(xlim=(0, 6000), ylim=(1.7, 3.6), title='Whole learning trajectory')
axes[1].set(xlim=(500, 6000), ylim=(1.7, 2.4), title='After the first 500 updates'); axes[1].legend(frameon=False)
save(fig, 'validation')
with (OUT / 'validation.csv').open('w', newline='', encoding='utf8') as handle:
    writer = csv.DictWriter(handle, fieldnames=['model','seed','step','bpb']); writer.writeheader(); writer.writerows(rows)

length = np.arange(1, 1025)
fig, ax = plt.subplots(figsize=(5.8, 2.6), layout='constrained')
ax.plot(length, np.full_like(length, 8192), label='FLM: 8 KiB', color=COLORS['flm'])
ax.plot(length, np.full_like(length, 800), label='GRU: 800 B', color=COLORS['gru'])
ax.plot(length, np.minimum(length, 95) * 2 * 2 * 108 * 4, label='Transformer: 160.3 KiB max', color=COLORS['transformer'])
ax.set(yscale='log', xlabel='Tokens already processed', ylabel='Persistent sequence state (bytes)', xlim=(1,1024)); ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5,1.02), ncol=3, fontsize=7.5); ax.grid(axis='y', alpha=.2)
save(fig, 'state-memory')

package = ROOT / 'public/models/flm-wikitext'; config = json.loads((package / 'model.json').read_text())
payload = (package / 'weights.bin').read_bytes()
def array(name):
    spec = config['arrays'][name]
    return np.frombuffer(payload, dtype='<u4' if spec['dtype']=='uint32' else '<f4', count=spec['length'], offset=spec['offset']).reshape(spec['shape'])
fig, ax = plt.subplots(figsize=(5.8, 2.6), layout='constrained')
for name, color in [('alpha',COLORS['flm']),('beta',COLORS['gru'])]:
    half = np.log(.5) / np.log1p(-array(name).astype(np.float64))
    ax.hist(half, bins=np.geomspace(.15,400,38), histtype='step', linewidth=1.7, color=color, label='Fast filter' if name=='alpha' else 'Slow filter')
ax.set(xscale='log', xlabel='Isolated-filter half-life (token updates)', ylabel='Neurons'); ax.legend(frameon=False)
save(fig, 'time-scales')

anatomy = json.loads((package / 'anatomy.json').read_text())
context = np.array(anatomy['context_positions']); context = context[(context[:,2]>=9000)&(context[:,2]<=58000)]
positions = np.array([p if p is not None else [np.nan]*3 for p in anatomy['positions']]); valid = np.isfinite(positions).all(1)
angle = np.deg2rad(25)
def projection(p): return -p[:,0]*.008, -(p[:,1]*np.sin(angle)+p[:,2]*np.cos(angle))*.008
fig, ax = plt.subplots(figsize=(6.2,3.3), layout='constrained')
ax.scatter(*projection(context), s=.4, color='#77776e', alpha=.18, linewidths=0, label='Anatomical context')
for sign, color, label in [(1,COLORS['flm'],'Positive source sign'),(-1,COLORS['gru'],'Negative source sign')]:
    mask=valid&(np.array(anatomy['source_sign'])==sign)
    ax.scatter(*projection(positions[mask]), s=4, color=color, alpha=.75, linewidths=0, label=label)
ax.set(aspect='equal', xlabel='Mirrored anatomical x (µm)', ylabel='Rotated anatomical coordinate (µm)'); ax.legend(frameon=False, markerscale=2, loc='lower center', bbox_to_anchor=(.5,1.02), ncol=3, fontsize=7)
save(fig, 'anatomy')

common = sorted(set.intersection(*[set(p['step'] for p in r['validation']) for r in runs]))[-1]
table = []
for run in runs:
    point = next(p for p in run['validation'] if p['step']==common)
    table.append(f"{NAMES[run['variant']]} & {run['parameters']:,} & {point['bits_per_byte']:.4f} & {point['token_perplexity']:.2f} " + r'\\')
(ROOT / 'papers/measured.tex').write_text('\\newcommand{\\MatchedUpdate}{'+str(common)+'}\n'+
    '\\newcommand{\\BrowserUpdate}{'+str(config['checkpoint_step'])+'}\n'+
    '\\newcommand{\\MatchedRows}{'+ '\n'.join(table)+'}\n',encoding='utf8')
print(f'Generated four figures and the matched-update table at step {common}.')
