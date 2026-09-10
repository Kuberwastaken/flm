"""Plot measured wiring controls and completed factorial results, without selection."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'public/research'; FIGURES=PUBLIC/'figures'
ORANGE='#a74c20'; GREEN='#497569'; BACKGROUND='#faf8f5'


def save(fig,name):
    fig.savefig(FIGURES/f'{name}.svg'); fig.savefig(FIGURES/f'{name}.png',dpi=160); plt.close(fig)


def main():
    report=json.loads((PUBLIC/'wiring-learning.json').read_text(encoding='utf8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'path',
        'figure.facecolor':BACKGROUND,'axes.facecolor':BACKGROUND,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained',sharex=True,sharey=True)
    for ax,name,title in zip(axes,['central-256','central-256-null101'],['Measured wiring','Artificial control · seed 101']):
        with np.load(ROOT/'data/graphs'/name/'graph.npz',allow_pickle=False) as g:
            matrix=np.zeros((256,256)); matrix[g['row'],g['col']]=np.sign(g['weight'])
        ax.imshow(matrix,cmap=ListedColormap([GREEN,BACKGROUND,ORANGE]),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),interpolation='nearest',origin='upper')
        ax.set(title=title,xlabel='Presynaptic index',xticks=[0,64,128,192,255],yticks=[0,64,128,192,255])
    axes[0].set_ylabel('Postsynaptic index')
    fig.suptitle('Same nodes, signed degrees, input magnitudes and self-connections',fontsize=12)
    fig.text(.5,-.025,'Orange: positive edge    Green: negative edge    Blank: no retained edge',ha='center',fontsize=10)
    # Keep the shared color explanation inside the saved figure.
    fig.set_constrained_layout_pads(h_pad=.16,w_pad=.04); fig.texts[-1].set_y(.015)
    save(fig,'wiring-matrix')
    for delay in (8,48):
        fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
        for index,(method,label) in enumerate(report['methods'].items()):
            for topology,offset,color,marker in [('measured',-.12,ORANGE,'o'),('null',.12,GREEN,'s')]:
                row=next(r for r in report['curves'] if r['task']=='context' and r['method']==method and r['topology']==topology and r['step']==900 and r['delay']==delay)
                stat=row['current_accuracy']; mean=stat['mean']*100
                ax.errorbar(mean,index+offset,xerr=[[mean-stat['minimum']*100],[stat['maximum']*100-mean]],fmt=marker,color=color,
                    capsize=3,markersize=5,label=('Measured wiring' if topology=='measured' else 'Artificial control') if index==0 else None)
        ax.set(yticks=np.arange(5),yticklabels=list(report['methods'].values()),xlabel='Final accuracy (%)',xlim=(-2,102),
            title=f'Delayed context · delay {delay} · update 900')
        ax.invert_yaxis(); ax.axvline(50,color='#77716b',linestyle=':',linewidth=1)
        ax.grid(axis='x',color='#d6d0c5',linewidth=.6); ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=10)
        save(fig,f'wiring-context-delay{delay}')
    fig,ax=plt.subplots(figsize=(7.2,4.4),layout='constrained')
    for topology,color in [('measured',ORANGE),('null',GREEN)]:
        for approximation,style in [('eligibility','-'),('instantaneous','--')]:
            rows=[r for r in report['gradient_diagnostics'] if r['task']=='context' and r['method']=='bptt' and r['topology']==topology and r['approximation']==approximation]
            steps=[0,300,600,900]; values=[[r['cosine'] for r in rows if r['step']==step] for step in steps]
            if any(len(v)!=3 or any(x is None for x in v) for v in values): raise ValueError('Incomplete/undefined plotted gradient observations')
            means=np.mean(values,axis=1); lows=np.min(values,axis=1); highs=np.max(values,axis=1)
            ax.plot(steps,means,style,color=color,marker='o',markersize=3,label=f'{"Measured" if topology=="measured" else "Artificial"} · {"eligibility" if approximation=="eligibility" else "no history"}')
            ax.fill_between(steps,lows,highs,color=color,alpha=.07)
    ax.set(xlabel='BPTT-trained checkpoint update',ylabel='Cosine with exact core gradient',xticks=steps,ylim=(-1.05,1.05),
        title='Supervised gradient alignment · delayed context · delay 8')
    ax.axhline(0,color='#77716b',linewidth=.8); ax.grid(axis='y',color='#d6d0c5',linewidth=.6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=9)
    save(fig,'wiring-gradient-alignment')
    rows=[]
    for point in report['curves']:
        for index,seed in enumerate(report['seeds']):
            rows.append({**{key:point[key] for key in ('task','method','topology','step','delay')},'seed':seed,
                'current_accuracy':point['current_accuracy']['values'][index],
                'original_accuracy':point['original_accuracy']['values'][index],
                'current_cross_entropy':point['current_cross_entropy']['values'][index]})
    with (FIGURES/'wiring-learning.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Plotted all requested wiring figures; exported {len(rows)} seed-level observations.')


if __name__=='__main__': main()
