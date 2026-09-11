"""Plot complete, checked BabyLM likelihood measurements without new inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.babylm_result_tables import collect, require, COMPONENTS, SUBSETS

COLORS = dict(flm='#a74c20', gru='#497569', transformer='#666277')
NAMES = dict(flm='FLM', gru='GRU', transformer='Transformer')
LABELS = ('Spoken BNC', 'CHILDES', 'Gutenberg', 'OpenSubtitles', 'Simple Wikipedia', 'Switchboard')
PANELS = tuple((subset, scale) for subset in SUBSETS for scale in ('10m', '100m'))
PLOT_FILES = {name+suffix for name in ('babylm-pooled','babylm-components') for suffix in ('.png','.svg')}


def lookup_scores(report):
    rows = report['scores']
    lookup = {(r['scale'], r['seed'], r['variant'], r['component'], r['subset']): r for r in rows}
    expected = {(scale, seed, variant, component, subset) for scale in ('10m', '100m')
                for seed in (42, 43) for variant in NAMES for component in ('all', *COMPONENTS) for subset in SUBSETS}
    require(len(rows) == len(lookup) and set(lookup) == expected, 'Every figure requires the complete 168-row table')
    return lookup


def component_points(report):
    """Pair the same seed, scale, source and analysis; never subtract seed means."""
    lookup = lookup_scores(report); points = []
    for subset, scale in PANELS:
        for component in COMPONENTS:
            for other in ('gru', 'transformer'):
                for seed in (42, 43):
                    a, b = [lookup[(scale, seed, variant, component, subset)] for variant in ('flm', other)]
                    require(a['available'] == b['available'], 'Paired availability differs')
                    require((a['bytes'], a['tokens'], a['blocks']) == (b['bytes'], b['tokens'], b['blocks']),
                            'Paired figure denominators differ')
                    points.append(dict(scale=scale, subset=subset, component=component, comparator=other,
                        seed=seed, difference_bpb=a['bits_per_byte']-b['bits_per_byte'] if a['available'] else None))
    return points


def draw(report, output, *, fixture):
    # Lazy import keeps the real incomplete-study gate free of plotting work.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    lookup = lookup_scores(report); points = component_points(report)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'svg.fonttype':'path', 'svg.hashsalt':'flm-babylm-results-v1'})
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    prefix = 'SYNTHETIC FIXTURE · NOT A BENCHMARK\n' if fixture else 'BabyLM 2026 · '
    def canvas(title, subtitle):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.subplots_adjust(left=.15, right=.96, top=.79, bottom=.22, hspace=.55, wspace=.35)
        fig.text(.05, .965, prefix+title, fontsize=16, weight='bold', va='top')
        fig.text(.05, .865, subtitle, fontsize=10)
        return fig, axes
    def finish(fig, name, note, legend):
        fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.045,.105), ncol=len(legend), frameon=False, fontsize=9)
        fig.text(.05,.075,note,fontsize=9)
        fig.text(.05,.045,'18,432,000 presented tokens per fit; equal updates cover different fractions of the 10M/100M corpora.',fontsize=9)
        for suffix in ('.png','.svg'):
            fig.savefig(output/(name+suffix),dpi=160,facecolor='#faf8f5',metadata={'Date':None} if suffix=='.svg' else {})
        plt.close(fig)
    seed_legend = [Line2D([],[],color='#333333',marker=m,linestyle='none',label=label,markersize=6)
                   for m,label in (('o','Training seed 42'),('^','Training seed 43'),('D','Two-seed mean'))]
    fig, axes = canvas('Held-out language prediction', 'Pooled bits per UTF-8 byte; lower is better. All panels use the same horizontal scale.')
    all_values = [r['bits_per_byte'] for r in report['scores'] if r['component']=='all' and r['available']]
    lower, upper = min(all_values), max(all_values); margin = max(upper-lower,.02)*.15
    for ax,(subset,scale) in zip(axes.flat,PANELS):
        for y,variant in enumerate(NAMES):
            values = [lookup[(scale,seed,variant,'all',subset)]['bits_per_byte'] for seed in (42,43)]
            require(all(v is not None for v in values),'No pooled score for a declared panel')
            for delta,marker,value in zip((-.13,.13),('o','^'),values):
                ax.plot(value,y+delta,marker=marker,color=COLORS[variant],markersize=6)
            ax.plot(sum(values)/2,y,marker='D',color=COLORS[variant],markersize=5)
        count=lookup[(scale,42,'flm','all',subset)]
        ax.set_title(f'{scale.upper()} words · '+('official mixture' if subset=='official' else 'overlap-filtered'),loc='left',fontsize=11,pad=28)
        ax.text(0,1.02,f"{count['bytes']:,} scored bytes · {count['blocks']:,} blocks",transform=ax.transAxes,fontsize=8)
        ax.set(yticks=range(3),yticklabels=list(NAMES.values()),ylim=(2.6,-.6),
               xlim=(max(0,lower-margin),upper+margin),xlabel='Test bits / UTF-8 byte')
        ax.grid(axis='x',alpha=.15);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.locator_params(axis='x',nbins=5)
    finish(fig,'babylm-pooled', 'Markers show two fitted seeds and their mean; no uncertainty interval is implied.', seed_legend)

    fig,axes=canvas('Source-specific model differences', 'FLM minus comparator in test bits per UTF-8 byte. Negative favors FLM; positive favors the comparator.')
    extent=max([abs(r['difference_bpb']) for r in points if r['difference_bpb'] is not None]+[.005])*1.2
    for ax,(subset,scale) in zip(axes.flat,PANELS):
        for y,component in enumerate(COMPONENTS):
            for offset,other in ((-.22,'gru'),(.22,'transformer')):
                rows=[r for r in points if (r['subset'],r['scale'],r['component'],r['comparator'])==(subset,scale,component,other)]
                values=[r['difference_bpb'] for r in rows]
                require(len(values)==2 and ((values[0] is None)==(values[1] is None)),'Incomplete component seed pair')
                if values[0] is None:
                    if other=='gru':ax.text(.98,y,'No retained blocks',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color='#666666')
                    continue
                for delta,marker,value in zip((-.10,.10),('o','^'),values):
                    ax.plot(value,y+offset+delta,marker=marker,color=COLORS[other],markersize=4)
        ax.set_title(f'{scale.upper()} words · '+('official mixture' if subset=='official' else 'overlap-filtered'),loc='left',fontsize=11)
        ax.set(yticks=range(6),yticklabels=LABELS,ylim=(5.65,-.65),xlim=(-extent,extent),xlabel='Difference in test bits / UTF-8 byte')
        ax.axvline(0,color='#777777',ls='--',linewidth=.8);ax.grid(axis='x',alpha=.15)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.locator_params(axis='x',nbins=5)
    other_legend=[Line2D([],[],color=COLORS[v],marker='o',linestyle='none',label='FLM − '+NAMES[v]) for v in ('gru','transformer')]
    finish(fig,'babylm-components','Both training seeds shown; no component confidence intervals or claims of domain competence.',other_legend+seed_legend[:2])
    return points


def build(root, output):
    root=Path(root);output=Path(output)
    require(not output.exists(),'Preserve existing figures; choose a fresh destination')
    report=collect(root)  # Complete-study arithmetic/input gate precedes plotting imports or output creation.
    points=draw(report,output,fixture=False)
    require({p.name for p in output.iterdir()}==PLOT_FILES and all((output/name).is_file() and (output/name).stat().st_size>0 for name in PLOT_FILES),
            'The complete four-file figure inventory is required')
    for name,expected in {**report['input_sha256'],**report['evaluator_source_sha256']}.items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,'A figure input changed during rendering')
    data=dict(tables=report,component_points=points,synthetic_fixture=False,
        figure_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Plots saved, independently arithmetic-checked likelihoods. No inference or bootstrap replication.')
    (output/'figure-data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.iterdir())}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.output)
