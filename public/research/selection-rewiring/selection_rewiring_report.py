"""Summarize audited structural controls without treating them as language fits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flm.provenance import sha256,write_json

SEEDS=(101,103,107)
GROUPS={
    'candidate':('KC-centered','#aa572d','o'),
    'contact_ranked':('Contact-ranked','#353936','s'),
    'uniform':('Uniform random','#4c7793','^'),
    'stratified':('Stratified random','#487361','D'),
}


def group(selection):
    label=selection.split('/')[1]
    if label in ('candidate','contact_ranked'): return label
    for prefix in ('uniform','stratified'):
        if label in [f'{prefix}_s{seed}' for seed in (201,203,207)]: return prefix
    raise ValueError('Unknown selection group')


def summarize(source,manifest):
    expected={name+f'/null{seed}' for name in source['graphs'] for seed in SEEDS}
    if set(manifest['records']) != expected or manifest['planned'] != len(expected):
        raise ValueError('Incomplete structural inventory')
    rows=[]; selections=[]
    for name,entry in source['graphs'].items():
        category=group(name); n=entry['config']['neurons']; e=entry['edges']
        if n < 1 or e < 1: raise ValueError('Invalid source dimensions')
        local=[]; originals=[]
        for seed in SEEDS:
            receipt=manifest['records'][name+f'/null{seed}']
            row=dict(selection=name,group=category,graph_seed=seed,status=receipt['status'],
                failure_reason='',neurons=n,fast_edges=e,edge_density_percent=100*e/(n*n),
                original_edge_overlap_percent=None,unchanged_endpoint_slots_percent=None,
                original_largest_scc_percent=None,rewired_largest_scc_percent=None,
                original_reciprocal_percent=None,rewired_reciprocal_percent=None,
                accepted_swaps=None,proposed_swaps=None,acceptance_fraction=None)
            if receipt['status']=='failed':
                row['failure_reason']=receipt['reason']
                if not row['failure_reason']: raise ValueError('Missing failure reason')
            elif receipt['status']=='complete':
                report=receipt['report']; original=report['original']; changed=report['randomized']
                if any(stats['neurons'] != n or stats['edges'] != e or
                       sum(stats['strong_component_sizes']) != n for stats in (original,changed)):
                    raise ValueError('Topology dimensions disagree with source')
                if report['seed'] != seed or report['accepted_swaps'] != 10*e:
                    raise ValueError('Seed or declared swap count changed')
                originals.append(original)
                row.update(original_edge_overlap_percent=100*report['original_edge_overlap_fraction'],
                    unchanged_endpoint_slots_percent=100*report['unchanged_endpoint_slots_fraction'],
                    original_largest_scc_percent=100*max(original['strong_component_sizes'])/n,
                    rewired_largest_scc_percent=100*max(changed['strong_component_sizes'])/n,
                    original_reciprocal_percent=100*original['reciprocal_off_diagonal_fraction'],
                    rewired_reciprocal_percent=100*changed['reciprocal_off_diagonal_fraction'],
                    accepted_swaps=report['accepted_swaps'],proposed_swaps=report['proposed_swaps'],
                    acceptance_fraction=report['acceptance_fraction'])
                for field,value in row.items():
                    if field.endswith('_percent') and (not np.isfinite(value) or not 0 <= value <= 100):
                        raise ValueError('Invalid structural percentage')
            else: raise ValueError('Nonterminal structural case')
            rows.append(row); local.append(row)
        if originals and any(original != originals[0] for original in originals):
            raise ValueError('Original diagnostics differ between control seeds')
        complete=[row for row in local if row['status']=='complete']
        aggregate=dict(selection=name,group=category,neurons=n,fast_edges=e,
            edge_density_percent=100*e/(n*n),complete=len(complete),failed=3-len(complete),
            plotted=len(complete)==3)
        if aggregate['plotted']:
            aggregate['original_largest_scc_percent']=complete[0]['original_largest_scc_percent']
            for key in ('original_edge_overlap_percent','rewired_largest_scc_percent'):
                values=[row[key] for row in complete]
                aggregate[key]=dict(mean=float(np.mean(values)),minimum=min(values),maximum=max(values))
        selections.append(aggregate)
    complete=sum(row['status']=='complete' for row in rows)
    if complete != manifest['complete'] or len(rows)-complete != manifest['failed']:
        raise ValueError('Terminal totals disagree')
    groups={}
    for category in GROUPS:
        selected=[r for r in selections if r['group']==category]
        observed=[r for r in rows if r['group']==category and r['status']=='complete']
        groups[category]=dict(original_selections=len(selected),plotted_selections=sum(r['plotted'] for r in selected),
            successful_controls=len(observed),failed_controls=3*len(selected)-len(observed),
            overlap_percent_range=[min(r['original_edge_overlap_percent'] for r in observed),
                                   max(r['original_edge_overlap_percent'] for r in observed)] if observed else None)
    return dict(planned=len(rows),complete=complete,failed=len(rows)-complete,
        original_selections=len(selections),plotted_selections=sum(r['plotted'] for r in selections),
        groups=groups,selections=selections,cases=rows,
        scope='Descriptive structural diagnostics. One plotted point per original selection, mean and range across three graph seeds; no independent biological replications, mixing guarantee or language performance.')


def verify_manifest(archived,local,expected):
    if hashlib.sha256(archived).hexdigest() != expected or json.loads(archived) != json.loads(local):
        raise ValueError('Archived or local manifest changed')
    return json.loads(archived)


def main():
    directory=Path('reports/selection-rewiring')
    release=json.loads((directory/'release.json').read_text(encoding='utf8'))
    manifest_path=directory/'manifest.json'; archive_path=Path('public/research/selection-rewiring.zip')
    if (sha256(archive_path) != release['archive_sha256']
            or not release['arrays_degrees_signs_weights_self_edges_and_diagnostics_verified']):
        raise ValueError('Audited structural release identity changed')
    with zipfile.ZipFile(archive_path) as archive:
        # The archive uses LF bytes; the repository JSON writer uses native
        # newlines. Bind exact archived bytes and require equal parsed records.
        manifest=verify_manifest(archive.read('rewiring-manifest.json'),manifest_path.read_bytes(),release['rewiring_manifest_sha256'])
        source=json.loads(archive.read('source-manifest.json'))
    summary=summarize(source,manifest)
    summary.update(release_sha256=sha256(directory/'release.json'),local_manifest_sha256=sha256(manifest_path),
        archive_manifest_sha256=release['rewiring_manifest_sha256'],
        report_source_sha256=sha256(Path(__file__)))
    write_json(directory/'summary.json',summary)
    output=Path('public/research/figures'); output.mkdir(parents=True,exist_ok=True)
    csv_path=output/'selection-rewiring.csv'
    with csv_path.open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary['cases'][0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(summary['cases'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'path'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,6.4))
    fig.subplots_adjust(left=.08,right=.97,top=.73,bottom=.23,wspace=.30)
    for category,(label,color,marker) in GROUPS.items():
        rows=[r for r in summary['selections'] if r['group']==category and r['plotted']]
        for ax,key,xkey in ((axes[0],'original_edge_overlap_percent','edge_density_percent'),
                            (axes[1],'rewired_largest_scc_percent','original_largest_scc_percent')):
            x=[r[xkey] for r in rows]; y=[r[key]['mean'] for r in rows]
            low=[max(0.,r[key]['mean']-r[key]['minimum']) for r in rows]
            high=[max(0.,r[key]['maximum']-r[key]['mean']) for r in rows]
            if rows: ax.errorbar(x,y,yerr=[low,high],fmt=marker,color=color,markersize=5,
                alpha=.8,elinewidth=.8,capsize=2,label=label)
    axes[0].set_xscale('log'); axes[0].set_xlabel('Fast-edge density, E / N² (%)')
    axes[0].set_ylabel('Original edges retained after rewiring (%)')
    axes[0].set_title('Edge overlap',fontsize=11,pad=12)
    axes[1].plot([0,100],[0,100],color='#b9b4ab',linestyle='--',linewidth=.8,zorder=0)
    axes[1].set_xlim(-3,103); axes[1].set_xlabel('Original largest strongly connected component (%)',fontsize=9)
    axes[1].set_ylabel('Rewired largest component (%)'); axes[1].set_title('Strong connectivity',fontsize=11,pad=12)
    for ax in axes:
        ax.set_ylim(-3,103); ax.grid(color='#dedbd5',linewidth=.6); ax.set_axisbelow(True)
        ax.tick_params(axis='both',length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.045,.945,'Selection controls: what the rewiring changes',fontsize=15,weight='bold')
    fig.text(.045,.895,f"{summary['complete']} successful controls; {summary['failed']} failures. All graphs in this preparation are untrained.",fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.855),frameon=False,ncol=4,fontsize=9)
    fig.text(.045,.13,f"Dots: {summary['plotted_selections']} original selections with all three rewires. Bars: seed range, not confidence intervals.",fontsize=9)
    fig.text(.045,.085,'Overlap includes fixed self edges. Neither plot establishes sufficient mixing or a language-learning benefit.',fontsize=9)
    fig.text(.045,.04,f"Selections with failed controls are not averaged or plotted: {summary['original_selections']-summary['plotted_selections']}. All cases remain in the CSV.",fontsize=9)
    artifacts=[csv_path]
    for suffix in ('png','svg'):
        path=output/f'selection-rewiring.{suffix}'; fig.savefig(path,dpi=160,facecolor='white')
        if suffix=='svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
        artifacts.append(path)
    plt.close(fig)
    write_json(directory/'figures.json',dict(summary_sha256=sha256(directory/'summary.json'),
        plot_source_sha256=sha256(Path(__file__)),artifacts={str(p).replace('\\','/'):sha256(p) for p in artifacts},
        plotted_selections=summary['plotted_selections'],scope=summary['scope']))
    print(json.dumps({key:summary[key] for key in ('planned','complete','failed','original_selections','plotted_selections','groups')},indent=2))


if __name__=='__main__': main()
