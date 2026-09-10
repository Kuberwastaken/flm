"""Audit every SCAN loss mask and action codec; plot data, not model scores."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flm.provenance import sha256, write_json
from flm.scan import FILES, REVISION, parse
from flm.scan_task import ACTION_BYTES, action_ids, decode_action_ids, encode_example
from flm.tokenizer import Lexicon


def main():
    card_path = ROOT / 'data/cards/scan.json'
    card = json.loads(card_path.read_text(encoding='utf8'))
    if card['parser_sha256'] != sha256(ROOT / 'flm/scan.py'):
        raise ValueError('Regenerate the dataset card after changing its parser')
    raw = ROOT / 'data/raw/scan' / REVISION
    for name, (size, digest) in FILES.items():
        if (raw/name).stat().st_size != size or sha256(raw/name) != digest:
            raise ValueError('Pinned source changed: ' + name)
    lexicon = Lexicon(ROOT / 'data/tokenizers/wikitext2-4096/tokenizer.json')
    rows = parse((raw/'tasks.txt').read_bytes(), 'tasks.txt')
    measured = []
    for row in rows:
        lengths = {}
        for form in ('literal', 'byte'):
            example = encode_example(row, lexicon, action_format=form)
            selected = [token for token, active in zip(example['target'], example['loss_mask']) if active]
            if selected[-1] != 1:
                raise ValueError('EOS is absent from the supervised targets')
            if form == 'byte':
                expected = {'actions':row['actions'], 'terminated':True}
                if decode_action_ids(selected, lexicon) != expected or len(selected) != len(row['actions']) + 1:
                    raise ValueError('Action codec changed the target sequence')
            else:
                if lexicon.decode(selected[:-1]) != ' ' + ' '.join(row['actions']):
                    raise ValueError('Literal target mask changed target bytes')
            if lexicon.decode(example['generation_prefix'][1:]) != 'IN: ' + row['command'] + '\nOUT:':
                raise ValueError('Generation prefix contains target data or changed input')
            if sum(example['loss_mask']) != len(selected):
                raise ValueError('Loss accounting changed')
            lengths[form] = len(example['input'])
        if lengths['byte'] > 96:
            raise ValueError('An action-coded example exceeds the comparison window')
        measured.append(dict(command_id=row['id'], actions=len(row['actions']),
                             literal_tokens=lengths['literal'], byte_tokens=lengths['byte']))
    figures = ROOT / 'public/research/figures'; figures.mkdir(parents=True, exist_ok=True)
    table = figures / 'scan-encoding.csv'
    with table.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(measured)
    summaries = {}
    for form in ('literal', 'byte'):
        values = [row[form+'_tokens'] for row in measured]
        summaries[form] = dict(minimum=min(values), maximum=max(values),
                               above_96=sum(value > 96 for value in values),
                               histogram=dict(sorted(Counter(values).items())))
    report = dict(study='SCAN instruction-transfer data preparation', status='Data audited; model comparison not run',
        canonical_commands=len(rows), source_revision=REVISION, dataset_card_sha256=sha256(card_path),
        tokenizer_sha256=lexicon.sha256, comparison_context=96,
        action_codec={action:dict(character=character, existing_token_id=action_ids(lexicon)[action]) for action,character in ACTION_BYTES.items()},
        encodings=summaries, all_command_masks_and_action_roundtrips_verified=True,
        partitions=card['partitions'], overlap_audit=card['overlap_audit'],
        measured_csv=dict(path='research/figures/scan-encoding.csv', bytes=table.stat().st_size, sha256=sha256(table)),
        sources={name:sha256(ROOT/name) for name in ('flm/scan.py','flm/scan_task.py','scripts/scan_data_report.py')},
        interpretation=['A symbol-coded output uses one existing vocabulary ID per action, without new learned parameters.',
            'The byte sequence is deliberately not re-encoded with BPE; every output position remains one action.',
            'Prediction must remain unconstrained over the original vocabulary; unknown action IDs and missing EOS count as errors.',
            'Literal action labels are a representation diagnostic, not a second fitted model experiment.',
            'Official test structural metadata is audited here; no model output, loss or checkpoint selection is inspected.'])
    write_json(ROOT/'reports/scan/data-preparation.json',report)
    write_json(ROOT/'public/research/scan-data.json',report)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'flm-scan-data-v1'})
    fig, axes = plt.subplots(1,2,figsize=(11,4.35),facecolor='#faf8f4')
    fig.subplots_adjust(left=.075,right=.97,bottom=.24,top=.73,wspace=.28)
    fig.text(.075,.91,'SCAN · data and representation audit',fontsize=18,weight='bold',color='#302f29')
    fig.text(.075,.82,'20,910 unique commands · no neural-model results',fontsize=11,color='#625e55')
    counts=Counter(row['actions'] for row in measured)
    axes[0].bar(list(counts),list(counts.values()),width=.8,color='#487568')
    axes[0].set(xlabel='Actions in the target sequence',ylabel='Unique commands',xlim=(0,50))
    for form,color,label in [('literal','#ad4b20','Spelled-out action labels'),('byte','#487568','One existing token per action')]:
        values=np.sort([row[form+'_tokens'] for row in measured])
        axes[1].plot(values,np.arange(1,len(values)+1)/len(values)*100,color=color,label=label,linewidth=1.8)
    axes[1].axvline(96,color='#706b61',linestyle='--',linewidth=1)
    axes[1].text(102,7,'96-token\nwindow',fontsize=9,color='#625e55')
    axes[1].set(xlabel='Full teacher-forced input tokens',ylabel='Commands within this length (%)',xlim=(0,480),ylim=(0,105))
    axes[1].legend(loc='lower right',frameon=False,fontsize=8)
    for ax in axes:
        ax.set_facecolor('#faf8f4')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.075,.08,'Action-coded inputs all fit the fixed window. Raw commands and official action sequences are preserved.',fontsize=9,color='#625e55')
    fig.text(.075,.035,'Source: pinned brendenlake/SCAN files · each point is a command, not a trained model or independent trial.',fontsize=9,color='#625e55')
    provenance=f'Source revision {REVISION}; CSV SHA-256 {sha256(table)}; generated by scripts/scan_data_report.py.'
    for extension in ('png','svg'):
        path=figures/('scan-data.'+extension)
        metadata=dict(Title='SCAN data and action representation audit',Description=provenance)
        if extension=='svg': metadata['Date']=None
        fig.savefig(path,dpi=160,metadata=metadata)
        if extension=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
    plt.close(fig)
    print(json.dumps(dict(commands=len(rows), encodings={form:{k:v for k,v in summary.items() if k!='histogram'} for form,summary in summaries.items()}, figure=str(figures/'scan-data.png')),indent=2))


if __name__=='__main__':
    main()
