"""Export every completed primary language checkpoint, with validation-only default selection."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from flm.language_train import restore
from flm.tokenizer import Lexicon

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def f32(x): return x.detach().cpu().numpy().astype('<f4')

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    graph=ROOT/'data/graphs/central-1024/graph.npz'
    catalog={}; inputs={}
    for study,path,tok in [('babylm','reports/babylm/selection.json','babylm-2026-4096'),
                           ('wikitext','reports/wikitext2/selection.json','wikitext2-4096')]:
        selection=ROOT/path; inputs[path]=sha(selection)
        lexicon=Lexicon(ROOT/'data/tokenizers'/tok/'tokenizer.json')
        for row in read(selection)['runs']:
            variant,seed=row['variant'],row['seed']; scale=row.get('scale','2')
            key=f'{study}-{scale}-{variant}-s{seed}' if study=='babylm' else f'{study}-{variant}-s{seed}'
            checkpoint=ROOT/row['checkpoint']; assert sha(checkpoint)==row['checkpoint_sha256']
            model,saved=restore(checkpoint,graph,lexicon); model.eval(); c=model.config
            directory=ROOT/'public/models'/key
            if directory.exists(): raise ValueError('Preserve existing browser export: '+key)
            directory.mkdir(parents=True)
            label=f'{variant.upper() if variant!="transformer" else "Transformer"} · {"BabyLM "+scale.upper() if study=="babylm" else "WikiText-2"} · seed {seed}'
            if variant=='flm':
                template=read(ROOT/'public/models/flm-wikitext/model.json')
                w,alpha,beta,gain=model.constants(); rows,cols=model.row.numpy(),model.col.numpy(); order=np.lexsort((cols,rows))
                weights=(w.to_dense() if w.is_sparse else w)[model.row,model.col]*gain
                arrays=dict(embedding=f32(model.embedding.weight),input_weight=f32(model.input.weight),input_bias=f32(model.input.bias),
                    offsets=np.r_[0,np.cumsum(np.bincount(rows,minlength=c.neurons))].astype('<u4'),sources=cols[order].astype('<u4'),
                    weights=f32(weights)[order],alpha=f32(alpha),beta=f32(beta),pool=model.pool_index.numpy().astype('<u4'),pool_sizes=f32(model.pool_sizes),
                    norm_weight=f32(model.norm.weight),norm_bias=f32(model.norm.bias),projection_weight=f32(model.readout.weight),
                    projection_bias=f32(model.readout.bias),readout_bias=f32(model.output_bias))
                shutil.copyfile(ROOT/'public/models/flm-wikitext/anatomy.json',directory/'anatomy.json')
                config=dict(template,architecture='flm',norm_epsilon=model.norm.eps,variant='flm',anatomy_sha256=sha(directory/'anatomy.json'))
            else:
                arrays={name:f32(value) for name,value in model.state_dict().items()}
                config=dict(format='flm-baseline-browser-v1',architecture=variant,architecture_config=asdict(c),features=c.embedding,
                    neurons=0,retained_edges=0,source_neurons=166700,bos=0,eos=1,
                    license='MIT original implementation; underlying corpus components retain their source rights')
            binary=bytearray(); index={}
            for name,array in arrays.items():
                index[name]=dict(offset=len(binary),length=array.size,shape=list(array.shape),dtype='uint32' if array.dtype.kind=='u' else 'float32')
                binary.extend(array.tobytes(order='C'))
            (directory/'weights.bin').write_bytes(binary)
            shutil.copyfile(ROOT/'data/tokenizers'/tok/'browser-tokenizer.json',directory/'tokenizer.json')
            config.update(name=label,model_id=key,dataset='BabyLM 2026 English '+scale.upper() if study=='babylm' else 'WikiText-2 raw',
                arrays=index,weights_bytes=len(binary),weights_sha256=sha(directory/'weights.bin'),
                checkpoint_sha256=row['checkpoint_sha256'],checkpoint_step=saved['step'],vocabulary=lexicon.vocabulary,
                tokenizer_sha256=lexicon.sha256,browser_tokenizer_sha256=sha(directory/'tokenizer.json'),training=saved['run'],
                trained_parameters=saved['run']['parameter_card']['trainable_parameters'],
                description='Trained from scratch; selected by the frozen validation protocol. Complete benchmark reports are published.',
                capability='Base language model with experimental chat formatting; no instruction tuning.',
                selection_validation_bpb=row['selection_validation_bpb'],selection_manifest_sha256=sha(selection))
            write(directory/'model.json',config)
            tokens=([0]+lexicon.encode('The little bird returned to the garden. A conversation begins with a question.\n')*20)[:130]
            state=None; cases=[]
            for i,token in enumerate(tokens):
                logits,state=model(torch.tensor([[token]]),state)
                if i in (0,11,96,129):
                    case=dict(length=i+1,logits=f32(logits[0,-1]).tolist())
                    if variant=='flm': case.update(h=f32(state[0][0]).tolist(),slow=f32(state[1][0]).tolist())
                    cases.append(case)
            write(directory/'parity.json',dict(tokens=tokens,cases=cases,scope='Fresh one-token PyTorch forward references; includes context beyond the 96-token attention window. No held-out corpus input.'))
            catalog[key]=dict(path=key,label=label,name=variant.upper() if variant!='transformer' else 'Transformer',lexical=True,
                architecture=variant,study=study,scale=scale,seed=seed,validation_bpb=row['selection_validation_bpb'],
                manifest_sha256=sha(directory/'model.json'))
            print('Exported '+key,flush=True)
    candidates={k:v for k,v in catalog.items() if v['study']=='babylm' and v['architecture']=='flm'}
    default=min(candidates,key=lambda k:(candidates[k]['validation_bpb'],k))
    result=dict(default_flm=default,selection='Lowest fixed-panel BabyLM validation BPB among completed FLM checkpoints; no test loss or sample quality used.',
        inputs=inputs,models=catalog)
    write(ROOT/'web/model-catalog.json',result); write(ROOT/'public/models/catalog.json',result)
    print('Default: '+default)

if __name__=='__main__': main()
