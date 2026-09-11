"""Audit one finished BabyLM fit without fitting, inference or test-cache access."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NUMERICAL_SOURCES = ('language_train.py','model.py','baselines.py','train.py',
                     'tokenizer.py','corpus_cache.py','graph.py','provenance.py')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def declared_protocol(root, scale):
    tokenizer=root/'data/tokenizers/babylm-2026-4096/tokenizer.json'
    cache=root/'data/processed/babylm-2026-bpe'
    return dict(steps=12000,batch=16,sequence=96,warmup=16,learning_rate=.002,
        final_learning_rate=.0002,lr_warmup_updates=100,weight_decay=.01,eval_tokens=49152,threads=4,
        tokenizer_sha256=sha(tokenizer),train_cache_sha256=sha(cache/f'train-{scale}/manifest.json'),
        validation_cache_sha256=sha(cache/'validation/manifest.json'),
        graph_sha256=sha(root/'data/graphs/central-1024/graph.npz'),
        validation_panel_sha256=sha(tokenizer.with_name('validation-panel.json')),
        evaluation_unit='block',cache_identity='SHA-256 of verified mmap manifest')


def validate_declaration(root, scale, seed, variant, complete, run):
    expected=declared_protocol(root,scale)
    require(complete['steps']==12000 and complete['protocol']==run['protocol']==expected,
            'Completed fit differs from the fixed protocol')
    require(run['seed']==seed and run['parameter_card']['config']['variant']==variant
            and run['test_set_used_for_training'] is False,'Completed run identity changed')
    study=read(root/f'runs/babylm-{scale}/study.json')
    require(study==dict(dataset=f'BabyLM 2026 {scale}',steps=12000,seeds=[42,43],variants=['flm','gru','transformer'],
        protocol_sha256=sha(root/'docs/BABYLM-PROTOCOL.md'),tokenizer_sha256=expected['tokenizer_sha256'],
        train_manifest_sha256=expected['train_cache_sha256'],validation_manifest_sha256=expected['validation_cache_sha256'],
        validation_panel_sha256=expected['validation_panel_sha256']),'Study declaration changed')
    return expected


def audit(root, scale, seed, variant):
    root=Path(root).resolve()
    require(scale in ('10m','100m') and seed in (42,43) and variant in ('flm','gru','transformer'),
            'Choose a registered BabyLM condition')
    folder=root/f'runs/babylm-{scale}/{variant}-s{seed}'
    required=[folder/name for name in ('complete.json','run.json','best.pt','last.pt')]
    required += [folder/f'validation-{step:06d}.json' for step in range(500,12001,500)]
    require(all(p.is_file() for p in required),'Completed fit and all 24 validation records are required')
    require(root==ROOT.resolve(),'Run this auditor from the target repository checkout')
    # Imports and validation data access happen only after the complete-fit gate.
    import torch
    from flm.babylm_test import verify_completed_payloads, validation_coverage
    from flm.corpus_cache import read_mmap
    from flm.tokenizer import Lexicon
    tokenizer=root/'data/tokenizers/babylm-2026-4096/tokenizer.json'
    cache=root/'data/processed/babylm-2026-bpe'
    required += [tokenizer,tokenizer.with_name('validation-panel.json'),root/'docs/BABYLM-PROTOCOL.md',
        root/f'runs/babylm-{scale}/study.json',cache/f'train-{scale}/manifest.json',cache/'validation/manifest.json',
        root/'data/graphs/central-1024/graph.npz',root/'flm/babylm_test.py',Path(__file__).resolve()]
    required += [root/'flm'/name for name in NUMERICAL_SOURCES]
    before={p:sha(p) for p in required}
    complete,run=read(folder/'complete.json'),read(folder/'run.json')
    protocol=validate_declaration(root,scale,seed,variant,complete,run)
    require(sha(folder/'best.pt')==complete['best_checkpoint_sha256'],'Completed checkpoint hash changed')
    require(str(torch.__version__)==run['python_torch'],'Training framework version changed')
    commit=run['source_commit'];require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Invalid training source commit')
    original_hashes={}
    for name in NUMERICAL_SOURCES:
        original=subprocess.check_output(['git','show',f'{commit}:flm/{name}'],cwd=root)
        expected=hashlib.sha256(original).hexdigest()
        require(sha(root/'flm'/name)==expected,'Training numerical source changed: '+name)
        original_hashes['flm/'+name]=expected
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        lexicon=Lexicon(tokenizer)
        documents,_,_=read_mmap(cache/'validation',lexicon.sha256)
        coverage=validation_coverage(documents,read(tokenizer.with_name('validation-panel.json')),lexicon)
        require(len(coverage)==48,'The declared validation panel requires 48 blocks')
        selected=dict(scale=scale,seed=seed,variant=variant,
            checkpoint=(folder/'best.pt').relative_to(root).as_posix(),checkpoint_sha256=sha(folder/'best.pt'),
            selection_validation_bpb=complete['best_validation_bpb'],training_source_commit=commit)
        selected.update(verify_completed_payloads(root,selected,protocol,lexicon,coverage))
    finally:torch.set_num_threads(previous_threads)
    require(all(sha(p)==value for p,value in before.items()),'An audit input changed during verification')
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        scope='Single completed fit: declared protocol, source identity, checkpoint restoration and all selection observations. No model inference or test-cache access.',
        completed_run=selected,validation_records_checked=24,validation_panel_blocks=48,
        input_sha256={p.relative_to(root).as_posix():value for p,value in before.items()},
        training_source_sha256=original_hashes,verifier_sha256=sha(root/'flm/babylm_test.py'),
        audit_source_sha256=sha(Path(__file__)),selection_frozen_by_this_audit=False,test_inference_performed=False)


def write_audit(root, scale, seed, variant, output):
    output=Path(output)
    require(not output.exists(),'Preserve existing audit records; choose a fresh output')
    report=audit(root,scale,seed,variant)
    content=json.dumps(report,indent=2,allow_nan=False)+'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8',newline='\n') as stream:stream.write(content)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--scale',choices=('10m','100m'),required=True)
    parser.add_argument('--seed',type=int,choices=(42,43),required=True)
    parser.add_argument('--variant',choices=('flm','gru','transformer'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=write_audit(args.root,args.scale,args.seed,args.variant,args.output)
    print(json.dumps(report['completed_run']))
