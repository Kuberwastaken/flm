"""Verify deployed physical results, every replay payload, and current source archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def verify(base, output):
    if output.exists(): raise ValueError('Preserve prior deployment verification')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest=json.loads((ROOT/'public/research/food-core-replay/manifest.json').read_bytes())
    files=['body/recorded-food/model.json','body/recorded-food/geometry.bin',
        'models/flm-wikitext/anatomy.json','research/food-core-replay/manifest.json',
        'research/food-core-physical-records.zip','research/paper-source.zip',
        'research/food-core-physical-results.md','research/food-core-physical-reference.md',
        'research/physical-state-semantics.md','research/food-response-plan.md','licenses/BODY-PROVENANCE.md',
        'research/food-readout-learning.md','research/food-readout/food_readout_learning.py',
        'research/food-readout/test_food_readout_learning.py', 'research/babylm-validation.json',
        'research/figures/food-core-physical.png','research/figures/food-core-physical.svg','research/figures/food-core-physical.csv']
    files += [p.relative_to(ROOT/'public').as_posix() for p in (ROOT/'public/research/food-core-physical').iterdir() if p.is_file()]
    for entry in manifest['trials']:
        files.append('research/food-core-replay/'+entry['metadata_file'])
        metadata=json.loads((ROOT/'public/research/food-core-replay'/entry['metadata_file']).read_bytes())
        files.append('research/food-core-replay/'+metadata['binary_file'])
    # Verify new Markdown links as published, including their public path rewrites.
    links=0
    for name in files:
        if name.endswith('.md'):
            path=ROOT/'public'/name
            for target in re.findall(r'\]\(([^)\s]+)\)',path.read_text(encoding='utf8')):
                if '://' in target or target.startswith('#'): continue
                resolved=(ROOT/'public'/target.lstrip('/') if target.startswith('/') else path.parent/target).resolve()
                if not resolved.is_relative_to((ROOT/'public').resolve()) or not resolved.is_file():
                    raise ValueError('Broken public Markdown link: '+name+' -> '+target)
                links+=1
    def fetch(name):
        with urllib.request.urlopen(base.rstrip('/')+'/'+name+'?release='+head,timeout=90) as response:
            data=response.read()
        local=(ROOT/'public'/name).read_bytes()
        if data!=local: raise ValueError('Published bytes differ: '+name)
        return name,dict(bytes=len(data),sha256=sha(data))
    with ThreadPoolExecutor(max_workers=4) as executor: checked=dict(executor.map(fetch,sorted(set(files))))
    release=json.loads((ROOT/'reports/food-core-physical/release.json').read_bytes())
    if checked['research/food-core-physical-records.zip']!=release['files']['public/research/food-core-physical-records.zip']:
        raise ValueError('Live physical archive differs from audited release')
    source=(ROOT/'public/research/paper-source.zip').read_bytes()
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        inventory=json.loads(archive.read('source-manifest.json'))
        if archive.testzip() or set(archive.namelist())!={*inventory,'source-manifest.json'}: raise ValueError('Source archive inventory differs')
        for name,entry in inventory.items():
            data=archive.read(name)
            if data!=(ROOT/name).read_bytes() or len(data)!=entry['bytes'] or sha(data)!=entry['sha256']:
                raise ValueError('Archived source differs from reviewed working tree: '+name)
    with urllib.request.urlopen(base.rstrip('/')+'/?release='+head,timeout=90) as response:
        html=response.read().decode('utf8')
    for text in ('food-replay-neuron-index','food-core-physical-results.md','Language-trained cores before food-task adaptation'):
        if text not in html: raise ValueError('Deployed interface missing: '+text)
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),content_commit=head,base_url=base,
        verifier_sha256=sha(Path(__file__).read_bytes()),http_files=checked,
        public_relative_links_checked=links,current_source_archive_files=len(inventory),
        all_24_replays_verified=True,physical_observations=4824,
        scope='HTTP byte verification of every physical replay payload and current public research artifacts; no new physics, training or transfer claim.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8',newline='\n') as handle: json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps(dict(http_files=len(checked),source_files=len(inventory),links_checked=links,content_commit=head)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='https://flm.kuber.studio'); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); verify(args.base,args.output)
