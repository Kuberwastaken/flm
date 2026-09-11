import copy
import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from scripts import babylm_result_tables as tables


def fixture():
    """Six artificial sources, unequal block bytes, one wholly excluded source."""
    metadata = []
    for component in tables.COMPONENTS:
        for i, size in enumerate((100, 900)):
            metadata.append(dict(id=f'{component}/{i}', component=component, utf8_bytes=size,
                token_start=0, token_end=size//2+2,
                overlap_filtered_eligible=component != 'bnc_spoken' and i == 1))
    manifest = dict(blocks=12, text_tokens=3000, utf8_bytes=6000)
    selected = []; results = []
    for scale, seed, variant in tables.RUNS:
        choice = dict(scale=scale, seed=seed, variant=variant, checkpoint_step=11500,
                      checkpoint_sha256=f'fixture-{scale}-{seed}-{variant}',
                      final_exposure=dict(presented_tokens=18432000,presented_bytes=60000000,scored_bytes=50000000))
        selected.append(choice)
        factor = {'flm': 1.4, 'gru': 1.2, 'transformer': 1.}[variant] + (seed-42)*.1 + (scale == '100m')*.2
        blocks = []
        for meta in metadata:
            size = meta['utf8_bytes']; nll = (70. if size == 100 else 180.) * factor
            blocks.append(dict(document=meta['id'], component=meta['component'], bytes=size, tokens=size//2,
                overlap_filtered_eligible=meta['overlap_filtered_eligible'], nll=nll,
                bits_per_byte=nll/size/math.log(2)))
        components = {}
        for component in ('all', *tables.COMPONENTS):
            group = [r for r in blocks if component == 'all' or r['component'] == component]
            excluded = [r for r in group if not r['overlap_filtered_eligible']]
            row = dict(excluded_blocks=len(excluded), excluded_bytes=sum(r['bytes'] for r in excluded))
            for subset in tables.SUBSETS:
                chosen = group if subset == 'official' else [r for r in group if r['overlap_filtered_eligible']]
                if not chosen: row[subset] = None; continue
                nll = sum(r['nll'] for r in chosen); size = sum(r['bytes'] for r in chosen)
                tokens = sum(r['tokens'] for r in chosen)
                row[subset] = dict(blocks=len(chosen), nll=nll, bytes=size, tokens=tokens,
                                  bits_per_byte=nll/size/math.log(2), token_perplexity=math.exp(nll/tokens))
            components[component] = row
        results.append(dict(**choice, identity=dict(**choice, test_manifest_sha256='fixture-manifest'),
                            parameters=1000, components=components, blocks=blocks))
    selection = dict(runs=selected, manifest_sha256=dict(test='fixture-manifest'))
    aggregates = []; pairs = []
    for scale in ('10m', '100m'):
        for subset in tables.SUBSETS:
            for component in ('all', *tables.COMPONENTS):
                for variant in ('flm', 'gru', 'transformer'):
                    scores = [r['components'][component][subset] for r in results if r['scale'] == scale and r['variant'] == variant]
                    if scores[0] is None: continue
                    values = [r['bits_per_byte'] for r in scores]
                    aggregates.append(dict(scale=scale, subset=subset, component=component, variant=variant,
                        seeds=[42,43], bits_per_byte=values, mean_bpb=statistics.mean(values),
                        seed_standard_deviation=statistics.stdev(values)))
            for seed in (42,43):
                scores = {r['variant']:r['components']['all'][subset]['bits_per_byte'] for r in results
                          if r['scale'] == scale and r['seed'] == seed}
                for other in ('gru', 'transformer'):
                    delta = scores['flm']-scores[other]
                    pairs.append(dict(scale=scale, subset=subset, training_seed=seed, first='flm', second=other,
                        difference_bpb=delta, lower_95=delta-.01, upper_95=delta+.01, replicates=10000, seed=31415,
                        unit='paired blocks, resampled within each source component', caution='Artificial endpoints, not resampling evidence'))
    summary = dict(runs=[{k:v for k,v in r.items() if k != 'blocks'} for r in results],
                   aggregates=aggregates, paired_comparisons=pairs)
    return copy.deepcopy((selection, results, summary, metadata, manifest))


def write_fixture(root):
    selection, results, summary, metadata, manifest = fixture()
    def write(path, value):
        path=root/path; path.parent.mkdir(parents=True, exist_ok=True)
        data=(json.dumps(value)+'\n').encode();path.write_bytes(data);return tables.digest(data)
    path=root/tables.CACHE/'blocks.jsonl';path.parent.mkdir(parents=True)
    data=''.join(json.dumps(r)+'\n' for r in metadata).encode();path.write_bytes(data)
    manifest['files']={'blocks.jsonl':tables.digest(data)}
    mh=write(tables.CACHE/'manifest.json',manifest)
    selection['manifest_sha256']['test']=mh
    sh=write(tables.REPORTS/'selection.json',selection)
    sources={}
    for name in ('babylm_test.py', 'corpus_evaluation.py', 'language_train.py', 'model.py', 'baselines.py'):
        p=root/'flm'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'# artificial source\n')
        sources[name]=tables.digest(p.read_bytes())
    for result in results:
        result['identity'].update(test_manifest_sha256=mh,selection_sha256=sh,evaluator_source_sha256=sources)
        write(tables.REPORTS/f"test-{result['scale']}-{result['variant']}-s{result['seed']}.json",result)
    summary['runs']=[{k:v for k,v in r.items() if k != 'blocks'} for r in results]
    write(tables.REPORTS/'summary.json',summary)


class BabyLMResultTableTests(unittest.TestCase):
    def test_weighted_pooled_scores_seed_sd_nulls_and_paired_sign(self):
        report=tables.derive(*fixture())
        self.assertEqual(len(report['scores']),168)
        self.assertEqual(len(report['seed_aggregates']),78)
        self.assertEqual(len(report['paired_comparisons']),16)
        row=next(r for r in report['scores'] if (r['scale'],r['seed'],r['variant'],r['component'],r['subset']) ==
                 ('10m',42,'flm','all','official'))
        self.assertAlmostEqual(row['bits_per_byte'], .25*1.4/math.log(2))
        self.assertNotAlmostEqual(row['bits_per_byte'], .45*1.4/math.log(2))  # Incorrect block-average rate.
        missing=[r for r in report['scores'] if not r['available']]
        self.assertEqual(len(missing),12)
        self.assertTrue(all(r['component']=='bnc_spoken' and r['bits_per_byte'] is None for r in missing))
        self.assertTrue(all(r['difference_bpb']>0 for r in report['paired_comparisons']))
        a=report['seed_aggregates'][0]
        self.assertAlmostEqual(a['seed_standard_deviation'],abs(a['seed_43_bpb']-a['seed_42_bpb'])/math.sqrt(2))

    def test_missing_duplicate_and_wrong_run_inventory(self):
        for mutation in (lambda r:r.pop(), lambda r:r.append(copy.deepcopy(r[0])), lambda r:r[0].update(seed=44)):
            with self.subTest(mutation=mutation):
                args=list(fixture());mutation(args[1])
                with self.assertRaises(ValueError):tables.derive(*args)

    def test_block_counts_flags_components_duplicates_and_nonfinite_losses(self):
        mutations=[lambda r:r.update(bytes=101),lambda r:r.update(tokens=51),
                   lambda r:r.update(overlap_filtered_eligible=0),lambda r:r.update(component='other'),
                   lambda r:r.update(nll=float('nan')),lambda r:r.update(nll=-1),lambda r:r.update(bits_per_byte=0)]
        for change in mutations:
            args=list(fixture());change(args[1][0]['blocks'][0])
            with self.subTest(change=change),self.assertRaises(ValueError):tables.derive(*args)
        args=list(fixture());args[1][0]['blocks'][1]=copy.deepcopy(args[1][0]['blocks'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):tables.derive(*args)

    def test_score_exclusions_nulls_and_mean_cannot_be_silently_changed(self):
        for mode in ('score','exclusions','null','mean','seed-order','extra-aggregate','missing-aggregate'):
            args=list(fixture());result=args[1][0];summary=args[2]
            if mode=='score':result['components']['all']['official']['bits_per_byte']+=.1
            elif mode=='exclusions':result['components']['all']['excluded_bytes']+=1
            elif mode=='null':result['components']['bnc_spoken']['overlap_filtered']={}
            elif mode=='mean':summary['aggregates'][0]['mean_bpb']+=.1
            elif mode=='seed-order':summary['aggregates'][0]['seeds'].reverse()
            elif mode=='extra-aggregate':summary['aggregates'].append(copy.deepcopy(summary['aggregates'][0]))
            else:summary['aggregates'].pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):tables.derive(*args)

    def test_pair_sign_inventory_and_interval_provenance_are_checked(self):
        for name,value in [('difference_bpb',-.5),('replicates',50),('seed',17),('lower_95',float('nan')),
                           ('lower_95',99),('unit','independent documents')]:
            args=list(fixture());args[2]['paired_comparisons'][0][name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):tables.derive(*args)
        args=list(fixture());args[2]['paired_comparisons'].pop()
        with self.assertRaises(ValueError):tables.derive(*args)

    def test_incomplete_gate_reads_no_payload_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'export'
            with patch.object(Path,'read_bytes',side_effect=AssertionError('No read before inventory gate')):
                with self.assertRaisesRegex(ValueError,'Complete BabyLM'):tables.export(root,out)
            self.assertFalse(out.exists())

    def test_file_export_hashes_csv_nulls_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root);out=root/'export';tables.export(root,out)
            manifest=json.loads((out/'manifest.json').read_bytes())
            self.assertEqual(set(manifest),{'tables.json','scores.csv','seed_aggregates.csv','paired_comparisons.csv'})
            for name,record in manifest.items():
                data=(out/name).read_bytes();self.assertEqual(record,dict(bytes=len(data),sha256=tables.digest(data)))
            rows=list(csv.DictReader(io.StringIO((out/'scores.csv').read_text())))
            self.assertEqual(len(rows),168)
            self.assertEqual(sum(r['bits_per_byte']=='' for r in rows),12)
            original={p.name:p.read_bytes() for p in out.iterdir()}
            with self.assertRaisesRegex(ValueError,'Preserve existing'):tables.export(root,out)
            self.assertEqual(original,{p.name:p.read_bytes() for p in out.iterdir()})

    def test_bound_metadata_selection_and_source_corruption_refuse(self):
        for path in (tables.CACHE/'blocks.jsonl',tables.REPORTS/'selection.json',Path('flm/model.py')):
            with tempfile.TemporaryDirectory() as d,self.subTest(path=path):
                root=Path(d);write_fixture(root);p=root/path;p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaises(ValueError):tables.export(root,root/'export')
                self.assertFalse((root/'export').exists())

    def test_late_input_mutation_refuses_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root);original=tables.derive
            def change(*args):
                report=original(*args);p=root/tables.REPORTS/'summary.json';p.write_bytes(p.read_bytes()+b' ');return report
            with patch.object(tables,'derive',side_effect=change):
                with self.assertRaisesRegex(ValueError,'changed during export'):tables.export(root,root/'export')
            self.assertFalse((root/'export').exists())

    def test_shortened_source_binding_and_wrong_exposure_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root)
            p=root/tables.REPORTS/'test-10m-flm-s42.json';result=json.loads(p.read_bytes())
            del result['identity']['evaluator_source_sha256']['model.py']
            p.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'source inventory'):tables.collect(root)
        args=list(fixture())
        # Consistent changes to selection, run and summary still cannot change the stated exposure.
        for row in (args[0]['runs'][0],args[1][0],args[1][0]['identity'],args[2]['runs'][0],args[2]['runs'][0]['identity']):
            row['final_exposure']['presented_tokens']=1536
        with self.assertRaisesRegex(ValueError,'training exposure'):tables.derive(*args)


if __name__=='__main__':unittest.main()
