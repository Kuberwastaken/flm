import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
from test_babylm_result_tables import fixture, write_fixture
from scripts import babylm_result_tables as tables
from scripts import babylm_result_figures as figures


class BabyLMResultFigureTests(unittest.TestCase):
    def test_same_seed_component_differences_preserve_all_panels_and_nulls(self):
        report=tables.derive(*fixture());points=figures.component_points(report)
        self.assertEqual(len(points),96)
        self.assertEqual(sum(r['difference_bpb'] is None for r in points),8)
        self.assertTrue(all(r['difference_bpb']>0 for r in points if r['difference_bpb'] is not None))
        changed=copy.deepcopy(report)
        row=next(r for r in changed['scores'] if (r['scale'],r['seed'],r['variant'],r['component'],r['subset']) ==
                 ('10m',43,'flm','childes','official'))
        row['bits_per_byte']+=.2
        after=figures.component_points(changed)
        modified=[(a,b) for a,b in zip(points,after) if a!=b]
        self.assertEqual(len(modified),2)
        for a,b in modified:
            self.assertEqual((b['scale'],b['seed'],b['component'],b['subset']),('10m',43,'childes','official'))
            self.assertAlmostEqual(b['difference_bpb']-a['difference_bpb'],.2)

    def test_missing_rows_and_unmatched_denominators_refuse(self):
        report=tables.derive(*fixture());report['scores'].pop()
        with self.assertRaisesRegex(ValueError,'complete 168-row'):figures.component_points(report)
        report=tables.derive(*fixture())
        next(r for r in report['scores'] if r['variant']=='flm' and r['component']=='childes')['bytes']+=1
        with self.assertRaisesRegex(ValueError,'denominators'):figures.component_points(report)

    def test_real_incomplete_gate_does_not_invoke_renderer_or_create_output(self):
        with tempfile.TemporaryDirectory() as d,patch.object(figures,'draw') as draw:
            root=Path(d);out=root/'out'
            with self.assertRaisesRegex(ValueError,'Complete BabyLM'):figures.build(root,out)
            draw.assert_not_called();self.assertFalse(out.exists())

    def test_existing_output_is_preserved_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as d,patch.object(figures,'collect') as collect:
            out=Path(d)/'out';out.mkdir();(out/'keep.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Preserve existing'):figures.build(Path(d),out)
            collect.assert_not_called();self.assertEqual((out/'keep.txt').read_text(),'preserve')

    def test_input_change_during_drawing_cannot_receive_a_completion_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root);out=root/'out'
            def changed(report,output,*,fixture):
                output.mkdir()
                for name in figures.PLOT_FILES:(output/name).write_bytes(b'artificial renderer')
                p=root/tables.REPORTS/'summary.json';p.write_bytes(p.read_bytes()+b' ')
                return figures.component_points(report)
            with patch.object(figures,'draw',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'changed during rendering'):figures.build(root,out)
            self.assertTrue((out/'babylm-pooled.svg').exists());self.assertFalse((out/'manifest.json').exists())

    def test_complete_export_manifest_binds_all_four_plots_and_underlying_values(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root);out=root/'out'
            def render(report,output,*,fixture):
                output.mkdir()
                for name in figures.PLOT_FILES:(output/name).write_bytes(('artificial '+name).encode())
                return figures.component_points(report)
            with patch.object(figures,'draw',side_effect=render):record=figures.build(root,out)
            self.assertEqual(len(record['component_points']),96)
            manifest=json.loads((out/'manifest.json').read_bytes())
            self.assertEqual(set(manifest),figures.PLOT_FILES|{'figure-data.json'})
            for name,expected in manifest.items():
                content=(out/name).read_bytes()
                self.assertEqual(expected,dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))

    def test_shortened_renderer_output_cannot_receive_completion_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root);out=root/'out'
            def render(report,output,*,fixture):
                output.mkdir();(output/'babylm-pooled.svg').write_bytes(b'artificial partial output')
                return figures.component_points(report)
            with patch.object(figures,'draw',side_effect=render):
                with self.assertRaisesRegex(ValueError,'four-file figure inventory'):figures.build(root,out)
            self.assertFalse((out/'manifest.json').exists())


if __name__=='__main__':unittest.main()
