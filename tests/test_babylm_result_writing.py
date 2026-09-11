"""Regression for the real result-writing path with artificial scored blocks."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flm import babylm_test as study


class ResultWritingTests(unittest.TestCase):
    def test_main_writes_all_results_when_selection_already_contains_checkpoint_step(self):
        selected=[dict(scale=scale,seed=seed,variant=variant,checkpoint_step=11500,
            checkpoint_sha256='fixture',selection_validation_bpb=2.)
            for scale in study.SCALES for seed in study.SEEDS for variant in study.VARIANTS]
        selection=dict(runs=selected,manifest_sha256={'test':'fixture'})
        blocks=[dict(document='fixture',component='fixture',tokens=2,bytes=3,nll=1.,overlap_filtered_eligible=True)]
        model=SimpleNamespace(parameter_card=lambda:{'trainable_parameters':7})
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            output=Path(folder)
            for name,value in [('OUTPUT',output),('freeze_selection',lambda:selection),
                ('Lexicon',lambda _:SimpleNamespace(sha256='fixture')),
                ('read_mmap',lambda *args:([],[],dict(text_tokens=2,utf8_bytes=3))),
                ('restore_selected',lambda *args:(model,dict(step=11500))),
                ('scored_blocks',lambda *args:(blocks,0.01))]:
                stack.enter_context(patch.object(study,name,value))
            stack.enter_context(patch('sys.argv',['babylm_test','--threads','1']))
            stack.enter_context(patch.object(study.subprocess,'check_output',return_value='fixture-commit'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            study.main()
            summary=study.read_json(output/'summary.json')
            self.assertEqual(len(summary['runs']),12)
            self.assertEqual(len(summary['paired_comparisons']),16)
            for row in selected:
                result=study.read_json(output/f"test-{row['scale']}-{row['variant']}-s{row['seed']}.json")
                self.assertEqual(result['checkpoint_step'],11500)
                self.assertEqual(result['identity']['checkpoint_step'],11500)
                self.assertEqual(result['blocks'],blocks)
                self.assertEqual(result['components']['all']['official']['tokens'],2)
                for name,value in row.items():self.assertEqual(result[name],value)


if __name__=='__main__':unittest.main()
