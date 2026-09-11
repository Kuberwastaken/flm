import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_babylm_completion as command


def declarations(root):
    for path in ('data/tokenizers/babylm-2026-4096/tokenizer.json',
                 'data/tokenizers/babylm-2026-4096/validation-panel.json',
                 'data/processed/babylm-2026-bpe/train-100m/manifest.json',
                 'data/processed/babylm-2026-bpe/validation/manifest.json',
                 'data/graphs/central-1024/graph.npz','docs/BABYLM-PROTOCOL.md'):
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('artificial '+path)
    protocol=command.declared_protocol(root,'100m')
    study=dict(dataset='BabyLM 2026 100m',steps=12000,seeds=[42,43],variants=['flm','gru','transformer'],
        protocol_sha256=command.sha(root/'docs/BABYLM-PROTOCOL.md'),tokenizer_sha256=protocol['tokenizer_sha256'],
        train_manifest_sha256=protocol['train_cache_sha256'],validation_manifest_sha256=protocol['validation_cache_sha256'],
        validation_panel_sha256=protocol['validation_panel_sha256'])
    p=root/'runs/babylm-100m/study.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(study))
    complete=dict(steps=12000,protocol=copy.deepcopy(protocol))
    run=dict(protocol=copy.deepcopy(protocol),seed=43,parameter_card=dict(config=dict(variant='flm')),test_set_used_for_training=False)
    return complete,run,study


class BabyLMCompletionCommandTests(unittest.TestCase):
    def test_incomplete_fit_refuses_before_metadata_reads_and_output_creation(self):
        with tempfile.TemporaryDirectory() as d,patch.object(command,'read',side_effect=AssertionError('No early metadata read')):
            root=Path(d);out=root/'audit.json'
            with self.assertRaisesRegex(ValueError,'all 24 validation'):command.write_audit(root,'100m',43,'flm',out)
            self.assertFalse(out.exists())

    def test_unsupported_condition_and_existing_output_refuse_before_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'registered'):command.audit(root,'100m',44,'flm')
            out=root/'audit.json';out.write_text('preserve')
            with patch.object(command,'audit') as audit:
                with self.assertRaisesRegex(ValueError,'Preserve existing'):command.write_audit(root,'100m',43,'flm',out)
                audit.assert_not_called();self.assertEqual(out.read_text(),'preserve')

    def test_protocol_uses_declared_budget_instead_of_another_runs_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);complete,run,_=declarations(root)
            protocol=command.validate_declaration(root,'100m',43,'flm',complete,run)
            self.assertEqual((protocol['steps'],protocol['batch'],protocol['sequence'],protocol['warmup']),(12000,16,96,16))
            self.assertEqual((protocol['learning_rate'],protocol['final_learning_rate'],protocol['threads']),(.002,.0002,4))
            for field,value in (('steps',11000),('warmup',15),('train_cache_sha256','changed'),('learning_rate',.001)):
                a,b=copy.deepcopy((complete,run));a['protocol'][field]=value;b['protocol'][field]=value
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'fixed protocol'):
                    command.validate_declaration(root,'100m',43,'flm',a,b)

    def test_run_identity_and_study_inventory_cannot_change_together_unnoticed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);complete,run,study=declarations(root)
            for field,value in (('seed',42),('test_set_used_for_training',True)):
                changed=copy.deepcopy(run);changed[field]=value
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'run identity'):
                    command.validate_declaration(root,'100m',43,'flm',complete,changed)
            study['variants']=['flm'];(root/'runs/babylm-100m/study.json').write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError,'Study declaration'):
                command.validate_declaration(root,'100m',43,'flm',complete,run)


if __name__=='__main__':unittest.main()
