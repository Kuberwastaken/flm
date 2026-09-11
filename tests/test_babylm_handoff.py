"""Supervisor fixtures; no real benchmark, sampling or timing command runs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch

import psutil

from scripts import continue_babylm_research as handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.output=self.root/'work/handoff'
        for name in handoff.SOURCES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name,encoding='utf8')
        self.prompts=[dict(id=f'prompt-{index}') for index in range(12)]
        self.write('data/prompts/babylm-original.json',dict(prompts=self.prompts))
        for scale,variant,seed in handoff.RUNS:
            path=self.root/f'runs/babylm-{scale}/{variant}-s{seed}'
            path.mkdir(parents=True);(path/'best.pt').write_bytes(b'artificial payload')
            self.write((path/'complete.json').relative_to(self.root),dict(steps=12000,best_checkpoint_sha256=handoff.sha(path/'best.pt')))
        self.target=dict(pid=123,created=1.,argv=['python','-m','flm.babylm_suite'],cwd=str(self.root))
        self.events=[];self.executed=[];self.waited=False

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf8')

    def event(self,event,**details): self.events.append(dict(event=event,**details))

    def waiter(self,target,event):
        self.assertEqual(target,self.target);self.waited=True;event('fixture_trainer_exited')

    def runner(self,root,stage,output,event):
        self.assertTrue(self.waited); self.executed.append(stage[0])
        rows=[dict(scale=scale,variant=variant,seed=seed) for scale,variant,seed in sorted(handoff.RUNS)]
        if stage[0]=='babylm-test':record=dict(runs=rows)
        elif stage[0]=='babylm-samples':
            record=dict(models=[dict(selected=row,passages=[dict(prompt_id=p['id'],sampling_seed=s)
                for p in self.prompts for s in (17,29)]) for row in rows])
        else:record=dict(conditions=[dict(selection=f'graph-{index}') for index in range(64)],
                         benchmark_budget_selected=False,training_matrix_frozen=False)
        self.write(stage[2],record)

    def run_fixture(self,**kwargs):
        with patch.object(handoff,'capture_trainer',return_value=self.target):
            return handoff.run(self.root,123,self.output,runner=kwargs.get('runner',self.runner),
                               waiter=kwargs.get('waiter',self.waiter),idle=kwargs.get('idle',lambda *args:None))

    def test_complete_handoff_waits_then_runs_only_declared_stages_in_order(self):
        self.run_fixture()
        self.assertEqual(self.executed,['babylm-test','babylm-samples','selection-pilot'])
        complete=handoff.read(self.output/'complete.json')
        self.assertEqual([row['path'] for row in complete['results']],[stage[2] for stage in handoff.STAGES])
        self.assertTrue(all(handoff.sha(self.root/row['path'])==row['sha256'] for row in complete['results']))
        self.assertFalse((self.root/'reports/selection-language/study-identity.json').exists())

    def test_completed_training_mode_runs_without_a_live_trainer(self):
        self.waited=True
        with patch.object(handoff,'capture_trainer') as capture, patch.object(handoff,'wait_for_exit') as wait:
            handoff.run(self.root,None,self.output,runner=self.runner,waiter=wait,idle=lambda *args:None)
        capture.assert_not_called();wait.assert_not_called()
        self.assertEqual(self.executed,['babylm-test','babylm-samples','selection-pilot'])
        self.assertIsNone(handoff.read(self.output/'identity.json')['trainer'])

    def test_completed_training_mode_rejects_missing_fit_before_creating_output(self):
        (self.root/'runs/babylm-100m/transformer-s43/complete.json').unlink()
        with self.assertRaisesRegex(ValueError,'all twelve'):
            handoff.run(self.root,None,self.output,runner=self.runner,idle=lambda *args:None)
        self.assertFalse(self.output.exists());self.assertEqual(self.executed,[])

    def test_completed_training_mode_does_not_bypass_idle_gate(self):
        def busy(*args): raise RuntimeError('Fixture busy job')
        with self.assertRaisesRegex(RuntimeError,'busy job'):
            handoff.run(self.root,None,self.output,runner=self.runner,idle=busy)
        self.assertEqual(self.executed,[]);self.assertFalse((self.output/'complete.json').exists())

    def test_incomplete_trainer_exit_does_not_launch_a_stage(self):
        path=self.root/'runs/babylm-100m/transformer-s43/complete.json';path.rename(path.with_suffix('.missing'))
        with self.assertRaisesRegex(ValueError,'before all twelve'):self.run_fixture()
        self.assertEqual(self.executed,[]);self.assertFalse((self.output/'complete.json').exists())
        self.assertIn('stopped',(self.output/'events.jsonl').read_text())

    def test_changed_source_or_checkpoint_during_wait_prevents_test_access(self):
        def changed(target,event):
            self.waiter(target,event);(self.root/'flm/model.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'research source changed'):self.run_fixture(waiter=changed)
        self.assertEqual(self.executed,[])
        (self.root/'runs/babylm-10m/flm-s42/best.pt').write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError,'checksum changed'):handoff.verify_training(self.root)

    def test_failed_child_stops_without_retry_or_later_commands(self):
        def failed(root,stage,output,event):
            self.executed.append(stage[0]);raise subprocess.CalledProcessError(3,['artificial-command'])
        with self.assertRaises(subprocess.CalledProcessError):self.run_fixture(runner=failed)
        self.assertEqual(self.executed,['babylm-test'])
        self.assertFalse((self.root/'reports/babylm/samples.json').exists())
        self.assertFalse((self.root/'reports/selection-pilot/timing.json').exists())

    def test_existing_output_is_preserved_and_not_rerun(self):
        self.write('reports/babylm/summary.json',{'existing':'retain'})
        before=(self.root/'reports/babylm/summary.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture()
        self.assertEqual(self.executed,[]);self.assertEqual((self.root/'reports/babylm/summary.json').read_bytes(),before)

    def test_incomplete_stage_output_or_late_source_change_stops_the_chain(self):
        def incomplete(root,stage,output,event):
            self.runner(root,stage,output,event);record=handoff.read(root/stage[2]);record['runs'].pop();self.write(stage[2],record)
        with self.assertRaisesRegex(ValueError,'Incomplete BabyLM test'):self.run_fixture(runner=incomplete)
        self.assertEqual(self.executed,['babylm-test'])
        def changed(root,stage,output,event):
            self.runner(root,stage,output,event);(root/'flm/model.py').write_text('changed')
        self.output=self.root/'work/second';(self.root/'reports/babylm/summary.json').rename(self.root/'retained-partial.json')
        with self.assertRaisesRegex(ValueError,'research source changed'):self.run_fixture(runner=changed)

    def test_sampling_and_pilot_inventories_cannot_be_shortened(self):
        self.waited=True
        for stage in handoff.STAGES[1:]:
            self.runner(self.root,stage,self.output,self.event)
            handoff.verify_output(self.root,stage)
            record=handoff.read(self.root/stage[2])
            if stage[0]=='babylm-samples':record['models'][0]['passages'].pop()
            else:record['conditions'].pop()
            self.write(stage[2],record)
            with self.assertRaisesRegex(ValueError,'Incomplete'):handoff.verify_output(self.root,stage)

    def test_capture_requires_exact_live_suite_module_and_directory(self):
        process=Mock(pid=123);process.create_time.return_value=1.
        process.cmdline.return_value=self.target['argv'];process.cwd.return_value=str(self.root)
        with patch.object(handoff.psutil,'Process',return_value=process):
            self.assertEqual(handoff.capture_trainer(self.root,123),self.target)
            process.cmdline.return_value=['python','-m','unittest']
            with self.assertRaises(ValueError):handoff.capture_trainer(self.root,123)
            process.cmdline.return_value=self.target['argv'];process.cwd.return_value=str(self.root/'other')
            with self.assertRaises(ValueError):handoff.capture_trainer(self.root,123)

    def test_pid_reuse_is_exit_but_access_denial_and_command_changes_are_not(self):
        process=Mock();process.create_time.return_value=2.
        with patch.object(handoff.psutil,'Process',return_value=process):
            self.assertIsNone(handoff.live_process(self.target))
            process.create_time.return_value=1.;process.cmdline.return_value=['different']
            with self.assertRaisesRegex(ValueError,'changed its command'):handoff.live_process(self.target)
        with patch.object(handoff.psutil,'Process',side_effect=psutil.AccessDenied(123)):
            with self.assertRaises(psutil.AccessDenied):handoff.live_process(self.target)
        with patch.object(handoff.psutil,'Process',side_effect=psutil.NoSuchProcess(123)):
            self.assertIsNone(handoff.live_process(self.target))

    def test_observed_child_must_exit_even_after_parent_disappears(self):
        child_identity=dict(self.target,pid=124,argv=['python','-m','flm.language_train'])
        child=Mock(pid=124);child.create_time.return_value=1.;child.cmdline.return_value=child_identity['argv']
        child.cwd.return_value=str(self.root);child.children.return_value=[]
        parent=Mock();parent.children.return_value=[child]
        step=0
        def live(identity):
            if identity['pid']==123:return parent if step==0 else None
            return child if step<2 else None
        def sleep(seconds):
            nonlocal step
            self.assertEqual(seconds,5);step+=1
        with patch.object(handoff,'live_process',side_effect=live):
            handoff.wait_for_exit(self.target,self.event,sleep=sleep)
        self.assertEqual(step,2);self.assertEqual(self.events[-1]['event'],'trainer_and_observed_children_exited')
        self.assertEqual(self.events[-1]['processes'],2)

    def test_known_competing_work_is_waited_on_without_repeated_events(self):
        active=[dict(self.target,pid=456)];observations=iter([active,active,[]]);sleeps=[]
        with patch.object(handoff,'busy_processes',side_effect=lambda root:next(observations)):
            handoff.wait_for_idle(self.root,self.event,sleep=sleeps.append)
        self.assertEqual(sleeps,[5,5]);self.assertEqual(len(self.events),1)

    def test_exclusive_supervisor_lease_and_existing_output_directory(self):
        with handoff.lease(self.root/'work/babylm-post-training.lock'):
            with self.assertRaisesRegex(RuntimeError,'supervisor holds'):self.run_fixture()
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):self.run_fixture()

    def test_unreadable_unrelated_process_does_not_inherit_previous_command(self):
        known=Mock(pid=7);known.cmdline.return_value=['python','-m','unittest'];known.cwd.return_value=str(self.root/'other')
        unknown=Mock(pid=8);unknown.cmdline.side_effect=psutil.AccessDenied(8)
        with patch.object(handoff.psutil,'process_iter',return_value=iter([known,unknown])):
            self.assertEqual(handoff.busy_processes(self.root),[])
        known.cwd.side_effect=psutil.AccessDenied(7)
        with patch.object(handoff.psutil,'process_iter',return_value=iter([known])):
            with self.assertRaises(psutil.AccessDenied):handoff.busy_processes(self.root)

    def test_package_change_while_waiting_prevents_the_next_command(self):
        with patch.object(handoff,'package_versions',side_effect=[{'torch':'original'},{'torch':'changed'}]):
            with self.assertRaisesRegex(ValueError,'package versions changed'):self.run_fixture()
        self.assertEqual(self.executed,[])


if __name__=='__main__':unittest.main()
