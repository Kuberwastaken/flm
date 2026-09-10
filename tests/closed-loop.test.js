import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateFeedback, feedbackRows, degrees, CONTROLLERS, SCENARIOS } from '../web/closed-loop.js';

test('the shipped physical cohort renders all scenarios with its actual evidence', () => {
  const report = validateFeedback(JSON.parse(readFileSync(new URL('../public/research/closed-loop.json', import.meta.url), 'utf8')));
  for (const scenario of Object.keys(SCENARIOS)) {
    const rows = feedbackRows(report, scenario);
    assert.equal(rows.length, 5);
    assert.equal(rows.filter(row => row.frozen).length, 4);
    for (const model of ['bptt', 'reservoir', 'eligibility']) {
      const row = rows.find(item => item.model === model);
      assert.equal(row.live.physical_qpos_sha256, rows.at(-1).live.physical_qpos_sha256);
      assert.ok(row.contrast.live_minus_frozen_error_rad < 0);
    }
  }
});

// Analytic fixtures exercise reporting invariants; these are never published as observations.
function fixture() {
  const report = { schema_version: 1, complete: true, conditions: 27, rows: [], contrasts: [], physical_reproducibility_repeats: 1, repeat_arrays_exact: true, neural_models: [], runtime_parity: [], identical_physical_trajectory_groups: [] };
  for (const key of ['identity_sha256', 'protocol_sha256', 'source_report_sha256', 'verifier_sha256', 'generator_sha256']) report[key] = 'a'.repeat(64);
  for (const model of Object.keys(CONTROLLERS)) {
    if (model !== 'scripted') {
      report.neural_models.push({ id: model, training_seed: 17, checkpoint_step: model === 'initial' ? 0 : 900, sha256: 'a'.repeat(64), source_checkpoint_sha256: 'b'.repeat(64), graph_sha256: 'c'.repeat(64) });
      report.runtime_parity.push({ model, frames: 128, argmax_exact: true, maximum_absolute_error: { fast: 1e-8, slow: 1e-8, logits: 1e-8 } });
    }
    for (const scenario of Object.keys(SCENARIOS)) {
      for (const pose_mode of model === 'scripted' ? ['live'] : ['live', 'frozen']) {
        const label = `${model}-${pose_mode}-${scenario}`;
        const phases = (scenario === 'switch' ? [[0, 1, 20], [1, 2, -20]] : [[0, 2, scenario === 'negative' ? -20 : 20]]).map(([start_s, end_s, y]) => ({ start_s, end_s, target_mm: [40, y], start_distance_mm: 40, end_distance_mm: 35, progress_toward_target_mm: 5 }));
        report.rows.push({ case: { label, model, pose_mode, scenario }, metrics: { observed_seconds: 2, mean_absolute_bearing_error_rad: pose_mode === 'live' ? .2 : 1.1, final_absolute_bearing_error_rad: .1, final_target_distance_mm: 35, minimum_thorax_height_mm: .6, minimum_thorax_up_z: .95, mean_contact_count: 6, heading_change_rad: .3, descending_signal_switches: 12, phase_progress: phases }, controller_replay: { frames: 400, decisions: 36, sensory_and_commands_exact: true, maximum_state_logit_error: 0 }, trajectory_sha256: 'a'.repeat(64), report_sha256: 'b'.repeat(64), physical_qpos_sha256: report.rows.length.toString(16).padStart(64, '0') });
      }
      if (model !== 'scripted') report.contrasts.push({ model, scenario, live_minus_frozen_error_rad: -.9 });
    }
  }
  return report;
}

test('physical report keeps the scripted reference separate from unrun frozen controls', () => {
  const report = validateFeedback(fixture());
  for (const scenario of Object.keys(SCENARIOS)) {
    const rows = feedbackRows(report, scenario);
    assert.equal(rows.length, 5); assert.equal(rows.at(-1).model, 'scripted');
    assert.equal(rows.at(-1).frozen, undefined); assert.equal(rows.at(-1).contrast, undefined);
    assert.equal(rows[0].live.case.scenario, scenario);
  }
  assert.equal(degrees(Math.PI / 2), '90.00°'); assert.equal(degrees(-Math.PI / 4, true), '-45.00°');
});

test('physical report does not count identical body paths as distinct behaviors', () => {
  const report = fixture(), rows = report.rows;
  rows[1].physical_qpos_sha256 = rows[0].physical_qpos_sha256;
  assert.throws(() => validateFeedback(report), /identical physical trajectories/);
  report.identical_physical_trajectory_groups = [[rows[1].case.label, rows[0].case.label]];
  assert.equal(validateFeedback(report), report);
});

test('physical report rejects incomplete, mislabeled and inconsistent evidence', () => {
  for (const mutate of [
    r => { r.complete = false; }, r => r.rows.pop(), r => { r.rows[1] = r.rows[0]; },
    r => { r.repeat_arrays_exact = false; }, r => { r.identity_sha256 = ''; },
    r => { r.neural_models[0].checkpoint_step = 900; }, r => { r.neural_models[1].training_seed = 29; },
    r => { r.neural_models[1].graph_sha256 = 'd'.repeat(64); }, r => { r.runtime_parity[1].argmax_exact = false; },
    r => { r.rows[0].case.pose_mode = 'frozen'; }, r => { r.rows[0].metrics.observed_seconds = 1; },
    r => { r.rows[0].controller_replay.decisions = 35; }, r => { r.rows[0].controller_replay.sensory_and_commands_exact = false; },
    r => { r.rows[0].metrics.mean_absolute_bearing_error_rad = NaN; }, r => { r.rows[0].metrics.final_absolute_bearing_error_rad = 4; },
    r => { r.rows[0].metrics.phase_progress[0].progress_toward_target_mm = 8; },
    r => { r.rows[0].metrics.phase_progress[0].target_mm = [40, -20]; },
    r => { r.rows[0].physical_qpos_sha256 = 'x'; }, r => { r.contrasts[0].live_minus_frozen_error_rad = .9; },
    r => { r.contrasts[1] = r.contrasts[0]; }, r => r.contrasts.pop()
  ]) { const report = fixture(); mutate(report); assert.throws(() => validateFeedback(report)); }
});
