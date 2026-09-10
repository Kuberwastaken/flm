export const CONTROLLERS = Object.freeze({ initial: 'Untrained', bptt: 'BPTT', reservoir: 'Fixed core', eligibility: 'Eligibility', scripted: 'Scripted reference' });
export const SCENARIOS = Object.freeze({ positive: 'Fixed target at (40, 20) mm', negative: 'Fixed target at (40, −20) mm', switch: 'Target changes at 1 second' });
const NEURAL = Object.keys(CONTROLLERS).filter(name => name !== 'scripted');
const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const near = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-9;
const requireEvidence = (condition, message) => { if (!condition) throw new Error(`Physical feedback evidence is inconsistent: ${message}.`); };

// Reject incomplete cohorts and recompute the displayed contrasts and path groups.
// The downloadable Python audit additionally replays every neural input and command.
export function validateFeedback(report) {
  requireEvidence(report.schema_version === 1 && report.complete === true && report.conditions === 27 && report.rows?.length === 27, 'all 27 conditions are required');
  requireEvidence(report.physical_reproducibility_repeats === 1 && report.repeat_arrays_exact === true, 'the physical repeat is required');
  for (const key of ['identity_sha256', 'protocol_sha256', 'source_report_sha256', 'verifier_sha256', 'generator_sha256']) requireEvidence(hash(report[key]), `missing ${key}`);
  const models = report.neural_models;
  requireEvidence(models?.length === 4 && new Set(models.map(model => model.id)).size === 4, 'four fixed neural checkpoints are required');
  for (const name of NEURAL) {
    const model = models.find(item => item.id === name);
    requireEvidence(model?.training_seed === 17 && model.checkpoint_step === (name === 'initial' ? 0 : 900) && hash(model.sha256) && hash(model.source_checkpoint_sha256) && hash(model.graph_sha256), 'checkpoint identity');
  }
  requireEvidence(new Set(models.map(model => model.graph_sha256)).size === 1, 'shared measured graph');
  requireEvidence(report.runtime_parity?.length === 4 && new Set(report.runtime_parity.map(item => item.model)).size === 4, 'all runtime parity checks');
  for (const name of NEURAL) {
    const parity = report.runtime_parity.find(item => item.model === name);
    requireEvidence(parity?.frames === 128 && parity.argmax_exact === true && ['fast', 'slow', 'logits'].every(key => Number.isFinite(parity.maximum_absolute_error?.[key]) && parity.maximum_absolute_error[key] >= 0), 'verified numerical bridge');
  }
  const expected = new Set();
  for (const model of Object.keys(CONTROLLERS)) for (const scenario of Object.keys(SCENARIOS)) for (const mode of model === 'scripted' ? ['live'] : ['live', 'frozen']) expected.add(`${model}-${mode}-${scenario}`);
  const paths = new Map();
  for (const row of report.rows) {
    const trial = row.case, metrics = row.metrics, replay = row.controller_replay;
    requireEvidence(trial && trial.label === `${trial.model}-${trial.pose_mode}-${trial.scenario}` && expected.delete(trial.label), 'missing, duplicate or mislabeled condition');
    requireEvidence(metrics?.observed_seconds === 2 && replay?.frames === 400 && replay.decisions === 36 && replay.sensory_and_commands_exact === true && Number.isFinite(replay.maximum_state_logit_error) && replay.maximum_state_logit_error >= 0, 'complete causal replay');
    for (const key of ['mean_absolute_bearing_error_rad', 'final_absolute_bearing_error_rad']) requireEvidence(Number.isFinite(metrics[key]) && metrics[key] >= 0 && metrics[key] <= Math.PI, 'angular measurements');
    for (const key of ['final_target_distance_mm', 'minimum_thorax_height_mm', 'minimum_thorax_up_z', 'mean_contact_count', 'heading_change_rad']) requireEvidence(Number.isFinite(metrics[key]), 'finite physical measurements');
    requireEvidence(metrics.final_target_distance_mm >= 0 && Number.isInteger(metrics.descending_signal_switches) && metrics.descending_signal_switches >= 0 && metrics.descending_signal_switches <= 36, 'distance and command count');
    const phases = metrics.phase_progress;
    requireEvidence(phases?.length === (trial.scenario === 'switch' ? 2 : 1), 'complete target phases');
    phases.forEach((phase, i) => {
      const targetY = trial.scenario === 'negative' || (trial.scenario === 'switch' && i === 1) ? -20 : 20;
      requireEvidence(phase.start_s === (i ? 1 : 0) && phase.end_s === (phases.length === 2 && !i ? 1 : 2) && phase.target_mm?.[0] === 40 && phase.target_mm?.[1] === targetY && phase.start_distance_mm >= 0 && phase.end_distance_mm >= 0 && near(phase.start_distance_mm - phase.end_distance_mm, phase.progress_toward_target_mm), 'phase-specific progress');
    });
    for (const key of ['trajectory_sha256', 'report_sha256', 'physical_qpos_sha256']) requireEvidence(hash(row[key]), 'trace identity');
    if (!paths.has(row.physical_qpos_sha256)) paths.set(row.physical_qpos_sha256, []);
    paths.get(row.physical_qpos_sha256).push(trial.label);
  }
  requireEvidence(expected.size === 0 && report.contrasts?.length === 12, 'all paired comparisons');
  const paired = new Set();
  for (const contrast of report.contrasts) {
    const key = `${contrast.model}-${contrast.scenario}`;
    requireEvidence(NEURAL.includes(contrast.model) && Object.hasOwn(SCENARIOS, contrast.scenario) && !paired.has(key), 'unique live/frozen pairs'); paired.add(key);
    const live = report.rows.find(row => row.case.label === `${contrast.model}-live-${contrast.scenario}`);
    const frozen = report.rows.find(row => row.case.label === `${contrast.model}-frozen-${contrast.scenario}`);
    requireEvidence(near(contrast.live_minus_frozen_error_rad, live.metrics.mean_absolute_bearing_error_rad - frozen.metrics.mean_absolute_bearing_error_rad), 'live minus frozen contrast');
  }
  const canonical = groups => JSON.stringify(groups.map(group => [...group].sort()).sort((a, b) => a[0].localeCompare(b[0])));
  requireEvidence(Array.isArray(report.identical_physical_trajectory_groups) && canonical(report.identical_physical_trajectory_groups) === canonical([...paths.values()].filter(group => group.length > 1)), 'identical physical trajectories');
  return report;
}

export function feedbackRows(report, scenario) {
  requireEvidence(Object.hasOwn(SCENARIOS, scenario), 'known scenario');
  return Object.keys(CONTROLLERS).map(model => ({ model,
    live: report.rows.find(row => row.case.label === `${model}-live-${scenario}`),
    frozen: report.rows.find(row => row.case.label === `${model}-frozen-${scenario}`),
    contrast: report.contrasts.find(row => row.model === model && row.scenario === scenario)
  }));
}

export function degrees(value, signed = false) {
  return `${signed && value > 0 ? '+' : ''}${(value * 180 / Math.PI).toFixed(2)}°`;
}

function tableRow(values) {
  const element = document.createElement('tr');
  values.forEach((value, index) => {
    const cell = document.createElement(index ? 'td' : 'th');
    if (!index) cell.scope = 'row';
    cell.textContent = value; element.append(cell);
  });
  return element;
}

export async function loadFeedbackStudy() {
  const $ = id => document.getElementById(id);
  const section = $('feedback-study');
  if (!section) return;
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/closed-loop.json`);
    if (!response.ok) throw new Error('The complete physical feedback report is unavailable.');
    const report = validateFeedback(await response.json());
    const render = () => {
      const scenario = $('feedback-scenario').value, rows = feedbackRows(report, scenario);
      $('feedback-rows').replaceChildren(...rows.map(item => tableRow([CONTROLLERS[item.model], degrees(item.live.metrics.mean_absolute_bearing_error_rad), item.frozen ? degrees(item.frozen.metrics.mean_absolute_bearing_error_rad) : 'Not run', item.contrast ? degrees(item.contrast.live_minus_frozen_error_rad, true) : 'Not run'])));
      $('feedback-outcomes').replaceChildren(...rows.flatMap(item => [item.live, item.frozen].filter(Boolean).map(row => tableRow([
        `${CONTROLLERS[item.model]} · ${row.case.pose_mode} pose`, degrees(row.metrics.final_absolute_bearing_error_rad), `${row.metrics.final_target_distance_mm.toFixed(2)} mm`, row.metrics.phase_progress.map(phase => `${phase.start_s}–${phase.end_s} s: ${phase.progress_toward_target_mm.toFixed(2)} mm`).join('; '), String(row.metrics.descending_signal_switches)
      ]))));
      const equivalent = rows.filter(item => item.model !== 'scripted' && item.live.physical_qpos_sha256 === rows.at(-1).live.physical_qpos_sha256).map(item => CONTROLLERS[item.model]);
      $('feedback-selection').textContent = `${SCENARIOS[scenario]}. Lower mean angular error is better; negative live-minus-frozen differences favor pose feedback. ${equivalent.length ? `${equivalent.join(', ')} produce exactly the same recorded body trajectory as the scripted reference in this scenario.` : 'No neural condition has an exactly identical recorded body trajectory to the scripted reference in this scenario.'}`;
      $('feedback-figure').src = `${import.meta.env.BASE_URL}research/figures/closed-loop-${scenario}.svg`;
      $('feedback-figure').alt = `${SCENARIOS[scenario]}: all five live-pose error traces, and the eligibility network's live and frozen paths.`;
      $('feedback-figure-link').href = `${import.meta.env.BASE_URL}research/figures/closed-loop-${scenario}.png`;
    };
    $('feedback-scenario').addEventListener('change', render); render();
    $('feedback-status').textContent = '27 conditions completed; all sensory inputs, 10,800 control frames and 972 delayed decisions independently replayed. One additional physical repeat matches every recorded array exactly. One model seed and one physics seed.';
    section.hidden = false;
  } catch (error) {
    $('feedback-status').textContent = error.message; $('feedback-scenario').disabled = true;
    $('feedback-results').hidden = true; section.hidden = false;
  }
}
