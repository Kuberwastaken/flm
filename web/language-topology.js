import { loadLanguageStructure } from './language-structure.js';

const $ = id => document.getElementById(id);
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(4)}`;
const conditionName = run => run.reference ? 'Measured · fast and slow' : run.variant === 'no_slow' ? 'Retrained without slow state' : `Rewired · graph ${run.graph_seed}`;

function tableRow(values) {
  const row = document.createElement('tr');
  values.forEach((value, index) => {
    const cell = document.createElement(index ? 'td' : 'th');
    if (!index) cell.scope = 'row';
    cell.textContent = value; row.append(cell);
  });
  return row;
}

export function validateFinalResults(report, expectedIdentity) {
  if (!/^[a-f0-9]{64}$/.test(report.study_identity_sha256 ?? '') || (expectedIdentity !== undefined && report.study_identity_sha256 !== expectedIdentity)) throw new Error('Language test results belong to a different study identity.');
  if (!Array.isArray(report.runs) || report.runs.length !== 10) throw new Error('All ten selected language conditions are required.');
  const lookup = new Map(report.runs.map(run => [run.label, run]));
  if (lookup.size !== 10) throw new Error('Duplicate language test condition.');
  for (const seed of [42, 43]) {
    for (const [label, graphSeed, variant, reference] of [
      [`measured-s${seed}`, null, 'flm', true], [`no-slow-s${seed}`, null, 'no_slow', false],
      ...[101, 103, 107].map(graph => [`null${graph}-s${seed}`, graph, 'flm', false])
    ]) {
      const run = lookup.get(label), score = run?.score;
      if (!run || run.seed !== seed || run.graph_seed !== graphSeed || run.variant !== variant || run.reference !== reference) throw new Error('Language test design does not match the declared conditions.');
      if (!score || !Number.isFinite(score.bits_per_byte) || score.bits_per_byte < 0 || !Number.isFinite(score.token_perplexity) || score.token_perplexity < 1 || score.documents?.length !== 60 || new Set(score.documents.map(article => article.document)).size !== 60) throw new Error('Incomplete or invalid language test scores.');
      if (run.checkpoint_step < 500 || run.checkpoint_step > 6000 || run.checkpoint_step % 500) throw new Error('Invalid selected language checkpoint.');
    }
  }
  for (const [key, count] of [['topology_contrasts', 6], ['slow_state_contrasts', 2]]) {
    const contrasts = report[key];
    if (!Array.isArray(contrasts) || contrasts.length !== count) throw new Error('Missing paired language contrasts.');
    const seen = new Set();
    for (const contrast of contrasts) {
      const seed = contrast.training_seed;
      const control = key === 'topology_contrasts' ? `null${contrast.graph_seed}-s${seed}` : `no-slow-s${seed}`;
      if (!lookup.has(control) || seen.has(control)) throw new Error('Invalid or duplicate paired language contrast.');
      seen.add(control);
      const expected = lookup.get(`measured-s${seed}`).score.bits_per_byte - lookup.get(control).score.bits_per_byte;
      if (![contrast.difference_bpb, contrast.lower_95, contrast.upper_95].every(Number.isFinite) || contrast.lower_95 > contrast.upper_95 || Math.abs(expected - contrast.difference_bpb) > 1e-8) throw new Error('Paired language contrast does not match the selected scores.');
    }
  }
  for (const [field, rows] of [['topology_mean_difference_bpb', report.topology_contrasts], ['slow_state_mean_difference_bpb', report.slow_state_contrasts]]) {
    const mean = rows.reduce((sum, row) => sum + row.difference_bpb, 0) / rows.length;
    if (!Number.isFinite(report[field]) || Math.abs(mean - report[field]) > 1e-8) throw new Error('Language contrast mean is inconsistent.');
  }
  return report;
}

function renderFinalResults(report, expectedIdentity) {
  validateFinalResults(report, expectedIdentity);
  const runs = report.runs.map(run => tableRow([conditionName(run), run.seed,
    run.checkpoint_step.toLocaleString(), run.score.bits_per_byte.toFixed(4), run.score.token_perplexity.toFixed(2)]));
  const contrasts = [...report.topology_contrasts, ...report.slow_state_contrasts].map(point => tableRow([
    point.graph_seed ? `Measured − graph ${point.graph_seed}` : 'Measured − retrained no slow',
    point.training_seed, signed(point.difference_bpb), `${signed(point.lower_95)} to ${signed(point.upper_95)}`]));
  const graphMeans = [101, 103, 107].map(seed => {
    const rows = report.topology_contrasts.filter(row => row.graph_seed === seed);
    return tableRow([seed, signed(rows.reduce((sum, row) => sum + row.difference_bpb, 0) / rows.length)]);
  });
  $('language-topology-test-rows').replaceChildren(...runs);
  $('language-topology-intervals').replaceChildren(...contrasts);
  $('language-topology-graph-means').replaceChildren(...graphMeans);
  $('language-topology-mean').textContent = `Mean measured minus rewired loss: ${signed(report.topology_mean_difference_bpb)} bits/byte across the six graph/training-seed combinations. Mean measured minus retrained no-slow loss: ${signed(report.slow_state_mean_difference_bpb)} bits/byte across two training seeds. Negative differences favor measured fast/slow wiring.`;
  $('language-topology-final').hidden = false;
}

export function matchedValidation(report) {
  return report.runs.filter(run => !run.reference).map(control => {
    const measured = report.runs.find(run => run.reference && run.seed === control.seed);
    const shared = control.validation.filter(point => point.step <= control.saved_step &&
      measured?.validation.some(other => other.step === point.step));
    const point = shared.at(-1);
    const reference = measured?.validation.find(other => other.step === point?.step);
    return { control, point, reference, difference: point && reference ? reference.bits_per_byte - point.bits_per_byte : null };
  });
}

export async function loadLanguageTopology() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/language-topology-progress.json`, { cache: 'no-store' });
    if (!response.ok) throw new Error('Language topology snapshot unavailable.');
    const report = await response.json();
    void loadLanguageStructure(report.study_identity_sha256);
    $('language-topology-status').textContent = `${report.completed_new_runs} of ${report.new_runs} new runs complete; two completed measured references. Snapshot: ${new Date(report.snapshot_utc).toLocaleString()}. Test comparison: ${report.test_status}.`;
    const rows = matchedValidation(report).map(({ control, point, reference, difference }) => {
      const label = control.variant === 'no_slow' ? 'Retrained without slow state' : `Rewired · graph ${control.graph_seed}`;
      const cells = [label, control.seed, point ? point.step.toLocaleString() : 'Awaiting checkpoint',
        reference ? reference.bits_per_byte.toFixed(4) : '—', point ? point.bits_per_byte.toFixed(4) : '—',
        difference === null ? '—' : signed(difference)];
      return tableRow(cells);
    });
    $('language-topology-rows').replaceChildren(...rows);
    if (report.test_status === 'Published') {
      if (report.completed_new_runs !== 8) throw new Error('The language control study is incomplete.');
      const response = await fetch(`${import.meta.env.BASE_URL}research/language-topology-results.json`);
      if (!response.ok) throw new Error('The completed language test report is unavailable.');
      renderFinalResults(await response.json(), report.study_identity_sha256);
    }
  } catch (error) { $('language-topology-status').textContent = error.message; }
}
