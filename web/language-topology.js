import { loadLanguageStructure } from './language-structure.js';

const $ = id => document.getElementById(id);
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(4)}`;
const conditionName = run => run.reference ? 'Measured · fast and slow' : run.variant === 'no_slow' ? 'Retrained without slow state' : `Rewired · graph ${run.graph_seed}`;
const digest = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

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
  if (!digest(report.selection_sha256)) throw new Error('The selected language checkpoints have no valid identity.');
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
      if (!digest(run.checkpoint_sha256) || run.selection_sha256 !== report.selection_sha256 || typeof run.graph !== 'string') throw new Error('Language checkpoint identities are incomplete or inconsistent.');
      if (!score || !Number.isFinite(score.bits_per_byte) || score.bits_per_byte < 0 || !Number.isFinite(score.token_perplexity) || score.token_perplexity < 1 || score.documents?.length !== 60 || new Set(score.documents.map(article => article.document)).size !== 60) throw new Error('Incomplete or invalid language test scores.');
      if (!Number.isInteger(run.checkpoint_step) || run.checkpoint_step < 500 || run.checkpoint_step > 6000 || run.checkpoint_step % 500) throw new Error('Invalid selected language checkpoint.');
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

export function validateReleaseDownload(report, release, kind) {
  validateFinalResults(report);
  const archive = kind === 'records' ? 'language-topology-records.zip' : kind === 'inference' ? 'language-topology-inference.zip' : null;
  if (!archive || release?.archive !== `public/research/${archive}` || !digest(release.archive_sha256) || !Number.isSafeInteger(release.bytes) || release.bytes <= 0) throw new Error('Invalid language release archive.');
  const identity = kind === 'records' ? release.arithmetic_audit : release;
  if (identity?.study_identity_sha256 !== report.study_identity_sha256 || identity.selection_sha256 !== report.selection_sha256) throw new Error('Language release belongs to a different selected study.');
  if (kind === 'records') {
    if (release.full_checkpoint_and_score_gate_passed !== true || identity.verified_runs !== 10 || identity.unique_test_articles !== 60) throw new Error('Complete language score verification is required.');
    for (const key of ['topology_contrasts', 'slow_state_contrasts']) {
      const points = identity[key], expected = report[key];
      if (!Array.isArray(points) || points.length !== expected.length) throw new Error('Released paired comparisons are incomplete.');
      const names = new Set();
      for (const point of points) {
        const name = `${point.graph_seed ?? 'slow'}:${point.training_seed}`;
        const pair = expected.find(row => row.training_seed === point.training_seed && row.graph_seed === point.graph_seed);
        if (!pair || names.has(name) || ['difference_bpb', 'lower_95', 'upper_95'].some(key => !Number.isFinite(point[key]) || Math.abs(point[key] - pair[key]) > 1e-8)) throw new Error('Released paired comparisons differ from the displayed results.');
        names.add(name);
      }
    }
  } else {
    if (release.fresh_archive_cli_verified !== true) throw new Error('The complete model download has not passed standalone replay.');
    const selected = new Map(report.runs.map(run => [run.label, run]));
    const inventories = [release.models, release.parity, release.standalone_cli_verification?.cases];
    for (const [index, rows] of inventories.entries()) {
      const key = index ? 'condition' : 'id';
      if (!Array.isArray(rows) || rows.length !== 10 || new Set(rows.map(row => row[key])).size !== 10 || rows.some(row => !selected.has(row[key]))) throw new Error('All ten released models and replay checks are required.');
    }
    for (const model of release.models) {
      const run = selected.get(model.id);
      if (model.source_checkpoint_sha256 !== run.checkpoint_sha256 || model.checkpoint_step !== run.checkpoint_step || model.training_seed !== run.seed || model.variant !== run.variant || model.graph !== run.graph || model.graph_seed !== run.graph_seed || model.parameters !== run.parameters || model.reference !== run.reference || !Number.isFinite(model.published_test_bits_per_byte) || Math.abs(model.published_test_bits_per_byte - run.score.bits_per_byte) > 1e-8) throw new Error('A released model differs from the displayed checkpoint selection.');
    }
    if (release.parity.some(row => row.tensors_and_buffers_exact !== true || row.logits_and_states_exact_at_probe !== true || row.fixed_prompt_continuations_exact !== 4) || release.standalone_cli_verification.cases.some(row => row.source_prompt_reproduced !== true || row.cli_exit_code !== 0)) throw new Error('Complete source and standalone replay checks are required.');
  }
  return { path: `research/${archive}`, bytes: release.bytes, sha256: release.archive_sha256 };
}

async function loadFinalDownloads(report) {
  const base = import.meta.env.BASE_URL;
  const kinds = ['records', 'inference'];
  const filenames = ['language-topology-release.json', 'language-topology-inference-release.json'];
  const responses = await Promise.allSettled(filenames.map(async filename => {
    const response = await fetch(`${base}research/${filename}`, { cache: 'no-store' });
    if (!response.ok) throw new Error('Release unavailable.');
    return response.json();
  }));
  let available = 0;
  responses.forEach((response, index) => {
    if (response.status !== 'fulfilled') return;
    const kind = kinds[index];
    try {
      const download = validateReleaseDownload(report, response.value, kind);
      const anchor = $(`language-topology-${kind}-link`);
      anchor.href = `${base}${download.path}`;
      anchor.textContent = `${kind === 'records' ? 'Complete score records and arithmetic checker' : 'Ten selected models and local inference runtime'} · ${(download.bytes / 2**20).toFixed(1)} MiB ZIP`;
      if (kind === 'records') {
        const image = $('language-topology-test-figure');
        $('language-topology-result-figure').hidden = false;
        image.onerror = () => { $('language-topology-result-figure').hidden = true; };
        image.src = `${base}research/figures/language-topology-test.svg`;
      }
      $(`language-topology-${kind}-download`).hidden = false;
      available++;
    } catch { /* Keep downloads for another selection or an unfinished release hidden. */ }
  });
  $('language-topology-download-status').textContent = available === 2 ? '' : `${available ? 'Additional release downloads' : 'Release downloads'} are not yet available for this selection.`;
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
  $('language-topology-mean').textContent = `Mean measured minus rewired loss: ${signed(report.topology_mean_difference_bpb)} bits/byte across the six graph/training-seed combinations. Mean measured minus retrained no-slow loss: ${signed(report.slow_state_mean_difference_bpb)} bits/byte across two training seeds. Negative differences favor the measured model with fast and slow state.`;
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
  $('language-topology-final').hidden = true;
  $('language-topology-records-download').hidden = true;
  $('language-topology-inference-download').hidden = true;
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
      const response = await fetch(`${import.meta.env.BASE_URL}research/language-topology-results.json`, { cache: 'no-store' });
      if (!response.ok) throw new Error('The completed language test report is unavailable.');
      const results = await response.json();
      renderFinalResults(results, report.study_identity_sha256);
      void loadFinalDownloads(results);
    }
  } catch (error) { $('language-topology-status').textContent = error.message; }
}
