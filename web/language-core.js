// Complete-study presentation, mounted only after both release records verify.
export const CORE_IDENTITY = '57dcbe9b8242cc0bb897486a54139553b8093192697830993a3a76085830aaee';
export const CORE_NAMES = Object.freeze({full:'Full FLM', fixed_dynamics:'Fixed recurrent dynamics', no_lateral:'No lateral recurrence', no_temporal_state:'No temporal state'});
const controls = Object.keys(CORE_NAMES);
const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const check = (condition, message) => { if (!condition) throw new Error(`Computation results unavailable: ${message}.`); };
const near = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-12 + 1e-9 * Math.abs(b);
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(6)}`;

export function validateCoreResults(report) {
  check(report?.study_identity_sha256 === CORE_IDENTITY && hash(report.selection_sha256), 'frozen study identity');
  check(Array.isArray(report.runs) && report.runs.length === 8, 'all eight selected conditions are required');
  const remaining = new Set(controls.flatMap(control => [42, 43].map(seed => `${control}-s${seed}`)));
  const runs = new Map();
  let coverage;
  for (const run of report.runs) {
    check(remaining.delete(run.label) && run.label === `${run.control}-s${run.seed}` && run.reference === (run.control === 'full'), 'unique declared model conditions');
    check(run.selection_sha256 === report.selection_sha256 && hash(run.checkpoint_sha256) && hash(run.test_cache_sha256), 'selected checkpoint and cache identities');
    check(run.graph === 'data/graphs/central-1024/graph.npz' && Number.isInteger(run.checkpoint_step) && run.checkpoint_step >= 500 && run.checkpoint_step <= 6000 && run.checkpoint_step % 500 === 0, 'graph and validation-selected update');
    check(run.allocated_parameters === 600003 && run.trainable_parameters === (run.control === 'fixed_dynamics' ? 521824 : 600003) && run.frozen_parameters === (run.control === 'fixed_dynamics' ? 78179 : 0), 'parameter inventory');
    const score = run.score;
    check(score?.documents?.length === 60 && hash(score.tokenizer_sha256), 'complete test article scores');
    const seen = new Set();
    let bytes = 0, tokens = 0, nll = 0;
    for (const article of score.documents) {
      check(typeof article.document === 'string' && !seen.has(article.document) && Number.isSafeInteger(article.bytes) && article.bytes > 0 && Number.isSafeInteger(article.tokens) && article.tokens > 0 && Number.isFinite(article.nll) && article.nll >= 0, 'valid unique article denominators');
      check(near(article.bits_per_byte, article.nll / article.bytes / Math.LN2), 'article codelength');
      seen.add(article.document); bytes += article.bytes; tokens += article.tokens; nll += article.nll;
    }
    check(bytes === 1287656 && tokens === 367981 && score.bytes === bytes && score.tokens === tokens && near(score.nll, nll) && near(score.bits_per_byte, nll / bytes / Math.LN2) && near(score.token_perplexity, Math.exp(nll / tokens)), 'complete test totals');
    const current = JSON.stringify({articles:score.documents.map(row => [row.document, row.bytes, row.tokens]).sort((a, b) => a[0].localeCompare(b[0])), tokenizer:score.tokenizer_sha256, cache:run.test_cache_sha256});
    check(coverage === undefined || current === coverage, 'matched article, byte, token and cache coverage'); coverage = current;
    if (!run.reference) check(score.mechanism === (run.control === 'no_temporal_state' ? 'Reset both states for every token, including within chunks' : 'Carry native fast/slow state within each article; reset per article'), 'reported state mechanism');
    runs.set(run.label, run);
  }
  check(remaining.size === 0, 'complete model inventory');
  for (const [name, pairs] of [
    ['primary_contrasts', controls.slice(1).flatMap(second => [42, 43].map(seed => ['full', second, seed]))],
    ['independent_unit_memory_contrasts', [42, 43].map(seed => ['no_lateral', 'no_temporal_state', seed])]
  ]) {
    const expected = new Set(pairs.map(row => row.join('/')));
    check(report[name]?.length === expected.size, 'all declared paired contrasts');
    for (const contrast of report[name]) {
      check(expected.delete([contrast.first, contrast.second, contrast.training_seed].join('/')), 'unique contrast identities');
      const a = runs.get(`${contrast.first}-s${contrast.training_seed}`), b = runs.get(`${contrast.second}-s${contrast.training_seed}`);
      check(near(contrast.difference_bpb, a.score.bits_per_byte - b.score.bits_per_byte) && Number.isFinite(contrast.lower_95) && Number.isFinite(contrast.upper_95) && contrast.lower_95 <= contrast.upper_95, 'contrast sign and finite interval');
      check(contrast.replicates === 10000 && contrast.seed === 31415 && contrast.unit === 'paired article resampling', 'declared bootstrap settings');
    }
  }
  check(report.primary_means?.length === 3 && new Set(report.primary_means.map(row => row.control)).size === 3, 'three descriptive primary means');
  for (const mean of report.primary_means) {
    check(controls.slice(1).includes(mean.control) && Object.keys(mean).sort().join() === 'control,mean_difference_bpb', 'descriptive mean without an invented interval');
    check(near(mean.mean_difference_bpb, report.primary_contrasts.filter(row => row.second === mean.control).reduce((total, row) => total + row.difference_bpb, 0) / 2), 'primary mean arithmetic');
  }
  check(near(report.independent_unit_memory_mean_difference_bpb, report.independent_unit_memory_contrasts.reduce((sum, row) => sum + row.difference_bpb, 0) / 2), 'secondary memory mean');
  return report;
}

export function validateCoreRecordRelease(report, release) {
  validateCoreResults(report);
  check(release?.archive === 'public/research/language-core-records.zip' && hash(release.archive_sha256) && hash(release.summary_sha256) && Number.isSafeInteger(release.bytes) && release.bytes > 0, 'score archive metadata');
  check(release.study_identity_sha256 === report.study_identity_sha256 && release.selection_sha256 === report.selection_sha256, 'release belongs to the displayed selection');
  for (const field of ['full_checkpoint_and_score_gate_passed','fresh_archive_arithmetic_verified','extraction_outside_repository','repository_pythonpath_removed','flm_and_torch_imports_disabled']) check(release[field] === true, 'complete source and standalone arithmetic verification');
  const audit = release.arithmetic_audit;
  check(audit?.verified_runs === 8 && audit.unique_test_articles === 60 && audit.scored_bytes_per_run === 1287656 && audit.scored_tokens_per_run === 367981 && audit.study_identity_sha256 === report.study_identity_sha256 && audit.selection_sha256 === report.selection_sha256, 'audited inventory and denominators');
  for (const key of ['primary_contrasts', 'independent_unit_memory_contrasts']) {
    const expected = new Map(report[key].map(row => [[row.first, row.second, row.training_seed].join('/'), row]));
    check(audit[key]?.length === expected.size, 'complete audited contrasts');
    for (const row of audit[key]) {
      const name = [row.first, row.second, row.training_seed].join('/'), original = expected.get(name);
      check(original && ['difference_bpb','lower_95','upper_95'].every(metric => near(row[metric], original[metric])), 'displayed intervals match the independent audit');
      expected.delete(name);
    }
  }
  check(audit.primary_means?.length === 3 && new Set(audit.primary_means.map(row => row.control)).size === 3, 'audited primary means');
  for (const row of audit.primary_means) check(near(row.mean_difference_bpb, report.primary_means.find(item => item.control === row.control)?.mean_difference_bpb), 'audited mean matches display');
  check(near(audit.independent_unit_memory_mean_difference_bpb, report.independent_unit_memory_mean_difference_bpb), 'audited secondary mean matches display');
  return release;
}

const element = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
function table(caption, headers, rows) {
  const wrap = element('div'); wrap.className = 'table-scroll'; wrap.tabIndex = 0; wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', caption);
  const result = element('table'), head = element('thead'), body = element('tbody'), heading = element('tr');
  result.append(element('caption', caption));
  for (const title of headers) { const cell = element('th', title); cell.scope = 'col'; heading.append(cell); }
  head.append(heading); result.append(head, body); wrap.append(result);
  for (const values of rows) {
    const row = element('tr');
    values.forEach((value, i) => { const cell = element(i ? 'td' : 'th', value); if (!i) cell.scope = 'row'; row.append(cell); }); body.append(row);
  }
  return wrap;
}

export async function decodeCoreResults(bytes, release) {
  check(hash(release?.summary_sha256), 'released summary checksum');
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
  check(digest === release.summary_sha256, 'summary bytes match release');
  const report = JSON.parse(new TextDecoder().decode(bytes));
  validateCoreRecordRelease(report, release); return report;
}

export async function fetchCoreRecords(base = '/', {fetcher = fetch, signal} = {}) {
  const prefix = base.endsWith('/') ? base : `${base}/`;
  const paths = ['language-core-results.json', 'language-core-release.json'];
  const responses = await Promise.all(paths.map(path => fetcher(`${prefix}research/${path}`, {cache:'no-store', signal})));
  for (const response of responses) {
    if (!response.ok) throw new Error(`Completed language comparison unavailable (${response.status}).`);
  }
  const [bytes, release] = await Promise.all([responses[0].arrayBuffer(), responses[1].json()]);
  const report = await decodeCoreResults(bytes, release);
  // Return only a fully checked pair. The caller replaces its existing content
  // after this resolves; this loader itself does not publish or mutate the DOM.
  return {report, release, base:prefix};
}

export async function loadLanguageCore() {
  const container = document.getElementById('language-core-results');
  if (!container) return;
  try {
    const {report, release, base} = await fetchCoreRecords(import.meta.env.BASE_URL);
    container.replaceChildren(coreResultsView(report, release, base));
  } catch {
    const status = element('p', 'The completed computation comparison could not be verified. Reload to try again, or open the published score records.');
    const link = element('a', 'Download computation score records');
    link.href = `${import.meta.env.BASE_URL}research/language-core-records.zip`;
    container.replaceChildren(status, link);
  }
}

// The caller uses decodeCoreResults before mounting a fetched report.
// Python publication gates verify real checkpoints and recompute bootstrap CIs;
// this browser component checks the supplied data, not independent model inference.
export function coreResultsView(report, release, base = '/') {
  validateCoreRecordRelease(report, release);
  const section = element('section'); section.setAttribute('aria-labelledby', 'core-results-title');
  const title = element('h3', 'Completed language computation controls'); title.id = 'core-results-title'; section.append(title);
  section.append(element('p', 'Six control fits were trained from scratch and selected on validation, alongside two reused full-FLM references. All eight scores cover the same 60 complete test articles. These comparisons test computation within the measured subset, separately from whether its topology helps.'));
  const details = element('details'); details.className = 'method-note'; details.append(element('summary', 'All eight selected checkpoints and every paired contrast'));
  const figure = element('figure'); figure.className = 'measured-figure';
  const figureLink = element('a'); figureLink.href = `${base}research/figures/language-core-test.svg`;
  figureLink.setAttribute('aria-label', 'Open every computation contrast at full size');
  const plot = element('img'); plot.loading = 'lazy'; plot.src = figureLink.href;
  plot.alt = 'Four panels show all six primary and two secondary paired differences in test loss, with separately labeled horizontal scales and conditional article intervals.';
  figureLink.append(plot); figure.append(figureLink, element('figcaption', 'Each panel has its own horizontal scale. Negative favors the first named model; intervals condition on the fitted pair.'));
  details.append(figure);
  details.append(table('Complete test split · lower bits per UTF-8 byte is better', ['Condition', 'Training seed', 'Selected update', 'Test BPB', 'Trainable / allocated parameters'],
    controls.flatMap(control => [42, 43].map(seed => {
      const run = report.runs.find(row => row.control === control && row.seed === seed);
      return [CORE_NAMES[control], String(seed), run.checkpoint_step.toLocaleString('en-US'), run.score.bits_per_byte.toFixed(6), `${run.trainable_parameters.toLocaleString('en-US')} / ${run.allocated_parameters.toLocaleString('en-US')}`];
    }))));
  for (const [key, caption] of [['primary_contrasts', 'Primary comparisons · full minus control'], ['independent_unit_memory_contrasts', 'Secondary comparison · no lateral recurrence minus no temporal state']]) {
    details.append(table(caption, ['First − second', 'Training seed', 'Difference BPB', '95% paired-article interval'], report[key].map(row => [
      `${CORE_NAMES[row.first]} − ${CORE_NAMES[row.second]}`, String(row.training_seed), signed(row.difference_bpb), `${signed(row.lower_95)} to ${signed(row.upper_95)}`])));
  }
  section.append(details);
  const means = report.primary_means.map(row => `${CORE_NAMES[row.control]}: ${signed(row.mean_difference_bpb)} BPB`).join('; ');
  section.append(element('p', `Mean full-minus-control differences across the two training seeds: ${means}. Secondary no-lateral-minus-no-temporal mean: ${signed(report.independent_unit_memory_mean_difference_bpb)} BPB. Negative favors the first named model.`));
  details.append(element('p', 'The eight intervals condition on the fitted pairs; shared references and articles are not independent replications. Two initializations do not establish training uncertainty. The means are descriptive, without mean confidence intervals. This is not an equivalence test or a topology-by-trainability factorial.'));
  details.append(element('p', 'Fixed dynamics still trains the lexical interface through time; it is not readout-only or matched in trainable count. No lateral recurrence retains fast and slow memory. No temporal state resets both states every token; 76,131 allocated edge/gain entries in both disabled-lateral controls are disconnected from the loss.'));
  const links = element('div'); links.className = 'document-links';
  for (const [path, label] of [['language-core-results.json', 'All article scores and checkpoint identities'], ['language-core-records.zip', `Complete score records and NumPy audit · ${(release.bytes / 2**20).toFixed(1)} MiB ZIP`], ['language-core-inference.zip', 'Eight selected models and standalone inference · ZIP'], ['language-core-samples.json', 'All 32 unedited example continuations'], ['language-core-protocol.md', 'Frozen computation protocol']]) {
    const anchor = element('a', label); anchor.href = `${base}research/${path}`; links.append(anchor);
  }
  section.append(links); return section;
}
