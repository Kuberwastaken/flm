export const METHODS = Object.freeze({bptt:'BPTT', reservoir:'Fixed core', eligibility:'Eligibility', instantaneous:'No trace history', reward:'Reward + eligibility'});
export const STEPS = Object.freeze([0, 300, 600, 900]);
const requireRecord = (condition, message) => { if (!condition) throw new Error(`Choice replay unavailable: ${message}.`); };
const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

export function validateReplay(report) {
  requireRecord(report.schema_version === 1 && report.complete === true && report.training_seed === 17 && report.physics_seed === 17, 'study identity');
  for (const key of ['source_report_sha256', 'source_summary_sha256', 'generator_sha256']) requireRecord(hash(report[key]), 'source identity');
  requireRecord(report.original_samples_per_trial === 1001 && report.full_recorded_arrays_equal_within_each_action === true, 'original trajectory verification');
  requireRecord(report.retained_indices?.length === 101 && report.retained_indices.every((n, i) => n === 10 * i), 'complete sampled frame inventory');
  requireRecord(report.paths && Object.keys(report.paths).sort().join() === 'action-0,action-1', 'both recorded commands');
  const bounds = report.bounds_mm;
  requireRecord(bounds && ['x_min', 'x_max', 'y_min', 'y_max'].every(key => Number.isFinite(bounds[key])) &&
    bounds.x_max > bounds.x_min && bounds.y_max > bounds.y_min &&
    Math.abs(bounds.x_max - bounds.x_min - (bounds.y_max - bounds.y_min)) < 1e-9, 'shared square coordinates');
  for (const path of Object.values(report.paths)) {
    for (const name of ['time_s', 'x_mm', 'y_mm', 'yaw_rad']) requireRecord(path[name]?.length === 101 && path[name].every(Number.isFinite), 'finite recorded frames');
    requireRecord(path.time_s.every((t, i) => Math.abs(t - (i === 100 ? 1 : .0001 + i * .01)) < 1e-12), 'original timestamps');
    requireRecord(path.x_mm.every(x => x >= bounds.x_min && x <= bounds.x_max) && path.y_mm.every(y => y >= bounds.y_min && y <= bounds.y_max), 'unclipped coordinates');
  }
  const expected = new Set(Object.keys(METHODS).flatMap(method => STEPS.flatMap(step => [0, 1].map(cue => `${method}-${step}-${cue}`))));
  requireRecord(report.cases?.length === 40, 'all forty trials');
  const initials = new Map();
  for (const row of report.cases) {
    requireRecord(expected.delete(`${row.method}-${row.step}-${row.cue}`), 'unique declared conditions');
    const p = row.action_probabilities;
    requireRecord(p?.length === 2 && p.every(n => Number.isFinite(n) && n >= 0 && n <= 1) && Math.abs(p[0] + p[1] - 1) < 1e-6, 'recorded action probabilities');
    requireRecord(row.chosen_action === (p[0] >= p[1] ? 0 : 1) && row.expected_action === (row.step === 600 ? 1 - row.cue : row.cue) && row.path === `action-${row.chosen_action}`, 'neural decision and physical path mapping');
    requireRecord(hash(row.checkpoint_sha256) && hash(row.trajectory_sha256), 'checkpoint and trajectory hashes');
    if (row.step === 0) {
      requireRecord(!initials.has(row.cue) || initials.get(row.cue).every((n, i) => n === p[i]), 'shared initial response');
      initials.set(row.cue, p);
    }
  }
  requireRecord(expected.size === 0, 'complete condition coverage');
  return report;
}

export function comparisonAt(report, method, step, cue, frame) {
  requireRecord(Object.hasOwn(METHODS, method) && [300, 600, 900].includes(step) && [0, 1].includes(cue) && Number.isInteger(frame) && frame >= 0 && frame <= 100, 'selection');
  return [0, step].map(update => {
    const row = report.cases.find(item => item.method === method && item.step === update && item.cue === cue);
    requireRecord(row && report.paths[row.path], 'selected trial');
    const path = report.paths[row.path];
    return {row, path, time: path.time_s[frame], x:path.x_mm[frame], y:path.y_mm[frame], yaw:path.yaw_rad[frame], frame};
  });
}

const svgNode = (name, attributes, text = '') => {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  element.textContent = text; return element;
};

function draw(svg, sample, bounds, label) {
  const scale = 280 / (bounds.x_max - bounds.x_min);
  const x = value => 40 + (value - bounds.x_min) * scale;
  const y = value => 312 - (value - bounds.y_min) * scale;
  svg.replaceChildren(svgNode('title', {}, `${label}. Recorded position and heading at ${sample.time.toFixed(4)} simulated seconds. Fly icon is schematic.`));
  for (let tick = Math.ceil(bounds.y_min / 5) * 5; tick <= bounds.y_max; tick += 5) {
    svg.append(svgNode('path', {d:`M40 ${y(tick)}H320`, fill:'none', stroke:'var(--border)'}));
    svg.append(svgNode('text', {x:32, y:y(tick)+4, 'text-anchor':'end'}, String(tick)));
  }
  for (let tick = Math.ceil(bounds.x_min / 5) * 5; tick <= bounds.x_max; tick += 5) {
    svg.append(svgNode('path', {d:`M${x(tick)} 32V312`, fill:'none', stroke:'var(--border)'}));
    svg.append(svgNode('text', {x:x(tick), y:332, 'text-anchor':'middle'}, String(tick)));
  }
  svg.append(svgNode('text', {x:40, y:18}, 'y (mm)'));
  svg.append(svgNode('text', {x:320, y:352, 'text-anchor':'end'}, 'x (mm)'));
  const path = end => sample.path.x_mm.slice(0, end).map((value, i) => `${i ? 'L' : 'M'}${x(value)} ${y(sample.path.y_mm[i])}`).join(' ');
  svg.append(svgNode('path', {d:path(101), fill:'none', stroke:'var(--muted)', 'stroke-width':1.5, 'stroke-dasharray':'3 3'}));
  svg.append(svgNode('path', {d:path(sample.frame+1), fill:'none', stroke:'var(--accent)', 'stroke-width':2.5}));
  svg.append(svgNode('circle', {cx:x(sample.path.x_mm[0]), cy:y(sample.path.y_mm[0]), r:3, fill:'var(--surface)', stroke:'var(--muted)'}));
  const fly = svgNode('g', {transform:`translate(${x(sample.x)} ${y(sample.y)}) rotate(${-sample.yaw * 180 / Math.PI})`, 'data-recorded-fly':''});
  fly.append(svgNode('path', {d:'M-2 -3L-8 -9L-13 -10 M1 -3L2 -10L-2 -14 M5 -2L10 -7L15 -7 M-2 3L-8 9L-13 10 M1 3L2 10L-2 14 M5 2L10 7L15 7', fill:'none', stroke:'#624b35', 'stroke-width':1.4, 'stroke-linecap':'round'}));
  fly.append(svgNode('ellipse', {cx:-5, cy:0, rx:9, ry:4.5, fill:'#9f723c', stroke:'#624b35'}));
  for (const side of [-1, 1]) fly.append(svgNode('ellipse', {cx:-5, cy:side*5, rx:10, ry:3.6, fill:'#ede8de', 'fill-opacity':.8, stroke:'#a39883', transform:`rotate(${side * -20} 1 0)`}));
  fly.append(svgNode('ellipse', {cx:3, cy:0, rx:5, ry:4.5, fill:'#b08855', stroke:'#624b35'}));
  fly.append(svgNode('ellipse', {cx:9, cy:0, rx:3.5, ry:4, fill:'#913c2b'}));
  svg.append(fly);
}

export async function loadChoiceReplay() {
  const $ = id => document.getElementById(id);
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/choice-replay.json`);
    if (!response.ok) throw new Error('Recorded choice trajectories are unavailable.');
    const bytes = await response.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
    requireRecord(digest === REPLAY_SHA256, 'published record checksum');
    const report = validateReplay(JSON.parse(new TextDecoder().decode(bytes)));
    const render = () => {
      const method = $('choice-replay-method').value, step = Number($('choice-replay-step').value), cue = Number($('choice-replay-cue').value), frame = Number($('choice-replay-time').value);
      const pair = comparisonAt(report, method, step, cue, frame);
      $('choice-replay-clock').value = `${pair[0].time.toFixed(4)} s`;
      $('choice-replay-time').setAttribute('aria-valuetext', `${pair[0].time.toFixed(4)} simulated seconds`);
      $('choice-replay-after-title').textContent = `After ${step} updates · ${(step * 8).toLocaleString()} episodes`;
      pair.forEach((sample, i) => {
        const name = i ? 'after' : 'before', row = sample.row;
        draw($(`choice-replay-${name}`), sample, report.bounds_mm, i ? `After ${step} updates` : 'Untrained');
        $(`choice-replay-${name}-note`).textContent = `Chose action ${row.chosen_action} (${(row.action_probabilities[row.chosen_action] * 100).toFixed(1)}%). Task expects ${row.expected_action}: ${row.chosen_action === row.expected_action ? 'correct' : 'incorrect'}. Heading change: ${((sample.yaw-sample.path.yaw_rad[0])*180/Math.PI).toFixed(1)}°.`;
      });
      $('choice-replay-status').textContent = `${METHODS[method]}, cue ${cue}. ${step === 600 ? 'The trained checkpoint was last taught the reversed rule; the untrained reference uses the original rule.' : step === 900 ? 'The trained checkpoint has relearned the original rule after reversal.' : 'The trained checkpoint has completed the first original-rule phase.'} ${pair[0].row.path === pair[1].row.path ? 'Both choose the same motor command and follow exactly the same recorded path.' : 'The two chosen commands produce different recorded paths.'}`;
    };
    for (const name of ['method', 'step', 'cue']) $(`choice-replay-${name}`).addEventListener('change', render);
    $('choice-replay-time').addEventListener('input', render);
    render(); $('choice-replay-controls').hidden = false;
  } catch (error) { $('choice-replay-status').textContent = error.message; $('choice-replay-controls').hidden = true; }
}

// Exact derived artifact, verified against every original trial by scripts/choice_replay.py.
const REPLAY_SHA256 = '0f9567a857f0bc572b054c232df5e4f566732c83d134931d1a56dcbec83699f7';
