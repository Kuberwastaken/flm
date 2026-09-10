import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {METHODS, validateReplay, comparisonAt} from '../web/choice-replay.js';

const source = JSON.parse(readFileSync(new URL('../public/research/choice-replay.json', import.meta.url)));
const fresh = () => structuredClone(source);

test('all forty recorded conditions retain exact sampled times and both physical commands', () => {
  const report = validateReplay(fresh());
  assert.equal(report.cases.length, 40);
  for (const path of Object.values(report.paths)) {
    assert.equal(path.time_s[0], .0001);
    assert.equal(path.time_s.at(-1), 1);
  }
});

test('every selection pairs its own initialization with its trained checkpoint at an actual common frame', () => {
  const report = validateReplay(fresh());
  for (const method of Object.keys(METHODS)) for (const step of [300, 600, 900]) for (const cue of [0, 1]) for (const frame of [0, 17, 100]) {
    const pair = comparisonAt(report, method, step, cue, frame);
    assert.deepEqual(pair.map(item => item.row.step), [0, step]);
    assert.ok(pair.every(item => item.row.method === method && item.row.cue === cue));
    assert.equal(pair[0].time, pair[1].time);
    for (const item of pair) {
      assert.equal(item.x, report.paths[`action-${item.row.chosen_action}`].x_mm[frame]);
      assert.equal(item.yaw, report.paths[`action-${item.row.chosen_action}`].yaw_rad[frame]);
    }
  }
});

test('reversal keeps its own task target and incorrect choices remain visible', () => {
  const report = validateReplay(fresh());
  const [initial, reversed] = comparisonAt(report, 'bptt', 600, 0, 100);
  assert.equal(initial.row.expected_action, 0);
  assert.equal(reversed.row.expected_action, 1);
  assert.ok(report.cases.some(row => row.expected_action !== row.chosen_action));
  const changed = fresh(); changed.cases.find(row => row.step === 600).expected_action ^= 1;
  assert.throws(() => validateReplay(changed), /decision/);
});

test('missing, duplicate, unknown and mislabeled conditions fail closed', () => {
  for (const change of [r => r.cases.pop(), r => r.cases[0] = r.cases[1], r => r.cases[0].method = 'unknown', r => r.cases[0].path = 'action-0']) {
    const report = fresh(); change(report); assert.throws(() => validateReplay(report));
  }
  for (const selection of [['bptt', 0, 0, 0], ['bptt', 300, 2, 0], ['bptt', 300, 0, 101], ['bptt', 300, 0, 1.5]]) {
    assert.throws(() => comparisonAt(source, ...selection), /selection/);
  }
});

test('truncated, nonfinite, shifted-time and clipped trajectories reject', () => {
  for (const change of [r => r.paths['action-0'].x_mm.pop(), r => r.paths['action-0'].yaw_rad[1] = NaN,
    r => r.paths['action-0'].time_s[0] = 0, r => r.bounds_mm.x_max = r.bounds_mm.x_min + 1,
    r => r.retained_indices[5] = 51, r => r.paths['action-1'].x_mm[1] = 1000]) {
    const report = fresh(); change(report); assert.throws(() => validateReplay(report));
  }
});

test('invalid probabilities, missing provenance and altered shared initialization reject', () => {
  for (const change of [r => r.cases[0].action_probabilities[0] = NaN, r => r.source_report_sha256 = '',
    r => r.cases[0].checkpoint_sha256 = '', r => r.cases[0].action_probabilities = [.3, .7],
    r => r.full_recorded_arrays_equal_within_each_action = false]) {
    const report = fresh(); change(report); assert.throws(() => validateReplay(report));
  }
});
