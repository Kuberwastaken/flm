import test from 'node:test';
import assert from 'node:assert/strict';
import { matchedValidation, validateFinalResults } from '../web/language-topology.js';

test('topology display compares matching updates and training seeds only', () => {
  const report = { runs: [
    { reference: true, seed: 42, validation: [{ step: 500, bits_per_byte: 2.5 }, { step: 6000, bits_per_byte: 1.9 }] },
    { reference: true, seed: 43, validation: [{ step: 500, bits_per_byte: 2.6 }] },
    { reference: false, seed: 42, saved_step: 500, validation: [{ step: 500, bits_per_byte: 2.4 }, { step: 1000, bits_per_byte: 2.2 }] },
    { reference: false, seed: 43, saved_step: 500, validation: [{ step: 500, bits_per_byte: 2.4 }] },
    { reference: false, seed: 42, saved_step: 0, validation: [] }
  ] };
  const rows = matchedValidation(report);
  assert.equal(rows[0].point.step, 500);
  assert.equal(rows[0].reference.bits_per_byte, 2.5);
  assert.ok(Math.abs(rows[0].difference - .1) < 1e-12);
  assert.ok(Math.abs(rows[1].difference - .2) < 1e-12);
  assert.equal(rows[2].difference, null);
});

test('topology display leaves unmatched reference measurements empty', () => {
  const rows = matchedValidation({ runs: [
    { reference: true, seed: 42, validation: [{ step: 6000, bits_per_byte: 1.9 }] },
    { reference: false, seed: 42, saved_step: 500, validation: [{ step: 500, bits_per_byte: 2.4 }] }
  ] });
  assert.equal(rows[0].point, undefined);
  assert.equal(rows[0].difference, null);
});

function completeFixture() {
  const runs = [], topology_contrasts = [], slow_state_contrasts = [];
  const documents = Array.from({ length: 60 }, (_, i) => ({ document: `article-${i}` }));
  for (const seed of [42, 43]) {
    runs.push({ label: `measured-s${seed}`, seed, graph_seed: null, reference: true, variant: 'flm',
      checkpoint_step: 6000, score: { bits_per_byte: 2, token_perplexity: 100, documents } });
    runs.push({ label: `no-slow-s${seed}`, seed, graph_seed: null, reference: false, variant: 'no_slow',
      checkpoint_step: 5500, score: { bits_per_byte: 2.1, token_perplexity: 110, documents } });
    slow_state_contrasts.push({ training_seed: seed, difference_bpb: -.1, lower_95: -.2, upper_95: .01 });
    for (const graph of [101, 103, 107]) {
      runs.push({ label: `null${graph}-s${seed}`, seed, graph_seed: graph, reference: false, variant: 'flm',
        checkpoint_step: 6000, score: { bits_per_byte: 1.9, token_perplexity: 90, documents } });
      topology_contrasts.push({ training_seed: seed, graph_seed: graph, difference_bpb: .1, lower_95: -.01, upper_95: .2 });
    }
  }
  return { study_identity_sha256: 'a'.repeat(64), runs, topology_contrasts, slow_state_contrasts, topology_mean_difference_bpb: .1, slow_state_mean_difference_bpb: -.1 };
}

test('completed topology display accepts both signs of observed effects', () => {
  const report = completeFixture();
  assert.equal(validateFinalResults(report), report);
  assert.ok(report.topology_mean_difference_bpb > 0);
  assert.ok(report.slow_state_mean_difference_bpb < 0);
  assert.throws(() => validateFinalResults(report, 'b'.repeat(64)), /different study/);
});

test('completed topology display rejects incomplete or inconsistent evidence', () => {
  for (const mutate of [
    r => r.runs.pop(), r => { r.runs[1] = r.runs[0]; },
    r => { r.runs[0].score.documents = r.runs[0].score.documents.slice(0, 59); },
    r => { r.runs[0].score.bits_per_byte = NaN; },
    r => { r.runs[0].checkpoint_step = 9999; },
    r => { r.topology_contrasts[0].difference_bpb = -.1; },
    r => { r.topology_contrasts[1] = r.topology_contrasts[0]; },
    r => { r.slow_state_contrasts.pop(); },
    r => { r.topology_mean_difference_bpb = 0; }
  ]) {
    const report = completeFixture(); mutate(report);
    assert.throws(() => validateFinalResults(report));
  }
});
