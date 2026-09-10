import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { matchedValidation, validateFinalResults, validateReleaseDownload } from '../web/language-topology.js';

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
  const selection_sha256 = 'b'.repeat(64);
  for (const run of runs) Object.assign(run, { selection_sha256, parameters: 600003,
    checkpoint_sha256: createHash('sha256').update(run.label).digest('hex'),
    graph: `data/graphs/central-1024${run.graph_seed ? `-null${run.graph_seed}` : ''}/graph.npz` });
  return { study_identity_sha256: 'a'.repeat(64), selection_sha256, runs, topology_contrasts, slow_state_contrasts, topology_mean_difference_bpb: .1, slow_state_mean_difference_bpb: -.1 };
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
    r => { delete r.runs[0].checkpoint_step; },
    r => { delete r.runs[0].checkpoint_sha256; },
    r => { r.runs[0].selection_sha256 = 'f'.repeat(64); },
    r => { r.topology_contrasts[0].difference_bpb = -.1; },
    r => { r.topology_contrasts[1] = r.topology_contrasts[0]; },
    r => { r.slow_state_contrasts.pop(); },
    r => { r.topology_mean_difference_bpb = 0; }
  ]) {
    const report = completeFixture(); mutate(report);
    assert.throws(() => validateFinalResults(report));
  }
});

function recordRelease(report) {
  return { archive: 'public/research/language-topology-records.zip', archive_sha256: 'c'.repeat(64), bytes: 1234,
    full_checkpoint_and_score_gate_passed: true, arithmetic_audit: {
      study_identity_sha256: report.study_identity_sha256, selection_sha256: report.selection_sha256,
      verified_runs: 10, unique_test_articles: 60, topology_contrasts: structuredClone(report.topology_contrasts),
      slow_state_contrasts: structuredClone(report.slow_state_contrasts) } };
}

function inferenceRelease(report) {
  return { archive: 'public/research/language-topology-inference.zip', archive_sha256: 'd'.repeat(64), bytes: 123456,
    study_identity_sha256: report.study_identity_sha256, selection_sha256: report.selection_sha256,
    fresh_archive_cli_verified: true, models: report.runs.map(run => ({ id: run.label,
      training_seed: run.seed, variant: run.variant, graph: run.graph, graph_seed: run.graph_seed,
      parameters: run.parameters, reference: run.reference, published_test_bits_per_byte: run.score.bits_per_byte,
      source_checkpoint_sha256: run.checkpoint_sha256, checkpoint_step: run.checkpoint_step })),
    parity: report.runs.map(run => ({ condition: run.label, tensors_and_buffers_exact: true,
      logits_and_states_exact_at_probe: true, fixed_prompt_continuations_exact: 4 })),
    standalone_cli_verification: { cases: report.runs.map(run => ({ condition: run.label, source_prompt_reproduced: true, cli_exit_code: 0 })) } };
}

test('final downloads require the same study and selected-checkpoint identity', () => {
  const report = completeFixture();
  assert.equal(validateReleaseDownload(report, recordRelease(report), 'records').path, 'research/language-topology-records.zip');
  assert.equal(validateReleaseDownload(report, inferenceRelease(report), 'inference').path, 'research/language-topology-inference.zip');
  for (const kind of ['records', 'inference']) {
    const release = kind === 'records' ? recordRelease(report) : inferenceRelease(report);
    const identity = kind === 'records' ? release.arithmetic_audit : release;
    identity.selection_sha256 = 'f'.repeat(64);
    assert.throws(() => validateReleaseDownload(report, release, kind), /different selected study/);
  }
});

test('score download rejects partial verification, changed contrasts and external paths', () => {
  const report = completeFixture();
  for (const mutate of [r => { r.full_checkpoint_and_score_gate_passed = false; },
    r => { r.arithmetic_audit.verified_runs = 9; }, r => { r.arithmetic_audit.unique_test_articles = 59; },
    r => { r.arithmetic_audit.topology_contrasts[0].difference_bpb = 99; },
    r => { r.arithmetic_audit.slow_state_contrasts.pop(); },
    r => { r.arithmetic_audit.topology_contrasts[1] = r.arithmetic_audit.topology_contrasts[0]; },
    r => { r.archive = 'https://example.com/other.zip'; }, r => { r.bytes = -1; }]) {
    const release = recordRelease(report); mutate(release);
    assert.throws(() => validateReleaseDownload(report, release, 'records'));
  }
});

test('model download requires all ten exact models and successful standalone replays', () => {
  const report = completeFixture();
  for (const mutate of [r => { r.fresh_archive_cli_verified = false; },
    r => { r.models.pop(); }, r => { r.models[1] = r.models[0]; },
    r => { r.models[0].source_checkpoint_sha256 = 'f'.repeat(64); },
    r => { r.models[0].graph = 'different.npz'; }, r => { r.models[0].variant = 'no_slow'; },
    r => { r.models[0].published_test_bits_per_byte = 0; },
    r => { r.parity[0].fixed_prompt_continuations_exact = 3; },
    r => { r.standalone_cli_verification.cases[0].cli_exit_code = 1; },
    r => { r.standalone_cli_verification.cases[1] = r.standalone_cli_verification.cases[0]; }]) {
    const release = inferenceRelease(report); mutate(release);
    assert.throws(() => validateReleaseDownload(report, release, 'inference'));
  }
});
