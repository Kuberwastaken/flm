import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {validateCoreResults, validateCoreRecordRelease, decodeCoreResults} from '../web/language-core.js';
import {syntheticCoreFixture} from './helpers/core-results-fixture.js';

test('synthetic complete model inventory retains selected steps, parameters and every signed contrast', () => {
  const {report, release} = syntheticCoreFixture();
  assert.equal(validateCoreResults(report), report);
  assert.equal(validateCoreRecordRelease(report, release), release);
  assert.equal(report.runs.filter(row => row.checkpoint_step === 500).length, 4);
  assert.ok(report.primary_contrasts.some(row => row.difference_bpb > 0));
  assert.ok(report.primary_contrasts.some(row => row.difference_bpb < 0));
});

test('partial cohorts, duplicate models and wrong selected study or mechanism reject', () => {
  for (const mutate of [r => r.runs.pop(), r => r.runs[0] = r.runs[1],
    r => r.study_identity_sha256 = 'f'.repeat(64), r => r.runs[1].selection_sha256 = 'f'.repeat(64),
    r => r.runs[2].trainable_parameters = 600003, r => r.runs[6].score.mechanism = 'Carry temporal state',
    r => r.runs[0].checkpoint_step = 6100]) {
    const {report} = syntheticCoreFixture(); mutate(report); assert.throws(() => validateCoreResults(report));
  }
});

test('article arithmetic, identity and common test denominators must match', () => {
  for (const mutate of [r => r.runs[1].score.documents[0].document = 'changed-article',
    r => r.runs[1].score.documents[0].bytes++, r => r.runs[0].score.nll++,
    r => r.runs[1].score.documents[0].nll = Infinity,
    r => r.runs[1].score.tokenizer_sha256 = 'e'.repeat(64), r => r.runs[1].test_cache_sha256 = 'e'.repeat(64),
    r => r.runs[0].score.token_perplexity++]) {
    const {report} = syntheticCoreFixture(); mutate(report); assert.throws(() => validateCoreResults(report));
  }
});

test('direction, secondary-pair identity, bootstrap settings and descriptive means are checked', () => {
  for (const mutate of [r => r.primary_contrasts[0].difference_bpb *= -1,
    r => r.independent_unit_memory_contrasts[0].first = 'full',
    r => r.primary_contrasts[0].seed = 1, r => r.primary_contrasts[0].replicates = 999,
    r => r.primary_means[0].lower_95 = -.1,
    r => r.primary_means[0].mean_difference_bpb++,
    r => r.independent_unit_memory_mean_difference_bpb++]) {
    const {report} = syntheticCoreFixture(); mutate(report); assert.throws(() => validateCoreResults(report));
  }
});

test('displayed confidence intervals cannot differ from the independent release audit', () => {
  for (const mutate of [r => r.fresh_archive_arithmetic_verified = false,
    r => r.selection_sha256 = 'e'.repeat(64), r => r.arithmetic_audit.scored_bytes_per_run--,
    r => r.arithmetic_audit.primary_contrasts[0].lower_95 -= .01,
    r => r.arithmetic_audit.primary_contrasts[1] = r.arithmetic_audit.primary_contrasts[0],
    r => r.arithmetic_audit.primary_means[0].mean_difference_bpb++, r => r.archive = '../other.zip']) {
    const {report, release} = syntheticCoreFixture(); mutate(release); assert.throws(() => validateCoreRecordRelease(report, release));
  }
});

test('serialized summary is checksum-bound before decoding or presentation', async () => {
  const {report, release} = syntheticCoreFixture();
  const bytes = new TextEncoder().encode(JSON.stringify(report));
  release.summary_sha256 = createHash('sha256').update(bytes).digest('hex');
  assert.deepEqual(await decodeCoreResults(bytes, release), report);
  bytes[bytes.length-2] ^= 1;
  await assert.rejects(decodeCoreResults(bytes, release), /summary bytes match release/);
  await assert.rejects(decodeCoreResults(new Uint8Array([0]), {}), /released summary checksum/);
});
