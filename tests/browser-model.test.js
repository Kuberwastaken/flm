import testCase from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { FLM, random, sample, softmax } from '../web/model.js';

for (const packageName of ['flm-compact', 'flm-wikitext']) {
const directory = new URL(`../public/models/${packageName}/`, import.meta.url);
const config = JSON.parse(readFileSync(new URL('model.json', directory)));
const raw = readFileSync(new URL('weights.bin', directory));
const binary = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const truth = JSON.parse(readFileSync(new URL('parity.json', directory)));
const model = () => new FLM(config, binary.slice(0));
const difference = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));
const test = (name, fn) => testCase(`${packageName}: ${name}`, fn);

test('published binary matches its checksum', () => {
  assert.equal(createHash('sha256').update(raw).digest('hex'), config.weights_sha256);
});
test('JavaScript logits and recurrent states match PyTorch', () => {
  const m = model();
  for (const token of truth.tokens) m.step(token);
  assert.ok(difference(m.logits, truth.logits) < 0.0005);
  assert.ok(difference(m.h, truth.h) < 0.00005);
  assert.ok(difference(m.slow, truth.slow) < 0.00005);
  assert.ok(difference(m.encoded, truth.features) < 0.0001);
});
test('reset removes preceding document state', () => {
  const m = model(); const before = Array.from(m.step(97));
  m.step(98); m.step(99); m.reset();
  assert.deepEqual(Array.from(m.step(97)), before);
});
test('fixed seed produces a reproducible continuation', () => {
  const generate = () => { const m = model(), rng = random(12), tokens = []; let token = config.bos ?? 256;
    const allowed = m.lexical ? Array.from({length: m.vocab}, (_, i) => i !== config.bos) : null;
    for (let i = 0; i < 20; i++) { token = sample(m.step(token), rng, {allowed}); tokens.push(token); }
    return tokens; };
  assert.deepEqual(generate(), generate());
});
test('local learning improves the next-token loss on a repeated observed feature', () => {
  const m = model(); for (const token of truth.tokens) m.step(token);
  const before = -Math.log(softmax(m.logits)[97]);
  for (let k = 0; k < 12; k++) { m.reset(); for (const token of truth.tokens) m.step(token); m.learn(97); }
  m.reset(); for (const token of truth.tokens) m.step(token);
  assert.ok(-Math.log(softmax(m.logits)[97]) < before);
  m.clearLearning(); m.reset(); for (const token of truth.tokens) m.step(token);
  assert.ok(difference(m.logits, truth.logits) < 0.0005);
});
test('adapter roundtrip works and incompatible/nonfinite adapters are rejected', () => {
  const a = model(), b = model(); a.step(97); a.learn(98);
  b.importLearning(a.exportLearning()); assert.deepEqual(b.adapter, a.adapter);
  const bad = a.exportLearning(); bad.weights[0] = NaN;
  assert.throws(() => b.importLearning(bad));
  const wrong = a.exportLearning(); wrong.modelHash = 'different';
  assert.throws(() => b.importLearning(wrong));
});
test('silenced neurons remain zero in both timescales', () => {
  const m = model(); m.disabled[3] = 1;
  for (const t of truth.tokens) { m.step(t); assert.equal(m.h[3], 0); assert.equal(m.slow[3], 0); }
});
test('recurrence intervention changes predictions on the same prefix', () => {
  const a = model(), b = model(); b.recurrenceEnabled = false;
  for (const t of truth.tokens) { a.step(t); b.step(t); }
  assert.ok(difference(a.logits, b.logits) > 0.0001);
});
test('invalid dimensions, tokens and sampling controls fail clearly', () => {
  assert.throws(() => new FLM(config, new ArrayBuffer(4)));
  const m = model(); assert.throws(() => m.step(-1)); assert.throws(() => m.step(0.5));
  assert.throws(() => sample(m.logits, random(), {temperature: 0}));
});
}

