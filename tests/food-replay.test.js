import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { decodeTrial, replayFrame, replayManifestSHA } from '../web/food-replay-data.js';

const read = name => readFileSync(new URL(`../public/${name}`, import.meta.url));
const digest = data => createHash('sha256').update(data).digest('hex');
const manifestBytes = read('research/food-core-replay/manifest.json'), manifest = JSON.parse(manifestBytes);
function load(entry = manifest.trials[0]) {
  const meta = read(`research/food-core-replay/${entry.metadata_file}`);
  assert.equal(digest(meta), entry.metadata_sha256);
  const metadata = JSON.parse(meta), binary = read(`research/food-core-replay/${metadata.binary_file}`);
  assert.equal(digest(binary), entry.binary_sha256);
  assert.equal(binary.length, entry.binary_bytes);
  return { metadata, buffer: binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength) };
}

test('complete replay binds all 24 records, exact samples, anatomy and compiled body assets', () => {
  assert.equal(digest(manifestBytes), replayManifestSHA);
  assert.equal(manifest.observations, 4824);
  assert.equal(new Set(manifest.trials.map(x => x.label)).size, 24);
  const anatomyBytes = read('models/flm-wikitext/anatomy.json');
  assert.equal(digest(anatomyBytes), manifest.anatomy_sha256);
  assert.equal(JSON.parse(anatomyBytes).body_ids.length, 1024);
  const bodyBytes = read('body/recorded-food/model.json'), body = JSON.parse(bodyBytes);
  assert.equal(digest(bodyBytes), manifest.body_manifest_sha256);
  assert.equal(digest(read('body/recorded-food/geometry.bin')), body.geometry_sha256);
  assert.equal(body.geometries.length, 69);
  const models = ['initial-s42','language-s42','initial-s43','language-s43'];
  for (const model of models) assert.equal(manifest.trials.filter(x => x.model_id === model).length, 6);
  for (const entry of manifest.trials) {
    const {metadata, buffer} = load(entry), trial = decodeTrial(metadata, buffer);
    assert.equal(metadata.label, entry.label);
    for (const frame of [0, 1, 72, 80, 200]) {
      const state = replayFrame(trial, frame);
      assert.equal(state.h.length, 1024); assert.equal(state.a.length, 1024); assert.equal(state.geometry.length, 828);
      assert.equal(state.h[0], new DataView(buffer).getFloat32(metadata.arrays.fast.offset + frame * 4096, true));
      assert.equal(state.a[1023], new DataView(buffer).getFloat32(metadata.arrays.slow.offset + (frame * 1024 + 1023) * 4, true));
      // A geometry's native row-major rotation must remain orthonormal after export.
      for (let mesh = 0; mesh < 69; mesh++) {
        const r = state.geometry.subarray(mesh * 12 + 3, (mesh + 1) * 12);
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
          const dot = r[a*3]*r[b*3] + r[a*3+1]*r[b*3+1] + r[a*3+2]*r[b*3+2];
          assert.ok(Math.abs(dot - (a === b ? 1 : 0)) < 3e-7);
        }
      }
    }
  }
});

test('wrong-source outcomes, missing-odor contact and censored cases remain available', () => {
  const records = manifest.trials.map(entry => load(entry).metadata);
  assert.equal(records.filter(x => x.metrics.contact_latency_censored).length, 3);
  assert.ok(records.filter(x => x.case.label === 'odor-a-right').every(x => x.metrics.first_contact_sources[0] === 'b' && x.metrics.first_contact_sugar === 0));
  const missing = records.find(x => x.label === 'language-s43--odor-a-left-missing');
  assert.equal(missing.metrics.first_contact_sugar, 1);
  assert.ok(missing.sensory.every(row => row.slice(0,4).every(x => x === 0)));
});

test('corrupt byte layouts, nonfinite values and invalid observations are rejected', () => {
  const source = load();
  for (const mutate of [m => m.arrays.fast.offset += 4, m => m.arrays.geometry.shape[1] = 68,
    m => m.arrays.fast.dtype = 'float64', m => m.time_s[1] = .015, m => m.time_s[0] = NaN,
    m => m.sensory.pop(), m => m.probabilities[0] = [1,1,1], m => m.cached_thorax_mm[0][0] = Infinity,
    m => m.status = 'failed']) {
    const metadata = structuredClone(source.metadata); mutate(metadata);
    assert.throws(() => decodeTrial(metadata, source.buffer));
  }
  assert.throws(() => decodeTrial(source.metadata, source.buffer.slice(0,-4)));
  const corrupt = source.buffer.slice(0); new DataView(corrupt).setFloat32(0, NaN, true);
  assert.throws(() => decodeTrial(source.metadata, corrupt), /Nonfinite/);
  const trial = decodeTrial(source.metadata, source.buffer);
  for (const index of [-1, .5, 201, NaN]) assert.throws(() => replayFrame(trial, index));
});
