import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateFK, segmentTransforms, neutralPoseRad } from '../web/body/fk.js';
import { parseBinarySTL } from '../web/body/stl.js';
const model = JSON.parse(readFileSync(new URL('../public/body/model.json', import.meta.url)));
test('articulated fly matches all reference joint coordinates', () => {
  assert.ok(validateFK(model) < 1e-8);
  const neutral = neutralPoseRad(model), before = segmentTransforms(model, neutral);
  neutral['c_thorax-c_head-yaw'] = 0.2;
  const after = segmentTransforms(model, neutral);
  assert.notDeepEqual(before.c_head, after.c_head);
  assert.deepEqual(before.c_thorax, after.c_thorax);
});
test('every body segment has valid finite geometry', () => {
  const files = new Set(Object.values(model.meshes).map(x => x.file));
  assert.equal(files.size, 39); let triangles = 0;
  for (const file of files) {
    const buffer = readFileSync(new URL(`../public/body/meshes/${file}`, import.meta.url));
    const positions = parseBinarySTL(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    assert.ok(positions.every(Number.isFinite)); triangles += positions.length / 9;
  }
  assert.equal(triangles, 66574);
});
