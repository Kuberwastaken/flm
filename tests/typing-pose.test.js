import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseBinarySTL, transformVertices } from '../web/body/stl.js';
import { neutralPoseRad, segmentTransforms, mat4ApplyPoint } from '../web/body/fk.js';
import { makeTypingRig, typingPose, footPosition, KEY_TOP, KEY_TRAVEL, FLOOR_Z, TYPING_PERIOD } from '../web/body/typing-pose.js';

const model = JSON.parse(readFileSync(new URL('../public/body/model.json', import.meta.url)));
const original = JSON.stringify(model), neutral = segmentTransforms(model, neutralPoseRad(model)), tips = {};
for (const leg of ['lf', 'rf', 'lm', 'rm', 'lh', 'rh']) {
  const name = `${leg}_tarsus5`, spec = model.meshes[name];
  const file = readFileSync(new URL(`../public/body/meshes/${spec.file}`, import.meta.url));
  const positions = transformVertices(parseBinarySTL(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)), model.meshScale, spec.mirror);
  let minimum = Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const vertex = Array.from(positions.slice(i, i + 3)), z = mat4ApplyPoint(neutral[name], vertex)[2];
    if (z < minimum) { minimum = z; tips[leg] = vertex; }
  }
}
const rig = makeTypingRig(model, tips);

test('the actual fly meshes reach keyboard keys and keep four support feet planted', () => {
  assert.equal(JSON.stringify(model), original);
  for (const leg of ['lf', 'rf']) {
    for (const [pose, height] of [[rig.rest, KEY_TOP], [rig.poses[leg].raised, KEY_TOP + 0.17], [rig.poses[leg].pressed, KEY_TOP - KEY_TRAVEL]]) {
      const point = footPosition(model, pose, leg, tips[leg]);
      const target = [...rig.targets[leg].slice(0, 2), height];
      assert.ok(Math.hypot(...point.map((v, i) => v - target[i])) < 0.002, `${leg} toe misses key target`);
    }
  }
  const support = ['lm', 'rm', 'lh', 'rh'];
  for (let i = 0; i <= 72; i++) {
    const { angles, depression } = typingPose(model, rig, i / 100, true);
    assert.ok(Object.values(angles).every(Number.isFinite));
    for (const leg of support) {
      const point = footPosition(model, angles, leg, tips[leg]);
      assert.ok(Math.abs(point[2] - (FLOOR_Z + 0.005)) < 0.002);
      assert.deepEqual(point, footPosition(model, rig.rest, leg, tips[leg]));
    }
    for (const leg of ['lf', 'rf']) {
      const point = footPosition(model, angles, leg, tips[leg]);
      assert.ok(point[2] >= KEY_TOP - depression[leg] - 0.006, 'toe penetrates key');
      assert.ok(depression[leg] >= 0 && depression[leg] <= KEY_TRAVEL + 1e-12);
    }
  }
});

test('typing alternates forelegs, loops continuously and returns to a still rest pose', () => {
  const first = typingPose(model, rig, 0, true), half = typingPose(model, rig, TYPING_PERIOD / 2, true);
  assert.equal(first.depression.lf, 0); assert.ok(first.depression.rf > 0.034);
  assert.equal(half.depression.rf, 0); assert.ok(half.depression.lf > 0.034);
  assert.deepEqual(first, typingPose(model, rig, TYPING_PERIOD, true));
  assert.deepEqual(typingPose(model, rig, 100, false), { angles: rig.rest, depression: { lf: 0, rf: 0 } });
  assert.equal(JSON.stringify(model), original);
});
