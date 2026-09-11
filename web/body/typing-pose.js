import { neutralPoseRad, segmentTransforms, mat4ApplyPoint } from './fk.js';

// A display rig only: authored key presses, with no neural inputs or physics.
export const KEY_TOP = 0.32;
export const KEY_TRAVEL = 0.035;
export const FLOOR_Z = -0.10;
export const TYPING_PERIOD = 0.72;

export function footPosition(model, angles, leg, tip) {
  return mat4ApplyPoint(segmentTransforms(model, angles)[`${leg}_tarsus5`], tip);
}

function reach(model, start, leg, tip, target) {
  const angles = { ...start };
  const joints = model.dofs.filter(d => d.group === `${leg}_leg` && d.biological &&
    (d.child.endsWith('coxa') || d.child.endsWith('trochanterfemur') || d.child.endsWith('tibia')));
  // Small, bounded coordinate steps keep the pose close to the existing stance.
  for (let iteration = 0; iteration < 70; iteration++) {
    const point = footPosition(model, angles, leg, tip);
    if (Math.hypot(...point.map((v, i) => target[i] - v)) < 0.0005) break;
    for (const joint of joints) {
      const before = footPosition(model, angles, leg, tip), original = angles[joint.name];
      angles[joint.name] += 0.0001;
      const after = footPosition(model, angles, leg, tip);
      const derivative = after.map((v, i) => (v - before[i]) / 0.0001);
      const norm = derivative.reduce((sum, v) => sum + v * v, 0) + 1e-8;
      const step = derivative.reduce((sum, v, i) => sum + v * (target[i] - before[i]), 0) / norm;
      const limits = (joint.limitDeg || joint.rangeDeg).map(v => v * Math.PI / 180);
      angles[joint.name] = Math.max(limits[0], Math.min(limits[1], original + Math.max(-0.08, Math.min(0.08, step * 0.65))));
    }
  }
  return angles;
}

export function makeTypingRig(model, tips) {
  let rest = neutralPoseRad(model);
  // Four support feet stay planted; only the two front legs operate the keys.
  for (const leg of ['lm', 'rm', 'lh', 'rh']) {
    const target = footPosition(model, rest, leg, tips[leg]);
    target[2] = FLOOR_Z + 0.005;
    rest = reach(model, rest, leg, tips[leg], target);
  }
  const targets = { lf: [1.35, 0.84, KEY_TOP], rf: [1.35, -0.84, KEY_TOP] };
  for (const leg of ['lf', 'rf']) rest = reach(model, rest, leg, tips[leg], targets[leg]);
  const poses = {};
  for (const leg of ['lf', 'rf']) {
    const target = targets[leg];
    poses[leg] = {
      raised: reach(model, rest, leg, tips[leg], [target[0], target[1], KEY_TOP + 0.17]),
      pressed: reach(model, rest, leg, tips[leg], [target[0], target[1], KEY_TOP - KEY_TRAVEL]),
    };
  }
  return { rest, poses, targets };
}

export function typingPose(model, rig, seconds, typing) {
  const angles = { ...rig.rest }, depression = { lf: 0, rf: 0 };
  if (!typing) return { angles, depression };
  for (const [leg, offset] of [['lf', 0], ['rf', 0.5]]) {
    const cycle = ((seconds / TYPING_PERIOD + offset) % 1 + 1) % 1;
    const stroke = (1 - Math.cos(cycle * Math.PI * 2)) / 2;
    for (const joint of model.dofs) if (joint.group === `${leg}_leg`) {
      angles[joint.name] = rig.poses[leg].raised[joint.name] * (1 - stroke) + rig.poses[leg].pressed[joint.name] * stroke;
    }
    // The key moves only once its corresponding toe reaches the key surface.
    depression[leg] = Math.max(0, (stroke - 0.83) / 0.17) * KEY_TRAVEL;
  }
  return { angles, depression };
}
