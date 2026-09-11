export const replayManifestSHA = '971b87bcf4c7bde44e0e1eb0929cba7a80d352fd1918efabdb5fd05d4b8ac1dc';
export async function checkedBytes(path, expected) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Replay file unavailable (${response.status}).`);
  const bytes = await response.arrayBuffer();
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
  if (digest !== expected) throw new Error('Replay checksum differs. Reload for a consistent release.');
  return bytes;
}
export const decodeJSON = bytes => JSON.parse(new TextDecoder().decode(bytes));

export function decodeTrial(metadata, buffer) {
  if (metadata.status !== 'complete' || metadata.binary_bytes !== buffer.byteLength || metadata.time_s.length !== 201)
    throw new Error('Incomplete replay record.');
  const shapes = { geometry: [201, 69, 12], fast: [201, 1024], slow: [201, 1024] }, arrays = {};
  let offset = 0;
  for (const [name, shape] of Object.entries(shapes)) {
    const spec = metadata.arrays[name], count = shape.reduce((a, b) => a * b, 1);
    if (!spec || spec.dtype !== 'float32' || JSON.stringify(spec.shape) !== JSON.stringify(shape) || spec.offset !== offset)
      throw new Error('Replay array layout differs.');
    if (offset + count * 4 > buffer.byteLength) throw new Error('Truncated replay array.');
    // Explicit little-endian decoding also supports hosts with a different byte order.
    const view = new DataView(buffer, offset, count * 4), values = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      values[i] = view.getFloat32(i * 4, true);
      if (!Number.isFinite(values[i])) throw new Error('Nonfinite replay state.');
    }
    arrays[name] = values; offset += count * 4;
  }
  if (offset !== buffer.byteLength) throw new Error('Unexpected replay bytes.');
  for (const [name, width] of [['sensory', 6], ['probabilities', 3], ['descending_signal', 2], ['cached_thorax_mm', 3]]) {
    if (metadata[name]?.length !== 201 || metadata[name].some(row => row.length !== width || row.some(x => !Number.isFinite(x))))
      throw new Error('Replay observation layout differs.');
  }
  if (metadata.time_s.some((value, i) => !Number.isFinite(value) || Math.abs(value - i * .01) > 1e-12))
    throw new Error('Replay clock differs.');
  if (metadata.probabilities.some(row => row.some(x => x < 0 || x > 1) || Math.abs(row.reduce((a,b) => a+b, 0) - 1) > 1e-6))
    throw new Error('Invalid action probabilities.');
  return { metadata, arrays };
}

export function replayFrame(trial, frame) {
  if (!Number.isInteger(frame) || frame < 0 || frame > 200) throw new Error('Invalid replay frame.');
  return { geometry: trial.arrays.geometry.subarray(frame * 828, (frame + 1) * 828),
    h: trial.arrays.fast.subarray(frame * 1024, (frame + 1) * 1024),
    a: trial.arrays.slow.subarray(frame * 1024, (frame + 1) * 1024) };
}
