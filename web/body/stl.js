// Binary STL: three unindexed xyz vertices per triangle.

export function parseBinarySTL(buffer) {
  const dv = new DataView(buffer);
  if (dv.byteLength < 84) throw new Error('STL too small');
  const nTri = dv.getUint32(80, true);
  if (dv.byteLength !== 84 + nTri * 50) {
    throw new Error(`not a well-formed binary STL (len ${dv.byteLength}, tris ${nTri})`);
  }
  const out = new Float32Array(nTri * 9);
  let o = 84,
    k = 0;
  for (let t = 0; t < nTri; t++) {
    o += 12; // skip the face normal
    for (let v = 0; v < 9; v++) {
      out[k++] = dv.getFloat32(o, true);
      o += 4;
    }
    o += 2; // skip the attribute byte count
  }
  return out;
}

// Right-side meshes mirror across the sagittal plane (y → −y).
export function transformVertices(raw, scale, mirror) {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i += 3) {
    out[i] = raw[i] * scale;
    out[i + 1] = raw[i + 1] * scale * (mirror ? -1 : 1);
    out[i + 2] = raw[i + 2] * scale;
  }
  return out;
}
