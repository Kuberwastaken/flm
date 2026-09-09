import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { parseBinarySTL, transformVertices } from './body/stl.js';
import { neutralPoseRad, segmentTransforms, validateFK } from './body/fk.js';

const base = import.meta.env.BASE_URL;
async function get(path, binary = false) {
  const response = await fetch(`${base}${path}`);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status}).`);
  return binary ? response.arrayBuffer() : response.json();
}

class View {
  constructor(element, background) {
    this.element = element;
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(background);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.domElement.setAttribute('aria-hidden', 'true');
    element.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false; this.controls.minDistance = 1; this.controls.maxDistance = 14;
    this.controls.addEventListener('change', () => this.render());
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(element);
    this.renderer.domElement.addEventListener('webglcontextlost', event => {
      event.preventDefault(); element.dispatchEvent(new CustomEvent('viewerror', { bubbles: true, detail: 'The 3D context was lost. Text generation remains available. Reload to restore the view.' }));
    });
  }
  resize() {
    const width = this.element.clientWidth, height = this.element.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height); this.camera.aspect = width / height; this.camera.updateProjectionMatrix(); this.render();
  }
  render() { this.renderer.render(this.scene, this.camera); }
  home() { this.camera.position.copy(this.homePosition); this.controls.target.set(0, 0, 0); this.controls.update(); this.render(); }
  rotate(horizontal, vertical = 0) {
    const position = new THREE.Spherical().setFromVector3(this.camera.position);
    position.theta += horizontal; position.phi = THREE.MathUtils.clamp(position.phi + vertical, 0.1, Math.PI - 0.1);
    this.camera.position.setFromSpherical(position); this.controls.update(); this.render();
  }
  zoom(factor) { this.camera.position.multiplyScalar(factor).clampLength(1, 14); this.controls.update(); }
}

export class BrainView extends View {
  constructor(element, onSelect) {
    super(element, '#191a18'); this.onSelect = onSelect;
    this.homePosition = new THREE.Vector3(0, 0, 4.8); this.home();
    this.raycaster = new THREE.Raycaster(); this.raycaster.params.Points.threshold = 0.04;
    let down;
    this.renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    this.renderer.domElement.addEventListener('pointerup', e => {
      if (!this.activePoints || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const box = this.renderer.domElement.getBoundingClientRect();
      this.raycaster.setFromCamera(new THREE.Vector2((e.clientX - box.left) / box.width * 2 - 1, 1 - (e.clientY - box.top) / box.height * 2), this.camera);
      const hit = this.raycaster.intersectObject(this.activePoints)[0];
      if (hit) this.onSelect(this.renderedIndices[hit.index]);
    });
  }
  async load(config) {
    const buffer = await get(`${config.package_path || 'models/flm-compact'}/anatomy.json`, true);
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), x => x.toString(16).padStart(2, '0')).join('');
    if (digest !== config.anatomy_sha256) throw new Error('Anatomy checksum mismatch. Reload to fetch a consistent release.');
    this.anatomy = JSON.parse(new TextDecoder().decode(buffer));
    // Display crop follows the source viewer's brain range, excluding the VNC.
    // This affects context framing only, never which neurons are simulated.
    const reference = (this.anatomy.context_positions || this.anatomy.positions.filter(Boolean))
      .filter(p => p[2] >= 9000 && p[2] <= 58000);
    const bounds = new THREE.Box3().setFromPoints(reference.map(p => new THREE.Vector3(...p)));
    const center = bounds.getCenter(new THREE.Vector3()), extent = bounds.getSize(new THREE.Vector3());
    const scale = 3.9 / Math.max(extent.x, extent.y, extent.z);
    // A rigid 25-degree tilt matches the readable frontal anatomical orientation.
    const sine = Math.sin(25 * Math.PI / 180), cosine = Math.cos(25 * Math.PI / 180);
    const position = p => [-(p[0] - center.x) * scale,
      -((p[1] - center.y) * sine + (p[2] - center.z) * cosine) * scale,
      ((p[1] - center.y) * cosine - (p[2] - center.z) * sine) * scale];
    const contextGeometry = new THREE.BufferGeometry();
    contextGeometry.setAttribute('position', new THREE.Float32BufferAttribute(reference.flatMap(position), 3));
    this.contextPoints = new THREE.Points(contextGeometry, new THREE.PointsMaterial({ color: '#81837b', size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0.38 }));
    this.scene.add(this.contextPoints);
    this.renderedIndices = [];
    const positions = [];
    this.anatomy.positions.forEach((p, i) => { if (p) { this.renderedIndices.push(i); positions.push(...position(p)); } });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(positions.length).fill(0.15), 3));
    this.activePoints = new THREE.Points(geometry, new THREE.PointsMaterial({ vertexColors: true, size: 2.7, sizeAttenuation: false }));
    this.scene.add(this.activePoints);
    const selectionGeometry = new THREE.BufferGeometry();
    selectionGeometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    this.selection = new THREE.Points(selectionGeometry, new THREE.PointsMaterial({ color: '#ffffff', size: 8, sizeAttenuation: false, depthTest: false }));
    this.selection.visible = false; this.scene.add(this.selection); this.resize();
    return this.anatomy;
  }
  select(index) {
    if (!this.activePoints) return;
    const offset = this.renderedIndices.indexOf(index); this.selection.visible = offset >= 0;
    if (offset >= 0) {
      const p = this.activePoints.geometry.attributes.position;
      this.selection.geometry.attributes.position.setXYZ(0, p.getX(offset), p.getY(offset), p.getZ(offset));
      this.selection.geometry.attributes.position.needsUpdate = true;
    }
    this.render();
  }
  update(state, mode = 'h') {
    if (!this.activePoints) return;
    const colors = this.activePoints.geometry.attributes.color;
    const negative = new THREE.Color('#82b5ad'), positive = new THREE.Color('#edb170'), zero = new THREE.Color('#686b62');
    const color = new THREE.Color();
    this.renderedIndices.forEach((index, j) => {
      const value = state[mode][index]; color.copy(zero).lerp(value >= 0 ? positive : negative, Math.min(1, Math.abs(value) * 2.4));
      colors.setXYZ(j, color.r, color.g, color.b);
    });
    colors.needsUpdate = true; this.render();
  }
  context(visible) { if (this.contextPoints) { this.contextPoints.visible = visible; this.render(); } }
}

export class FlyView extends View {
  constructor(element) {
    super(element, '#f2eee6'); this.camera.up.set(0, 0, 1);
    this.homePosition = new THREE.Vector3(2.6, -3.2, 2.2); this.home();
    this.scene.add(new THREE.HemisphereLight('#fffaf0', '#807660', 2.8));
    const light = new THREE.DirectionalLight('#ffffff', 3); light.position.set(2, -3, 7); this.scene.add(light);
    this.fly = new THREE.Group(); this.scene.add(this.fly); this.meshes = [];
  }
  async load() {
    this.model = await get('body/model.json');
    if (validateFK(this.model) > 1e-8) throw new Error('Body kinematics failed the reference-pose check.');
    const files = [...new Set(Object.values(this.model.meshes).map(x => x.file))], raw = new Map();
    let cursor = 0;
    await Promise.all(Array.from({ length: 4 }, async () => {
      while (cursor < files.length) { const file = files[cursor++]; raw.set(file, parseBinarySTL(await get(`body/meshes/${file}`, true))); }
    }));
    for (const [name, spec] of Object.entries(this.model.meshes)) {
      const geometry = new THREE.BufferGeometry();
      const positions = transformVertices(raw.get(spec.file), this.model.meshScale, spec.mirror);
      // Mirroring reverses winding. Repair triangles before calculating normals.
      if (spec.mirror) for (let i = 0; i < positions.length; i += 9)
        for (let j = 0; j < 3; j++) [positions[i + 3 + j], positions[i + 6 + j]] = [positions[i + 6 + j], positions[i + 3 + j]];
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3)); geometry.computeVertexNormals();
      const wing = name.includes('wing'), eye = name.endsWith('eye');
      const material = new THREE.MeshStandardMaterial({ color: eye ? '#912f22' : wing ? '#c3b9a5' : '#9f723c',
        roughness: eye ? 0.4 : 0.73, metalness: 0, transparent: wing, opacity: wing ? 0.43 : 1,
        side: wing ? THREE.DoubleSide : THREE.FrontSide, depthWrite: !wing });
      const mesh = new THREE.Mesh(geometry, material); mesh.matrixAutoUpdate = false;
      this.fly.add(mesh); this.meshes.push({ name, mesh });
    }
    this.pose();
    const bounds = new THREE.Box3().setFromObject(this.fly), center = bounds.getCenter(new THREE.Vector3());
    this.fly.position.sub(center);
    const size = bounds.getSize(new THREE.Vector3());
    this.fly.scale.setScalar(3.5 / Math.max(size.x, size.y, size.z));
    this.fly.position.multiplyScalar(this.fly.scale.x); this.resize();
  }
  pose(state = null, enabled = true) {
    if (!this.model) return;
    const angles = neutralPoseRad(this.model);
    if (state && enabled) {
      const mean = (values, start, end) => values.slice(start, end).reduce((sum, x) => sum + x, 0) / Math.max(1, end - start);
      const half = Math.floor(state.h.length / 2);
      const asymmetry = mean(state.h, 0, half) - mean(state.h, half, state.h.length);
      const persistence = mean(state.slow, 0, state.slow.length);
      angles['c_thorax-c_head-yaw'] += asymmetry * 0.8;
      angles['c_thorax-c_head-pitch'] += persistence * 0.6;
      for (const side of ['l', 'r']) angles[`c_thorax-${side}_wing-roll`] += state.meanActivity * (side === 'l' ? 0.6 : -0.6);
    }
    const transforms = segmentTransforms(this.model, angles);
    for (const { name, mesh } of this.meshes) { mesh.matrix.set(...transforms[name]); mesh.matrixWorldNeedsUpdate = true; }
    this.render();
  }
}
