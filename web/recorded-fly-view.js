import * as THREE from 'three';
import { View } from './views.js';
import { checkedBytes, decodeJSON } from './food-replay-data.js';

export class RecordedFlyView extends View {
  constructor(element) {
    super(element, '#f2eee6'); this.camera.up.set(0, 0, 1);
    this.controls.maxDistance = 50; this.follow = new THREE.Vector3(); this.meshes = [];
    this.homePosition = new THREE.Vector3(5, -8, 6); this.home();
    this.scene.add(new THREE.HemisphereLight('#fffaf0', '#807660', 2.8));
    const light = new THREE.DirectionalLight('#ffffff', 3); light.position.set(2, -3, 7); this.scene.add(light);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 30), new THREE.MeshStandardMaterial({color: '#e5dfd2', roughness: 1}));
    ground.position.set(12, 0, -.02); this.scene.add(ground);
    this.patches = new THREE.Group(); this.scene.add(this.patches);
  }
  home() {
    const target = this.follow || new THREE.Vector3();
    this.camera.position.copy(this.homePosition).add(target); this.controls.target.copy(target); this.controls.update(); this.render();
  }
  async load(manifest, base) {
    const model = decodeJSON(await checkedBytes(`${base}body/recorded-food/model.json`, manifest.body_manifest_sha256));
    const bytes = await checkedBytes(`${base}body/recorded-food/geometry.bin`, model.geometry_sha256);
    if (model.geometries.length !== 69 || bytes.byteLength !== model.geometry_bytes) throw new Error('Body mesh inventory differs.');
    const view = new DataView(bytes);
    for (const row of model.geometries) {
      const positions = new Float32Array(row.vertices.count * 3), indices = new Uint32Array(row.faces.count * 3);
      for (let i = 0; i < positions.length; i++) positions[i] = view.getFloat32(row.vertices.offset + i * 4, true);
      for (let i = 0; i < indices.length; i++) indices[i] = view.getUint32(row.faces.offset + i * 4, true);
      const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1)); geometry.computeVertexNormals();
      const wing = row.segment.includes('wing'), eye = row.segment.endsWith('eye');
      const material = new THREE.MeshStandardMaterial({color: eye ? '#912f22' : wing ? '#c3b9a5' : '#9f723c',
        roughness: .73, transparent: wing, opacity: wing ? .43 : 1, side: wing ? THREE.DoubleSide : THREE.FrontSide, depthWrite: !wing});
      const mesh = new THREE.Mesh(geometry, material); mesh.matrixAutoUpdate = false; this.scene.add(mesh); this.meshes.push(mesh);
    }
    this.resize();
  }
  field(field) {
    this.patches.traverse(node => { node.geometry?.dispose(); node.material?.map?.dispose(); node.material?.dispose(); });
    this.patches.clear();
    for (const source of field.sources) {
      const patch = new THREE.Mesh(new THREE.CircleGeometry(source.contact_radius_mm, 48), new THREE.MeshBasicMaterial({ color: '#aab9a7', side: THREE.DoubleSide }));
      patch.position.set(source.position_mm[0], source.position_mm[1], .005); this.patches.add(patch);
      const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 64;
      const context = canvas.getContext('2d'); context.font = '24px sans-serif'; context.textAlign = 'center'; context.fillStyle = '#353e32';
      context.fillText(`${source.name.toUpperCase()} · ${source.sugar ? 'sugar' : 'neutral'}`, 128, 38);
      const texture = new THREE.CanvasTexture(canvas), label = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, depthTest: false}));
      label.position.copy(patch.position).add(new THREE.Vector3(0, 0, .55)); label.scale.set(2, .5, 1); this.patches.add(label);
    }
  }
  update(frame, thorax) {
    for (let i = 0; i < this.meshes.length; i++) {
      const v = frame.geometry.subarray(i * 12, (i + 1) * 12);
      this.meshes[i].matrix.set(v[3],v[4],v[5],v[0], v[6],v[7],v[8],v[1], v[9],v[10],v[11],v[2], 0,0,0,1);
      this.meshes[i].matrixWorldNeedsUpdate = true;
    }
    const target = new THREE.Vector3(...thorax), delta = target.clone().sub(this.follow);
    this.camera.position.add(delta); this.controls.target.copy(target); this.follow.copy(target); this.controls.update(); this.render();
  }
}
