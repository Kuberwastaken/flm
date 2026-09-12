import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { FlyView } from './views.js';
import { neutralPoseRad, segmentTransforms, mat4ApplyPoint } from './body/fk.js';
import { makeTypingRig, typingPose, KEY_TOP, FLOOR_Z } from './body/typing-pose.js';

// Reuses the exact body meshes and joint hierarchy of the anatomical fly view.
export class TypingFlyView extends FlyView {
  constructor(element) {
    super(element);
    this.controls.enabled = false;
    this.renderer.domElement.style.pointerEvents = 'none';
    this.applyTheme = () => { this.scene.background.set(getComputedStyle(document.documentElement).getPropertyValue('--scene-background').trim() || '#faf8f5'); this.render(); };
    document.addEventListener('flm-theme', this.applyTheme); this.applyTheme();
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.children.filter(x => x.isLight).forEach(x => this.scene.remove(x));
    this.scene.add(new THREE.HemisphereLight('#fffaf2', '#80766b', 2.0));
    const light = new THREE.DirectionalLight('#fff4df', 3.2);
    light.position.set(-1, -3, 6); light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    Object.assign(light.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.1, far: 15 });
    light.shadow.normalBias = 0.015; light.shadow.bias = -0.0002;
    this.scene.add(light);
    const fill = new THREE.DirectionalLight('#ffffff', 0.8); fill.position.set(3, 4, 3); this.scene.add(fill);
    this.camera.fov = 34;
    this.camera.position.set(3.9, -6.2, 4.4);
    this.controls.target.set(0.35, 0, 0.55); this.controls.update();
  }

  async load() {
    await super.load();
    this.fly.scale.setScalar(1); this.fly.position.set(0, 0, 0);
    const neutral = segmentTransforms(this.model, neutralPoseRad(this.model)), tips = {};
    for (const { name, mesh } of this.meshes) {
      mesh.castShadow = !name.includes('wing'); mesh.receiveShadow = true;
      if (name.includes('wing')) { mesh.material.opacity = 0.30; mesh.material.roughness = 0.38; }
      if (name.startsWith('c_abdomen')) mesh.material.color.set('#84512e');
      if (!name.endsWith('_tarsus5')) continue;
      const positions = mesh.geometry.attributes.position;
      let lowest = Infinity;
      for (let i = 0; i < positions.count; i++) {
        const vertex = [positions.getX(i), positions.getY(i), positions.getZ(i)];
        const z = mat4ApplyPoint(neutral[name], vertex)[2];
        if (z < lowest) { lowest = z; tips[name.slice(0, 2)] = vertex; }
      }
    }
    this.rig = makeTypingRig(this.model, tips);
    this.createKeyboard();
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ color: '#574936', opacity: 0.23 }));
    floor.position.z = FLOOR_Z; floor.receiveShadow = true; this.scene.add(floor);
    this.draw(0, false); this.resize();
  }

  createKeyboard() {
    const caseMaterial = new THREE.MeshStandardMaterial({ color: '#aaa399', roughness: 0.68, metalness: 0.15 });
    const chassis = new THREE.Mesh(new RoundedBoxGeometry(1.64, 3.08, 0.22, 2, 0.055), caseMaterial);
    chassis.position.set(1.86, 0, FLOOR_Z + 0.115); chassis.castShadow = true; chassis.receiveShadow = true; this.scene.add(chassis);
    const plate = new THREE.Mesh(new RoundedBoxGeometry(1.50, 2.96, 0.07, 2, 0.025), new THREE.MeshStandardMaterial({ color: '#3d3934', roughness: 0.85 }));
    plate.position.set(1.86, 0, 0.165); plate.receiveShadow = true; this.scene.add(plate);
    const keyGeometry = new RoundedBoxGeometry(0.22, 0.245, 0.13, 2, 0.024);
    const ivory = new THREE.MeshStandardMaterial({ color: '#e6dfd2', roughness: 0.65 });
    const accent = new THREE.MeshStandardMaterial({ color: '#b48156', roughness: 0.7 });
    this.keys = {};
    for (let row = 0; row < 5; row++) for (let column = 0; column < 11; column++) {
      const key = new THREE.Mesh(keyGeometry, row === 0 && [2, 8].includes(column) ? accent : ivory);
      key.position.set(1.35 + row * 0.26, (column - 5) * 0.28, KEY_TOP - 0.065);
      key.castShadow = true; key.receiveShadow = true; this.scene.add(key);
      if (row === 0 && column === 2) this.keys.rf = key;
      if (row === 0 && column === 8) this.keys.lf = key;
    }
    // The wide near-edge space bar makes the object readable as a keyboard.
    const space = new THREE.Mesh(new RoundedBoxGeometry(0.21, 1.31, 0.13, 2, 0.025), ivory);
    space.position.set(1.08, 0, KEY_TOP - 0.065); space.castShadow = true; space.receiveShadow = true; this.scene.add(space);
  }

  draw(seconds, typing) {
    if (!this.rig) return;
    const { angles, depression } = typingPose(this.model, this.rig, seconds, typing);
    const transforms = segmentTransforms(this.model, angles);
    for (const { name, mesh } of this.meshes) { mesh.matrix.set(...transforms[name]); mesh.matrixWorldNeedsUpdate = true; }
    for (const leg of ['lf', 'rf']) this.keys[leg].position.z = KEY_TOP - 0.065 - depression[leg];
    this.render();
  }

  dispose() {
    document.removeEventListener('flm-theme', this.applyTheme);
    this.observer.disconnect(); this.controls.dispose();
    const geometries = new Set(), materials = new Set();
    this.scene.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) materials.add(object.material);
      if (object.shadow) object.shadow.dispose();
    });
    geometries.forEach(value => value.dispose()); materials.forEach(value => value.dispose());
    this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
