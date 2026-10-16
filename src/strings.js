/**
 * CyberPuppet Holographic Tension Filaments (Strings)
 * 3D glowing catenary/bezier laser filaments connecting the controller's hand
 * to the robot's physical anchor points with dynamic droop, tension vibrations,
 * and traveling energy pulses.
 */

import * as THREE from 'three';

export class StringManager {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = "TensionFilaments";
    this.scene.add(this.group);

    this.numSegments = 24;
    this.strings = [];
    this.energyPulses = [];

    // 5 Filament definitions matching viewer screen space (thumb on viewer left -> robot right)
    this.filamentDefs = [
      { name: "thumb", color: 0x00f0ff, anchorKey: "r_hand", handIdx: 4 },
      { name: "index", color: 0x38bdf8, anchorKey: "head", handIdx: 8 },
      { name: "middle", color: 0xc084fc, anchorKey: "chest", handIdx: 12 },
      { name: "ring", color: 0xec4899, anchorKey: "l_hand", handIdx: 16 },
      { name: "pinky", color: 0xf43f5e, anchorKey: "l_foot", handIdx: 20 },
    ];

    this._initFilaments();
  }

  _initFilaments() {
    this.filamentDefs.forEach((def, i) => {
      const positions = new Float32Array((this.numSegments + 1) * 3);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      const mat = new THREE.LineBasicMaterial({
        color: def.color,
        transparent: true,
        opacity: 0.75,
        linewidth: 1.5,
        blending: THREE.AdditiveBlending
      });

      const line = new THREE.Line(geom, mat);
      this.group.add(line);

      // Sleek laser pulse bead (micro-spark moving along filament)
      const beadGeom = new THREE.SphereGeometry(0.018, 8, 8);
      const beadMat = new THREE.MeshBasicMaterial({
        color: def.color,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending
      });
      const bead = new THREE.Mesh(beadGeom, beadMat);
      this.group.add(bead);

      this.strings.push({
        def,
        line,
        geom,
        mat,
        bead,
        pulsePhase: Math.random()
      });
    });
  }

  setThemeColor(colorHex) {
    this.strings.forEach(s => {
      s.mat.color.setHex(colorHex);
      s.bead.material.color.setHex(colorHex);
    });
  }

  setVisible(visible) {
    this.group.visible = visible;
  }

  update(delta, time, robotAnchors, handPos3D, fingerCurls) {
    if (!robotAnchors || !this.group.visible) return;

    // Default top-plane virtual hand control bar if hand isn't active
    const defaultHandBar = [
      new THREE.Vector3(-0.9, 2.7, 0.4), // Thumb
      new THREE.Vector3(-0.4, 2.8, 0.5), // Index
      new THREE.Vector3(0.0, 2.9, 0.4),  // Middle
      new THREE.Vector3(0.4, 2.8, 0.5),  // Ring
      new THREE.Vector3(0.9, 2.7, 0.4),  // Pinky
    ];

    this.strings.forEach((str, i) => {
      const def = str.def;
      const anchor = robotAnchors[def.anchorKey] || new THREE.Vector3(0, 1.5, 0);

      // Source origin: from projected 3D fingertip or default control bar
      let topPoint;
      if (handPos3D && handPos3D[def.name]) {
        topPoint = handPos3D[def.name];
      } else {
        topPoint = defaultHandBar[i];
      }

      // Curl determines string droop: curled = loose catenary droop, extended = taut vibration
      const curl = fingerCurls ? (fingerCurls[def.name] ?? 0.2) : 0.2;
      const droopAmount = curl * 0.45;
      const tensionVibe = (1.0 - curl) * 0.015 * Math.sin(time * 35.0 + i * 2.0);

      // Mid-point sag
      const mid = new THREE.Vector3()
        .addVectors(topPoint, anchor)
        .multiplyScalar(0.5);
      mid.y -= droopAmount;
      mid.x += tensionVibe;

      // Quadratic bezier curve interpolation
      const curve = new THREE.QuadraticBezierCurve3(topPoint, mid, anchor);
      const points = curve.getPoints(this.numSegments);

      const posAttr = str.geom.attributes.position;
      for (let s = 0; s <= this.numSegments; s++) {
        const pt = points[s];
        posAttr.setXYZ(s, pt.x, pt.y, pt.z);
      }
      posAttr.needsUpdate = true;

      // Update energy pulse bead
      str.pulsePhase = (str.pulsePhase + delta * (1.2 + (1.0 - curl) * 2.0)) % 1.0;
      const beadPos = curve.getPoint(str.pulsePhase);
      str.bead.position.copy(beadPos);
      str.bead.scale.setScalar(0.7 + Math.sin(str.pulsePhase * Math.PI) * 0.6);
    });
  }
}
