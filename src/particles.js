/**
 * CyberPuppet Particle FX Engine
 * High-performance VFX for mecha thruster plumes, superhero landing shockwaves,
 * combat punch sparks, blade slash arcs, and dancing beat rings.
 */

import * as THREE from 'three';

export class ParticleFX {
  constructor(scene) {
    this.scene = scene;

    // Sparks / Plasma particles
    this.maxParticles = 800;
    this.particleCount = 0;
    
    const geom = new THREE.BufferGeometry();
    this.positions = new Float32Array(this.maxParticles * 3);
    this.velocities = new Float32Array(this.maxParticles * 3);
    this.colors = new Float32Array(this.maxParticles * 3);
    this.lifetimes = new Float32Array(this.maxParticles);
    this.maxLifetimes = new Float32Array(this.maxParticles);
    this.sizes = new Float32Array(this.maxParticles);

    geom.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));

    // Custom circular spark point texture using HTML5 2D canvas
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.2, 'rgba(255, 50, 80, 0.95)');
    grad.addColorStop(0.55, 'rgba(255, 0, 40, 0.4)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    const sparkTexture = new THREE.CanvasTexture(canvas);

    this.pMaterial = new THREE.PointsMaterial({
      size: 0.22,
      vertexColors: true,
      map: sparkTexture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.points = new THREE.Points(geom, this.pMaterial);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    // Shockwave Rings Pool
    this.shockwaves = [];
    this.maxShockwaves = 20;
    const ringGeom = new THREE.RingGeometry(0.1, 0.28, 48);
    ringGeom.rotateX(-Math.PI / 2);

    for (let i = 0; i < this.maxShockwaves; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0xff003c,
        transparent: true,
        opacity: 0.0,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });
      const mesh = new THREE.Mesh(ringGeom, mat);
      mesh.visible = false;
      this.scene.add(mesh);
      this.shockwaves.push({
        mesh,
        life: 0,
        maxLife: 0.6,
        scaleSpeed: 5.5,
        baseScale: 1.0
      });
    }
  }

  spawnShockwave(position, colorHex = 0xff003c, maxScale = 2.4, duration = 0.55) {
    const sw = this.shockwaves.find(s => !s.mesh.visible);
    if (!sw) return;

    sw.mesh.position.copy(position);
    sw.mesh.position.y = Math.max(0.04, position.y);
    sw.mesh.scale.set(0.1, 0.1, 0.1);
    sw.mesh.material.color.setHex(colorHex);
    sw.mesh.material.opacity = 0.95;
    sw.mesh.visible = true;
    sw.life = 0;
    sw.maxLife = duration;
    sw.scaleSpeed = maxScale / duration;
  }

  spawnThrusterSparks(position, count = 3, colorHex = 0xff3300) {
    const c = new THREE.Color(colorHex);
    for (let i = 0; i < count; i++) {
      if (this.particleCount >= this.maxParticles) break;
      const idx = this.particleCount;
      const i3 = idx * 3;

      this.positions[i3] = position.x + (Math.random() - 0.5) * 0.14;
      this.positions[i3 + 1] = position.y - 0.16;
      this.positions[i3 + 2] = position.z + (Math.random() - 0.5) * 0.14;

      // Downward jet thrust velocity with flare spread
      this.velocities[i3] = (Math.random() - 0.5) * 1.1;
      this.velocities[i3 + 1] = -3.0 - Math.random() * 2.5;
      this.velocities[i3 + 2] = (Math.random() - 0.5) * 1.1;

      this.colors[i3] = c.r;
      this.colors[i3 + 1] = c.g;
      this.colors[i3 + 2] = c.b;

      this.lifetimes[idx] = 0;
      this.maxLifetimes[idx] = 0.35 + Math.random() * 0.25;

      this.particleCount++;
    }
  }

  spawnCombatImpact(position, colorHex = 0xff003c, count = 18) {
    const c = new THREE.Color(colorHex);
    for (let i = 0; i < count; i++) {
      if (this.particleCount >= this.maxParticles) break;
      const idx = this.particleCount;
      const i3 = idx * 3;

      this.positions[i3] = position.x;
      this.positions[i3 + 1] = position.y;
      this.positions[i3 + 2] = position.z;

      // Explosive radial burst
      const theta = Math.random() * Math.PI * 2;
      const phi = (Math.random() - 0.5) * Math.PI;
      const speed = 2.5 + Math.random() * 4.5;

      this.velocities[i3] = Math.cos(theta) * Math.cos(phi) * speed;
      this.velocities[i3 + 1] = Math.sin(phi) * speed + 1.0;
      this.velocities[i3 + 2] = Math.sin(theta) * Math.cos(phi) * speed;

      this.colors[i3] = c.r;
      this.colors[i3 + 1] = c.g;
      this.colors[i3 + 2] = c.b;

      this.lifetimes[idx] = 0;
      this.maxLifetimes[idx] = 0.45 + Math.random() * 0.35;

      this.particleCount++;
    }
  }

  spawnBladeSlashArc(position, colorHex = 0xff003c, count = 15) {
    const c = new THREE.Color(colorHex);
    for (let i = 0; i < count; i++) {
      if (this.particleCount >= this.maxParticles) break;
      const idx = this.particleCount;
      const i3 = idx * 3;

      const angle = (i / count) * Math.PI * 0.8 - 0.4;
      const radius = 0.3 + (i / count) * 0.4;

      this.positions[i3] = position.x + Math.sin(angle) * radius;
      this.positions[i3 + 1] = position.y + Math.cos(angle) * radius * 0.5;
      this.positions[i3 + 2] = position.z + 0.1;

      this.velocities[i3] = Math.sin(angle) * 2.0;
      this.velocities[i3 + 1] = (Math.random() - 0.5) * 0.5;
      this.velocities[i3 + 2] = 1.0 + Math.random() * 1.5;

      this.colors[i3] = c.r;
      this.colors[i3 + 1] = c.g;
      this.colors[i3 + 2] = c.b;

      this.lifetimes[idx] = 0;
      this.maxLifetimes[idx] = 0.3 + Math.random() * 0.2;

      this.particleCount++;
    }
  }

  update(delta) {
    // 1. Update shockwaves
    for (let i = 0; i < this.shockwaves.length; i++) {
      const sw = this.shockwaves[i];
      if (!sw.mesh.visible) continue;

      sw.life += delta;
      const progress = sw.life / sw.maxLife;

      if (progress >= 1.0) {
        sw.mesh.visible = false;
      } else {
        const s = 1.0 + progress * sw.scaleSpeed * 3.5;
        sw.mesh.scale.set(s, s, s);
        sw.mesh.material.opacity = (1.0 - progress) * 0.95;
      }
    }

    // 2. Update particles
    let active = 0;
    for (let i = 0; i < this.particleCount; i++) {
      const i3 = i * 3;
      this.lifetimes[i] += delta;

      if (this.lifetimes[i] < this.maxLifetimes[i]) {
        // Integrate physics
        this.positions[i3] += this.velocities[i3] * delta;
        this.positions[i3 + 1] += this.velocities[i3 + 1] * delta;
        this.positions[i3 + 2] += this.velocities[i3 + 2] * delta;

        // Gravity drag
        this.velocities[i3 + 1] -= 4.2 * delta;

        // If alive, keep in active pool
        if (active !== i) {
          const a3 = active * 3;
          this.positions[a3] = this.positions[i3];
          this.positions[a3 + 1] = this.positions[i3 + 1];
          this.positions[a3 + 2] = this.positions[i3 + 2];

          this.velocities[a3] = this.velocities[i3];
          this.velocities[a3 + 1] = this.velocities[i3 + 1];
          this.velocities[a3 + 2] = this.velocities[i3 + 2];

          this.colors[a3] = this.colors[i3];
          this.colors[a3 + 1] = this.colors[i3 + 1];
          this.colors[a3 + 2] = this.colors[i3 + 2];

          this.lifetimes[active] = this.lifetimes[i];
          this.maxLifetimes[active] = this.maxLifetimes[i];
        }
        active++;
      }
    }
    this.particleCount = active;

    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.color.needsUpdate = true;
    this.points.geometry.setDrawRange(0, this.particleCount);
  }
}
