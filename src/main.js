/**
 * CyberPuppet Main Application
 * Integrates Three.js PBR scene, OrbitControls, Mecha Robot, Particle VFX,
 * Tension Filaments, MediaPipe Hand Tracker, and Cyberpunk HUD UI.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Robot3D } from './robot.js';
import { ParticleFX } from './particles.js';
import { StringManager } from './strings.js';
import { HandTracker } from './handTracker.js';

class App {
  constructor() {
    this.container = document.getElementById('canvas-container');
    this.clock = new THREE.Clock();

    // Camera view presets
    this.cameraPresets = {
      front: { pos: new THREE.Vector3(0, 1.8, 4.2), target: new THREE.Vector3(0, 1.5, 0) },
      action: { pos: new THREE.Vector3(2.5, 1.2, 3.4), target: new THREE.Vector3(0, 1.4, 0) },
      close: { pos: new THREE.Vector3(0, 2.3, 1.8), target: new THREE.Vector3(0, 2.2, 0) }
    };
    this.targetCameraPos = this.cameraPresets.front.pos.clone();
    this.targetCameraLookAt = this.cameraPresets.front.target.clone();

    this.isWireframe = false;
    this.frameCount = 0;
    this.lastFpsUpdate = 0;

    this._initThree();
    this._initSceneLighting();
    this._initMecha();
    this._initFilamentsAndParticles();
    this._initHandTracker();
    this._initHUD();
    this._initEventListeners();

    // Support URL parameters for direct view/theme/action presets
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('theme')) {
      const t = urlParams.get('theme');
      const btn = document.querySelector(`.theme-btn[data-theme="${t}"]`);
      if (btn) btn.click();
    }
    if (urlParams.has('action')) {
      const act = urlParams.get('action').toUpperCase();
      this.robot.triggerAction(act);
      this.robot.actionLocked = true;
      this._syncActiveActionUI(act);
      this.elLockVal.innerText = "ON";
      this.elLockBtn.classList.add('active');
    }
    if (urlParams.has('view')) {
      const v = urlParams.get('view');
      const btn = document.querySelector(`.view-btn[data-view="${v}"]`);
      if (btn) btn.click();
    }

    // Start loop
    this._animate();
  }

  _initThree() {
    // 1. Scene & Sci-Fi Atmosphere Fog
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060913);
    this.scene.fog = new THREE.FogExp2(0x060913, 0.045);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      100
    );
    this.camera.position.copy(this.cameraPresets.front.pos);

    // 3. WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.02;
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.target.copy(this.cameraPresets.front.target);
    this.controls.maxPolarAngle = Math.PI / 2 + 0.05; // Prevent dipping beneath floor
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 12.0;

    // Window resize
    window.addEventListener('resize', () => this._onWindowResize());
  }

  _initSceneLighting() {
    // Ambient fill (dark navy for deep sci-fi shadows)
    const ambLight = new THREE.AmbientLight(0x0e172a, 0.7);
    this.scene.add(ambLight);

    // Hemisphere sky/ground radiance
    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x020617, 0.7);
    this.scene.add(hemiLight);

    // Key Sun Light with soft shadow casting
    this.dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    this.dirLight.position.set(4, 8, 5);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.width = 2048;
    this.dirLight.shadow.mapSize.height = 2048;
    this.dirLight.shadow.camera.near = 0.5;
    this.dirLight.shadow.camera.far = 25;
    this.dirLight.shadow.camera.left = -3;
    this.dirLight.shadow.camera.right = 3;
    this.dirLight.shadow.camera.top = 4;
    this.dirLight.shadow.camera.bottom = -1;
    this.dirLight.shadow.bias = -0.0004;
    this.scene.add(this.dirLight);

    // Cyberpunk Rim Light 1: Cyan / Electric Teal (Left Rear)
    const rimLightLeft = new THREE.DirectionalLight(0x00f0ff, 1.6);
    rimLightLeft.position.set(-4, 3, -3);
    this.scene.add(rimLightLeft);

    // Cyberpunk Rim Light 2: Electric Magenta / Violet (Right Rear)
    const rimLightRight = new THREE.DirectionalLight(0xc084fc, 1.8);
    rimLightRight.position.set(4, 3, -3);
    this.scene.add(rimLightRight);

    // Undercarriage Ground Uplight
    const upLight = new THREE.PointLight(0x00ffd5, 0.8, 5);
    upLight.position.set(0, 0.2, 0.5);
    this.scene.add(upLight);

    // Ground Platform & Cyber Grid
    this._initGroundPlatform();
  }

  _initGroundPlatform() {
    // 1. Grid Helper
    const grid = new THREE.GridHelper(30, 60, 0x00f0ff, 0x1e293b);
    grid.position.y = 0.001;
    grid.material.opacity = 0.45;
    grid.material.transparent = true;
    this.scene.add(grid);

    // 2. High-tech Circular Stage Pad
    const stageGeom = new THREE.CylinderGeometry(2.4, 2.6, 0.08, 48);
    const stageMat = new THREE.MeshStandardMaterial({
      color: 0x090d16,
      roughness: 0.25,
      metalness: 0.85
    });
    const stage = new THREE.Mesh(stageGeom, stageMat);
    stage.position.y = -0.04;
    stage.receiveShadow = true;
    this.scene.add(stage);

    // 3. Glowing Concentric Hologram Rings
    const ring1Geom = new THREE.RingGeometry(2.35, 2.42, 64);
    ring1Geom.rotateX(-Math.PI / 2);
    const ring1Mat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending
    });
    const ring1 = new THREE.Mesh(ring1Geom, ring1Mat);
    ring1.position.y = 0.005;
    this.scene.add(ring1);

    const ring2Geom = new THREE.RingGeometry(1.4, 1.45, 48);
    ring2Geom.rotateX(-Math.PI / 2);
    const ring2Mat = new THREE.MeshBasicMaterial({
      color: 0xc084fc,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending
    });
    const ring2 = new THREE.Mesh(ring2Geom, ring2Mat);
    ring2.position.y = 0.006;
    this.scene.add(ring2);

    // 4. Ground Shadow Receiver Plane
    const planeGeom = new THREE.PlaneGeometry(30, 30);
    const planeMat = new THREE.ShadowMaterial({ opacity: 0.55 });
    const shadowPlane = new THREE.Mesh(planeGeom, planeMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = 0;
    shadowPlane.receiveShadow = true;
    this.scene.add(shadowPlane);
  }

  _initMecha() {
    this.robot = new Robot3D();
    this.scene.add(this.robot.group);
  }

  _initFilamentsAndParticles() {
    // Particles
    this.particles = new ParticleFX(this.scene);

    // Screen Shake state
    this.cameraShakeIntensity = 0;

    // Hook robot events to particle system & camera shake
    this.robot.onShockwave = (pos, color, scale) => {
      this.particles.spawnShockwave(pos, color, scale);
      this.particles.spawnCombatImpact(pos, color, 16);
    };

    this.robot.onThrusterSpark = (pos, count, color) => {
      this.particles.spawnThrusterSparks(pos, count, color || 0xff3300);
    };

    this.robot.onSlashArc = (pos, color, count) => {
      this.particles.spawnBladeSlashArc(pos, color, count);
    };

    this.robot.onCameraShake = (intensity) => {
      this.cameraShakeIntensity = Math.max(this.cameraShakeIntensity, intensity);
    };

    // Strings (Tension Filaments)
    this.strings = new StringManager(this.scene);
  }

  _initHandTracker() {
    this.videoEl = document.getElementById('webcam-video');
    this.canvasEl = document.getElementById('hand-canvas');

    this.tracker = new HandTracker({
      videoEl: this.videoEl,
      canvasEl: this.canvasEl,
      onGestureChange: (action) => {
        if (!this.robot.actionLocked) {
          this.robot.triggerAction(action);
          this._syncActiveActionUI(action);
        }
      },
      onStatusLog: (msg) => this._logTerminal(msg)
    });
  }

  _initHUD() {
    this.elActiveAction = document.getElementById('active-action-text');
    this.elActionBadge = document.getElementById('action-badge');
    this.elLockBtn = document.getElementById('lock-toggle-btn');
    this.elLockVal = document.getElementById('lock-status-val');
    this.elFps = document.getElementById('fps-display');
    this.elTerminalLogs = document.getElementById('terminal-logs');
    this.elLogStatus = document.getElementById('log-status');
    this.elGestureBadge = document.getElementById('detected-gesture-badge');

    // Tension fill bars
    this.tensionBars = {
      thumb: document.getElementById('bar-thumb'),
      index: document.getElementById('bar-index'),
      middle: document.getElementById('bar-middle'),
      ring: document.getElementById('bar-ring'),
      pinky: document.getElementById('bar-pinky'),
    };
  }

  _initEventListeners() {
    // 1. Action Dock Buttons
    document.querySelectorAll('.action-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.getAttribute('data-action');
        if (action) {
          this.robot.triggerAction(action);
          this._syncActiveActionUI(action);
          this._logTerminal(`[MANUAL REMOTE] Action Triggered: ${action}`);
        }
      });
    });

    // 2. Action Lock Toggle Button
    this.elLockBtn.addEventListener('click', () => {
      const locked = this.robot.toggleActionLock();
      this.elLockVal.innerText = locked ? "ON" : "OFF";
      this.elLockBtn.classList.toggle('active', locked);
      this._logTerminal(`[LOCK] Action Continuous Loop: ${locked ? 'LOCKED' : 'UNLOCKED'}`);
    });

    // 3. Theme Buttons
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const theme = btn.getAttribute('data-theme');
        this.robot.setTheme(theme);
        this._logTerminal(`[THEME] Armor Coating Applied: ${theme.toUpperCase()}`);
      });
    });

    // 4. Camera View Presets
    document.querySelectorAll('.view-btn[data-view]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.view-btn[data-view]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.getAttribute('data-view');
        this._setCameraPreset(view);
      });
    });

    // 5. Wireframe Toggle
    const wireframeBtn = document.getElementById('wireframe-toggle-btn');
    wireframeBtn.addEventListener('click', () => {
      this.isWireframe = !this.isWireframe;
      this.robot.setWireframe(this.isWireframe);
      wireframeBtn.classList.toggle('active', this.isWireframe);
      this._logTerminal(`[SHADING] Wireframe Mode: ${this.isWireframe ? 'ENABLED' : 'DISABLED'}`);
    });

    // 6. Reset Pose
    const resetBtn = document.getElementById('reset-pose-btn');
    resetBtn.addEventListener('click', () => {
      this.robot.triggerAction('MARIONETTE');
      this.robot._resetStance();
      this._setCameraPreset('front');
      this._syncActiveActionUI('MARIONETTE');
      this._logTerminal(`[RESET] Kinematic Pose & Camera Stance Reset`);
    });

    // 7. Webcam & Demo Buttons
    const camToggleBtn = document.getElementById('cam-toggle-btn');
    const demoModeBtn = document.getElementById('demo-mode-btn');

    camToggleBtn.addEventListener('click', async () => {
      if (this.tracker.isCameraActive) {
        this.tracker.stopCamera();
        camToggleBtn.innerText = "START CAM";
        camToggleBtn.classList.remove('active');
        demoModeBtn.innerText = "DEMO: ON";
        demoModeBtn.classList.add('active');
      } else {
        await this.tracker.startCamera();
        if (this.tracker.isCameraActive) {
          camToggleBtn.innerText = "STOP CAM";
          camToggleBtn.classList.add('active');
          demoModeBtn.innerText = "DEMO: OFF";
          demoModeBtn.classList.remove('active');
        }
      }
    });

    demoModeBtn.addEventListener('click', () => {
      const demoState = this.tracker.toggleDemo();
      demoModeBtn.innerText = `DEMO: ${demoState ? 'ON' : 'OFF'}`;
      demoModeBtn.classList.toggle('active', demoState);
      if (!demoState && !this.tracker.isCameraActive) {
        camToggleBtn.innerText = "START CAM";
        camToggleBtn.classList.remove('active');
      }
    });

    // 8. Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      const key = e.key.toUpperCase();
      if (key === '1') this._triggerDockAction('JUMP');
      else if (key === '2') this._triggerDockAction('DANCE');
      else if (key === '3') this._triggerDockAction('RUN');
      else if (key === '4') this._triggerDockAction('FIGHT');
      else if (key === '5') this._triggerDockAction('MARIONETTE');
      else if (key === '6') this._triggerDockAction('MIMIC');
      else if (key === '7') this._triggerDockAction('SLASH');
      else if (key === 'L') this.elLockBtn.click();
      else if (key === 'W') wireframeBtn.click();
      else if (key === 'R') resetBtn.click();
      else if (key === 'C') camToggleBtn.click();
      else if (key === 'D') demoModeBtn.click();
    });
  }

  _triggerDockAction(action) {
    const btn = document.querySelector(`.action-btn[data-action="${action}"]`);
    if (btn) btn.click();
  }

  _setCameraPreset(presetKey) {
    const p = this.cameraPresets[presetKey];
    if (!p) return;
    this.targetCameraPos.copy(p.pos);
    this.targetCameraLookAt.copy(p.target);
  }

  _syncActiveActionUI(action) {
    if (this.elActiveAction) this.elActiveAction.innerText = action;

    // Update Bottom Dock active button
    document.querySelectorAll('.action-btn').forEach(btn => {
      const bAct = btn.getAttribute('data-action');
      btn.classList.toggle('active', bAct === action);
    });

    // Update PIP badge
    if (this.elGestureBadge) {
      const mode = this.tracker.isDemoActive ? "DEMO" : (this.tracker.hasHand ? "HAND DETECTED" : "NO HAND");
      this.elGestureBadge.innerText = `${mode} // ACTION: ${action}`;
    }

    // Highlight Remote guide list item
    document.querySelectorAll('.guide-item').forEach(item => {
      const gAct = item.getAttribute('data-action');
      item.classList.toggle('active', gAct === action);
    });
  }

  _logTerminal(text) {
    if (!this.elTerminalLogs) return;
    const line = document.createElement('div');
    line.className = 'log-line';
    if (text.startsWith('[OK]')) line.classList.add('ok');
    else if (text.startsWith('[WARN]') || text.startsWith('[ERR]')) line.classList.add('warn');
    else if (text.startsWith('[INFO]')) line.classList.add('info');
    else line.classList.add('system');

    line.innerText = text;
    this.elTerminalLogs.appendChild(line);

    // Keep max 8 lines in log
    while (this.elTerminalLogs.children.length > 9) {
      this.elTerminalLogs.removeChild(this.elTerminalLogs.children[1]);
    }
    this.elTerminalLogs.scrollTop = this.elTerminalLogs.scrollHeight;
  }

  _updateTelemetry(handData, fps) {
    // 1. FPS display
    if (this.elFps) this.elFps.innerText = `FPS: ${Math.round(fps)}`;

    // 2. Tension meter bars
    if (handData && handData.curls) {
      for (const finger in this.tensionBars) {
        const bar = this.tensionBars[finger];
        if (bar) {
          const curl = handData.curls[finger] ?? 0;
          // Tension is higher when pulling taut (1 - curl)
          const tension = Math.round((1.0 - curl) * 100);
          bar.style.width = `${tension}%`;
        }
      }
    }

    // 3. Dynamic terminal status
    if (this.elLogStatus) {
      if (this.tracker.isCameraActive) {
        this.elLogStatus.innerText = this.tracker.hasHand
          ? `LIVE HAND TRACKING // ${this.robot.action}`
          : "SEARCHING WEBCAM FOR HAND...";
      } else {
        this.elLogStatus.innerText = `AUTO-DEMO ACTIVE // ${this.robot.action}`;
      }
    }
  }

  _onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());

    const delta = Math.min(this.clock.getDelta(), 0.08); // Clamp large delta steps
    const time = this.clock.getElapsedTime();

    // 1. Update Hand Tracker (either MediaPipe live frames or Demo simulator)
    this.tracker.update(delta);
    const handData = this.tracker.handData;

    // 2. Update Robot Kinematics & Procedural Action Engine
    this.robot.update(delta, handData);

    // 3. Update 3D Holographic Tension Filaments
    this.strings.update(delta, time, this.robot.stringAnchors, handData.fingerPos3D, handData.curls);

    // 4. Update Particle VFX
    this.particles.update(delta);

    // 5. Smooth Camera Preset Lerp & Screen Shake (if user is not actively dragging OrbitControls)
    if (!this.controls.state || this.controls.state === -1) {
      this.camera.position.lerp(this.targetCameraPos, 0.045);
      this.controls.target.lerp(this.targetCameraLookAt, 0.045);
    }
    if (this.cameraShakeIntensity > 0.005) {
      this.camera.position.x += (Math.random() - 0.5) * this.cameraShakeIntensity * 0.16;
      this.camera.position.y += (Math.random() - 0.5) * this.cameraShakeIntensity * 0.16;
      this.cameraShakeIntensity *= 0.86;
    }
    this.controls.update();

    // 6. Render 3D Frame
    this.renderer.render(this.scene, this.camera);

    // 7. Telemetry & FPS Counter
    this.frameCount++;
    if (time - this.lastFpsUpdate > 0.4) {
      const fps = this.frameCount / (time - this.lastFpsUpdate);
      this._updateTelemetry(handData, fps);
      this.lastFpsUpdate = time;
      this.frameCount = 0;
    }
  }
}

// Instantiate application on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  new App();
});
