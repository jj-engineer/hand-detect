/**
 * CyberPuppet Hand Tracker & Neural Gesture Classifier
 * Integrates MediaPipe Hands with a 1-Euro adaptive jitter filter,
 * kinematic curl calculation, 3D hand orientation, gesture recognition,
 * and an autonomous organic demo simulator.
 */

import * as THREE from 'three';

// 1-Euro Filter for jitter suppression without lag
class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  alpha(rate, cutoff) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    const te = 1.0 / rate;
    return 1.0 / (1.0 + tau / te);
  }

  filter(x, timestamp = performance.now() / 1000) {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.tPrev = timestamp;
      return x;
    }

    const dt = Math.max(timestamp - this.tPrev, 1e-4);
    this.tPrev = timestamp;
    const rate = 1.0 / dt;

    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(rate, this.dCutoff);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    this.dxPrev = dxHat;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(rate, cutoff);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat;

    return xHat;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
}

export class HandTracker {
  constructor({ videoEl, canvasEl, onGestureChange, onStatusLog }) {
    this.video = videoEl;
    this.canvas = canvasEl;
    this.ctx = canvasEl ? canvasEl.getContext('2d') : null;
    this.onGestureChange = onGestureChange;
    this.onStatusLog = onStatusLog || console.log;

    this.isCameraActive = false;
    this.isDemoActive = true;
    this.hasHand = false;

    // Filters for 21 3D landmarks
    this.filters = Array.from({ length: 21 }, () => ({
      x: new OneEuroFilter(1.2, 0.008),
      y: new OneEuroFilter(1.2, 0.008),
      z: new OneEuroFilter(1.2, 0.008)
    }));

    // Output state
    this.handData = {
      landmarks: null,
      palmCenter: new THREE.Vector2(0.5, 0.5),
      scale: 0.25,
      pitch: 0,
      yaw: 0,
      roll: 0,
      curls: { thumb: 0.2, index: 0.1, middle: 0.1, ring: 0.1, pinky: 0.1 },
      fingerPos3D: {},
      action: "MARIONETTE"
    };

    // Autonomous Demo Mode sequencer
    this.demoTime = 0.0;
    this.demoPhase = 0;
    this.demoActions = ["FIGHT", "SLASH", "JUMP", "DANCE", "RUN", "MARIONETTE", "MIMIC"];
    this.demoActionDuration = 5.0; // seconds per action

    this._initMediaPipe();
  }

  _initMediaPipe() {
    if (typeof window.Hands === 'undefined') {
      this.onStatusLog("[WARN] MediaPipe Hands script loading or unavailable, running in Demo Mode");
      return;
    }

    try {
      this.mpHands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      this.mpHands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.65,
        minTrackingConfidence: 0.65
      });

      this.mpHands.onResults((results) => this._onMediaPipeResults(results));
      this.onStatusLog("[OK] MediaPipe Neural Engine Loaded");
    } catch (err) {
      console.warn("MediaPipe init error:", err);
      this.onStatusLog("[ERR] MediaPipe init fallback to Demo Mode");
    }
  }

  async startCamera() {
    if (this.isCameraActive) return;

    try {
      this.onStatusLog("[INFO] Requesting Webcam Feed...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
      });
      this.video.srcObject = stream;
      await this.video.play();

      this.isCameraActive = true;
      this.isDemoActive = false;
      this.onStatusLog("[OK] Webcam Stream Connected");

      if (window.Camera && this.mpHands) {
        this.camera = new window.Camera(this.video, {
          onFrame: async () => {
            if (this.isCameraActive) {
              await this.mpHands.send({ image: this.video });
            }
          },
          width: 640,
          height: 480
        });
        this.camera.start();
      }
    } catch (err) {
      console.error("Camera access error:", err);
      this.onStatusLog("[WARN] Camera access denied or unavailable. Fallback to Demo Mode");
      this.isDemoActive = true;
    }
  }

  stopCamera() {
    if (this.video && this.video.srcObject) {
      this.video.srcObject.getTracks().forEach(track => track.stop());
      this.video.srcObject = null;
    }
    this.isCameraActive = false;
    this.isDemoActive = true;
    this.onStatusLog("[INFO] Camera Paused, Resumed Demo Simulator");
  }

  toggleDemo(forceState = null) {
    this.isDemoActive = forceState !== null ? forceState : !this.isDemoActive;
    if (this.isDemoActive && this.isCameraActive) {
      this.stopCamera();
    }
    return this.isDemoActive;
  }

  _onMediaPipeResults(results) {
    if (!this.isCameraActive) return;

    if (this.canvas) {
      this.canvas.width = this.canvas.clientWidth || 320;
      this.canvas.height = this.canvas.clientHeight || 240;
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      this.hasHand = true;
      const rawLm = results.multiHandLandmarks[0];
      const now = performance.now() / 1000;

      // Filtered landmarks
      const lm = rawLm.map((pt, i) => ({
        x: this.filters[i].x.filter(pt.x, now),
        y: this.filters[i].y.filter(pt.y, now),
        z: this.filters[i].z.filter(pt.z, now)
      }));

      this._computeHandKinematics(lm);
      this._drawSkeleton(lm);
    } else {
      this.hasHand = false;
    }
  }

  _computeHandKinematics(lm) {
    // 1. Palm Center
    const palmX = (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5;
    const palmY = (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5;
    this.handData.palmCenter.set(1.0 - palmX, palmY); // mirror horizontal for natural feel

    // 2. Hand Scale
    const dx = lm[9].x - lm[0].x;
    const dy = lm[9].y - lm[0].y;
    const dz = lm[9].z - lm[0].z;
    this.handData.scale = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // 3. Orientation (pitch, yaw, roll)
    const wrist = lm[0];
    const middleMcp = lm[9];
    const indexMcp = lm[5];
    const pinkyMcp = lm[17];

    const upVector = { x: middleMcp.x - wrist.x, y: middleMcp.y - wrist.y, z: middleMcp.z - wrist.z };
    const rightVector = { x: pinkyMcp.x - indexMcp.x, y: pinkyMcp.y - indexMcp.y, z: pinkyMcp.z - indexMcp.z };

    this.handData.pitch = -Math.atan2(upVector.z, Math.hypot(upVector.x, upVector.y));
    this.handData.yaw = Math.atan2(upVector.x, -upVector.y);
    this.handData.roll = Math.atan2(rightVector.y, rightVector.x);

    // 4. Calculate finger curls [0.0 = extended, 1.0 = fully curled into palm]
    const computeCurl = (tipIdx, mcpIdx) => {
      const tip = lm[tipIdx];
      const mcp = lm[mcpIdx];
      const dist = Math.hypot(tip.x - mcp.x, tip.y - mcp.y);
      // Normalized distance vs palm scale
      const norm = dist / (this.handData.scale * 1.5);
      return Math.max(0.0, Math.min(1.0, 1.0 - (norm - 0.2) / 0.8));
    };

    const curls = {
      thumb: computeCurl(4, 2),
      index: computeCurl(8, 5),
      middle: computeCurl(12, 9),
      ring: computeCurl(16, 13),
      pinky: computeCurl(20, 17)
    };
    this.handData.curls = curls;

    // 5. 3D Fingertip points for 3D tension strings
    const toWorld3D = (pt) => {
      return new THREE.Vector3(
        (0.5 - pt.x) * 3.2,
        -(pt.y - 0.5) * 2.4 + 2.2,
        -(pt.z) * 2.0 + 0.3
      );
    };
    this.handData.fingerPos3D = {
      thumb: toWorld3D(lm[4]),
      index: toWorld3D(lm[8]),
      middle: toWorld3D(lm[12]),
      ring: toWorld3D(lm[16]),
      pinky: toWorld3D(lm[20])
    };

    // 6. Neural Gesture Classifier
    const action = this._classifyGesture(curls, lm);
    if (action !== this.handData.action) {
      this.handData.action = action;
      if (this.onGestureChange) this.onGestureChange(action);
    }
  }

  _classifyGesture(curls, lm) {
    const ext = {
      thumb: curls.thumb < 0.45,
      index: curls.index < 0.40,
      middle: curls.middle < 0.40,
      ring: curls.ring < 0.45,
      pinky: curls.pinky < 0.45
    };

    // Fist: all fingers curled
    if (!ext.index && !ext.middle && !ext.ring && !ext.pinky) {
      return "FIGHT";
    }

    // Horns / Rock: Index & Pinky extended, middle & ring curled
    if (ext.index && ext.pinky && !ext.middle && !ext.ring) {
      return "DANCE";
    }

    // Peace: Index & Middle extended, ring & pinky curled
    if (ext.index && ext.middle && !ext.ring && !ext.pinky) {
      // Check if pointing downward for Two-Finger Walk / Run
      const idxTip = lm[8];
      const idxMcp = lm[5];
      if (idxTip.y > idxMcp.y + 0.05) {
        return "RUN";
      }
      return "JUMP";
    }

    // Single finger pointing: Blade Slash Frenzy
    if (ext.index && !ext.middle && !ext.ring && !ext.pinky) {
      return "SLASH";
    }

    // Open Palm: all fingers extended
    if (ext.index && ext.middle && ext.ring && ext.pinky) {
      return "MARIONETTE";
    }

    return this.handData.action || "MARIONETTE";
  }

  _drawSkeleton(lm) {
    if (!this.ctx || !this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    // Bone connections
    const bones = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // Index
      [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [0, 13], [13, 14], [14, 15], [15, 16],// Ring
      [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
      [5, 9], [9, 13], [13, 17]             // Palm arch
    ];

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "rgba(0, 240, 255, 0.75)";
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 6;

    bones.forEach(([i, j]) => {
      ctx.beginPath();
      // Mirror X for display
      ctx.moveTo((1 - lm[i].x) * w, lm[i].y * h);
      ctx.lineTo((1 - lm[j].x) * w, lm[j].y * h);
      ctx.stroke();
    });

    // Joint knuckles
    lm.forEach((pt, i) => {
      const isTip = [4, 8, 12, 16, 20].includes(i);
      ctx.beginPath();
      ctx.arc((1 - pt.x) * w, pt.y * h, isTip ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = isTip ? "#ff007f" : "#ffffff";
      ctx.fill();
    });

    ctx.shadowBlur = 0;
  }

  update(delta) {
    if (this.isDemoActive || !this.isCameraActive || !this.hasHand) {
      this._updateDemoMode(delta);
    }
  }

  _updateDemoMode(delta) {
    this.demoTime += delta;
    const t = this.demoTime;

    // Cycle through actions
    const currentActionIndex = Math.floor(t / this.demoActionDuration) % this.demoActions.length;
    const currentAction = this.demoActions[currentActionIndex];

    if (currentAction !== this.handData.action) {
      this.handData.action = currentAction;
      if (this.onGestureChange) this.onGestureChange(currentAction);
    }

    // Procedural virtual hand motion
    const swayX = Math.sin(t * 1.4) * 0.15;
    const swayY = Math.cos(t * 1.8) * 0.12;
    this.handData.palmCenter.set(0.5 + swayX, 0.5 + swayY);
    this.handData.scale = 0.24 + Math.sin(t * 0.8) * 0.04;
    this.handData.pitch = Math.sin(t * 1.2) * 0.2;
    this.handData.yaw = Math.sin(t * 0.9) * 0.25;
    this.handData.roll = Math.cos(t * 1.5) * 0.15;

    // Organic finger curls based on current demo action
    switch (currentAction) {
      case "JUMP": // Peace sign
        this.handData.curls = {
          thumb: 0.85,
          index: 0.05,
          middle: 0.05,
          ring: 0.9,
          pinky: 0.9
        };
        break;
      case "DANCE": // Rock / Horns sign with rhythmic bounce
        this.handData.curls = {
          thumb: 0.75,
          index: 0.05,
          middle: 0.95,
          ring: 0.95,
          pinky: 0.05
        };
        break;
      case "RUN": // Two-finger stride
        const stride = Math.sin(t * 7.5);
        this.handData.curls = {
          thumb: 0.8,
          index: 0.2 + Math.max(0, stride) * 0.6,
          middle: 0.2 + Math.max(0, -stride) * 0.6,
          ring: 0.9,
          pinky: 0.9
        };
        break;
      case "FIGHT": // Clenched fist with strike pulses
        const punch = (Math.sin(t * 6.0) + 1.0) * 0.15;
        this.handData.curls = {
          thumb: 0.95,
          index: 0.95 - punch,
          middle: 0.95,
          ring: 0.95,
          pinky: 0.95
        };
        break;
      case "SLASH": // Index blade pointing with blade slash frenzy
        this.handData.curls = {
          thumb: 0.85,
          index: 0.05,
          middle: 0.92,
          ring: 0.92,
          pinky: 0.92
        };
        break;
      case "MIMIC":
        this.handData.curls = {
          thumb: 0.2 + Math.sin(t * 2.0) * 0.2,
          index: 0.2 + Math.cos(t * 2.5) * 0.25,
          middle: 0.3 + Math.sin(t * 3.0) * 0.2,
          ring: 0.2 + Math.cos(t * 1.8) * 0.25,
          pinky: 0.3 + Math.sin(t * 2.2) * 0.2
        };
        break;
      case "MARIONETTE":
      default: // Open palm with dynamic marionette pull
        this.handData.curls = {
          thumb: 0.25 + Math.sin(t * 3.1) * 0.25,
          index: 0.2 + Math.sin(t * 2.4) * 0.25,
          middle: 0.3 + Math.cos(t * 2.8) * 0.3,
          ring: 0.25 + Math.sin(t * 3.5) * 0.25,
          pinky: 0.3 + Math.cos(t * 2.1) * 0.3
        };
        break;
    }

    // Generate virtual 3D fingertip positions for strings
    const baseY = 2.6 + Math.sin(t * 1.2) * 0.15;
    const baseX = swayX * 2.0;
    this.handData.fingerPos3D = {
      thumb: new THREE.Vector3(baseX - 0.85, baseY - this.handData.curls.thumb * 0.3, 0.4),
      index: new THREE.Vector3(baseX - 0.40, baseY + 0.15 - this.handData.curls.index * 0.3, 0.5),
      middle: new THREE.Vector3(baseX + 0.0, baseY + 0.25 - this.handData.curls.middle * 0.3, 0.45),
      ring: new THREE.Vector3(baseX + 0.40, baseY + 0.15 - this.handData.curls.ring * 0.3, 0.5),
      pinky: new THREE.Vector3(baseX + 0.85, baseY - this.handData.curls.pinky * 0.3, 0.4)
    };

    // Render demo skeleton onto canvas
    if (this.ctx && this.canvas) {
      this._drawDemoCanvas(t, currentAction);
    }
  }

  _drawDemoCanvas(t, action) {
    const w = this.canvas.width = this.canvas.clientWidth || 320;
    const h = this.canvas.height = this.canvas.clientHeight || 240;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    // Subtle sci-fi scanner grid in PIP
    ctx.strokeStyle = "rgba(0, 240, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw animated holographic demo hand
    const cx = w * 0.5 + Math.sin(t * 1.4) * 25;
    const cy = h * 0.55 + Math.cos(t * 1.8) * 15;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.sin(t * 1.2) * 0.15);

    // Palm circle
    ctx.beginPath();
    ctx.arc(0, 0, 24, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0, 240, 255, 0.6)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // 5 Finger Rays
    const fingerAngles = [-0.65, -0.32, 0.0, 0.32, 0.65];
    const fingerKeys = ["thumb", "index", "middle", "ring", "pinky"];

    fingerAngles.forEach((ang, idx) => {
      const curl = this.handData.curls[fingerKeys[idx]];
      const len = 50 * (1.0 - curl * 0.65);

      const ex = Math.sin(ang) * len;
      const ey = -Math.cos(ang) * len;

      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = "rgba(0, 240, 255, 0.8)";
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Tip circle
      ctx.beginPath();
      ctx.arc(ex, ey, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = curl < 0.4 ? "#00ffd5" : "#ff007f";
      ctx.fill();
    });

    ctx.restore();
  }
}
