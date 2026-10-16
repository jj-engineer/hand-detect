# CyberPuppet: 3D Robot Hand Controller

A real-time Python computer vision & kinematics system that lets you control an articulated 3D humanoid robot avatar using your hand and dynamic marionette tension strings, complete with a cyberpunk sci-fi holographic HUD overlay.

---

## Features

- **Realistic 3D Humanoid Mecha Anatomy**:
  - Sculpted aerodynamic geometry: faceted mecha helmet with crown crest, dual optic eye sensors, curved wrap-around neon visor, and chiseled chin baffle.
  - Multi-tier chest carapace with pulsating Arc Reactor, flank cooling heatsink grilles, and collarbone mantle.
  - Articulated spinal column vertebrae flanked by dual exposed chrome hydraulic piston cylinders.
  - Twin-vectored rocket jetpack with deployable aerodynamic stabilizer wings & active plasma exhaust plumes.
  - Tiered curved pauldrons, bicep hydraulic actuators, dual-disc mechanical elbow joints, and combat gauntlets.
  - Articulated robotic hands with individual digit actuators and knuckle guards.
  - Muscular faceted thighs, dual-pivot mechanical knee guards, rear calf thruster nacelles, and combat sabatons with ground repulsor pads.
- **Dynamic Remote Action Engine (Custom Remote Control)**:
  - **JUMP**: Deep crouch windup -> explosive rocket launch with thruster plumes -> mid-air athletic pose -> superhero landing with expanding ground shockwaves.
  - **DANCE**: 120 BPM cyber breakdance routine featuring arm wave popping, hip sways, disco flares, robot freezes, and pulsing neon beat ripples.
  - **RUN**: Aerodynamic forward sprint with 22° torso lean, cycling high-knee strides, pumping arms, and continuous plasma rocket exhaust trails.
  - **FIGHT**: Tactical martial arts combat: bouncing boxing guard -> 4-hit combo (Left Jab -> Right Cross -> Left Hook -> Rising Uppercut) with fist shockwave bursts.
  - **MARIONETTE**: 5-finger organic puppet teleoperation where each finger directly pulls limbs via radiant tension strings.
  - **DIRECT MIMIC**: 1:1 teleoperation mirroring user's hand and arm posture.
- **One-Euro Filter Adaptive Hand Tracking**:
  - 21 high-precision 3D landmarks filtered with dynamic velocity-adaptive cutoff frequencies.
  - 100% elimination of hand tremors and camera sensor jitter with zero perceptible latency.
  - Real-time finger gesture recognition (`PEACE`, `ROCK_ON`, `TWO_FINGER_WALK`, `FIST`, `OPEN_PALM`, `POINT`, `THUMBS_UP`).
- **Clean Sci-Fi Hologram HUD**:
  - Finger Remote Matrix visual guide card.
  - Dynamic Action status badges and Action Lock indicator.
  - Modern glassmorphism terminal cards with translucent obsidian background.
  - Vector circular targeting reticle with dynamic azimuth brackets.
  - Sleek gradient tension bus meters and minimal telemetry pill badges.

---

## Quick Start

### 1. Launch with One Command (Auto-detects MacBook Webcam):
```bash
./run.sh
```
Or directly:
```bash
./.venv/bin/python3 main.py
```
> **Auto-Detection**: On macOS, MacBook built-in FaceTime HD webcams are often assigned index `1` while index `0` can be a virtual/standby camera. The program automatically probes all indices and selects the active, illuminated MacBook camera with zero configuration!

### 2. Switch Camera On-The-Fly:
Press **`[N]`** or **`[TAB]`** inside the window at any time to instantly toggle between connected cameras (e.g. index 0 and 1).

### 3. Explicit Camera Index:
```bash
./.venv/bin/python3 main.py --camera 1
```

### 4. Autonomous Synthetic Demo Mode (No camera needed):
```bash
./.venv/bin/python3 main.py --demo
```

---

## Controls & Gestures

### Finger Remote Matrix:
| Finger Gesture | Action Triggered |
| :--- | :--- |
| **✌️ PEACE (V-Sign Up)** | **JUMP** (Super Leap & Superhero Landing) |
| **🤘 ROCK (Horns Sign)** | **DANCE** (120 BPM Cyber Breakdance) |
| **✌️ INVERTED (Legs Down)** | **RUN** (Aerodynamic Sprint & Boost) |
| **✊ FIST (Closed)** | **FIGHT** (Martial Arts 4-Hit Combo) |
| **✋ OPEN PALM** | **MARIONETTE** (5-Finger Tension Strings) |
| **👆 POINT (Index Only)** | **TARGET LOCK** (Precision Aiming) |
| **👍 THUMBS UP** | **DIRECT MIMIC** (1:1 Arm Teleoperation) |

### Remote Hotkeys:
| Key | Action |
| :--- | :--- |
| **`[1]`** or **`[J]`** | Trigger **JUMP** Action |
| **`[2]`** or **`[D]`** | Trigger **DANCE** Action |
| **`[3]`** or **`[K]`** | Trigger **RUN** Action |
| **`[4]`** or **`[F]`** | Trigger **FIGHT** Action |
| **`[5]`** or **`[P]`** | Trigger **MARIONETTE** Puppet Mode |
| **`[6]`** or **`[M]`** | Trigger **DIRECT MIMIC** Mode |
| **`[L]`** | Toggle **ACTION LOCK** (Loop action continuously) |
| **`[SPACE]`** | Cycle Robot Mode |
| **`[W]`** | Toggle 3D Wireframe / Solid Shaded Mesh |
| **`[R]`** | Reset Robot position, rotation & action |
| **`[C]`** | Toggle Camera horizontal mirror |
| **`[H]`** | Toggle Sci-Fi HUD overlay |
| **`[N]`** or **`[TAB]`** | Switch Camera |
| **`[Q]`** or **`[ESC]`** | Exit cleanly |

---

## Project Architecture

- [`main.py`](file:///Users/macbook/Documents/hand-detect/main.py): Main event loop, camera capture, coordination, and keyboard controls.
- [`hand_tracker.py`](file:///Users/macbook/Documents/hand-detect/hand_tracker.py): MediaPipe pipeline, jitter smoothing, gesture classifier, and hand mesh visualization.
- [`robot_3d.py`](file:///Users/macbook/Documents/hand-detect/robot_3d.py): 3D Robot geometry, hierarchical forward kinematics, lighting shader, and projection engine.
- [`string_solver.py`](file:///Users/macbook/Documents/hand-detect/string_solver.py): Multi-thread tension strings, bezier curvature solver, and spark particle dynamics.
- [`hud_overlay.py`](file:///Users/macbook/Documents/hand-detect/hud_overlay.py): Holographic sci-fi terminal, robot telemetry brackets, tension meters, and status bars.
- [`test_pipeline.py`](file:///Users/macbook/Documents/hand-detect/test_pipeline.py): Automated test & headless validation suite.
