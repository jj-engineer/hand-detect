/**
 * CyberPuppet 3D Mecha Robot (Three.js Implementation)
 * Ultra-realistic, dangerous combat humanoid mecha in stark Arctic Ceramic White,
 * Blood Crimson Red, and Deep Obsidian Carbon Slate.
 * Features predatory optic slits, twin swept-back demon horn blades,
 * dual forearm vibro-blades, heavy spiked pauldrons, dorsal rocket jetpack,
 * and high-impact cinematic combat kinematics.
 */

import * as THREE from 'three';

export class Robot3D {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = "MechaRobot";

    // World tracking anchors for tension strings & VFX
    this.stringAnchors = {
      head: new THREE.Vector3(),
      l_hand: new THREE.Vector3(),
      r_hand: new THREE.Vector3(),
      chest: new THREE.Vector3(),
      l_foot: new THREE.Vector3(),
      r_foot: new THREE.Vector3(),
      l_blade: new THREE.Vector3(),
      r_blade: new THREE.Vector3()
    };

    // Color themes - APEX (Red, Dark, White) is the primary default theme
    this.themes = {
      apex: {
        armorMain: 0xf1f5f9,     // Stark Arctic Ceramic White
        armorAccent: 0xdc2626,   // Blood Crimson Red
        innerFrame: 0x070a12,    // Deep Obsidian Carbon Slate
        chrome: 0x334155,        // Dark Titanium Gunmetal
        chromeBright: 0xe2e8f0,  // Mirror Blade Chrome
        gold: 0xb91c1c,          // Anodized Crimson
        visor: 0xff003c,         // Menacing Predatory Laser Crimson
        core: 0xff1744,          // Crimson Singularity Core
        blade: 0xff003c,         // Glowing Laser Blade Edge
        thruster: 0xff3300       // Fiery Orange/Red Rocket Plasma
      },
      stealth: {
        armorMain: 0x0b0f19,     // Matte Stealth Slate
        armorAccent: 0xef4444,   // Crimson Red
        innerFrame: 0x030712,    // Deep Carbon Black
        chrome: 0x475569,        // Dark Nickel
        chromeBright: 0x94a3b8,  // Burnished Steel
        gold: 0xd97706,          // Amber Gold
        visor: 0xff0055,         // Laser Crimson
        core: 0xff2255,          // Red Arc Core
        blade: 0xff0055,
        thruster: 0xff4500
      },
      pearl: {
        armorMain: 0xe2e8f0,     // Platinum Pearl
        armorAccent: 0x8b5cf6,   // Cyber Violet / Iris
        innerFrame: 0x0f172a,    // Dark Titanium Carbon
        chrome: 0x64748b,        // Polished Aluminum
        chromeBright: 0xf8fafc,  // Mirror Chrome
        gold: 0xf59e0b,          // Anodized Gold
        visor: 0x00f0ff,         // Hologram Cyan
        core: 0x00ffd5,          // Arc Cyan-Teal
        blade: 0x00f0ff,
        thruster: 0x00b4d8
      },
      solar: {
        armorMain: 0xf59e0b,     // Solar Gold Composite
        armorAccent: 0x18181b,   // Carbon Onyx
        innerFrame: 0x09090b,    // Dark Chassis
        chrome: 0xd97706,        // Bronze Chrome
        chromeBright: 0xfef3c7,  // Bright Gold
        gold: 0xfbbf24,          // Gold Accents
        visor: 0xffea00,         // Electric Yellow
        core: 0xffaa00,          // Sunfire Core
        blade: 0xffea00,
        thruster: 0xff7700
      }
    };
    this.currentTheme = "apex";

    // PBR Materials
    this._initMaterials();

    // Hierarchical Bones & Joints
    this.joints = {};
    this._buildMechaAnatomy();

    // Kinematic target angles (radians)
    this.angles = {
      head_yaw: 0,
      head_pitch: 0.08, // Sinister slight downward tilt
      head_roll: 0,
      torso_pitch: 0.05,
      torso_yaw: 0,
      torso_roll: 0,
      l_shoulder_pitch: -0.35,
      l_shoulder_roll: 0.40,
      l_elbow_pitch: 0.85,
      r_shoulder_pitch: -0.25,
      r_shoulder_roll: -0.40,
      r_elbow_pitch: 0.75,
      l_hip_pitch: 0.20,
      l_hip_roll: 0.12,
      l_knee_pitch: -0.40,
      r_hip_pitch: -0.25,
      r_hip_roll: -0.12,
      r_knee_pitch: -0.50,
    };
    this.targetAngles = { ...this.angles };

    // Kinematic translations & root transforms
    this.basePos = new THREE.Vector3(0, 0, 0);
    this.targetPos = new THREE.Vector3(0, 0, 0);
    this.baseRot = new THREE.Euler(0, 0, 0);
    this.targetRot = new THREE.Euler(0, 0, 0);

    // Action state
    this.action = "MARIONETTE"; // JUMP, DANCE, RUN, FIGHT, SLASH, MARIONETTE, MIMIC, POINT
    this.actionTime = 0.0;
    this.actionLocked = false;

    // Callbacks for particles & screen shake
    this.onShockwave = null;
    this.onThrusterSpark = null;
    this.onSlashArc = null;
    this.onCameraShake = null;
  }

  _initMaterials() {
    const t = this.themes[this.currentTheme];

    // 1. Primary Exterior Armor: Stark Arctic Ceramic White with automotive gloss clearcoat
    this.matArmorMain = new THREE.MeshStandardMaterial({
      color: t.armorMain,
      metalness: 0.18,
      roughness: 0.24,
      clearcoat: 0.90,
      clearcoatRoughness: 0.10
    });

    // 2. High-Grade Danger Blood Crimson Alloy
    this.matArmorAccent = new THREE.MeshStandardMaterial({
      color: t.armorAccent,
      emissive: 0x330005,
      emissiveIntensity: 0.4,
      metalness: 0.45,
      roughness: 0.22,
      clearcoat: 0.85,
      clearcoatRoughness: 0.12
    });

    // 3. Deep Obsidian Carbon Slate Inner Frame
    this.matInnerFrame = new THREE.MeshStandardMaterial({
      color: t.innerFrame,
      metalness: 0.85,
      roughness: 0.35
    });

    // 4. Dark Titanium Gunmetal Bearings & Shock Rods
    this.matChrome = new THREE.MeshStandardMaterial({
      color: t.chrome,
      metalness: 0.95,
      roughness: 0.16
    });

    this.matChromeBright = new THREE.MeshStandardMaterial({
      color: t.chromeBright,
      metalness: 0.98,
      roughness: 0.08
    });

    // 5. Anodized Crimson / Gold Fasteners
    this.matGold = new THREE.MeshStandardMaterial({
      color: t.gold,
      metalness: 0.85,
      roughness: 0.22
    });

    // 6. Predatory Optic Visor Slits
    this.matVisor = new THREE.MeshStandardMaterial({
      color: t.visor,
      emissive: t.visor,
      emissiveIntensity: 3.2,
      roughness: 0.05,
      metalness: 0.5
    });

    // 7. Crimson Singularity Power Reactor
    this.matCore = new THREE.MeshStandardMaterial({
      color: t.core,
      emissive: t.core,
      emissiveIntensity: 3.6,
      roughness: 0.1
    });

    // 8. Laser Edge Vibro-Blades
    this.matBlade = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: t.blade,
      emissiveIntensity: 2.4,
      metalness: 0.95,
      roughness: 0.08
    });

    // 9. Burnished Rocket Nozzles
    this.matNozzle = new THREE.MeshStandardMaterial({
      color: 0x11131c,
      metalness: 0.95,
      roughness: 0.32
    });
  }

  setTheme(themeName) {
    if (!this.themes[themeName]) return;
    this.currentTheme = themeName;
    const t = this.themes[themeName];

    this.matArmorMain.color.setHex(t.armorMain);
    this.matArmorAccent.color.setHex(t.armorAccent);
    this.matInnerFrame.color.setHex(t.innerFrame);
    this.matChrome.color.setHex(t.chrome);
    this.matChromeBright.color.setHex(t.chromeBright);
    this.matGold.color.setHex(t.gold);
    this.matVisor.color.setHex(t.visor);
    this.matVisor.emissive.setHex(t.visor);
    this.matCore.color.setHex(t.core);
    this.matCore.emissive.setHex(t.core);
    this.matBlade.emissive.setHex(t.blade);

    if (this.coreLight) this.coreLight.color.setHex(t.core);
    if (this.visorLight) this.visorLight.color.setHex(t.visor);
  }

  setWireframe(enabled) {
    [
      this.matArmorMain, this.matArmorAccent, this.matInnerFrame,
      this.matChrome, this.matChromeBright, this.matGold,
      this.matVisor, this.matCore, this.matBlade, this.matNozzle
    ].forEach(m => { m.wireframe = enabled; });
  }

  _buildMechaAnatomy() {
    // ----------------------------------------------------
    // 1. Pelvis Root Assembly (Heavy combat chassis)
    // ----------------------------------------------------
    const pelvis = new THREE.Group();
    pelvis.position.y = 1.48;
    this.group.add(pelvis);
    this.joints.pelvis = pelvis;

    // Pelvis Core Chassis (Dark obsidian octagonal hub)
    const pelvChassis = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.14, 0.16, 8),
      this.matInnerFrame
    );
    pelvis.add(pelvChassis);

    // Armored V-Codpiece Shield with Crimson Center Inlay
    const codpiece = new THREE.Mesh(
      new THREE.ConeGeometry(0.13, 0.20, 5),
      this.matArmorAccent
    );
    codpiece.rotation.x = Math.PI * 0.88;
    codpiece.position.set(0, -0.04, 0.13);
    pelvis.add(codpiece);

    // Hip Heavy Ball Bearings
    const hipSocketGeom = new THREE.SphereGeometry(0.09, 12, 12);
    const hipSocketL = new THREE.Mesh(hipSocketGeom, this.matChrome);
    hipSocketL.position.set(0.18, -0.06, 0);
    pelvis.add(hipSocketL);
    const hipSocketR = new THREE.Mesh(hipSocketGeom, this.matChrome);
    hipSocketR.position.set(-0.18, -0.06, 0);
    pelvis.add(hipSocketR);

    // ----------------------------------------------------
    // 2. Articulated Lumbar Spine & Shock Dampers
    // ----------------------------------------------------
    const spine = new THREE.Group();
    spine.position.y = 0.08;
    pelvis.add(spine);
    this.joints.spine = spine;

    // Vertebrae Discs flanked by glowing red conduits
    for (let i = 0; i < 3; i++) {
      const vGeom = new THREE.CylinderGeometry(0.095 - i * 0.008, 0.105 - i * 0.008, 0.05, 8);
      const vMesh = new THREE.Mesh(vGeom, i % 2 === 0 ? this.matInnerFrame : this.matArmorAccent);
      vMesh.position.y = 0.06 + i * 0.065;
      spine.add(vMesh);
    }

    // Dual Hydraulic Lumbar Pushrods
    const pRodGeom = new THREE.CylinderGeometry(0.015, 0.015, 0.22, 8);
    const pLeft = new THREE.Mesh(pRodGeom, this.matChromeBright);
    pLeft.position.set(0.10, 0.14, -0.02);
    spine.add(pLeft);
    const pRight = new THREE.Mesh(pRodGeom, this.matChromeBright);
    pRight.position.set(-0.10, 0.14, -0.02);
    spine.add(pRight);

    // ----------------------------------------------------
    // 3. V-Tapered Torso & Crimson Singularity Reactor
    // ----------------------------------------------------
    const chest = new THREE.Group();
    chest.position.y = 0.28;
    spine.add(chest);
    this.joints.chest = chest;

    // Dark Titanium Inner Framework
    const innerTorso = new THREE.Mesh(
      new THREE.CylinderGeometry(0.25, 0.17, 0.38, 8),
      this.matInnerFrame
    );
    innerTorso.position.y = 0.14;
    innerTorso.castShadow = true;
    chest.add(innerTorso);

    // Stark Ceramic White Pectoral Armor Plates (Left & Right)
    const pecGeom = new THREE.BoxGeometry(0.21, 0.22, 0.11);
    const pecL = new THREE.Mesh(pecGeom, this.matArmorMain);
    pecL.position.set(0.12, 0.20, 0.12);
    pecL.rotation.set(0.12, 0.20, -0.08);
    pecL.castShadow = true;
    chest.add(pecL);

    const pecR = new THREE.Mesh(pecGeom, this.matArmorMain);
    pecR.position.set(-0.12, 0.20, 0.12);
    pecR.rotation.set(0.12, -0.20, 0.08);
    pecR.castShadow = true;
    chest.add(pecR);

    // Red Chevron Hazard Stripes across the pectoral plates
    const chevGeom = new THREE.BoxGeometry(0.09, 0.04, 0.12);
    const chevL = new THREE.Mesh(chevGeom, this.matArmorAccent);
    chevL.position.set(0.12, 0.25, 0.13);
    chevL.rotation.set(0.12, 0.20, -0.08);
    chest.add(chevL);
    const chevR = new THREE.Mesh(chevGeom, this.matArmorAccent);
    chevR.position.set(-0.12, 0.25, 0.13);
    chevR.rotation.set(0.12, -0.20, 0.08);
    chest.add(chevR);

    // Lower Rib Abdominal Radiator Cowlings (Glowing red heat vents)
    const ribGeom = new THREE.BoxGeometry(0.16, 0.10, 0.09);
    const ribL = new THREE.Mesh(ribGeom, this.matArmorAccent);
    ribL.position.set(0.09, 0.06, 0.11);
    ribL.rotation.set(0.05, 0.15, -0.05);
    chest.add(ribL);
    const ribR = new THREE.Mesh(ribGeom, this.matArmorAccent);
    ribR.position.set(-0.09, 0.06, 0.11);
    ribR.rotation.set(0.05, -0.15, 0.05);
    chest.add(ribR);

    // White Ceramic Collarbone Mantle
    const collarGeom = new THREE.BoxGeometry(0.50, 0.07, 0.22);
    const collarMesh = new THREE.Mesh(collarGeom, this.matArmorMain);
    collarMesh.position.set(0, 0.32, 0.03);
    collarMesh.castShadow = true;
    chest.add(collarMesh);

    // Crimson Singularity Core Unit (Recessed with radiating intake ring)
    const coreBezel = new THREE.Mesh(
      new THREE.CylinderGeometry(0.09, 0.09, 0.04, 16),
      this.matArmorAccent
    );
    coreBezel.rotation.x = Math.PI / 2;
    coreBezel.position.set(0, 0.18, 0.16);
    chest.add(coreBezel);

    const coreMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.065, 0.065, 0.05, 16),
      this.matCore
    );
    coreMesh.rotation.x = Math.PI / 2;
    coreMesh.position.set(0, 0.18, 0.17);
    chest.add(coreMesh);

    // Core Dynamic Light (Intense sinister red glow)
    this.coreLight = new THREE.PointLight(0xff1744, 2.4, 3.2);
    this.coreLight.position.set(0, 0.18, 0.24);
    chest.add(this.coreLight);

    // Flank Heatsink Exhaust Grilles
    const ventGeom = new THREE.BoxGeometry(0.05, 0.18, 0.10);
    const ventL = new THREE.Mesh(ventGeom, this.matInnerFrame);
    ventL.position.set(0.24, 0.15, 0.04);
    chest.add(ventL);
    const ventR = new THREE.Mesh(ventGeom, this.matInnerFrame);
    ventR.position.set(-0.24, 0.15, 0.04);
    chest.add(ventR);

    // ----------------------------------------------------
    // 4. Heavy Dorsal Jetpack & Rocket Booster Pods
    // ----------------------------------------------------
    const pack = new THREE.Group();
    pack.position.set(0, 0.18, -0.15);
    chest.add(pack);

    // Main Dark Carbon Avionics Pack
    const packBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.32, 0.34, 0.13),
      this.matInnerFrame
    );
    packBody.castShadow = true;
    pack.add(packBody);

    // Ceramic White Armored Cowling
    const packCover = new THREE.Mesh(
      new THREE.BoxGeometry(0.28, 0.30, 0.04),
      this.matArmorMain
    );
    packCover.position.z = -0.075;
    pack.add(packCover);

    // Twin Heavy Rocket Exhaust Nozzles
    const nozGeom = new THREE.ConeGeometry(0.07, 0.15, 12, 1, true);
    nozGeom.rotateX(Math.PI);
    const nozL = new THREE.Mesh(nozGeom, this.matNozzle);
    nozL.position.set(0.10, -0.19, -0.02);
    pack.add(nozL);
    const nozR = new THREE.Mesh(nozGeom, this.matNozzle);
    nozR.position.set(-0.10, -0.19, -0.02);
    pack.add(nozR);

    // Swept-Back Crimson Razor Stabilizer Wings
    const wingGeom = new THREE.BoxGeometry(0.22, 0.015, 0.18);
    const wingL = new THREE.Mesh(wingGeom, this.matArmorAccent);
    wingL.position.set(0.22, 0.10, -0.02);
    wingL.rotation.z = -0.35;
    wingL.rotation.y = 0.18;
    pack.add(wingL);
    const wingR = new THREE.Mesh(wingGeom, this.matArmorAccent);
    wingR.position.set(-0.22, 0.10, -0.02);
    wingR.rotation.z = 0.35;
    wingR.rotation.y = -0.18;
    pack.add(wingR);

    // ----------------------------------------------------
    // 5. Head: Menacing Predatory Demon War Helmet
    // ----------------------------------------------------
    const neck = new THREE.Group();
    neck.position.y = 0.38;
    chest.add(neck);

    const neckCyl = new THREE.Mesh(
      new THREE.CylinderGeometry(0.065, 0.075, 0.10, 10),
      this.matChrome
    );
    neck.add(neckCyl);

    const head = new THREE.Group();
    head.position.y = 0.12;
    neck.add(head);
    this.joints.head = head;

    // Stark Ceramic White Cranial Helmet Dome
    const dome = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.12, 0.13, 8, 16),
      this.matArmorMain
    );
    dome.position.set(0, 0.02, -0.01);
    dome.castShadow = true;
    head.add(dome);

    // Sharp Brow Ridge (Casts menacing shadow over eyes)
    const browGeom = new THREE.BoxGeometry(0.18, 0.04, 0.10);
    const brow = new THREE.Mesh(browGeom, this.matArmorMain);
    brow.position.set(0, 0.08, 0.08);
    brow.rotation.x = 0.22;
    head.add(brow);

    // DUAL PREDATORY SLANTED OPTIC SLITS (Sinister scowling red eyes)
    const eyeGeom = new THREE.BoxGeometry(0.05, 0.018, 0.04);
    
    // Left eye (tilted inward)
    const eyeL = new THREE.Mesh(eyeGeom, this.matVisor);
    eyeL.position.set(0.042, 0.045, 0.115);
    eyeL.rotation.set(0.08, 0.15, -0.22);
    head.add(eyeL);

    // Right eye (tilted inward)
    const eyeR = new THREE.Mesh(eyeGeom, this.matVisor);
    eyeR.position.set(-0.042, 0.045, 0.115);
    eyeR.rotation.set(0.08, -0.15, 0.22);
    head.add(eyeR);

    // Visor Headlight (Intense focused crimson beam)
    this.visorLight = new THREE.PointLight(0xff003c, 1.8, 2.2);
    this.visorLight.position.set(0, 0.05, 0.18);
    head.add(this.visorLight);

    // Razor Chin Guard & Rebreather Spike (Dark obsidian)
    const chinGeom = new THREE.ConeGeometry(0.05, 0.10, 4);
    const chinMesh = new THREE.Mesh(chinGeom, this.matInnerFrame);
    chinMesh.position.set(0, -0.08, 0.07);
    chinMesh.rotation.x = Math.PI * 0.9;
    head.add(chinMesh);

    // Swept-Back Angled Demon Horn Blades (Crimson red combat horns)
    const hornGeom = new THREE.ConeGeometry(0.03, 0.24, 5);
    
    const hornL = new THREE.Mesh(hornGeom, this.matArmorAccent);
    hornL.position.set(0.13, 0.15, -0.04);
    hornL.rotation.set(-0.45, 0, -0.38);
    head.add(hornL);

    const hornR = new THREE.Mesh(hornGeom, this.matArmorAccent);
    hornR.position.set(-0.13, 0.15, -0.04);
    hornR.rotation.set(-0.45, 0, 0.38);
    head.add(hornR);

    // Forehead Combat Crest Blade
    const crestGeom = new THREE.BoxGeometry(0.02, 0.13, 0.24);
    const crestMesh = new THREE.Mesh(crestGeom, this.matArmorAccent);
    crestMesh.position.set(0, 0.16, 0.01);
    crestMesh.rotation.x = -0.20;
    head.add(crestMesh);

    // ----------------------------------------------------
    // 6. Arms Assembly with Forearm High-Frequency Vibro-Blades
    // ----------------------------------------------------
    this._buildArm('l', 0.30, chest);
    this._buildArm('r', -0.30, chest);

    // ----------------------------------------------------
    // 7. Legs Assembly with Spiked Knee Guards & Sabatons
    // ----------------------------------------------------
    this._buildLeg('l', 0.16, pelvis);
    this._buildLeg('r', -0.16, pelvis);
  }

  _buildArm(side, xPos, parent) {
    const isL = side === 'l';
    const sign = isL ? 1 : -1;

    // Shoulder Pivot
    const shoulder = new THREE.Group();
    shoulder.position.set(xPos, 0.24, 0);
    parent.add(shoulder);
    this.joints[`${side}_shoulder`] = shoulder;

    // Chrome Spherical Bearing
    const ballMesh = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 12), this.matChrome);
    shoulder.add(ballMesh);

    // Spiked Multi-Tiered Combat Pauldron
    const pauGroup = new THREE.Group();
    pauGroup.position.set(sign * 0.05, 0.05, 0);
    pauGroup.rotation.z = sign * 0.24;
    shoulder.add(pauGroup);

    // White Ceramic Main Curved Shield
    const pauMain = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.11, 0.17, 6, 12),
      this.matArmorMain
    );
    pauMain.rotation.x = Math.PI / 2;
    pauMain.castShadow = true;
    pauGroup.add(pauMain);

    // Crimson Spiked Blade Fin jutting outward
    const pauBlade = new THREE.Mesh(
      new THREE.ConeGeometry(0.05, 0.20, 4),
      this.matArmorAccent
    );
    pauBlade.position.set(sign * 0.12, 0.04, 0);
    pauBlade.rotation.z = -sign * Math.PI / 2;
    pauBlade.rotation.x = 0.25;
    pauGroup.add(pauBlade);

    // Bicep (Ceramic white muscle plate with chrome hydraulic ram)
    const bicep = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.055, 0.20, 6, 12),
      this.matArmorMain
    );
    bicep.position.y = -0.17;
    bicep.castShadow = true;
    shoulder.add(bicep);

    const bPiston = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, 0.20, 8),
      this.matChromeBright
    );
    bPiston.position.set(0, -0.17, 0.055);
    shoulder.add(bPiston);

    // Elbow Joint (Double-disc chrome hinge with crimson hub)
    const elbow = new THREE.Group();
    elbow.position.y = -0.32;
    shoulder.add(elbow);
    this.joints[`${side}_elbow`] = elbow;

    const elbowDisc = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.06, 0.08, 12),
      this.matChrome
    );
    elbowDisc.rotation.z = Math.PI / 2;
    elbow.add(elbowDisc);

    const axisCap = new THREE.Mesh(
      new THREE.CylinderGeometry(0.025, 0.025, 0.095, 8),
      this.matArmorAccent
    );
    axisCap.rotation.z = Math.PI / 2;
    elbow.add(axisCap);

    // Forearm Gauntlet (White ceramic upper plate, dark titanium chassis)
    const forearm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.062, 0.052, 0.26, 8),
      this.matArmorMain
    );
    forearm.position.y = -0.17;
    forearm.castShadow = true;
    elbow.add(forearm);

    // HIGH-FREQUENCY LASER VIBRO-BLADE (Retractable combat arm blade)
    const bladeGeom = new THREE.BoxGeometry(0.015, 0.26, 0.06);
    const blade = new THREE.Mesh(bladeGeom, this.matBlade);
    blade.position.set(sign * 0.075, -0.22, 0.04);
    blade.rotation.x = -0.25;
    elbow.add(blade);
    this.joints[`${side}_blade`] = blade;

    // Articulated Combat Hand & Claw Digits
    const hand = new THREE.Group();
    hand.position.y = -0.32;
    elbow.add(hand);
    this.joints[`${side}_hand`] = hand;

    // Palm Carapace (Dark obsidian carbon)
    const palmMesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.09, 0.045),
      this.matInnerFrame
    );
    hand.add(palmMesh);

    // 4 Claw-Tipped Articulated Fingers
    for (let f = 0; f < 4; f++) {
      const fX = (f - 1.5) * 0.022;
      const finger = new THREE.Mesh(
        new THREE.CapsuleGeometry(0.008, 0.045, 4, 6),
        this.matChromeBright
      );
      finger.position.set(fX, -0.065, 0.005);
      hand.add(finger);
    }

    // Opposable Thumb
    const thumb = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.009, 0.038, 4, 6),
      this.matChromeBright
    );
    thumb.position.set(sign * 0.045, -0.04, 0.02);
    thumb.rotation.z = -sign * 0.5;
    hand.add(thumb);
  }

  _buildLeg(side, xPos, parent) {
    const isL = side === 'l';
    const sign = isL ? 1 : -1;

    // Hip Pivot
    const hip = new THREE.Group();
    hip.position.set(xPos, -0.06, 0);
    parent.add(hip);
    this.joints[`${side}_hip`] = hip;

    // Upper Thigh Armor (Stark white ceramic quad plate)
    const thigh = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.085, 0.32, 8, 12),
      this.matArmorMain
    );
    thigh.position.y = -0.22;
    thigh.castShadow = true;
    hip.add(thigh);

    // Side Hydraulic Power Strut
    const tStrut = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, 0.28, 6),
      this.matChromeBright
    );
    tStrut.position.set(sign * 0.09, -0.22, 0);
    hip.add(tStrut);

    // Knee Mechanism & FORWARD-JUTTING CRIMSON IMPACT RAM SPIKE
    const knee = new THREE.Group();
    knee.position.y = -0.42;
    hip.add(knee);
    this.joints[`${side}_knee`] = knee;

    const kneePivot = new THREE.Mesh(
      new THREE.CylinderGeometry(0.065, 0.065, 0.12, 12),
      this.matChrome
    );
    kneePivot.rotation.z = Math.PI / 2;
    knee.add(kneePivot);

    const kneeCap = new THREE.Mesh(
      new THREE.ConeGeometry(0.06, 0.14, 4),
      this.matArmorAccent
    );
    kneeCap.position.set(0, 0, 0.09);
    kneeCap.rotation.x = Math.PI / 2;
    knee.add(kneeCap);

    // Lower Shin & Calf Assembly (Flared white ceramic shin guard)
    const shin = new THREE.Mesh(
      new THREE.CylinderGeometry(0.068, 0.055, 0.34, 8),
      this.matArmorMain
    );
    shin.position.y = -0.22;
    shin.castShadow = true;
    knee.add(shin);

    // Rear Calf Rocket Thruster (High-speed dash booster)
    const calfNoz = new THREE.Mesh(
      new THREE.ConeGeometry(0.05, 0.11, 8, 1, true),
      this.matNozzle
    );
    calfNoz.position.set(0, -0.24, -0.08);
    calfNoz.rotation.x = -Math.PI / 2;
    knee.add(calfNoz);

    // Articulated Combat Sabaton Foot (Split-toe predator claws)
    const foot = new THREE.Group();
    foot.position.y = -0.44;
    knee.add(foot);
    this.joints[`${side}_foot`] = foot;

    // Dark Obsidian Sole Chassis
    const footMesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.13, 0.07, 0.28),
      this.matInnerFrame
    );
    footMesh.position.set(0, 0, 0.04);
    footMesh.castShadow = true;
    foot.add(footMesh);

    // Crimson Armored Toe Claw
    const toeCap = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.05, 0.10),
      this.matArmorAccent
    );
    toeCap.position.set(0, -0.01, 0.165);
    foot.add(toeCap);

    // Glowing Laser Heel Repulsor Pad
    const repPad = new THREE.Mesh(
      new THREE.CylinderGeometry(0.04, 0.04, 0.015, 12),
      this.matVisor
    );
    repPad.position.set(0, -0.04, 0.02);
    foot.add(repPad);
  }

  triggerAction(actionName) {
    const act = actionName.toUpperCase().trim();
    if (["JUMP", "DANCE", "RUN", "FIGHT", "SLASH", "MARIONETTE", "MIMIC", "POINT"].includes(act)) {
      this.action = act;
      this.actionTime = 0.0;
    }
  }

  toggleActionLock() {
    this.actionLocked = !this.actionLocked;
    return this.actionLocked;
  }

  update(delta, handData = null) {
    this.actionTime += delta;
    const t = this.actionTime;

    // Menacing Sinister Pulse: Core Reactor & Optic Slits
    const pulse = 2.4 + Math.sin(t * 8.0) * 0.8;
    this.matCore.emissiveIntensity = pulse;
    this.matVisor.emissiveIntensity = pulse * 0.9;
    if (this.coreLight) this.coreLight.intensity = pulse * 0.85;
    if (this.visorLight) this.visorLight.intensity = pulse * 0.7;

    // Hand Tracking or Sinister Idle Stance
    if (handData && handData.landmarks) {
      const palm = handData.palmCenter;
      const scale = handData.scale;
      const curls = handData.curls;
      const gestureAction = handData.action;

      if (!this.actionLocked && gestureAction) {
        if (gestureAction !== this.action) {
          this.action = gestureAction;
          this.actionTime = 0.0;
        }
      }

      // 3D position tracking
      this.basePos.x = (palm.x - 0.5) * 2.6;
      this.basePos.y = -(palm.y - 0.5) * 1.8 + 0.1;
      this.basePos.z = (scale - 0.2) * -2.5;

      // 3D rotation tracking
      this.baseRot.x = handData.pitch * 0.5;
      this.baseRot.y = handData.yaw * 0.6;
      this.baseRot.z = -handData.roll * 0.4;

      if (this.action === "MARIONETTE") {
        this._updateMarionette(curls);
      } else if (this.action === "MIMIC") {
        this._updateMimic(curls, handData);
      }
    } else {
      // Menacing predatory breathing hover
      this.basePos.set(0, Math.sin(t * 1.8) * 0.05, 0);
      this.baseRot.set(0.04, Math.sin(t * 0.9) * 0.12, 0);
    }

    // Dynamic Action Kinematics
    this.targetPos.copy(this.basePos);
    this.targetRot.copy(this.baseRot);

    switch (this.action) {
      case "JUMP":
        this._execJump(t);
        break;
      case "DANCE":
        this._execDance(t);
        break;
      case "RUN":
        this._execRun(t);
        break;
      case "FIGHT":
        this._execFight(t);
        break;
      case "SLASH":
        this._execSlash(t);
        break;
      case "POINT":
        this._execPoint();
        break;
    }

    // Snappy responsive joint & body smoothing
    const smooth = 0.24;
    this.group.position.lerp(this.targetPos, smooth);
    this.group.rotation.x += smooth * (this.targetRot.x - this.group.rotation.x);
    this.group.rotation.y += smooth * (this.targetRot.y - this.group.rotation.y);
    this.group.rotation.z += smooth * (this.targetRot.z - this.group.rotation.z);

    // Apply joint angles to Three.js transforms
    this._applyJoints(smooth);

    // Update String Anchor World Positions
    this._updateStringAnchors();
  }

  // ----------------------------------------------------
  // UPGRADED HIGH-OCTANE ACTIONS
  // ----------------------------------------------------

  _execJump(t) {
    const cycle = t % 1.9;

    if (cycle < 0.25) {
      // Phase 1: Kinetic Pre-Ignition Crouch
      const crouch = cycle / 0.25;
      this.targetPos.y += -0.45 * crouch;
      this.targetAngles.l_hip_pitch = 0.95 * crouch;
      this.targetAngles.r_hip_pitch = 0.95 * crouch;
      this.targetAngles.l_knee_pitch = -1.45 * crouch;
      this.targetAngles.r_knee_pitch = -1.45 * crouch;
      this.targetAngles.l_shoulder_pitch = 0.6 * crouch;
      this.targetAngles.r_shoulder_pitch = 0.6 * crouch;
      this.targetAngles.torso_pitch = 0.40 * crouch;

      // Rocket thrusters warming up
      if (this.onThrusterSpark && Math.random() < 0.4) {
        this.onThrusterSpark(this.stringAnchors.chest, 2, 0xff3300);
      }
    } else if (cycle < 0.80) {
      // Phase 2: Supersonic Rocket Launch! (+1.9Y altitude)
      const air = (cycle - 0.25) / 0.55;
      const jumpHeight = Math.sin(air * Math.PI) * 1.9;
      this.targetPos.y += jumpHeight;

      // Interceptor aerodynamic posture
      this.targetAngles.torso_pitch = -0.22;
      this.targetAngles.l_shoulder_pitch = -1.75;
      this.targetAngles.r_shoulder_pitch = -1.75;
      this.targetAngles.l_shoulder_roll = 0.20;
      this.targetAngles.r_shoulder_roll = -0.20;
      this.targetAngles.l_elbow_pitch = 0.2;
      this.targetAngles.r_elbow_pitch = 0.2;

      this.targetAngles.l_hip_pitch = -0.45;
      this.targetAngles.r_hip_pitch = 0.35;
      this.targetAngles.l_knee_pitch = -0.3;
      this.targetAngles.r_knee_pitch = -0.8;

      // Continuous plasma plume sparks from back jetpack & calf nozzles
      if (this.onThrusterSpark) {
        this.onThrusterSpark(this.stringAnchors.chest, 3, 0xff3300);
        this.onThrusterSpark(this.stringAnchors.r_foot, 2, 0xff1744);
      }
    } else if (cycle < 1.25) {
      // Phase 3: Superhero 3-Point Crater Ground Slam!
      const land = (cycle - 0.80) / 0.45;
      this.targetPos.y += -0.42 * (1.0 - land);

      // Brutal landing posture: right fist smashed into ground, left knee down
      this.targetAngles.torso_pitch = 0.55 * (1.0 - land * 0.4);
      this.targetAngles.r_shoulder_pitch = -1.35 * (1.0 - land);
      this.targetAngles.r_elbow_pitch = 1.65;
      this.targetAngles.l_shoulder_pitch = 0.45;
      this.targetAngles.l_elbow_pitch = 0.30;

      this.targetAngles.l_hip_pitch = 1.25 * (1.0 - land * 0.4);
      this.targetAngles.r_hip_pitch = 0.45;
      this.targetAngles.l_knee_pitch = -1.75 * (1.0 - land * 0.4);
      this.targetAngles.r_knee_pitch = -0.75;

      // Trigger massive expanding shockwave rings and camera shake on impact moment
      if (cycle - 0.80 < 0.06) {
        if (this.onShockwave) {
          this.onShockwave(this.stringAnchors.r_hand, 0xff003c, 2.8);
          this.onShockwave(this.stringAnchors.l_foot, 0xff3300, 2.0);
        }
        if (this.onCameraShake) this.onCameraShake(0.35);
      }
    } else {
      // Phase 4: Lethal Recovery
      this._resetStance();
    }
  }

  _execFight(t) {
    const cycle = 1.8;
    const phase = Math.floor((t % cycle) / (cycle / 4));
    const subT = ((t % cycle) % (cycle / 4)) / (cycle / 4);

    this.targetAngles.l_hip_pitch = 0.40;
    this.targetAngles.r_hip_pitch = -0.30;
    this.targetAngles.l_knee_pitch = -0.75;
    this.targetAngles.r_knee_pitch = -0.90;

    if (phase === 0) {
      // 1. Sonic Left Dash Jab & Blade Slash
      const strike = Math.sin(subT * Math.PI);
      this.targetAngles.l_shoulder_pitch = -1.75 * strike;
      this.targetAngles.l_shoulder_roll = 0.10;
      this.targetAngles.l_elbow_pitch = 0.05 + (1 - strike) * 1.4;
      this.targetAngles.torso_yaw = -0.45 * strike;
      this.targetAngles.torso_pitch = 0.20;

      this.targetAngles.r_shoulder_pitch = -0.40;
      this.targetAngles.r_elbow_pitch = 1.70;

      if (subT > 0.4 && subT < 0.55 && this.onShockwave) {
        this.onShockwave(this.stringAnchors.l_hand, 0xff003c, 1.2);
        if (this.onCameraShake) this.onCameraShake(0.18);
      }
    } else if (phase === 1) {
      // 2. Heavy Right 360 Spinning Cross
      const spin = Math.sin(subT * Math.PI);
      this.targetAngles.r_shoulder_pitch = -1.85 * spin;
      this.targetAngles.r_shoulder_roll = -0.15;
      this.targetAngles.r_elbow_pitch = 0.05 + (1 - spin) * 1.5;
      this.targetAngles.torso_yaw = 0.65 * spin;
      this.targetAngles.torso_pitch = 0.25;

      this.targetAngles.l_shoulder_pitch = -0.35;
      this.targetAngles.l_elbow_pitch = 1.65;

      if (subT > 0.4 && subT < 0.55) {
        if (this.onShockwave) this.onShockwave(this.stringAnchors.r_hand, 0xff3300, 1.6);
        if (this.onCameraShake) this.onCameraShake(0.25);
      }
    } else if (phase === 2) {
      // 3. Flying Knee Thruster Ram Strike!
      const jump = Math.sin(subT * Math.PI);
      this.targetPos.y += 0.35 * jump;
      this.targetAngles.r_hip_pitch = 1.25 * jump;
      this.targetAngles.r_knee_pitch = -1.70 * jump;
      this.targetAngles.l_hip_pitch = -0.50 * jump;
      this.targetAngles.torso_pitch = -0.25 * jump;

      this.targetAngles.l_shoulder_pitch = 0.45;
      this.targetAngles.r_shoulder_pitch = 0.45;

      if (this.onThrusterSpark && subT < 0.6) {
        this.onThrusterSpark(this.stringAnchors.r_foot, 3, 0xff1744);
      }
      if (subT > 0.45 && subT < 0.6 && this.onShockwave) {
        this.onShockwave(this.stringAnchors.r_foot, 0xff003c, 1.8);
        if (this.onCameraShake) this.onCameraShake(0.28);
      }
    } else {
      // 4. Cataclysmic Dual-Gauntlet Ground Sledgehammer!
      const smash = Math.sin(subT * Math.PI);
      if (subT < 0.4) {
        // Raise high
        this.targetPos.y += 0.25;
        this.targetAngles.l_shoulder_pitch = -1.85;
        this.targetAngles.r_shoulder_pitch = -1.85;
        this.targetAngles.l_elbow_pitch = 0.8;
        this.targetAngles.r_elbow_pitch = 0.8;
        this.targetAngles.torso_pitch = -0.20;
      } else {
        // Slam down into floor!
        this.targetPos.y += -0.28 * smash;
        this.targetAngles.l_shoulder_pitch = -0.65;
        this.targetAngles.r_shoulder_pitch = -0.65;
        this.targetAngles.l_elbow_pitch = 1.75;
        this.targetAngles.r_elbow_pitch = 1.75;
        this.targetAngles.torso_pitch = 0.50;

        if (subT > 0.45 && subT < 0.6) {
          if (this.onShockwave) {
            this.onShockwave(this.stringAnchors.l_hand, 0xff003c, 2.5);
            this.onShockwave(this.stringAnchors.r_hand, 0xff003c, 2.5);
          }
          if (this.onCameraShake) this.onCameraShake(0.38);
        }
      }
    }
  }

  _execSlash(t) {
    // Dual Forearm Blade Hurricane Cross-Slash
    const cycle = 1.2;
    const phase = (t % cycle) / cycle;

    const cross = Math.sin(phase * Math.PI * 2);
    this.targetRot.y = Math.sin(t * 12.0) * 0.35;
    this.targetAngles.torso_yaw = this.targetRot.y * 1.2;
    this.targetAngles.torso_pitch = 0.15;

    // Cross arms in X-slash
    this.targetAngles.l_shoulder_pitch = -1.25 - cross * 0.4;
    this.targetAngles.l_shoulder_roll = -0.35 + cross * 0.7;
    this.targetAngles.l_elbow_pitch = 1.35;

    this.targetAngles.r_shoulder_pitch = -1.25 + cross * 0.4;
    this.targetAngles.r_shoulder_roll = 0.35 - cross * 0.7;
    this.targetAngles.r_elbow_pitch = 1.35;

    if (this.onSlashArc && Math.random() < 0.5) {
      this.onSlashArc(this.stringAnchors.l_hand, 0xff003c, 10);
      this.onSlashArc(this.stringAnchors.r_hand, 0xff3300, 10);
    }
  }

  _execDance(t) {
    const beat = (t * 2.4) % 8.0;
    const bounce = Math.abs(Math.sin(t * 4.8)) * 0.14;
    this.targetPos.y += bounce;

    const sway = Math.sin(t * 4.8) * 0.26;
    this.targetRot.y += sway;
    this.targetAngles.torso_yaw = sway * 0.9;
    this.targetAngles.head_pitch = Math.sin(t * 9.6) * 0.22;

    if (beat < 2.0) {
      // 1. High-Tech Robotic Popping Wave
      const w = beat / 2.0;
      this.targetAngles.l_shoulder_pitch = -0.4 - Math.sin(w * Math.PI) * 1.4;
      this.targetAngles.l_elbow_pitch = 0.3 + Math.sin(w * Math.PI * 1.3) * 1.3;
      this.targetAngles.r_shoulder_pitch = -0.3 - Math.sin(Math.max(0, w - 0.25) * Math.PI) * 1.4;
      this.targetAngles.r_elbow_pitch = 0.3 + Math.sin(Math.max(0, w - 0.25) * Math.PI) * 1.3;
    } else if (beat < 4.0) {
      // 2. Cross-Body Disco Air-Freeze
      this.targetAngles.r_shoulder_pitch = -1.75;
      this.targetAngles.r_elbow_pitch = 0.15;
      this.targetAngles.l_shoulder_pitch = 0.25;
      this.targetAngles.l_elbow_pitch = 1.45;
      this.targetAngles.r_hip_pitch = -0.65;
      this.targetAngles.r_knee_pitch = -0.95;
    } else if (beat < 6.0) {
      // 3. Moonwalk Power Slide & Shoulder Rolls
      const stride = Math.sin(t * 7.0);
      this.targetAngles.l_hip_pitch = stride * 0.65;
      this.targetAngles.r_hip_pitch = -stride * 0.65;
      this.targetAngles.l_knee_pitch = -0.4 - Math.max(0, stride) * 0.8;
      this.targetAngles.r_knee_pitch = -0.4 - Math.max(0, -stride) * 0.8;
      this.targetAngles.l_shoulder_pitch = -0.8 + stride * 0.5;
      this.targetAngles.r_shoulder_pitch = -0.8 - stride * 0.5;
    } else {
      // 4. Double Pop Isolation Pose
      this.targetAngles.l_shoulder_pitch = -1.25;
      this.targetAngles.r_shoulder_pitch = -1.25;
      this.targetAngles.l_elbow_pitch = 1.65;
      this.targetAngles.r_elbow_pitch = 1.65;
    }
  }

  _execRun(t) {
    const stride = t * 9.2;
    this.targetRot.x = 0.38; // Predatory 38° forward aerodynamic lean
    const bob = Math.abs(Math.sin(stride * 2.0)) * 0.12;
    this.targetPos.y += -bob;

    const sinS = Math.sin(stride);
    this.targetAngles.l_hip_pitch = -sinS * 1.05;
    this.targetAngles.l_knee_pitch = -0.35 - Math.max(0, sinS) * 1.45;
    this.targetAngles.r_hip_pitch = sinS * 1.05;
    this.targetAngles.r_knee_pitch = -0.35 - Math.max(0, -sinS) * 1.45;

    // Aggressive counter-pumping arms with forearm blades swept back
    this.targetAngles.l_shoulder_pitch = sinS * 1.15 - 0.25;
    this.targetAngles.l_elbow_pitch = 0.95 + Math.max(0, sinS) * 0.6;
    this.targetAngles.r_shoulder_pitch = -sinS * 1.15 - 0.25;
    this.targetAngles.r_elbow_pitch = 0.95 + Math.max(0, -sinS) * 0.6;

    this.targetAngles.torso_yaw = -sinS * 0.22;

    // Pulse calf thrusters during sprint
    if (this.onThrusterSpark && Math.random() < 0.35) {
      this.onThrusterSpark(this.stringAnchors.r_foot, 1, 0xff3300);
      this.onThrusterSpark(this.stringAnchors.l_foot, 1, 0xff3300);
    }
  }

  _execPoint() {
    this.targetAngles.r_shoulder_pitch = -1.65;
    this.targetAngles.r_elbow_pitch = 0.10;
    this.targetAngles.l_shoulder_pitch = 0.1;
    this.targetAngles.l_elbow_pitch = 0.9;
    this.targetAngles.head_yaw = 0.35;
  }

  _updateMarionette(curls) {
    const idxPull = 1.0 - curls.index;
    const thumbPull = 1.0 - curls.thumb;
    const midPull = 1.0 - curls.middle;
    const ringPull = 1.0 - curls.ring;
    const pinkyPull = 1.0 - curls.pinky;

    this.targetAngles.l_shoulder_pitch = -0.25 - idxPull * 1.55;
    this.targetAngles.l_elbow_pitch = 0.35 + idxPull * 0.95;
    this.targetAngles.l_shoulder_roll = 0.25 + thumbPull * 0.35;

    this.targetAngles.torso_pitch = -0.1 + (1.0 - midPull) * 0.3;
    this.targetAngles.head_pitch = -0.15 + midPull * 0.35;

    this.targetAngles.r_shoulder_pitch = -0.2 - ringPull * 1.4;
    this.targetAngles.r_elbow_pitch = 0.35 + ringPull * 0.8;

    this.targetAngles.r_hip_pitch = -0.25 - pinkyPull * 0.75;
    this.targetAngles.r_knee_pitch = -0.3 - pinkyPull * 0.95;
  }

  _updateMimic(curls, handData) {
    this.targetAngles.l_shoulder_pitch = -0.3 - (1.0 - curls.index) * 1.6;
    this.targetAngles.r_shoulder_pitch = -0.3 - (1.0 - curls.ring) * 1.6;
    this.targetAngles.torso_pitch = -handData.pitch * 0.6;
    this.targetAngles.head_yaw = handData.yaw * 0.8;
  }

  _resetStance() {
    this.targetAngles.l_hip_pitch = 0.20;
    this.targetAngles.r_hip_pitch = -0.25;
    this.targetAngles.l_knee_pitch = -0.40;
    this.targetAngles.r_knee_pitch = -0.50;
    this.targetAngles.l_shoulder_pitch = -0.35;
    this.targetAngles.r_shoulder_pitch = -0.25;
    this.targetAngles.torso_pitch = 0.05;
    this.targetAngles.head_pitch = 0.08;
  }

  _applyJoints(smooth) {
    for (const key in this.targetAngles) {
      this.angles[key] += smooth * (this.targetAngles[key] - this.angles[key]);
    }

    if (this.joints.head) {
      this.joints.head.rotation.set(this.angles.head_pitch, this.angles.head_yaw, this.angles.head_roll);
    }
    if (this.joints.chest) {
      this.joints.chest.rotation.set(this.angles.torso_pitch, this.angles.torso_yaw, this.angles.torso_roll);
    }
    if (this.joints.l_shoulder) {
      this.joints.l_shoulder.rotation.set(this.angles.l_shoulder_pitch, 0, this.angles.l_shoulder_roll);
    }
    if (this.joints.l_elbow) {
      this.joints.l_elbow.rotation.x = this.angles.l_elbow_pitch;
    }
    if (this.joints.r_shoulder) {
      this.joints.r_shoulder.rotation.set(this.angles.r_shoulder_pitch, 0, this.angles.r_shoulder_roll);
    }
    if (this.joints.r_elbow) {
      this.joints.r_elbow.rotation.x = this.angles.r_elbow_pitch;
    }
    if (this.joints.l_hip) {
      this.joints.l_hip.rotation.set(this.angles.l_hip_pitch, 0, this.angles.l_hip_roll);
    }
    if (this.joints.l_knee) {
      this.joints.l_knee.rotation.x = this.angles.l_knee_pitch;
    }
    if (this.joints.r_hip) {
      this.joints.r_hip.rotation.set(this.angles.r_hip_pitch, 0, this.angles.r_hip_roll);
    }
    if (this.joints.r_knee) {
      this.joints.r_knee.rotation.x = this.angles.r_knee_pitch;
    }
  }

  _updateStringAnchors() {
    if (this.joints.head) this.joints.head.getWorldPosition(this.stringAnchors.head);
    if (this.joints.chest) this.joints.chest.getWorldPosition(this.stringAnchors.chest);
    if (this.joints.l_hand) this.joints.l_hand.getWorldPosition(this.stringAnchors.l_hand);
    if (this.joints.r_hand) this.joints.r_hand.getWorldPosition(this.stringAnchors.r_hand);
    if (this.joints.l_foot) this.joints.l_foot.getWorldPosition(this.stringAnchors.l_foot);
    if (this.joints.r_foot) this.joints.r_foot.getWorldPosition(this.stringAnchors.r_foot);
  }
}
