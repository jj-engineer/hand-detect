"""
Sleek Modern 3D Humanoid Robot / Dangerous High-Tech Combat Mecha (Python Implementation).
Rendered with Software Polygon Rasterizer, Metallic Lambertian & Specular Shading,
and Real-Time Hand Tracking Kinematics.

Dangerous Red, Dark & White War Mecha Design:
- Stark Arctic Ceramic White Armor Plates
- Blood Crimson Red Demon Horns, Vibro-Blades, Pauldron Spikes & Knee Ram Spikes
- Deep Obsidian Carbon Slate & Dark Titanium Chassis
- Predatory Dual Slanted Red Optic Slits & Crimson Singularity Core
- Brutal 4-Hit Martial Arts Combat Combo, Blade Frenzy Slash, Meteor Rocket Jump & Assassin Sprint.
"""

import math
import random
import numpy as np
import cv2


def rot_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def compute_normal(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm > 1e-6:
        return n / norm
    return np.array([0, 0, 1], dtype=np.float32)


def create_cylinder_mesh(r_top, r_bot, height, segments=10, base_color=(245, 245, 250), outline_color=(20, 15, 25)):
    """Creates a precision faceted cylinder/cone for mechanical actuators, pistons & joints."""
    verts = []
    hy = height / 2.0
    angles = [i * 2.0 * math.pi / segments for i in range(segments)]

    # Top ring
    for a in angles:
        verts.append([r_top * math.cos(a), -hy, r_top * math.sin(a)])
    # Bottom ring
    for a in angles:
        verts.append([r_bot * math.cos(a), hy, r_bot * math.sin(a)])

    verts = np.array(verts, dtype=np.float32)
    faces = []

    # Side quad faces
    for i in range(segments):
        nxt = (i + 1) % segments
        p_top_cur = i
        p_top_nxt = nxt
        p_bot_cur = i + segments
        p_bot_nxt = nxt + segments

        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), 0, math.sin(mid_a)], dtype=np.float32)
        faces.append(([p_top_cur, p_bot_cur, p_bot_nxt, p_top_nxt], n, base_color))

    # Top cap
    top_indices = list(range(segments))
    faces.append((top_indices, np.array([0, -1, 0], dtype=np.float32), base_color))
    # Bottom cap
    bot_indices = [i + segments for i in reversed(range(segments))]
    faces.append((bot_indices, np.array([0, 1, 0], dtype=np.float32), base_color))

    return verts, faces, outline_color


def create_faceted_capsule_mesh(r_top, r_bot, height, segments=8, base_color=(245, 245, 250), outline_color=(20, 15, 25)):
    """Creates an aerodynamic, curved capsule mesh for organic mecha limbs (limbs, calves, bicep)."""
    verts = []
    hy = height / 2.0
    angles = [i * 2.0 * math.pi / segments for i in range(segments)]

    # 0: top dome apex
    verts.append([0, -hy - (r_top * 0.35), 0])
    # 1 .. segments: top ring
    for a in angles:
        verts.append([r_top * math.cos(a), -hy, r_top * math.sin(a)])
    # segments+1 .. 2*segments: mid waist ring (slightly bulged for muscle contour)
    r_mid = (r_top + r_bot) * 0.56
    for a in angles:
        verts.append([r_mid * math.cos(a), 0, r_mid * math.sin(a)])
    # 2*segments+1 .. 3*segments: bottom ring
    for a in angles:
        verts.append([r_bot * math.cos(a), hy, r_bot * math.sin(a)])
    # 3*segments+1: bottom dome apex
    verts.append([0, hy + (r_bot * 0.35), 0])

    verts = np.array(verts, dtype=np.float32)
    faces = []

    # Top apex fan
    for i in range(segments):
        nxt = (i + 1) % segments
        p1 = 1 + i
        p2 = 1 + nxt
        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), -0.85, math.sin(mid_a)], dtype=np.float32)
        n /= np.linalg.norm(n)
        faces.append(([0, p1, p2], n, base_color))

    # Upper body quads
    for i in range(segments):
        nxt = (i + 1) % segments
        p_t1 = 1 + i
        p_t2 = 1 + nxt
        p_m1 = 1 + segments + i
        p_m2 = 1 + segments + nxt
        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), -0.15, math.sin(mid_a)], dtype=np.float32)
        n /= np.linalg.norm(n)
        faces.append(([p_t1, p_m1, p_m2, p_t2], n, base_color))

    # Lower body quads
    for i in range(segments):
        nxt = (i + 1) % segments
        p_m1 = 1 + segments + i
        p_m2 = 1 + segments + nxt
        p_b1 = 1 + 2 * segments + i
        p_b2 = 1 + 2 * segments + nxt
        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), 0.15, math.sin(mid_a)], dtype=np.float32)
        n /= np.linalg.norm(n)
        faces.append(([p_m1, p_b1, p_b2, p_m2], n, base_color))

    # Bottom apex fan
    bot_idx = len(verts) - 1
    for i in range(segments):
        nxt = (i + 1) % segments
        p1 = 1 + 2 * segments + i
        p2 = 1 + 2 * segments + nxt
        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), 0.85, math.sin(mid_a)], dtype=np.float32)
        n /= np.linalg.norm(n)
        faces.append(([bot_idx, p2, p1], n, base_color))

    return verts, faces, outline_color


def create_sculpted_armor_plate(w_top, w_bot, h, d_top, d_bot, bevel=0.18, base_color=(245, 245, 250), outline_color=(20, 15, 25)):
    """Creates a multi-faceted beveled mecha armor plate with aerodynamic center ridge."""
    hwt, hwb = w_top / 2.0, w_bot / 2.0
    hdt, hdb = d_top / 2.0, d_bot / 2.0
    hy = h / 2.0

    # 10 vertices: 6 on front face (with center ridge), 4 on back face
    ridge_front = -max(hdt, hdb) * 0.35
    v = np.array([
        [-hwb, hy, -hdb],              # 0: bot front left
        [0.0,  hy, -hdb + ridge_front],# 1: bot front ridge
        [ hwb, hy, -hdb],              # 2: bot front right
        [ hwt, -hy, -hdt],             # 3: top front right
        [0.0,  -hy, -hdt + ridge_front],# 4: top front ridge
        [-hwt, -hy, -hdt],             # 5: top front left
        [-hwb, hy,  hdb],              # 6: bot back left
        [ hwb, hy,  hdb],              # 7: bot back right
        [ hwt, -hy,  hdt],             # 8: top back right
        [-hwt, -hy,  hdt],             # 9: top back left
    ], dtype=np.float32)

    faces = [
        # Front left facet
        ([0, 1, 4, 5], np.array([-0.3, 0, -0.95], dtype=np.float32), base_color),
        # Front right facet
        ([1, 2, 3, 4], np.array([ 0.3, 0, -0.95], dtype=np.float32), base_color),
        # Back plate
        ([7, 6, 9, 8], np.array([ 0.0, 0,  1.0], dtype=np.float32), base_color),
        # Left side
        ([6, 0, 5, 9], np.array([-1.0, 0,  0.0], dtype=np.float32), base_color),
        # Right side
        ([2, 7, 8, 3], np.array([ 1.0, 0,  0.0], dtype=np.float32), base_color),
        # Top cap
        ([5, 4, 3, 8, 9], np.array([0.0, -1.0, 0.0], dtype=np.float32), base_color),
        # Bottom cap
        ([6, 7, 2, 1, 0], np.array([0.0,  1.0, 0.0], dtype=np.float32), base_color),
    ]
    return v, faces, outline_color


def create_blade_mesh(length, width, thickness, base_color=(60, 40, 255), outline_color=(20, 10, 30)):
    """Creates a sharp angular combat vibro-blade with glowing laser cutting edge."""
    hl = length / 2.0
    hw = width / 2.0
    ht = thickness / 2.0

    # 6 vertices: sharp diamond cross-section tapering to a razor tip
    v = np.array([
        [-hw, -hl,  0.0],      # 0: base spine
        [ hw, -hl,  0.0],      # 1: base cutting edge
        [ 0.0, -hl,  ht],      # 2: base left facet
        [ 0.0, -hl, -ht],      # 3: base right facet
        [-hw * 0.3, hl, 0.0],  # 4: tip spine
        [ hw * 1.2, hl * 1.1, 0.0], # 5: razor blade tip!
    ], dtype=np.float32)

    faces = [
        ([0, 1, 5, 4], np.array([0, 0, -1], dtype=np.float32), base_color),
        ([1, 2, 5], np.array([0.7, 0, 0.7], dtype=np.float32), base_color),
        ([0, 2, 5, 4], np.array([-0.7, 0, 0.7], dtype=np.float32), base_color),
        ([1, 3, 5], np.array([0.7, 0, -0.7], dtype=np.float32), base_color),
        ([0, 3, 5, 4], np.array([-0.7, 0, -0.7], dtype=np.float32), base_color),
    ]
    return v, faces, outline_color


def create_spike_mesh(radius, height, segments=5, base_color=(35, 35, 235), outline_color=(20, 10, 25)):
    """Creates an aggressive angular horn/ram spike."""
    verts = [[0, -height, 0]] # 0: apex
    angles = [i * 2.0 * math.pi / segments for i in range(segments)]
    for a in angles:
        verts.append([radius * math.cos(a), 0, radius * math.sin(a)])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(segments):
        nxt = (i + 1) % segments
        p1 = 1 + i
        p2 = 1 + nxt
        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), -0.5, math.sin(mid_a)], dtype=np.float32)
        n /= np.linalg.norm(n)
        faces.append(([0, p1, p2], n, base_color))

    bot_indices = list(reversed(range(1, segments + 1)))
    faces.append((bot_indices, np.array([0, 1, 0], dtype=np.float32), base_color))
    return verts, faces, outline_color


def create_nozzle_mesh(r_in, r_out, height, segments=8, base_color=(25, 22, 30), outline_color=(15, 12, 20)):
    """Creates a flared rocket thruster exhaust bell nozzle."""
    verts = []
    hy = height / 2.0
    angles = [i * 2.0 * math.pi / segments for i in range(segments)]

    for a in angles:
        verts.append([r_in * math.cos(a), -hy, r_in * math.sin(a)])
    for a in angles:
        verts.append([r_out * math.cos(a), hy, r_out * math.sin(a)])

    verts = np.array(verts, dtype=np.float32)
    faces = []

    for i in range(segments):
        nxt = (i + 1) % segments
        p_t1 = i
        p_t2 = nxt
        p_b1 = i + segments
        p_b2 = nxt + segments
        mid_a = (angles[i] + angles[nxt]) / 2.0
        n = np.array([math.cos(mid_a), -0.3, math.sin(mid_a)], dtype=np.float32)
        n /= np.linalg.norm(n)
        faces.append(([p_t1, p_b1, p_b2, p_t2], n, base_color))

    return verts, faces, outline_color


class Robot3D:
    def __init__(self):
        # Base transform in camera coordinates
        self.pos = np.array([-210.0, 10.0, 520.0], dtype=np.float32)
        self.rot = np.array([0.05, 0.45, 0.0], dtype=np.float32)

        self.vel_pos = np.zeros(3, dtype=np.float32)
        self.vel_rot = np.zeros(3, dtype=np.float32)
        self.vel_angles = {k: 0.0 for k in [
            'head_yaw', 'head_pitch', 'head_roll',
            'torso_pitch', 'torso_yaw', 'torso_roll',
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_elbow_pitch', 'l_wrist_pitch',
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_elbow_pitch', 'r_wrist_pitch',
            'l_hip_pitch', 'l_hip_roll', 'l_knee_pitch',
            'r_hip_pitch', 'r_hip_roll', 'r_knee_pitch'
        ]}

        self.scale = 2.0

        # DANGEROUS RED, DARK & WHITE COLOR PALETTE (BGR Format for OpenCV)
        self.COLOR_ARMOR_WHITE   = (248, 245, 242)  # Stark Arctic Ceramic White
        self.COLOR_CRIMSON_RED   = (35, 35, 235)    # Blood Crimson Red (BGR: R=235)
        self.COLOR_ARMOR_DARK    = (28, 22, 16)     # Dark Titanium Carbon
        self.COLOR_CARBON        = (16, 12, 8)      # Deep Obsidian Carbon Slate
        self.COLOR_JOINT_CHROME  = (85, 75, 65)     # Dark Titanium Gunmetal
        self.COLOR_CHROME_BRIGHT = (235, 230, 225)  # Mirror Blade Chrome
        self.COLOR_VISOR_RED     = (60, 0, 255)     # Luminous Laser Crimson Optic Slits (BGR)
        self.COLOR_CORE_RED      = (30, 20, 255)    # Crimson Singularity Core (BGR)
        self.COLOR_BLADE_RED     = (60, 40, 255)    # Laser Edge Vibro-Blades (BGR)
        self.COLOR_THRUSTER_FIRE = (25, 110, 255)   # Fiery Rocket Plasma Plume (BGR)
        self.COLOR_OUTLINE       = (15, 12, 10)     # Crisp Dark Outline

        # Articulated joints (radians)
        self.angles = {
            'head_yaw': 0.0,
            'head_pitch': 0.08, # Menacing slight downward scowl
            'head_roll': 0.0,
            'torso_pitch': 0.05,
            'torso_yaw': 0.0,
            'torso_roll': 0.0,
            'l_shoulder_pitch': -0.35,
            'l_shoulder_roll': 0.40,
            'l_elbow_pitch': 0.85,
            'l_wrist_pitch': 0.1,
            'r_shoulder_pitch': -0.25,
            'r_shoulder_roll': -0.40,
            'r_elbow_pitch': 0.75,
            'r_wrist_pitch': 0.1,
            'l_hip_pitch': 0.20,
            'l_hip_roll': 0.12,
            'l_knee_pitch': -0.40,
            'r_hip_pitch': -0.25,
            'r_hip_roll': -0.12,
            'r_knee_pitch': -0.50,
        }
        self.target_angles = dict(self.angles)

        # Marionette string attachment points in world space
        self.string_anchors = {
            'head': np.zeros(3, dtype=np.float32),
            'l_hand': np.zeros(3, dtype=np.float32),
            'r_hand': np.zeros(3, dtype=np.float32),
            'chest': np.zeros(3, dtype=np.float32),
            'l_foot': np.zeros(3, dtype=np.float32),
            'r_foot': np.zeros(3, dtype=np.float32),
        }
        self.screen_anchors = {k: (0, 0) for k in self.string_anchors}

        # Key directional light vector + rim light
        lv = np.array([0.5, 0.8, -0.65], dtype=np.float32)
        self.light_dir = lv / np.linalg.norm(lv)

        rv = np.array([-0.7, -0.3, 0.6], dtype=np.float32)
        self.rim_dir = rv / np.linalg.norm(rv)

        self.time = 0.0

        # --- Dynamic Action Remote Control System ---
        self.action_mode = "MARIONETTE"  # MARIONETTE, JUMP, DANCE, RUN, FIGHT, SLASH, MIMIC
        self.action_time = 0.0
        self.action_locked = False
        self.combo_index = 0
        self.particles = []        # Dynamic thruster plumes and sparks
        self.shockwaves = []       # Punch and landing shockwaves
        self.camera_shake = 0.0    # Screen shake intensity

        self._build_realistic_mecha_geometry()

    def _build_realistic_mecha_geometry(self):
        """Constructs dangerous red, dark & white mecha anatomy with demon horns & vibro-blades."""
        s = self.scale

        # 1. HEAD & SENSORY UNIT (Demon War Helmet)
        # Stark Arctic White Cranial Helmet
        self.m_head_v, self.m_head_f, _ = create_sculpted_armor_plate(19*s, 13*s, 22*s, 20*s, 16*s, bevel=0.2, base_color=self.COLOR_ARMOR_WHITE)
        # Forehead Crimson Combat Crest
        self.m_crest_v, self.m_crest_f, _ = create_sculpted_armor_plate(3*s, 2*s, 13*s, 15*s, 11*s, bevel=0.1, base_color=self.COLOR_CRIMSON_RED)
        
        # SWEPT-BACK ANGLED DEMON HORN BLADES (Blood Crimson Red)
        self.m_horn_l_v, self.m_horn_l_f, _ = create_spike_mesh(3.0*s, 22*s, segments=5, base_color=self.COLOR_CRIMSON_RED)
        self.m_horn_r_v, self.m_horn_r_f, _ = create_spike_mesh(3.0*s, 22*s, segments=5, base_color=self.COLOR_CRIMSON_RED)

        # DUAL PREDATORY SLANTED OPTIC SLITS (Laser Crimson Red)
        self.m_eye_l_v, self.m_eye_l_f, _ = create_sculpted_armor_plate(5*s, 3*s, 2.5*s, 4*s, 3*s, bevel=0.08, base_color=self.COLOR_VISOR_RED)
        self.m_eye_r_v, self.m_eye_r_f, _ = create_sculpted_armor_plate(5*s, 3*s, 2.5*s, 4*s, 3*s, bevel=0.08, base_color=self.COLOR_VISOR_RED)

        # Razor Chin Spike & Rebreather (Deep Obsidian Carbon)
        self.m_chin_v, self.m_chin_f, _ = create_sculpted_armor_plate(7*s, 4*s, 9*s, 9*s, 7*s, bevel=0.15, base_color=self.COLOR_CARBON)
        # Segmented Neck Actuator
        self.m_neck_v, self.m_neck_f, _ = create_cylinder_mesh(7*s, 8*s, 8*s, segments=8, base_color=self.COLOR_JOINT_CHROME)
        self.m_neck_piston_v, self.m_neck_piston_f, _ = create_cylinder_mesh(2.5*s, 2.5*s, 7*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)

        # 2. TORSO, CRIMSON SINGULARITY CORE & DORSAL JETPACK
        # Stark White Ceramic Pectoral Armor Plates
        self.m_chest_v, self.m_chest_f, _ = create_sculpted_armor_plate(44*s, 28*s, 34*s, 28*s, 20*s, bevel=0.22, base_color=self.COLOR_ARMOR_WHITE)
        # Crimson Hazard Chevrons across pectorals
        self.m_chev_v, self.m_chev_f, _ = create_sculpted_armor_plate(10*s, 7*s, 5*s, 4*s, 3*s, bevel=0.08, base_color=self.COLOR_CRIMSON_RED)

        # White Ceramic Collarbone Mantle
        self.m_collar_v, self.m_collar_f, _ = create_sculpted_armor_plate(48*s, 38*s, 10*s, 18*s, 14*s, bevel=0.15, base_color=self.COLOR_ARMOR_WHITE)
        # Crimson Singularity Core Outer Containment Ring
        self.m_core_outer_v, self.m_core_outer_f, _ = create_cylinder_mesh(8.5*s, 8.5*s, 4*s, segments=12, base_color=self.COLOR_CRIMSON_RED)
        # Pulsating Crimson Singularity Reactor Lens
        self.m_core_v, self.m_core_f, _ = create_cylinder_mesh(6.2*s, 5.5*s, 5*s, segments=10, base_color=self.COLOR_CORE_RED)
        # Flank Heatsink Radiator Vents
        self.m_vent_v, self.m_vent_f, _ = create_sculpted_armor_plate(6*s, 5*s, 16*s, 8*s, 6*s, bevel=0.1, base_color=self.COLOR_CARBON)

        # Segmented Spinal Column (Deep Obsidian Carbon)
        self.m_spine1_v, self.m_spine1_f, _ = create_cylinder_mesh(12*s, 13*s, 5*s, segments=8, base_color=self.COLOR_CARBON)
        self.m_spine2_v, self.m_spine2_f, _ = create_cylinder_mesh(11*s, 12*s, 5*s, segments=8, base_color=self.COLOR_JOINT_CHROME)
        self.m_spine3_v, self.m_spine3_f, _ = create_cylinder_mesh(13*s, 14*s, 5*s, segments=8, base_color=self.COLOR_CARBON)
        # Hydraulic Lumbar Shock Pistons
        self.m_piston_casing_v, self.m_piston_casing_f, _ = create_cylinder_mesh(3.5*s, 3.5*s, 14*s, segments=6, base_color=self.COLOR_CARBON)
        self.m_piston_rod_v, self.m_piston_rod_f, _ = create_cylinder_mesh(2*s, 2*s, 14*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)

        # Dorsal Rocket Jetpack Unit
        self.m_pack_v, self.m_pack_f, _ = create_sculpted_armor_plate(26*s, 18*s, 30*s, 16*s, 12*s, bevel=0.18, base_color=self.COLOR_CARBON)
        self.m_nozzle_v, self.m_nozzle_f, _ = create_nozzle_mesh(4.5*s, 7.5*s, 10*s, segments=10, base_color=self.COLOR_ARMOR_DARK)
        # Swept-Back Crimson Razor Stabilizer Wings
        self.m_pack_wing_v, self.m_pack_wing_f, _ = create_sculpted_armor_plate(14*s, 2*s, 20*s, 3*s, 1*s, bevel=0.1, base_color=self.COLOR_CRIMSON_RED)

        # 3. PELVIS & WAIST CHASSIS
        self.m_pelvis_v, self.m_pelvis_f, _ = create_sculpted_armor_plate(28*s, 34*s, 16*s, 20*s, 22*s, bevel=0.2, base_color=self.COLOR_CARBON)
        # Armored V-Codpiece (Crimson Red)
        self.m_groin_v, self.m_groin_f, _ = create_sculpted_armor_plate(12*s, 8*s, 14*s, 10*s, 8*s, bevel=0.15, base_color=self.COLOR_CRIMSON_RED)
        self.m_hip_skirt_v, self.m_hip_skirt_f, _ = create_sculpted_armor_plate(8*s, 6*s, 14*s, 12*s, 10*s, bevel=0.12, base_color=self.COLOR_ARMOR_WHITE)
        self.m_hip_ball_v, self.m_hip_ball_f, _ = create_cylinder_mesh(6.5*s, 6.5*s, 6*s, segments=8, base_color=self.COLOR_CHROME_BRIGHT)

        # 4. ARMS, SPIKED PAULDRONS & FOREARM VIBRO-BLADES
        # Spiked Multi-Tiered Pauldrons (White Ceramic + Spiked Blade Wing)
        self.m_shoulder_v, self.m_shoulder_f, _ = create_sculpted_armor_plate(24*s, 18*s, 18*s, 22*s, 16*s, bevel=0.25, base_color=self.COLOR_ARMOR_WHITE)
        self.m_pau_spike_v, self.m_pau_spike_f, _ = create_spike_mesh(3.5*s, 18*s, segments=4, base_color=self.COLOR_CRIMSON_RED)
        self.m_shoulder_sub_v, self.m_shoulder_sub_f, _ = create_sculpted_armor_plate(20*s, 14*s, 14*s, 18*s, 12*s, bevel=0.18, base_color=self.COLOR_CARBON)
        self.m_shoulder_ball_v, self.m_shoulder_ball_f, _ = create_cylinder_mesh(7*s, 7*s, 8*s, segments=8, base_color=self.COLOR_JOINT_CHROME)

        # Ceramic White Bicep + Hydraulic Ram
        self.m_bicep_v, self.m_bicep_f, _ = create_faceted_capsule_mesh(6.5*s, 5.5*s, 22*s, segments=8, base_color=self.COLOR_ARMOR_WHITE)
        self.m_bicep_piston_v, self.m_bicep_piston_f, _ = create_cylinder_mesh(2*s, 2*s, 16*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)

        # Double-Disc Elbow Joint
        self.m_elbow_v, self.m_elbow_f, _ = create_cylinder_mesh(6.5*s, 6.5*s, 8*s, segments=8, base_color=self.COLOR_JOINT_CHROME)
        self.m_elbow_axis_v, self.m_elbow_axis_f, _ = create_cylinder_mesh(3*s, 3*s, 9*s, segments=8, base_color=self.COLOR_CRIMSON_RED)

        # Ceramic White Forearm Gauntlet
        self.m_forearm_v, self.m_forearm_f, _ = create_sculpted_armor_plate(15*s, 11*s, 26*s, 15*s, 11*s, bevel=0.22, base_color=self.COLOR_ARMOR_WHITE)

        # RETRACTABLE HIGH-FREQUENCY VIBRO-BLADE (Laser Crimson Red)
        self.m_blade_v, self.m_blade_f, _ = create_blade_mesh(26*s, 6*s, 2.5*s, base_color=self.COLOR_BLADE_RED)

        # Claw-Tipped Articulated Hands
        self.m_hand_v, self.m_hand_f, _ = create_sculpted_armor_plate(10*s, 8*s, 12*s, 11*s, 8*s, bevel=0.2, base_color=self.COLOR_CARBON)
        self.m_finger_thumb_v, self.m_finger_thumb_f, _ = create_cylinder_mesh(2.2*s, 1.6*s, 7*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)
        self.m_finger_index_v, self.m_finger_index_f, _ = create_cylinder_mesh(2.2*s, 1.6*s, 9*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)
        self.m_finger_other_v, self.m_finger_other_f, _ = create_cylinder_mesh(2.2*s, 1.6*s, 8*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)

        # 5. LEGS, SPIKED KNEES & COMBAT SABATONS
        # Muscular Ceramic White Thighs
        self.m_thigh_v, self.m_thigh_f, _ = create_faceted_capsule_mesh(16*s, 12*s, 34*s, segments=8, base_color=self.COLOR_ARMOR_WHITE)
        self.m_thigh_plate_v, self.m_thigh_plate_f, _ = create_sculpted_armor_plate(14*s, 10*s, 24*s, 6*s, 4*s, bevel=0.15, base_color=self.COLOR_ARMOR_WHITE)
        self.m_hamstring_piston_v, self.m_hamstring_piston_f, _ = create_cylinder_mesh(2.5*s, 2.5*s, 20*s, segments=6, base_color=self.COLOR_CHROME_BRIGHT)

        # Dual-Pivot Knee Joint with FORWARD-JUTTING CRIMSON IMPACT RAM SPIKE
        self.m_knee_v, self.m_knee_f, _ = create_cylinder_mesh(7.5*s, 7.5*s, 9*s, segments=8, base_color=self.COLOR_JOINT_CHROME)
        self.m_knee_spike_v, self.m_knee_spike_f, _ = create_spike_mesh(4*s, 14*s, segments=4, base_color=self.COLOR_CRIMSON_RED)

        # Flared Shin Guard + Rear Calf Rocket Booster
        self.m_shin_v, self.m_shin_f, _ = create_sculpted_armor_plate(14*s, 18*s, 38*s, 16*s, 22*s, bevel=0.22, base_color=self.COLOR_ARMOR_WHITE)
        self.m_calf_thruster_v, self.m_calf_thruster_f, _ = create_sculpted_armor_plate(8*s, 10*s, 18*s, 10*s, 14*s, bevel=0.15, base_color=self.COLOR_CARBON)
        self.m_calf_nozzle_v, self.m_calf_nozzle_f, _ = create_nozzle_mesh(3*s, 5.5*s, 7*s, segments=8, base_color=self.COLOR_ARMOR_DARK)

        # Combat Sabaton Mecha Foot with Crimson Toe Claw
        self.m_foot_v, self.m_foot_f, _ = create_sculpted_armor_plate(16*s, 14*s, 9*s, 32*s, 26*s, bevel=0.2, base_color=self.COLOR_CARBON)
        self.m_toe_claw_v, self.m_toe_claw_f, _ = create_sculpted_armor_plate(14*s, 10*s, 6*s, 12*s, 8*s, bevel=0.15, base_color=self.COLOR_CRIMSON_RED)
        self.m_repulsor_v, self.m_repulsor_f, _ = create_cylinder_mesh(6*s, 6*s, 3*s, segments=8, base_color=self.COLOR_VISOR_RED)

    def trigger_action(self, action_name):
        """Remotely triggers a dynamic robot action (JUMP, DANCE, RUN, FIGHT, SLASH, MARIONETTE, MIMIC)."""
        clean_action = action_name.upper().strip()
        if clean_action in ["JUMP", "DANCE", "RUN", "FIGHT", "SLASH", "MARIONETTE", "MIMIC", "POINT"]:
            self.action_mode = clean_action
            self.action_time = 0.0
            print(f"[ACTION ENGINE] Activated: {self.action_mode}")

    def toggle_action_lock(self):
        """Toggles continuous looping for the current active action."""
        self.action_locked = not self.action_locked
        state_str = "LOCKED [CONTINUOUS]" if self.action_locked else "UNLOCKED [MOMENTARY]"
        print(f"[ACTION ENGINE] Action mode is now: {state_str}")
        return self.action_locked

    def update_from_hand(self, hand_data, screen_w, screen_h, mode_name="MARIONETTE"):
        """Fluid, natural hand-to-robot kinematic mapping with procedural remote actions."""
        self.time += 0.04
        dt = 0.033
        idle_sway_y = math.sin(self.time * 1.6) * 4.0

        # Update dynamic particle, shockwave, and screen shake lifetimes
        self._update_fx(dt)

        # 1. Evaluate Remote Gestures from Fingers
        detected_action = None
        pitch, yaw, roll = (0.0, 0.0, 0.0)
        norm_x, norm_y = 0.0, 0.0
        curls = {'thumb': 0.5, 'index': 0.5, 'middle': 0.5, 'ring': 0.5, 'pinky': 0.5}

        if hand_data is not None:
            palm = hand_data['palm_center']
            scale = hand_data['hand_scale']
            curls = hand_data['finger_curls']
            detected_action = hand_data.get('action', None)
            pitch, yaw, roll = hand_data['orientation']

            norm_x = (palm[0] / screen_w) - 0.5
            norm_y = (palm[1] / screen_h) - 0.5

            target_x = -210.0 + (norm_x * 240.0)
            target_y = 10.0 + (norm_y * 190.0) + idle_sway_y
            target_z = 520.0 - (scale - 75.0) * 1.6
            target_z = float(np.clip(target_z, 360.0, 780.0))
            self.base_target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)

            self.base_target_rot = np.array([
                np.clip(pitch * 0.7, -0.5, 0.5),
                np.clip(0.40 + (yaw * 0.65), -0.7, 0.9),
                np.clip(-roll * 0.45, -0.4, 0.4)
            ], dtype=np.float32)
        else:
            self.base_target_pos = np.array([-210.0, 10.0 + idle_sway_y, 520.0], dtype=np.float32)
            self.base_target_rot = np.array([0.05, 0.45, 0.0], dtype=np.float32)

        self.target_pos = np.copy(self.base_target_pos)
        self.target_rot = np.copy(self.base_target_rot)

        # Switch action based on finger gesture if not locked
        if not self.action_locked and detected_action is not None:
            if detected_action in ["JUMP", "DANCE", "RUN", "FIGHT", "SLASH", "MIMIC", "POINT"]:
                if self.action_mode != detected_action:
                    self.action_mode = detected_action
                    self.action_time = 0.0
            elif detected_action == "MARIONETTE" and self.action_mode not in ["MARIONETTE"]:
                self.action_mode = "MARIONETTE"
                self.action_time = 0.0

        # Override by explicit UI mode_name if specified
        if "COMBAT" in mode_name and self.action_mode not in ["FIGHT", "SLASH", "JUMP", "RUN", "DANCE"]:
            self.action_mode = "FIGHT"
        elif "MIMIC" in mode_name and self.action_mode not in ["FIGHT", "SLASH", "JUMP", "RUN", "DANCE"]:
            self.action_mode = "MIMIC"

        # 2. Execute Dynamic Action Routines
        self.action_time += dt

        if self.action_mode == "JUMP":
            self._exec_jump_action()
        elif self.action_mode == "DANCE":
            self._exec_dance_action()
        elif self.action_mode == "RUN":
            self._exec_run_action()
        elif self.action_mode == "FIGHT":
            self._exec_fight_action()
        elif self.action_mode == "SLASH":
            self._exec_slash_action()
        elif self.action_mode == "POINT":
            self._exec_slash_action()
        elif self.action_mode == "MIMIC":
            self._exec_mimic_action(curls, pitch, yaw, roll)
        else:
            self._exec_marionette_action(curls, norm_x)

        self._critically_damped_step(stiffness=16.0, damping=9.2)

    def _exec_jump_action(self):
        """Supersonic Meteor Jump & Superhero Ground Crater Impact."""
        t = self.action_time % 1.90

        if t < 0.25:
            # Phase 1: Pre-ignition kinetic crouch
            crouch = t / 0.25
            self.target_pos[1] = self.base_target_pos[1] + 60.0 * crouch
            self.target_angles['l_hip_pitch'] = 0.95 * crouch
            self.target_angles['r_hip_pitch'] = 0.95 * crouch
            self.target_angles['l_knee_pitch'] = -1.45 * crouch
            self.target_angles['r_knee_pitch'] = -1.45 * crouch
            self.target_angles['torso_pitch'] = 0.40 * crouch
            self.target_angles['l_shoulder_pitch'] = 0.6 * crouch
            self.target_angles['r_shoulder_pitch'] = 0.6 * crouch
            self.target_angles['head_pitch'] = -0.25

            if random.random() < 0.5:
                self._spawn_thruster_plumes(intensity=1.2)

        elif t < 0.80:
            # Phase 2: Supersonic Rocket Launch!
            air_t = (t - 0.25) / 0.55
            jump_h = math.sin(air_t * math.pi) * 190.0
            self.target_pos[1] = self.base_target_pos[1] - jump_h

            self.target_angles['torso_pitch'] = -0.22
            self.target_angles['l_shoulder_pitch'] = -1.75
            self.target_angles['r_shoulder_pitch'] = -1.75
            self.target_angles['l_shoulder_roll'] = 0.25
            self.target_angles['r_shoulder_roll'] = -0.25
            self.target_angles['l_elbow_pitch'] = 0.20
            self.target_angles['r_elbow_pitch'] = 0.20

            self.target_angles['l_hip_pitch'] = -0.45
            self.target_angles['r_hip_pitch'] = 0.35
            self.target_angles['l_knee_pitch'] = -0.30
            self.target_angles['r_knee_pitch'] = -0.80

            self._spawn_thruster_plumes(intensity=2.2)

        elif t < 1.25:
            # Phase 3: Superhero 3-Point Ground Crater Impact!
            land_t = (t - 0.80) / 0.45
            self.target_pos[1] = self.base_target_pos[1] + 45.0 * (1.0 - land_t)

            self.target_angles['torso_pitch'] = 0.55 * (1.0 - land_t * 0.4)
            self.target_angles['r_shoulder_pitch'] = -1.35 * (1.0 - land_t)
            self.target_angles['r_elbow_pitch'] = 1.65
            self.target_angles['l_shoulder_pitch'] = 0.45
            self.target_angles['l_elbow_pitch'] = 0.30

            self.target_angles['l_hip_pitch'] = 1.25 * (1.0 - land_t * 0.4)
            self.target_angles['r_hip_pitch'] = 0.45
            self.target_angles['l_knee_pitch'] = -1.75 * (1.0 - land_t * 0.4)
            self.target_angles['r_knee_pitch'] = -0.75

            if t - 0.80 < 0.07:
                self._spawn_shockwave(self.screen_anchors['r_hand'], max_r=110, color=(35, 35, 235))
                self.camera_shake = 0.40

        else:
            # Phase 4: Battle Recovery
            self._reset_stance()

    def _exec_fight_action(self):
        """4-Hit Brutal Martial Arts Combat Sequence."""
        cycle = 1.8
        phase = int((self.action_time % cycle) / (cycle / 4))
        sub_t = ((self.action_time % cycle) % (cycle / 4)) / (cycle / 4)

        self.target_angles['l_hip_pitch'] = 0.40
        self.target_angles['r_hip_pitch'] = -0.30
        self.target_angles['l_knee_pitch'] = -0.75
        self.target_angles['r_knee_pitch'] = -0.90

        if phase == 0:
            # 1. Sonic Left Dash Jab & Blade Slash
            strike = math.sin(sub_t * math.pi)
            self.target_angles['l_shoulder_pitch'] = -1.75 * strike
            self.target_angles['l_shoulder_roll'] = 0.10
            self.target_angles['l_elbow_pitch'] = 0.05 + (1 - strike) * 1.4
            self.target_angles['torso_yaw'] = -0.45 * strike
            self.target_angles['torso_pitch'] = 0.20

            self.target_angles['r_shoulder_pitch'] = -0.40
            self.target_angles['r_elbow_pitch'] = 1.70

            if 0.4 < sub_t < 0.55:
                self._spawn_shockwave(self.screen_anchors['l_hand'], max_r=75, color=(35, 35, 235))
                self.camera_shake = 0.20
        elif phase == 1:
            # 2. Heavy Right 360 Spinning Cross
            spin = math.sin(sub_t * math.pi)
            self.target_angles['r_shoulder_pitch'] = -1.85 * spin
            self.target_angles['r_shoulder_roll'] = -0.15
            self.target_angles['r_elbow_pitch'] = 0.05 + (1 - spin) * 1.5
            self.target_angles['torso_yaw'] = 0.65 * spin
            self.target_angles['torso_pitch'] = 0.25

            self.target_angles['l_shoulder_pitch'] = -0.35
            self.target_angles['l_elbow_pitch'] = 1.65

            if 0.4 < sub_t < 0.55:
                self._spawn_shockwave(self.screen_anchors['r_hand'], max_r=90, color=(25, 110, 255))
                self.camera_shake = 0.28
        elif phase == 2:
            # 3. Flying Knee Jet Thruster Ram
            jump = math.sin(sub_t * math.pi)
            self.target_pos[1] = self.base_target_pos[1] - 40.0 * jump
            self.target_angles['r_hip_pitch'] = 1.25 * jump
            self.target_angles['r_knee_pitch'] = -1.70 * jump
            self.target_angles['l_hip_pitch'] = -0.50 * jump
            self.target_angles['torso_pitch'] = -0.25 * jump

            self.target_angles['l_shoulder_pitch'] = 0.45
            self.target_angles['r_shoulder_pitch'] = 0.45

            if sub_t < 0.6:
                self._spawn_thruster_plumes(intensity=1.8)
            if 0.45 < sub_t < 0.6:
                self._spawn_shockwave(self.screen_anchors['r_foot'], max_r=95, color=(35, 35, 235))
                self.camera_shake = 0.30
        else:
            # 4. Cataclysmic Dual-Gauntlet Ground Sledgehammer
            smash = math.sin(sub_t * math.pi)
            if sub_t < 0.4:
                self.target_pos[1] = self.base_target_pos[1] - 30.0
                self.target_angles['l_shoulder_pitch'] = -1.85
                self.target_angles['r_shoulder_pitch'] = -1.85
                self.target_angles['l_elbow_pitch'] = 0.8
                self.target_angles['r_elbow_pitch'] = 0.8
            else:
                self.target_pos[1] = self.base_target_pos[1] + 35.0 * smash
                self.target_angles['l_shoulder_pitch'] = -0.65
                self.target_angles['r_shoulder_pitch'] = -0.65
                self.target_angles['l_elbow_pitch'] = 1.75
                self.target_angles['r_elbow_pitch'] = 1.75
                self.target_angles['torso_pitch'] = 0.50

                if 0.45 < sub_t < 0.6:
                    self._spawn_shockwave(self.screen_anchors['r_hand'], max_r=130, color=(35, 35, 235))
                    self._spawn_shockwave(self.screen_anchors['l_hand'], max_r=130, color=(35, 35, 235))
                    self.camera_shake = 0.45

    def _exec_slash_action(self):
        """Dual Forearm Vibro-Blade Hurricane Cross-Slash."""
        cycle = 1.2
        phase = (self.action_time % cycle) / cycle
        cross = math.sin(phase * math.pi * 2)

        self.target_rot[1] = self.base_target_rot[1] + math.sin(self.action_time * 12.0) * 0.35
        self.target_angles['torso_yaw'] = math.sin(self.action_time * 12.0) * 0.45
        self.target_angles['torso_pitch'] = 0.15

        self.target_angles['l_shoulder_pitch'] = -1.25 - cross * 0.4
        self.target_angles['l_shoulder_roll'] = -0.35 + cross * 0.7
        self.target_angles['l_elbow_pitch'] = 1.35

        self.target_angles['r_shoulder_pitch'] = -1.25 + cross * 0.4
        self.target_angles['r_shoulder_roll'] = 0.35 - cross * 0.7
        self.target_angles['r_elbow_pitch'] = 1.35

        if random.random() < 0.4:
            self._spawn_shockwave(self.screen_anchors['l_hand'], max_r=65, color=(60, 40, 255))
            self._spawn_shockwave(self.screen_anchors['r_hand'], max_r=65, color=(35, 35, 235))

    def _exec_dance_action(self):
        """Cyber Robotic Breakdance & Popping."""
        beat = (self.action_time * 2.4) % 8.0
        bounce = abs(math.sin(self.action_time * 4.8)) * 16.0
        self.target_pos[1] = self.base_target_pos[1] - bounce

        sway = math.sin(self.action_time * 4.8) * 0.26
        self.target_rot[1] = self.base_target_rot[1] + sway
        self.target_angles['torso_yaw'] = sway * 0.9
        self.target_angles['head_pitch'] = math.sin(self.action_time * 9.6) * 0.22

        if beat < 2.0:
            w = beat / 2.0
            self.target_angles['l_shoulder_pitch'] = -0.4 - math.sin(w * math.pi) * 1.4
            self.target_angles['l_elbow_pitch'] = 0.3 + math.sin(w * math.pi * 1.3) * 1.3
            self.target_angles['r_shoulder_pitch'] = -0.3 - math.sin(max(0, w - 0.25) * math.pi) * 1.4
            self.target_angles['r_elbow_pitch'] = 0.3 + math.sin(max(0, w - 0.25) * math.pi) * 1.3
        elif beat < 4.0:
            self.target_angles['r_shoulder_pitch'] = -1.75
            self.target_angles['r_elbow_pitch'] = 0.15
            self.target_angles['l_shoulder_pitch'] = 0.25
            self.target_angles['l_elbow_pitch'] = 1.45
            self.target_angles['r_hip_pitch'] = -0.65
            self.target_angles['r_knee_pitch'] = -0.95
        elif beat < 6.0:
            stride = math.sin(self.action_time * 7.0)
            self.target_angles['l_hip_pitch'] = stride * 0.65
            self.target_angles['r_hip_pitch'] = -stride * 0.65
            self.target_angles['l_knee_pitch'] = -0.4 - max(0, stride) * 0.8
            self.target_angles['r_knee_pitch'] = -0.4 - max(0, -stride) * 0.8
            self.target_angles['l_shoulder_pitch'] = -0.8 + stride * 0.5
            self.target_angles['r_shoulder_pitch'] = -0.8 - stride * 0.5
        else:
            self.target_angles['l_shoulder_pitch'] = -1.25
            self.target_angles['r_shoulder_pitch'] = -1.25
            self.target_angles['l_elbow_pitch'] = 1.65
            self.target_angles['r_elbow_pitch'] = 1.65

    def _exec_run_action(self):
        """Predatory Assassin Sprint."""
        stride = self.action_time * 9.2
        self.target_rot[0] = 0.38 # 38 degree forward aerodynamic lean
        bob = abs(math.sin(stride * 2.0)) * 12.0
        self.target_pos[1] = self.base_target_pos[1] - bob

        sin_s = math.sin(stride)
        self.target_angles['l_hip_pitch'] = -sin_s * 1.05
        self.target_angles['l_knee_pitch'] = -0.35 - max(0, sin_s) * 1.45
        self.target_angles['r_hip_pitch'] = sin_s * 1.05
        self.target_angles['r_knee_pitch'] = -0.35 - max(0, -sin_s) * 1.45

        self.target_angles['l_shoulder_pitch'] = sin_s * 1.15 - 0.25
        self.target_angles['l_elbow_pitch'] = 0.95 + max(0, sin_s) * 0.6
        self.target_angles['r_shoulder_pitch'] = -sin_s * 1.15 - 0.25
        self.target_angles['r_elbow_pitch'] = 0.95 + max(0, -sin_s) * 0.6

        self.target_angles['torso_yaw'] = -sin_s * 0.22

        if random.random() < 0.4:
            self._spawn_thruster_plumes(intensity=1.2)

    def _exec_point_action(self, norm_x):
        self.target_angles['r_shoulder_pitch'] = -1.65
        self.target_angles['r_elbow_pitch'] = 0.10
        self.target_angles['l_shoulder_pitch'] = 0.1
        self.target_angles['l_elbow_pitch'] = 0.9
        self.target_angles['head_yaw'] = float(np.clip(norm_x * 0.8, -0.6, 0.6))

    def _exec_mimic_action(self, curls, pitch, yaw, roll):
        self.target_angles['l_shoulder_pitch'] = -0.3 - (1.0 - curls['index']) * 1.6
        self.target_angles['r_shoulder_pitch'] = -0.3 - (1.0 - curls['ring']) * 1.6
        self.target_angles['torso_pitch'] = -pitch * 0.6
        self.target_angles['head_yaw'] = yaw * 0.8

    def _exec_marionette_action(self, curls, norm_x):
        idx_pull = 1.0 - curls['index']
        mid_pull = 1.0 - curls['middle']
        ring_pull = 1.0 - curls['ring']
        pinky_pull = 1.0 - curls['pinky']
        thumb_pull = 1.0 - curls['thumb']

        self.target_angles['l_shoulder_pitch'] = -0.25 - idx_pull * 1.55
        self.target_angles['l_elbow_pitch'] = 0.35 + idx_pull * 0.95
        self.target_angles['l_shoulder_roll'] = 0.25 + thumb_pull * 0.35

        self.target_angles['torso_pitch'] = -0.1 + (1.0 - mid_pull) * 0.3
        self.target_angles['head_pitch'] = -0.15 + mid_pull * 0.35
        self.target_angles['head_yaw'] = float(np.clip(norm_x * 0.6, -0.4, 0.4))

        self.target_angles['r_shoulder_pitch'] = -0.2 - ring_pull * 1.4
        self.target_angles['r_elbow_pitch'] = 0.35 + ring_pull * 0.8

        self.target_angles['r_hip_pitch'] = -0.25 - pinky_pull * 0.75
        self.target_angles['r_knee_pitch'] = -0.3 - pinky_pull * 0.95

    def _reset_stance(self):
        self.target_pos[1] = self.base_target_pos[1]
        self.target_angles['l_hip_pitch'] = 0.20
        self.target_angles['r_hip_pitch'] = -0.25
        self.target_angles['l_knee_pitch'] = -0.40
        self.target_angles['r_knee_pitch'] = -0.50
        self.target_angles['l_shoulder_pitch'] = -0.35
        self.target_angles['r_shoulder_pitch'] = -0.25
        self.target_angles['torso_pitch'] = 0.05
        self.target_angles['head_pitch'] = 0.08

    def _critically_damped_step(self, stiffness=16.0, damping=9.2):
        dt = 0.033
        # Spring-damper for position
        f_p = (self.target_pos - self.pos) * stiffness - self.vel_pos * damping
        self.vel_pos += f_p * dt
        self.pos += self.vel_pos * dt

        # Spring-damper for body rotation
        f_r = (self.target_rot - self.rot) * stiffness - self.vel_rot * damping
        self.vel_rot += f_r * dt
        self.rot += self.vel_rot * dt

        # Spring-damper for each joint
        for k in self.target_angles:
            f_a = (self.target_angles[k] - self.angles[k]) * stiffness - self.vel_angles[k] * damping
            self.vel_angles[k] += f_a * dt
            self.angles[k] += self.vel_angles[k] * dt

    def _spawn_thruster_plumes(self, intensity=1.5):
        pt_l = self.string_anchors['chest'] + np.array([12, 28, 20])
        pt_r = self.string_anchors['chest'] + np.array([-12, 28, 20])
        for origin in [pt_l, pt_r]:
            for _ in range(int(2 * intensity)):
                self.particles.append({
                    'pos': np.copy(origin) + np.random.uniform(-4, 4, 3),
                    'vel': np.array([np.random.uniform(-1, 1), np.random.uniform(4, 9), np.random.uniform(-1, 1)], dtype=np.float32),
                    'color': self.COLOR_THRUSTER_FIRE,
                    'life': 1.0,
                    'max_life': 1.0,
                    'size': random.uniform(3.5, 7.5) * intensity
                })

    def _spawn_shockwave(self, center_px, max_r=90, color=(35, 35, 235)):
        self.shockwaves.append({
            'center': (int(center_px[0]), int(center_px[1])),
            'r': 5.0,
            'max_r': float(max_r),
            'life': 1.0,
            'color': color
        })

    def _update_fx(self, dt):
        # Update particles
        alive_p = []
        for p in self.particles:
            p['life'] -= dt * 2.8
            p['pos'] += p['vel']
            p['size'] = max(1.0, p['size'] - 0.15)
            if p['life'] > 0:
                alive_p.append(p)
        self.particles = alive_p

        # Update shockwaves
        alive_w = []
        for w in self.shockwaves:
            w['life'] -= dt * 2.2
            w['r'] += (w['max_r'] - w['r']) * 0.28
            if w['life'] > 0:
                alive_w.append(w)
        self.shockwaves = alive_w

        # Decay camera shake
        if self.camera_shake > 0.01:
            self.camera_shake *= 0.85
        else:
            self.camera_shake = 0.0

    def render(self, frame, cam_fx=750.0, cam_fy=750.0, wireframe_mode=False):
        """Renders realistic 3D mecha robot with red/dark/white palette, demon horns, vibro-blades & VFX."""
        h, w, _ = frame.shape
        cx, cy = w / 2.0, h / 2.0
        s = self.scale

        # Apply screen shake offset if active
        if self.camera_shake > 0.02:
            cx += (random.random() - 0.5) * self.camera_shake * 30.0
            cy += (random.random() - 0.5) * self.camera_shake * 30.0

        polygons_to_draw = []

        # Root world transform
        R_world = rot_y(self.rot[1]) @ rot_x(self.rot[0]) @ rot_z(self.rot[2])
        T_world = self.pos

        def transform_part(verts, R_local, T_local, parent_R, parent_T):
            R_combined = parent_R @ R_local
            T_combined = parent_T + (parent_R @ T_local)
            w_verts = (R_combined @ verts.T).T + T_combined
            return w_verts, R_combined, T_combined

        def project(v3d):
            z = max(v3d[2], 1.0)
            px = int((v3d[0] * cam_fx / z) + cx)
            py = int((v3d[1] * cam_fy / z) + cy)
            return (px, py), z

        def shade_color(base_bgr, normal, R_mat):
            world_norm = R_mat @ normal
            world_norm /= (np.linalg.norm(world_norm) + 1e-6)
            dot = np.dot(world_norm, self.light_dir)

            diffuse = 0.45 + 0.55 * max(0.0, float(dot))
            rim = max(0.0, float(np.dot(world_norm, self.rim_dir))) ** 3 * 0.40

            view_dir = np.array([0, 0, -1], dtype=np.float32)
            half = self.light_dir + view_dir
            half /= (np.linalg.norm(half) + 1e-6)
            spec = max(0.0, float(np.dot(world_norm, half))) ** 12 * 0.50

            b = np.clip(base_bgr[0] * diffuse + (spec + rim) * 255, 0, 255)
            g = np.clip(base_bgr[1] * diffuse + (spec + rim) * 255, 0, 255)
            r = np.clip(base_bgr[2] * diffuse + (spec + rim) * 255, 0, 255)
            return (int(b), int(g), int(r))

        # --- 1. PELVIS & SPINAL COLUMN ---
        R_p = np.eye(3, dtype=np.float32)
        T_p = np.zeros(3, dtype=np.float32)
        w_pelvis, R_p_w, T_p_w = transform_part(self.m_pelvis_v, R_p, T_p, R_world, T_world)
        w_groin, _, _ = transform_part(self.m_groin_v, np.eye(3), np.array([0, 2*s, -10*s]), R_p_w, T_p_w)
        w_hip_sk_l, _, _ = transform_part(self.m_hip_skirt_v, rot_z(-0.2), np.array([17*s, 2*s, 0]), R_p_w, T_p_w)
        w_hip_sk_r, _, _ = transform_part(self.m_hip_skirt_v, rot_z(0.2), np.array([-17*s, 2*s, 0]), R_p_w, T_p_w)

        w_sp1, R_sp1_w, T_sp1_w = transform_part(self.m_spine1_v, np.eye(3), np.array([0, -7*s, 0]), R_p_w, T_p_w)
        w_sp2, R_sp2_w, T_sp2_w = transform_part(self.m_spine2_v, np.eye(3), np.array([0, -6*s, 0]), R_sp1_w, T_sp1_w)
        w_sp3, R_sp3_w, T_sp3_w = transform_part(self.m_spine3_v, np.eye(3), np.array([0, -6*s, 0]), R_sp2_w, T_sp2_w)

        w_pis_l, _, _ = transform_part(self.m_piston_casing_v, np.eye(3), np.array([10*s, -7*s, 0]), R_p_w, T_p_w)
        w_pis_r, _, _ = transform_part(self.m_piston_casing_v, np.eye(3), np.array([-10*s, -7*s, 0]), R_p_w, T_p_w)

        # --- 2. TORSO, REACTOR & DORSAL JETPACK ---
        R_torso = rot_x(self.angles['torso_pitch']) @ rot_y(self.angles['torso_yaw']) @ rot_z(self.angles['torso_roll'])
        T_torso = np.array([0, -18*s, 0], dtype=np.float32)
        w_chest, R_c_w, T_c_w = transform_part(self.m_chest_v, R_torso, T_torso, R_sp3_w, T_sp3_w)
        self.string_anchors['chest'] = T_c_w

        # Crimson Chevrons
        w_chev_l, _, _ = transform_part(self.m_chev_v, np.eye(3), np.array([10*s, -6*s, -12*s]), R_c_w, T_c_w)
        w_chev_r, _, _ = transform_part(self.m_chev_v, np.eye(3), np.array([-10*s, -6*s, -12*s]), R_c_w, T_c_w)

        w_collar, _, _ = transform_part(self.m_collar_v, np.eye(3), np.array([0, -16*s, 0]), R_c_w, T_c_w)

        # Pulsating Crimson Singularity Core
        w_core_out, _, _ = transform_part(self.m_core_outer_v, rot_x(math.pi/2), np.array([0, 2*s, -13*s]), R_c_w, T_c_w)
        w_core, _, _ = transform_part(self.m_core_v, rot_x(math.pi/2), np.array([0, 2*s, -14*s]), R_c_w, T_c_w)

        w_vent_l, _, _ = transform_part(self.m_vent_v, rot_y(-0.3), np.array([18*s, 0, -4*s]), R_c_w, T_c_w)
        w_vent_r, _, _ = transform_part(self.m_vent_v, rot_y(0.3), np.array([-18*s, 0, -4*s]), R_c_w, T_c_w)

        w_pack, R_pk_w, T_pk_w = transform_part(self.m_pack_v, np.eye(3), np.array([0, -4*s, 15*s]), R_c_w, T_c_w)
        w_noz_l, _, _ = transform_part(self.m_nozzle_v, rot_x(-0.2), np.array([8*s, 14*s, 2*s]), R_pk_w, T_pk_w)
        w_noz_r, _, _ = transform_part(self.m_nozzle_v, rot_x(-0.2), np.array([-8*s, 14*s, 2*s]), R_pk_w, T_pk_w)
        w_wing_l, _, _ = transform_part(self.m_pack_wing_v, rot_z(-0.4), np.array([16*s, -10*s, 0]), R_pk_w, T_pk_w)
        w_wing_r, _, _ = transform_part(self.m_pack_wing_v, rot_z(0.4), np.array([-16*s, -10*s, 0]), R_pk_w, T_pk_w)

        # --- 3. HEAD: DEMON WAR HELMET & PREDATORY OPTICS ---
        R_head = rot_y(self.angles['head_yaw']) @ rot_x(self.angles['head_pitch']) @ rot_z(self.angles['head_roll'])
        T_head = np.array([0, -28*s, 2*s], dtype=np.float32)

        w_neck, R_nk_w, T_nk_w = transform_part(self.m_neck_v, np.eye(3), np.array([0, -18*s, 2*s]), R_c_w, T_c_w)
        w_neck_p, _, _ = transform_part(self.m_neck_piston_v, np.eye(3), np.array([0, 0, -4*s]), R_nk_w, T_nk_w)

        w_head, R_h_w, T_h_w = transform_part(self.m_head_v, R_head, T_head, R_c_w, T_c_w)
        self.string_anchors['head'] = T_h_w + (R_h_w @ np.array([0, -14*s, 0]))

        # Helmet crest
        w_crest, _, _ = transform_part(self.m_crest_v, np.eye(3), np.array([0, -10*s, -1*s]), R_h_w, T_h_w)
        
        # SWEPT-BACK CRIMSON DEMON HORNS
        w_horn_l, _, _ = transform_part(self.m_horn_l_v, rot_z(-0.35) @ rot_x(-0.45), np.array([11*s, -8*s, 2*s]), R_h_w, T_h_w)
        w_horn_r, _, _ = transform_part(self.m_horn_r_v, rot_z(0.35) @ rot_x(-0.45), np.array([-11*s, -8*s, 2*s]), R_h_w, T_h_w)

        # PREDATORY INWARD-SLANTED RED OPTIC SLITS
        w_eye_l, _, _ = transform_part(self.m_eye_l_v, rot_z(-0.25) @ rot_y(0.15), np.array([4.5*s, 1*s, -11.5*s]), R_h_w, T_h_w)
        w_eye_r, _, _ = transform_part(self.m_eye_r_v, rot_z(0.25) @ rot_y(-0.15), np.array([-4.5*s, 1*s, -11.5*s]), R_h_w, T_h_w)

        w_chin, _, _ = transform_part(self.m_chin_v, np.eye(3), np.array([0, 10*s, -7*s]), R_h_w, T_h_w)

        # --- 4. ARMS, SPIKED PAULDRONS & FOREARM VIBRO-BLADES ---
        # LEFT ARM
        R_l_sh = rot_x(self.angles['l_shoulder_pitch']) @ rot_z(self.angles['l_shoulder_roll'])
        T_l_sh = np.array([28*s, -12*s, 0], dtype=np.float32)
        w_l_pau, R_l_sh_w, T_l_sh_w = transform_part(self.m_shoulder_v, R_l_sh, T_l_sh, R_c_w, T_c_w)
        w_l_pau_spk, _, _ = transform_part(self.m_pau_spike_v, rot_z(-math.pi/2) @ rot_x(0.2), np.array([14*s, 2*s, 0]), R_l_sh_w, T_l_sh_w)
        w_l_bic, _, _ = transform_part(self.m_bicep_v, np.eye(3), np.array([0, 13*s, 0]), R_l_sh_w, T_l_sh_w)

        R_l_el = rot_x(self.angles['l_elbow_pitch'])
        T_l_el = np.array([0, 24*s, 0], dtype=np.float32)
        w_l_el, R_l_el_w, T_l_el_w = transform_part(self.m_elbow_v, R_l_el, T_l_el, R_l_sh_w, T_l_sh_w)
        w_l_fore, _, _ = transform_part(self.m_forearm_v, np.eye(3), np.array([0, 14*s, 0]), R_l_el_w, T_l_el_w)
        
        # FOREARM LASER VIBRO-BLADE
        w_l_blade, _, _ = transform_part(self.m_blade_v, rot_x(-0.25), np.array([7*s, 20*s, 4*s]), R_l_el_w, T_l_el_w)

        w_l_hand, _, T_l_h_w = transform_part(self.m_hand_v, np.eye(3), np.array([0, 28*s, 0]), R_l_el_w, T_l_el_w)
        self.string_anchors['l_hand'] = T_l_h_w

        # RIGHT ARM
        R_r_sh = rot_x(self.angles['r_shoulder_pitch']) @ rot_z(self.angles['r_shoulder_roll'])
        T_r_sh = np.array([-28*s, -12*s, 0], dtype=np.float32)
        w_r_pau, R_r_sh_w, T_r_sh_w = transform_part(self.m_shoulder_v, R_r_sh, T_r_sh, R_c_w, T_c_w)
        w_r_pau_spk, _, _ = transform_part(self.m_pau_spike_v, rot_z(math.pi/2) @ rot_x(0.2), np.array([-14*s, 2*s, 0]), R_r_sh_w, T_r_sh_w)
        w_r_bic, _, _ = transform_part(self.m_bicep_v, np.eye(3), np.array([0, 13*s, 0]), R_r_sh_w, T_r_sh_w)

        R_r_el = rot_x(self.angles['r_elbow_pitch'])
        T_r_el = np.array([0, 24*s, 0], dtype=np.float32)
        w_r_el, R_r_el_w, T_r_el_w = transform_part(self.m_elbow_v, R_r_el, T_r_el, R_r_sh_w, T_r_sh_w)
        w_r_fore, _, _ = transform_part(self.m_forearm_v, np.eye(3), np.array([0, 14*s, 0]), R_r_el_w, T_r_el_w)
        
        # FOREARM LASER VIBRO-BLADE
        w_r_blade, _, _ = transform_part(self.m_blade_v, rot_x(-0.25), np.array([-7*s, 20*s, 4*s]), R_r_el_w, T_r_el_w)

        w_r_hand, _, T_r_h_w = transform_part(self.m_hand_v, np.eye(3), np.array([0, 28*s, 0]), R_r_el_w, T_r_el_w)
        self.string_anchors['r_hand'] = T_r_h_w

        # --- 5. LEGS, SPIKED KNEES & COMBAT SABATONS ---
        # LEFT LEG
        R_l_hp = rot_x(self.angles['l_hip_pitch']) @ rot_z(self.angles['l_hip_roll'])
        T_l_hp = np.array([16*s, 8*s, 0], dtype=np.float32)
        w_l_thi, R_l_hp_w, T_l_hp_w = transform_part(self.m_thigh_v, R_l_hp, T_l_hp, R_p_w, T_p_w)

        R_l_kn = rot_x(self.angles['l_knee_pitch'])
        T_l_kn = np.array([0, 36*s, 0], dtype=np.float32)
        w_l_kn, R_l_kn_w, T_l_kn_w = transform_part(self.m_knee_v, R_l_kn, T_l_kn, R_l_hp_w, T_l_hp_w)
        # Forward-jutting knee ram spike
        w_l_kn_spk, _, _ = transform_part(self.m_knee_spike_v, rot_x(math.pi/2), np.array([0, 0, -8*s]), R_l_kn_w, T_l_kn_w)
        w_l_shin, _, _ = transform_part(self.m_shin_v, np.eye(3), np.array([0, 18*s, 0]), R_l_kn_w, T_l_kn_w)

        w_l_ft, _, T_l_ft_w = transform_part(self.m_foot_v, np.eye(3), np.array([0, 38*s, 4*s]), R_l_kn_w, T_l_kn_w)
        w_l_toe, _, _ = transform_part(self.m_toe_claw_v, np.eye(3), np.array([0, 40*s, -12*s]), R_l_kn_w, T_l_kn_w)
        self.string_anchors['l_foot'] = T_l_ft_w

        # RIGHT LEG
        R_r_hp = rot_x(self.angles['r_hip_pitch']) @ rot_z(self.angles['r_hip_roll'])
        T_r_hp = np.array([-16*s, 8*s, 0], dtype=np.float32)
        w_r_thi, R_r_hp_w, T_r_hp_w = transform_part(self.m_thigh_v, R_r_hp, T_r_hp, R_p_w, T_p_w)

        R_r_kn = rot_x(self.angles['r_knee_pitch'])
        T_r_kn = np.array([0, 36*s, 0], dtype=np.float32)
        w_r_kn, R_r_kn_w, T_r_kn_w = transform_part(self.m_knee_v, R_r_kn, T_r_kn, R_r_hp_w, T_r_hp_w)
        # Forward-jutting knee ram spike
        w_r_kn_spk, _, _ = transform_part(self.m_knee_spike_v, rot_x(math.pi/2), np.array([0, 0, -8*s]), R_r_kn_w, T_r_kn_w)
        w_r_shin, _, _ = transform_part(self.m_shin_v, np.eye(3), np.array([0, 18*s, 0]), R_r_kn_w, T_r_kn_w)

        w_r_ft, _, T_r_ft_w = transform_part(self.m_foot_v, np.eye(3), np.array([0, 38*s, 4*s]), R_r_kn_w, T_r_kn_w)
        w_r_toe, _, _ = transform_part(self.m_toe_claw_v, np.eye(3), np.array([0, 40*s, -12*s]), R_r_kn_w, T_r_kn_w)
        self.string_anchors['r_foot'] = T_r_ft_w

        # Project screen anchors
        for k in self.string_anchors:
            self.screen_anchors[k], _ = project(self.string_anchors[k])

        # Subsystem meshes to queue for depth sorting
        all_parts = [
            (w_pelvis, self.m_pelvis_f, R_p_w, self.COLOR_CARBON),
            (w_groin, self.m_groin_f, R_p_w, self.COLOR_CRIMSON_RED),
            (w_hip_sk_l, self.m_hip_skirt_f, R_p_w, self.COLOR_ARMOR_WHITE),
            (w_hip_sk_r, self.m_hip_skirt_f, R_p_w, self.COLOR_ARMOR_WHITE),
            (w_sp1, self.m_spine1_f, R_sp1_w, self.COLOR_CARBON),
            (w_sp2, self.m_spine2_f, R_sp2_w, self.COLOR_JOINT_CHROME),
            (w_sp3, self.m_spine3_f, R_sp3_w, self.COLOR_CARBON),
            (w_pis_l, self.m_piston_casing_f, R_p_w, self.COLOR_CARBON),
            (w_pis_r, self.m_piston_casing_f, R_p_w, self.COLOR_CARBON),

            (w_chest, self.m_chest_f, R_c_w, self.COLOR_ARMOR_WHITE),
            (w_chev_l, self.m_chev_f, R_c_w, self.COLOR_CRIMSON_RED),
            (w_chev_r, self.m_chev_f, R_c_w, self.COLOR_CRIMSON_RED),
            (w_collar, self.m_collar_f, R_c_w, self.COLOR_ARMOR_WHITE),
            (w_core_out, self.m_core_outer_f, R_c_w, self.COLOR_CRIMSON_RED),
            (w_core, self.m_core_f, R_c_w, self.COLOR_CORE_RED),
            (w_vent_l, self.m_vent_f, R_c_w, self.COLOR_CARBON),
            (w_vent_r, self.m_vent_f, R_c_w, self.COLOR_CARBON),
            (w_pack, self.m_pack_f, R_pk_w, self.COLOR_CARBON),
            (w_noz_l, self.m_nozzle_f, R_pk_w, self.COLOR_ARMOR_DARK),
            (w_noz_r, self.m_nozzle_f, R_pk_w, self.COLOR_ARMOR_DARK),
            (w_wing_l, self.m_pack_wing_f, R_pk_w, self.COLOR_CRIMSON_RED),
            (w_wing_r, self.m_pack_wing_f, R_pk_w, self.COLOR_CRIMSON_RED),

            (w_neck, self.m_neck_f, R_nk_w, self.COLOR_JOINT_CHROME),
            (w_neck_p, self.m_neck_piston_f, R_nk_w, self.COLOR_CHROME_BRIGHT),
            (w_head, self.m_head_f, R_h_w, self.COLOR_ARMOR_WHITE),
            (w_crest, self.m_crest_f, R_h_w, self.COLOR_CRIMSON_RED),
            (w_horn_l, self.m_horn_l_f, R_h_w, self.COLOR_CRIMSON_RED),
            (w_horn_r, self.m_horn_r_f, R_h_w, self.COLOR_CRIMSON_RED),
            (w_eye_l, self.m_eye_l_f, R_h_w, self.COLOR_VISOR_RED),
            (w_eye_r, self.m_eye_r_f, R_h_w, self.COLOR_VISOR_RED),
            (w_chin, self.m_chin_f, R_h_w, self.COLOR_CARBON),

            (w_l_pau, self.m_shoulder_f, R_l_sh_w, self.COLOR_ARMOR_WHITE),
            (w_l_pau_spk, self.m_pau_spike_f, R_l_sh_w, self.COLOR_CRIMSON_RED),
            (w_l_bic, self.m_bicep_f, R_l_sh_w, self.COLOR_ARMOR_WHITE),
            (w_l_el, self.m_elbow_f, R_l_el_w, self.COLOR_JOINT_CHROME),
            (w_l_fore, self.m_forearm_f, R_l_el_w, self.COLOR_ARMOR_WHITE),
            (w_l_blade, self.m_blade_f, R_l_el_w, self.COLOR_BLADE_RED),
            (w_l_hand, self.m_hand_f, R_l_el_w, self.COLOR_CARBON),

            (w_r_pau, self.m_shoulder_f, R_r_sh_w, self.COLOR_ARMOR_WHITE),
            (w_r_pau_spk, self.m_pau_spike_f, R_r_sh_w, self.COLOR_CRIMSON_RED),
            (w_r_bic, self.m_bicep_f, R_r_sh_w, self.COLOR_ARMOR_WHITE),
            (w_r_el, self.m_elbow_f, R_r_el_w, self.COLOR_JOINT_CHROME),
            (w_r_fore, self.m_forearm_f, R_r_el_w, self.COLOR_ARMOR_WHITE),
            (w_r_blade, self.m_blade_f, R_r_el_w, self.COLOR_BLADE_RED),
            (w_r_hand, self.m_hand_f, R_r_el_w, self.COLOR_CARBON),

            (w_l_thi, self.m_thigh_f, R_l_hp_w, self.COLOR_ARMOR_WHITE),
            (w_l_kn, self.m_knee_f, R_l_kn_w, self.COLOR_JOINT_CHROME),
            (w_l_kn_spk, self.m_knee_spike_f, R_l_kn_w, self.COLOR_CRIMSON_RED),
            (w_l_shin, self.m_shin_f, R_l_kn_w, self.COLOR_ARMOR_WHITE),
            (w_l_ft, self.m_foot_f, R_l_kn_w, self.COLOR_CARBON),
            (w_l_toe, self.m_toe_claw_f, R_l_kn_w, self.COLOR_CRIMSON_RED),

            (w_r_thi, self.m_thigh_f, R_r_hp_w, self.COLOR_ARMOR_WHITE),
            (w_r_kn, self.m_knee_f, R_r_kn_w, self.COLOR_JOINT_CHROME),
            (w_r_kn_spk, self.m_knee_spike_f, R_r_kn_w, self.COLOR_CRIMSON_RED),
            (w_r_shin, self.m_shin_f, R_r_kn_w, self.COLOR_ARMOR_WHITE),
            (w_r_ft, self.m_foot_f, R_r_kn_w, self.COLOR_CARBON),
            (w_r_toe, self.m_toe_claw_f, R_r_kn_w, self.COLOR_CRIMSON_RED),
        ]

        # Extract polygons for painter's depth-sorting
        for w_verts, faces, R_mat, default_col in all_parts:
            for face_indices, normal, face_color in faces:
                pts_3d = w_verts[face_indices]
                pts_2d = []
                avg_z = 0.0
                for v3 in pts_3d:
                    p2, z = project(v3)
                    pts_2d.append(p2)
                    avg_z += z
                avg_z /= max(1, len(pts_3d))

                # Backface culling
                p0, p1, p2 = pts_3d[0], pts_3d[1], pts_3d[2]
                fn = compute_normal(p0, p1, p2)
                cam_to_face = p0 - np.array([0, 0, 0])
                if np.dot(fn, cam_to_face) < 0:
                    shaded = shade_color(face_color, normal, R_mat)
                    polygons_to_draw.append((avg_z, np.array(pts_2d, dtype=np.int32), shaded))

        # Sort polygons back to front (largest Z to smallest)
        polygons_to_draw.sort(key=lambda item: item[0], reverse=True)

        # Draw shaded polygons
        for avg_z, poly_pts, col in polygons_to_draw:
            if wireframe_mode:
                cv2.polylines(frame, [poly_pts], isClosed=True, color=(100, 240, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.fillPoly(frame, [poly_pts], color=col, lineType=cv2.LINE_AA)
                cv2.polylines(frame, [poly_pts], isClosed=True, color=self.COLOR_OUTLINE, thickness=1, lineType=cv2.LINE_AA)

        # Draw dynamic particle plumes & sparks
        for p in self.particles:
            pp, _ = project(p['pos'])
            alpha = p['life']
            r = max(1, int(p['size'] * alpha))
            col = tuple(int(c * alpha) for c in p['color'])
            cv2.circle(frame, pp, r, col, -1, lineType=cv2.LINE_AA)

        # Draw glowing shockwaves
        for sw in self.shockwaves:
            r = int(sw['r'])
            alpha = sw['life']
            col = tuple(int(c * alpha) for c in sw['color'])
            if r > 1:
                cv2.circle(frame, sw['center'], r, col, 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, sw['center'], max(1, r - 5), tuple(int(c * alpha * 0.6) for c in sw['color']), 1, lineType=cv2.LINE_AA)
