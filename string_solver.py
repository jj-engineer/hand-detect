"""
Dynamic Laser Tension String Solver & Marionette Physics module.
Generates ultra-fine, radiant tension filaments connecting fingertips to robot joints
with catenary/bezier physics, micro-vibrations, and traveling cyber-energy sparks.
"""

import math
import random
import numpy as np
import cv2


class TensionFilament:
    def __init__(self, finger_name, robot_anchor_name, color_core=(70, 240, 255), color_glow=(15, 150, 255)):
        self.finger_name = finger_name
        self.robot_anchor_name = robot_anchor_name
        self.color_core = color_core
        self.color_glow = color_glow
        self.particles = [random.random() for _ in range(3)]

    def update_particles(self, speed=0.035):
        for i in range(len(self.particles)):
            self.particles[i] = (self.particles[i] + speed) % 1.0


class StringSolver:
    def __init__(self):
        self.filaments = [
            TensionFilament('thumb',  'l_hand', color_core=(120, 140, 255), color_glow=(20, 20, 240)),
            TensionFilament('index',  'head',   color_core=(150, 170, 255), color_glow=(30, 30, 255)),
            TensionFilament('middle', 'chest',  color_core=(150, 170, 255), color_glow=(30, 30, 255)),
            TensionFilament('ring',   'r_hand', color_core=(120, 140, 255), color_glow=(20, 20, 240)),
            TensionFilament('pinky',  'r_foot', color_core=(100, 120, 255), color_glow=(15, 15, 230)),
        ]
        self.time = 0.0

    def solve_bezier_curve(self, p_start, p_end, tension=0.5, sag_bias=45.0, num_points=26):
        """Calculates smooth cubic bezier points with tension sagging."""
        p0 = np.array(p_start, dtype=np.float32)
        p3 = np.array(p_end, dtype=np.float32)

        dx = p3[0] - p0[0]
        dy = p3[1] - p0[1]
        dist = math.hypot(dx, dy)

        sag_amount = (1.0 - np.clip(tension, 0.0, 1.0)) * (dist * 0.25 + sag_bias)
        wave = math.sin(self.time * 7.5 + dist * 0.015) * (1.2 + (1.0 - tension) * 3.0)

        ctrl1 = p0 + np.array([dx * 0.35, dy * 0.35 + sag_amount + wave], dtype=np.float32)
        ctrl2 = p0 + np.array([dx * 0.70, dy * 0.70 + sag_amount * 0.75 + wave], dtype=np.float32)

        t = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
        curve = (
            ((1.0 - t) ** 3) * p0 +
            (3.0 * ((1.0 - t) ** 2) * t) * ctrl1 +
            (3.0 * (1.0 - t) * (t ** 2)) * ctrl2 +
            (t ** 3) * p3
        )
        return curve.astype(np.int32)

    def draw(self, frame, hand_data, robot_screen_anchors):
        """Draws glowing laser filaments, soft aura bloom, and streaming pulse sparks."""
        if hand_data is None:
            return

        self.time += 0.04
        curls = hand_data['finger_curls']
        tips = hand_data['tips']

        overlay = frame.copy()

        for f in self.filaments:
            f.update_particles(speed=0.03)
            finger_pt = tips.get(f.finger_name)
            robot_pt = robot_screen_anchors.get(f.robot_anchor_name)

            if finger_pt is None or robot_pt is None:
                continue

            h, w, _ = frame.shape
            if not (-120 <= robot_pt[0] <= w + 120 and -120 <= robot_pt[1] <= h + 120):
                continue

            finger_curl = curls.get(f.finger_name, 0.5)
            tension = np.clip(1.0 - finger_curl * 0.85, 0.1, 0.95)

            # Elegant 3-strand laser tendon bundle
            offsets = [-8.0, 0.0, 8.0]
            for sag_offset in offsets:
                curve = self.solve_bezier_curve(
                    finger_pt,
                    robot_pt,
                    tension=tension,
                    sag_bias=35.0 + sag_offset,
                    num_points=24
                )

                # Soft outer glow
                cv2.polylines(overlay, [curve], isClosed=False, color=f.color_glow, thickness=3, lineType=cv2.LINE_AA)
                # Radiant core line
                cv2.polylines(frame, [curve], isClosed=False, color=f.color_core, thickness=1, lineType=cv2.LINE_AA)

                # Traveling energy pulses
                for t_val in f.particles:
                    p_idx = int(t_val * (len(curve) - 1))
                    spark_pt = tuple(curve[p_idx])
                    cv2.circle(frame, spark_pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # Joint anchor ring at robot attachment point
            r_pt = (int(robot_pt[0]), int(robot_pt[1]))
            cv2.circle(frame, r_pt, 4, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, r_pt, 8, (245, 245, 250), 1, cv2.LINE_AA)

        # Alpha blend the glowing string aura
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
