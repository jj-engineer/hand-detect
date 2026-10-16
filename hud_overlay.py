"""
Clean, Modern Sci-Fi Hologram HUD & Vector Telemetry Overlay.
Upgraded from boxy retro style into a sleek, minimalist cyber-glass interface:
- Modern Glassmorphism Terminal Card with translucent obsidian background
- Dynamic Circular Vector Reticle & Azimuth brackets around 3D Robot
- Gradient Tension Bus Meters & Neon Status Pills
- High-precision telemetry metrics
"""

import time
import math
import numpy as np
import cv2


class HudOverlay:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()

        self.terminal_logs = [
            ("kit@vector:~$ ./agent --watch", (70, 180, 255)),
            ("[ok] neural link active [LATENCY: 0.8ms]", (100, 255, 160)),
            ("status: string_solver active [60Hz SYNC]", (0, 210, 255)),
            ("running string_solver (5 tension threads)", (40, 200, 255)),
            ("running gesture_compiler****** ok", (110, 255, 160)),
            ("IDEA 12: puppets that dream when idle", (60, 190, 255)),
            ("lifting: migrate memory store (96GB)", (210, 225, 245)),
            ("**", (220, 220, 220)),
            ("done - 98GB in 2.3s", (110, 255, 160)),
            ("lifting: compact heap (1.8G)", (210, 225, 245)),
            ("done - 1.8G in 1.5s", (110, 255, 160)),
            ("lifting: train epoch 12/40 (512 batches)", (210, 225, 245)),
            ("done - 512 batches in 5.2s", (110, 255, 160)),
            ("running puppet_daemon****** ok", (110, 255, 160)),
            ("IDEA 13: puppets that dream when idle", (60, 190, 255)),
        ]

    def update_fps(self):
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 0.5:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.last_fps_time = now

    def draw(self, frame, hand_data, robot, mode_name="MARIONETTE"):
        self.update_fps()
        h, w, _ = frame.shape
        t = time.time() - self.start_time

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_mono = cv2.FONT_HERSHEY_PLAIN

        # --- 1. MODERN SLEEK TERMINAL CARD (Top-Right) ---
        card_w = 440
        card_h = 300
        card_x = w - card_w - 24
        card_y = 20

        overlay = frame.copy()
        # Soft dark obsidian glass
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (14, 12, 20), -1)
        # Sleek minimal outline with soft cyan-violet glow
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (180, 150, 220), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Header tag
        cv2.putText(frame, "kit@vector:~$ ./agent --watch", (card_x + 16, card_y + 24), font, 0.45, (60, 180, 255), 1, cv2.LINE_AA)

        # Status badge line
        if hand_data is not None:
            track_str = "[OK] HAND TRACKED // LINK ACTIVE"
            track_col = (110, 255, 150)
        else:
            track_str = "[--] SEARCHING HAND TELEMETRY"
            track_col = (70, 150, 255)
        cv2.putText(frame, track_str, (card_x + 16, card_y + 46), font, 0.42, track_col, 1, cv2.LINE_AA)

        # Mode line
        cv2.putText(frame, f"SOLVER: {mode_name}", (card_x + 16, card_y + 66), font, 0.40, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (card_x + 12, card_y + 76), (card_x + card_w - 12, card_y + 76), (80, 70, 100), 1)

        # Dynamic log lines
        start_y = card_y + 94
        for i, (text, col) in enumerate(self.terminal_logs[3:13]):
            ly = start_y + (i * 18)
            cv2.putText(frame, text, (card_x + 16, ly), font_mono, 0.9, col, 1, cv2.LINE_AA)

        # Blinking cursor
        if int(t * 3.5) % 2 == 0:
            cursor_y = start_y + (10 * 18)
            cv2.putText(frame, "> _", (card_x + 16, cursor_y), font_mono, 0.95, (50, 230, 255), 1, cv2.LINE_AA)

        # --- 2. VECTOR TARGETING RETICLE & RADAR AROUND ROBOT ---
        bar_h = 44
        xs = [pt[0] for pt in robot.screen_anchors.values()]
        ys = [pt[1] for pt in robot.screen_anchors.values()]
        l = max(10, int(min(xs) - 40))
        r = min(w - 10, int(max(xs) + 40))
        top = max(10, int(min(ys) - 50))
        bot = min(h - bar_h - 20, int(max(ys) + 35))
        rx = int((l + r) / 2)
        ry = int((top + bot) / 2)

        reticle_col = (190, 140, 240)
        c_len = 24

        # Modern angled corner brackets
        cv2.line(frame, (l, top), (l + c_len, top), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (l, top), (l, top + c_len), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (r, top), (r - c_len, top), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (r, top), (r, top + c_len), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (l, bot), (l + c_len, bot), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (l, bot), (l, bot - c_len), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (r, bot), (r - c_len, bot), reticle_col, 1, cv2.LINE_AA)
        cv2.line(frame, (r, bot), (r, bot - c_len), reticle_col, 1, cv2.LINE_AA)

        # Rotating azimuth ticks
        rot_angle = t * 1.5
        cos_a, sin_a = math.cos(rot_angle), math.sin(rot_angle)
        rad = 32
        for offset_a in [0, math.pi/2, math.pi, 3*math.pi/2]:
            a = rot_angle + offset_a
            p1 = (int(rx + rad * math.cos(a)), int(ry + rad * math.sin(a)))
            p2 = (int(rx + (rad + 6) * math.cos(a)), int(ry + (rad + 6) * math.sin(a)))
            cv2.line(frame, p1, p2, (150, 110, 210), 1, cv2.LINE_AA)

        # Center crosshair mark
        cv2.circle(frame, (rx, ry), 4, (0, 220, 255), 1, cv2.LINE_AA)

        # Modern Mecha Tag
        cv2.putText(frame, "APEX-01 // PREDATOR WAR MECHA", (l + 4, top - 8), font, 0.40, (230, 200, 255), 1, cv2.LINE_AA)
        deg_yaw = int(math.degrees(robot.rot[1]))
        deg_l_arm = int(math.degrees(robot.angles['l_shoulder_pitch']))
        deg_r_arm = int(math.degrees(robot.angles['r_shoulder_pitch']))
        telem = f"YAW: {deg_yaw:+03d}* | L_ARM: {deg_l_arm:+03d}* | R_ARM: {deg_r_arm:+03d}*"
        cv2.putText(frame, telem, (l + 4, bot + 18), font_mono, 0.9, (180, 200, 230), 1, cv2.LINE_AA)

        # --- 3. SLEEK FLOATING BOTTOM HUD BAR ---
        bar_y = h - bar_h - 12
        overlay_bar = frame.copy()
        cv2.rectangle(overlay_bar, (20, bar_y), (w - 20, bar_y + bar_h), (14, 12, 20), -1)
        cv2.rectangle(overlay_bar, (20, bar_y), (w - 20, bar_y + bar_h), (70, 60, 90), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay_bar, 0.8, frame, 0.2, 0, frame)

        # Active Action Badge Pill
        action_name = getattr(robot, 'action_mode', 'MARIONETTE')
        lock_state = getattr(robot, 'action_locked', False)

        # Vibrant color per action
        action_colors = {
            'JUMP': (255, 220, 0),        # Radiant Gold/Teal
            'DANCE': (240, 100, 255),      # Neon Magenta
            'RUN': (255, 160, 40),        # Plasma Blue
            'FIGHT': (40, 40, 255),       # Blood Crimson
            'SLASH': (50, 20, 255),       # Laser Vibro-Blade Red
            'MARIONETTE': (0, 220, 140),  # Cyber Green
            'MIMIC': (0, 200, 240),       # Gold
            'POINT': (50, 20, 255)
        }
        pill_col = action_colors.get(action_name, (0, 215, 120))
        cv2.rectangle(frame, (28, bar_y + 8), (145, bar_y + bar_h - 8), pill_col, -1)
        cv2.putText(frame, f"{action_name}", (36, bar_y + 27), font, 0.46, (10, 10, 20), 2, cv2.LINE_AA)

        # Action Lock Status Pill
        lock_col = (0, 240, 140) if lock_state else (70, 65, 85)
        lock_txt = "LOCK: [ON]" if lock_state else "LOCK: [OFF]"
        cv2.rectangle(frame, (152, bar_y + 8), (242, bar_y + bar_h - 8), lock_col, -1 if lock_state else 1, cv2.LINE_AA)
        txt_col = (0, 0, 0) if lock_state else (160, 160, 180)
        cv2.putText(frame, lock_txt, (158, bar_y + 26), font_mono, 0.85, txt_col, 1, cv2.LINE_AA)

        # Telemetry info
        fps_info = f"FPS: {self.fps:.1f} | APEX RED/DARK/WHITE MECHA | CAMERA 60Hz"
        cv2.putText(frame, fps_info, (254, bar_y + 27), font_mono, 0.90, (190, 220, 255), 1, cv2.LINE_AA)

        # Keyboard Hints
        hints = "[1]Jump [2]Dance [3]Run [4]Fight [7]Slash [5]Puppet [L]Lock [SPACE]Mode [W]Wire [Q]Exit"
        cv2.putText(frame, hints, (w - 635, bar_y + 27), font, 0.36, (160, 175, 195), 1, cv2.LINE_AA)

        # --- 4. SLEEK GRADIENT TENSION BUS METERS (Top-Left) ---
        mx, my = 24, 28
        if hand_data is not None:
            curls = hand_data['finger_curls']
            fingers = [('THUMB', 'thumb'), ('INDEX', 'index'), ('MID', 'middle'), ('RING', 'ring'), ('PINKY', 'pinky')]

            cv2.putText(frame, "TENSION BUS", (mx, my - 6), font, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
            for idx, (label, key) in enumerate(fingers):
                fy = my + (idx * 16)
                tension = 1.0 - curls.get(key, 0.5)
                cv2.putText(frame, label, (mx, fy + 8), font_mono, 0.85, (170, 180, 210), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (mx + 54, fy), (mx + 130, fy + 7), (35, 30, 45), -1)
                fill_w = int(tension * 76)
                fill_col = (0, 210, 255) if tension > 0.45 else (40, 120, 220)
                cv2.rectangle(frame, (mx + 54, fy), (mx + 54 + fill_w, fy + 7), fill_col, -1)

        # --- 5. FINGER REMOTE CONTROL GUIDE CARD (Top-Left below meters) ---
        guide_y = my + 94
        guide_w = 200
        guide_h = 130
        overlay_guide = frame.copy()
        cv2.rectangle(overlay_guide, (mx - 6, guide_y), (mx + guide_w, guide_y + guide_h), (14, 12, 20), -1)
        cv2.rectangle(overlay_guide, (mx - 6, guide_y), (mx + guide_w, guide_y + guide_h), (60, 50, 75), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay_guide, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "FINGER REMOTE MATRIX", (mx, guide_y + 16), font, 0.38, (0, 215, 255), 1, cv2.LINE_AA)
        gestures_guide = [
            ("PEACE (V-Sign)", "-> JUMP LEAP", (255, 220, 0)),
            ("ROCK (Horns)",   "-> DANCE 120BPM", (240, 120, 255)),
            ("LEGS (Inverted)", "-> SPRINT RUN", (255, 170, 40)),
            ("FIST (Closed)",  "-> COMBAT COMBO", (60, 90, 255)),
            ("OPEN PALM",      "-> 5-FINGER PUPPET", (0, 230, 150)),
            ("POINT (Index)",  "-> VIBRO SLASH", (60, 40, 255)),
        ]
        for g_idx, (g_name, g_act, g_col) in enumerate(gestures_guide):
            gy = guide_y + 32 + (g_idx * 16)
            cv2.putText(frame, g_name, (mx, gy), font_mono, 0.82, (190, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, g_act, (mx + 98, gy), font_mono, 0.82, g_col, 1, cv2.LINE_AA)
