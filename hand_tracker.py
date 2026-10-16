"""
Robust Hand Tracking module using MediaPipe Hands with One-Euro velocity-adaptive
filtering for zero-jitter, zero-latency precision control.
"""

import math
import time
import numpy as np
import cv2
import mediapipe as mp


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter with velocity-based cutoff frequency.
    Eliminates high-frequency jitter at low speeds while maintaining instant response at high speeds.
    """
    def __init__(self, t0, x0, min_cutoff=1.2, beta=0.012, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def _smoothing_factor(self, dt, cutoff):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, t, x):
        t = float(t)
        x = np.array(x, dtype=np.float32)
        dt = max(t - self.t_prev, 1e-4)

        # Estimate derivative (velocity)
        d_x = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx = a_d * d_x + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff based on velocity
        speed = np.abs(dx)
        cutoff = self.min_cutoff + self.beta * speed
        # Smooth x
        if np.isscalar(cutoff):
            a = self._smoothing_factor(dt, cutoff)
        else:
            a = np.array([self._smoothing_factor(dt, c) for c in cutoff], dtype=np.float32)
        
        x_filtered = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_filtered
        self.dx_prev = dx
        self.t_prev = t
        return x_filtered

    def reset(self, t0, x0):
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t0)


class HandTracker:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.start_time = time.time()
        # 1-Euro filters for each of the 21 keypoints
        self.landmark_filters = None
        self.palm_filter = None
        self.scale_filter = None
        self.orientation_filter = None

        self.TIP_IDS = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        self.MCP_IDS = {
            'thumb': 2,
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
        self.pending_gesture = "OPEN_PALM"
        self.gesture_hold_count = 0
        self.stable_gesture = "OPEN_PALM"
        self.stable_action = "MARIONETTE"

    def process(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        t = time.time() - self.start_time
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        raw_hand = results.multi_hand_landmarks[0]
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        raw_pts = []
        raw_pixels = []
        for lm in raw_hand.landmark:
            raw_pts.append((lm.x, lm.y, lm.z))
            raw_pixels.append([lm.x * w, lm.y * h, lm.z * w])

        raw_pixels = np.array(raw_pixels, dtype=np.float32)

        # Initialize or apply 1-Euro filters
        if self.landmark_filters is None:
            self.landmark_filters = [
                OneEuroFilter(t, raw_pixels[i], min_cutoff=1.5, beta=0.015)
                for i in range(21)
            ]
        
        filtered_pixels = []
        for i in range(21):
            filtered_pt = self.landmark_filters[i].filter(t, raw_pixels[i])
            filtered_pixels.append(filtered_pt)
        pixel_pts = np.array(filtered_pixels, dtype=np.float32)

        # Fingertips smoothed positions
        tips = {
            'thumb': pixel_pts[self.TIP_IDS['thumb']][:2],
            'index': pixel_pts[self.TIP_IDS['index']][:2],
            'middle': pixel_pts[self.TIP_IDS['middle']][:2],
            'ring': pixel_pts[self.TIP_IDS['ring']][:2],
            'pinky': pixel_pts[self.TIP_IDS['pinky']][:2],
        }

        # Palm center & scale
        wrist = pixel_pts[0]
        index_mcp = pixel_pts[5]
        middle_mcp = pixel_pts[9]
        pinky_mcp = pixel_pts[17]

        raw_palm_center = (wrist + index_mcp + middle_mcp + pinky_mcp)[:2] / 4.0
        if self.palm_filter is None:
            self.palm_filter = OneEuroFilter(t, raw_palm_center, min_cutoff=1.0, beta=0.01)
        palm_center = self.palm_filter.filter(t, raw_palm_center)

        raw_hand_span = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))
        if self.scale_filter is None:
            self.scale_filter = OneEuroFilter(t, [raw_hand_span], min_cutoff=0.8, beta=0.005)
        hand_scale = float(self.scale_filter.filter(t, [raw_hand_span])[0])

        # Smooth Finger Curl Detection [0.0 = fully open/extended, 1.0 = curled into fist]
        finger_curls = {}
        for name, tip_id in self.TIP_IDS.items():
            mcp_id = self.MCP_IDS[name]
            pip_id = tip_id - 2
            d_tip_wrist = np.linalg.norm(pixel_pts[tip_id][:2] - wrist[:2])
            d_mcp_wrist = np.linalg.norm(pixel_pts[mcp_id][:2] - wrist[:2])

            if name == 'thumb':
                d_thumb_pinky = np.linalg.norm(pixel_pts[4][:2] - pixel_pts[17][:2])
                curl = np.clip(1.0 - (d_thumb_pinky / (hand_scale * 1.35 + 1e-5)), 0.0, 1.0)
            else:
                max_reach = hand_scale * 1.55
                reach = (d_tip_wrist - d_mcp_wrist) / (max_reach * 0.72 + 1e-5)
                curl = float(np.clip(1.0 - reach, 0.0, 1.0))
            finger_curls[name] = float(curl)

        # Pinch detection (Thumb tip to Index tip distance)
        pinch_dist = float(np.linalg.norm(pixel_pts[4][:2] - pixel_pts[8][:2]))
        is_pinching = pinch_dist < (hand_scale * 0.38)

        # Gestures classification
        num_extended = sum(1 for c in finger_curls.values() if c < 0.4)
        is_index_extended = finger_curls['index'] < 0.35
        is_middle_extended = finger_curls['middle'] < 0.35
        is_ring_extended = finger_curls['ring'] < 0.35
        is_pinky_extended = finger_curls['pinky'] < 0.35
        is_thumb_extended = finger_curls['thumb'] < 0.45

        # Check if fingers are pointing down (for inverted legs / running / walking gesture)
        index_pointing_down = (pixel_pts[8][1] > pixel_pts[5][1] + 15.0)
        middle_pointing_down = (pixel_pts[12][1] > pixel_pts[9][1] + 15.0)
        two_fingers_down = is_index_extended and is_middle_extended and index_pointing_down and middle_pointing_down

        # Rock/Horns sign (Index & Pinky extended, Middle & Ring curled tight)
        is_rock_horns = is_index_extended and is_pinky_extended and (finger_curls['middle'] > 0.42) and (finger_curls['ring'] > 0.42)

        if is_pinching:
            gesture = "PINCH"
            action = "PINCH"
        elif is_rock_horns:
            gesture = "ROCK_ON"
            action = "DANCE"
        elif two_fingers_down:
            gesture = "TWO_FINGER_WALK"
            action = "RUN"
        elif is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
            gesture = "PEACE"
            action = "JUMP"
        elif num_extended == 0 or (finger_curls['index'] > 0.65 and finger_curls['middle'] > 0.65 and finger_curls['ring'] > 0.65 and finger_curls['pinky'] > 0.65):
            gesture = "FIST"
            action = "FIGHT"
        elif is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            gesture = "POINT"
            action = "SLASH"
        elif is_thumb_extended and not is_index_extended and not is_middle_extended:
            gesture = "THUMBS_UP"
            action = "MIMIC"
        elif num_extended >= 4:
            gesture = "OPEN_PALM"
            action = "MARIONETTE"
        else:
            gesture = "PUPPET_CONTROL"
            action = "MARIONETTE"

        # Smooth debouncing: requires 2 consecutive frames before switching gesture
        if gesture == self.pending_gesture:
            self.gesture_hold_count += 1
            if self.gesture_hold_count >= 2:
                self.stable_gesture = gesture
                self.stable_action = action
        else:
            self.pending_gesture = gesture
            self.gesture_hold_count = 1

        # 3D Palm Orientation (Pitch, Yaw, Roll)
        v_y = middle_mcp[:3] - wrist[:3]
        v_y /= (np.linalg.norm(v_y) + 1e-6)
        v_x = pinky_mcp[:3] - index_mcp[:3]
        v_x /= (np.linalg.norm(v_x) + 1e-6)
        v_z = np.cross(v_x, v_y)
        v_z /= (np.linalg.norm(v_z) + 1e-6)

        raw_roll = math.atan2(v_y[0], -v_y[1])
        raw_pitch = math.asin(np.clip(v_y[2], -1.0, 1.0))
        raw_yaw = math.atan2(v_z[0], v_z[2])

        raw_orientation = np.array([raw_pitch, raw_yaw, raw_roll], dtype=np.float32)
        if self.orientation_filter is None:
            self.orientation_filter = OneEuroFilter(t, raw_orientation, min_cutoff=1.0, beta=0.01)
        orientation = self.orientation_filter.filter(t, raw_orientation)

        return {
            'landmarks_pixel': pixel_pts,
            'raw_landmarks': raw_pts,
            'tips': tips,
            'palm_center': palm_center,
            'hand_scale': hand_scale,
            'finger_curls': finger_curls,
            'gesture': self.stable_gesture,
            'action': self.stable_action,
            'is_pinching': is_pinching,
            'pinch_dist': pinch_dist,
            'orientation': tuple(orientation),
            'handedness': handedness
        }

    def draw_hand_mesh(self, frame, hand_data):
        """Renders refined glowing cyber nodes and elegant luminous lines on user's hand."""
        if hand_data is None:
            return

        pts = hand_data['landmarks_pixel']
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # index
            (5, 9), (9, 10), (10, 11), (11, 12),    # middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # ring
            (13, 17), (17, 18), (18, 19), (19, 20), # pinky
            (0, 17)                                 # palm base
        ]

        overlay = frame.copy()
        # Soft glowing skeletal links
        for p1, p2 in connections:
            pt1 = (int(pts[p1][0]), int(pts[p1][1]))
            pt2 = (int(pts[p2][0]), int(pts[p2][1]))
            cv2.line(overlay, pt1, pt2, (10, 140, 255), 2, cv2.LINE_AA)

        # Subtle palm web
        mcps = [5, 9, 13, 17]
        for i in range(len(mcps) - 1):
            pt1 = (int(pts[mcps[i]][0]), int(pts[mcps[i]][1]))
            pt2 = (int(pts[mcps[i+1]][0]), int(pts[mcps[i+1]][1]))
            cv2.line(overlay, pt1, pt2, (20, 170, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Luminous beacon nodes
        for i, pt in enumerate(pts):
            p = (int(pt[0]), int(pt[1]))
            if i in self.TIP_IDS.values():
                # Radiant halo
                cv2.circle(frame, p, 15, (0, 130, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, p, 10, (0, 190, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, p, 6, (50, 240, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
            elif i in [0, 5, 9, 13, 17]:
                cv2.circle(frame, p, 5, (0, 180, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 3, (0, 160, 255), -1, cv2.LINE_AA)

    def close(self):
        self.hands.close()
