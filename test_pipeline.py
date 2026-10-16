"""
Test & Validation script for Hand-Detect 3D Robot Puppet pipeline.
Runs the entire pipeline on simulated frames and saves test renders for quality verification.
"""

import sys
import os
import math
import numpy as np
import cv2

from hand_tracker import HandTracker
from robot_3d import Robot3D
from string_solver import StringSolver
from hud_overlay import HudOverlay


def test_full_pipeline():
    print("[TEST] Initializing subsystems...")
    tracker = HandTracker()
    robot = Robot3D()
    strings = StringSolver()
    hud = HudOverlay()

    h, w = 720, 1280
    test_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Dark cyber room background
    for gy in range(0, h, 40):
        cv2.line(test_canvas, (0, gy), (w, gy), (30, 25, 40), 1)
    for gx in range(0, w, 40):
        cv2.line(test_canvas, (gx, 0), (gx, h), (30, 25, 40), 1)

    # Simulated hand tracking data matching the reference image
    hand_x = 760
    hand_y = 480
    demo_hand = {
        'landmarks_pixel': np.array([
            [hand_x, hand_y + 90, 0],
            [hand_x - 30, hand_y + 40, 0],
            [hand_x - 45, hand_y + 10, 0],
            [hand_x - 55, hand_y - 20, 0],
            [hand_x - 65, hand_y - 45, 0], # thumb
            [hand_x - 20, hand_y - 30, 0],
            [hand_x - 20, hand_y - 60, 0],
            [hand_x - 20, hand_y - 90, 0],
            [hand_x - 20, hand_y - 130, 0], # index (pointing up)
            [hand_x, hand_y - 35, 0],
            [hand_x, hand_y - 70, 0],
            [hand_x, hand_y - 100, 0],
            [hand_x, hand_y - 125, 0], # middle
            [hand_x + 20, hand_y - 30, 0],
            [hand_x + 20, hand_y - 65, 0],
            [hand_x + 20, hand_y - 95, 0],
            [hand_x + 20, hand_y - 120, 0], # ring
            [hand_x + 40, hand_y - 20, 0],
            [hand_x + 40, hand_y - 50, 0],
            [hand_x + 40, hand_y - 75, 0],
            [hand_x + 40, hand_y - 110, 0], # pinky
        ]),
        'tips': {
            'thumb': (hand_x - 65, hand_y - 45),
            'index': (hand_x - 20, hand_y - 130),
            'middle': (hand_x, hand_y - 125),
            'ring': (hand_x + 20, hand_y - 120),
            'pinky': (hand_x + 40, hand_y - 110),
        },
        'palm_center': (hand_x, hand_y),
        'hand_scale': 85.0,
        'finger_curls': {
            'thumb': 0.35,
            'index': 0.1,  # pointing straight up
            'middle': 0.25,
            'ring': 0.3,
            'pinky': 0.2,
        },
        'gesture': "OPEN_PALM",
        'is_pinching': False,
        'orientation': (0.1, 0.2, 0.0),
        'handedness': "Right"
    }

    print("[TEST] Testing Hand Mesh drawing...")
    tracker.draw_hand_mesh(test_canvas, demo_hand)

    print("[TEST] Testing Robot Kinematics & 3D Render...")
    robot.update_from_hand(demo_hand, w, h)
    robot.render(test_canvas, cam_fx=750.0, cam_fy=750.0, wireframe_mode=False)

    print("[TEST] Testing String Solver...")
    strings.draw(test_canvas, demo_hand, robot.screen_anchors)

    print("[TEST] Testing HUD Overlay...")
    hud.draw(test_canvas, demo_hand, robot, mode_name="MARIONETTE")

    output_path = "test_output_render.png"
    cv2.imwrite(output_path, test_canvas)
    print(f"[TEST SUCCESS] Output saved to: {output_path}")

    tracker.close()
    return True


if __name__ == "__main__":
    if test_full_pipeline():
        print("ALL PIPELINE TESTS PASSED WITH ZERO ERRORS.")
        sys.exit(0)
    else:
        sys.exit(1)
