"""
Hand-Detect 3D Robot Puppet Controller.
Main Application Entry Point.
Controls a 3D Robot Mecha using Real-Time Hand Tracking & Tension String Physics.
"""

import sys
import time
import math
import argparse
import warnings
import cv2
import numpy as np

# Suppress protobuf and mediapipe deprecation warnings for clean console output
warnings.filterwarnings("ignore", category=UserWarning)

from hand_tracker import HandTracker
from robot_3d import Robot3D
from string_solver import StringSolver
from hud_overlay import HudOverlay


def find_active_webcam(preferred_id=None):
    """
    Scans camera devices and automatically selects the active, working webcam.
    On macOS, camera 0 can sometimes be a black/standby stream while camera 1
    is the active built-in MacBook FaceTime HD webcam.
    """
    candidates = [preferred_id] if preferred_id is not None else [1, 0, 2]
    # Remove None and duplicates while preserving order
    indices_to_test = []
    for idx in candidates:
        if idx is not None and idx not in indices_to_test:
            indices_to_test.append(idx)

    best_cam = None
    best_id = None

    print("[CAMERA] Probing available camera devices...")
    for cam_id in indices_to_test:
        try:
            cap = cv2.VideoCapture(cam_id, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_id)
            
            if cap.isOpened():
                # Read test frames to warm up sensor
                for _ in range(6):
                    cap.read()
                ret, frame = cap.read()
                if ret and frame is not None:
                    brightness = float(np.mean(frame))
                    print(f"  > Camera index {cam_id}: Detected (resolution: {frame.shape[1]}x{frame.shape[0]}, brightness: {brightness:.1f})")
                    if brightness > 3.0: # Active image with light
                        print(f"[CAMERA] Successfully selected active MacBook webcam at index {cam_id}!")
                        return cap, cam_id
                    elif best_cam is None:
                        best_cam = cap
                        best_id = cam_id
                        continue
                cap.release()
        except Exception as e:
            print(f"  > Camera index {cam_id} error: {e}")

    if best_cam is not None:
        print(f"[CAMERA] Using fallback camera index {best_id}")
        return best_cam, best_id

    print("[WARN] No physical camera found or permissions denied.")
    return None, -1


def main():
    parser = argparse.ArgumentParser(description="CyberPuppet: Hand-Tracked 3D Robot Controller")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (default: auto-detect)")
    parser.add_argument("--mirror", action="store_true", default=True, help="Mirror camera horizontally")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic hand demo (no camera required)")
    args = parser.parse_args()

    print("=======================================================")
    print("  CYBERPUPPET // APEX COMBAT MECHA CONTROLLER INITIALIZED ")
    print("  [PALETTE: ARCTIC WHITE / BLOOD RED / OBSIDIAN CARBON] ")
    print("=======================================================")
    print(" Finger Remote Gestures (Show Hand to Camera):")
    print("   [✌️ PEACE]     : JUMP (Meteor Rocket Leap & Ground Impact Slam)")
    print("   [🤘 HORNS]     : DANCE (120 BPM Cybernetic Breakdance)")
    print("   [✌️ INVERTED]  : RUN (Predator Assassin Sprint & Boosters)")
    print("   [✊ FIST]      : FIGHT (4-Hit Martial Arts Mecha Combo)")
    print("   [👆 POINT]     : SLASH (Dual Vibro-Blade Hurricane Frenzy)")
    print("   [✋ OPEN PALM] : MARIONETTE (5-Finger Laser Tension Strings)")
    print("   [👍 THUMBS UP] : MIMIC (Direct 1:1 Teleoperation)")
    print(" Remote Hotkeys:")
    print("   [1] or [J]   : Trigger JUMP Action")
    print("   [2] or [D]   : Trigger DANCE Action")
    print("   [3] or [K]   : Trigger RUN Action")
    print("   [4] or [F]   : Trigger FIGHT Action")
    print("   [7] or [S]   : Trigger SLASH Action (Vibro-Blades)")
    print("   [5] or [P]   : Trigger MARIONETTE Puppet Mode")
    print("   [6] or [M]   : Trigger DIRECT MIMIC Teleoperation")
    print("   [L]          : Toggle Action Lock (Continuous Loop)")
    print("   [SPACE]      : Cycle Mode")
    print("   [W]          : Toggle Wireframe / Shaded 3D Mesh")
    print("   [R]          : Reset Robot Position & Rotation")
    print("   [C]          : Toggle Mirror Mode")
    print("   [H]          : Toggle HUD Interface")
    print("   [N] or [TAB] : Switch Camera (Cam 0 <-> Cam 1)")
    print("   [Q] or [ESC] : Exit Application")
    print("=======================================================")

    current_cam_id = args.camera
    cap = None
    if not args.demo:
        cap, current_cam_id = find_active_webcam(args.camera)

    is_synthetic = (cap is None)
    if is_synthetic:
        print("[INFO] Running in Autonomous Synthetic Demo Mode.")

    # Initialize subsystems
    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    robot = Robot3D()
    strings = StringSolver()
    hud = HudOverlay()

    modes = ["MARIONETTE", "DIRECT_MIMIC", "COMBAT_READY"]
    mode_idx = 0

    wireframe_mode = False
    mirror_mode = args.mirror
    show_hud = True

    cv2.namedWindow("Hand-Detect 3D Robot Puppet Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand-Detect 3D Robot Puppet Controller", 1280, 720)

    demo_t = 0.0

    try:
        while True:
            if not is_synthetic:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Warm-up or momentary glitch recovery
                    time.sleep(0.02)
                    continue

                # Scale high-res camera feed (e.g. 1920x1080) to 1280x720 for optimal 60 FPS performance
                if frame.shape[1] > 1280 or frame.shape[0] > 720:
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

                if mirror_mode:
                    frame = cv2.flip(frame, 1)
            else:
                # Synthetic dark cyber backdrop
                h, w = 720, 1280
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                for gy in range(0, h, 60):
                    cv2.line(frame, (0, gy), (w, gy), (25, 20, 30), 1)
                for gx in range(0, w, 60):
                    cv2.line(frame, (gx, 0), (gx, h), (25, 20, 30), 1)

            h, w, _ = frame.shape

            # 1. Hand Tracking Phase
            hand_data = None
            if not is_synthetic:
                hand_data = tracker.process(frame)
            else:
                demo_t += 0.04
                hand_x = int(w * 0.65 + math.cos(demo_t * 0.8) * 120)
                hand_y = int(h * 0.55 + math.sin(demo_t * 1.2) * 80)

                # Showcase different actions over time in demo mode:
                # 0: MARIONETTE, 1: JUMP, 2: DANCE, 3: RUN, 4: FIGHT, 5: SLASH
                demo_cycle = int(demo_t / 4.5) % 6
                if demo_cycle == 0:
                    demo_gesture = "OPEN_PALM"
                    demo_action = "MARIONETTE"
                    c_thumb, c_index, c_mid, c_ring, c_pinky = 0.2, 0.2, 0.2, 0.2, 0.2
                elif demo_cycle == 1:
                    demo_gesture = "PEACE"
                    demo_action = "JUMP"
                    c_thumb, c_index, c_mid, c_ring, c_pinky = 0.8, 0.1, 0.1, 0.8, 0.8
                elif demo_cycle == 2:
                    demo_gesture = "ROCK_ON"
                    demo_action = "DANCE"
                    c_thumb, c_index, c_mid, c_ring, c_pinky = 0.8, 0.1, 0.8, 0.8, 0.1
                elif demo_cycle == 3:
                    demo_gesture = "TWO_FINGER_WALK"
                    demo_action = "RUN"
                    c_thumb, c_index, c_mid, c_ring, c_pinky = 0.8, 0.1, 0.1, 0.8, 0.8
                elif demo_cycle == 4:
                    demo_gesture = "FIST"
                    demo_action = "FIGHT"
                    c_thumb, c_index, c_mid, c_ring, c_pinky = 0.9, 0.9, 0.9, 0.9, 0.9
                else:
                    demo_gesture = "POINT"
                    demo_action = "SLASH"
                    c_thumb, c_index, c_mid, c_ring, c_pinky = 0.9, 0.1, 0.9, 0.9, 0.9

                hand_data = {
                    'landmarks_pixel': np.array([
                        [hand_x, hand_y + 90, 0],
                        [hand_x - 30, hand_y + 40, 0],
                        [hand_x - 45, hand_y + 10, 0],
                        [hand_x - 55, hand_y - 20, 0],
                        [hand_x - 65, hand_y - 45 + c_thumb * 20, 0],
                        [hand_x - 20, hand_y - 30, 0],
                        [hand_x - 20, hand_y - 60, 0],
                        [hand_x - 20, hand_y - 90, 0],
                        [hand_x - 20, hand_y - 120 + c_index * 30, 0],
                        [hand_x, hand_y - 35, 0],
                        [hand_x, hand_y - 70, 0],
                        [hand_x, hand_y - 100, 0],
                        [hand_x, hand_y - 130 + c_mid * 25, 0],
                        [hand_x + 20, hand_y - 30, 0],
                        [hand_x + 20, hand_y - 65, 0],
                        [hand_x + 20, hand_y - 95, 0],
                        [hand_x + 20, hand_y - 125 + c_ring * 25, 0],
                        [hand_x + 40, hand_y - 20, 0],
                        [hand_x + 40, hand_y - 50, 0],
                        [hand_x + 40, hand_y - 75, 0],
                        [hand_x + 40, hand_y - 100 + c_pinky * 30, 0],
                    ]),
                    'tips': {
                        'thumb': (hand_x - 65, hand_y - 45 + c_thumb * 20),
                        'index': (hand_x - 20, hand_y - 120 + c_index * 30),
                        'middle': (hand_x, hand_y - 130 + c_mid * 25),
                        'ring': (hand_x + 20, hand_y - 125 + c_ring * 25),
                        'pinky': (hand_x + 40, hand_y - 100 + c_pinky * 30),
                    },
                    'palm_center': (hand_x, hand_y),
                    'hand_scale': 85.0 + math.sin(demo_t * 0.5) * 15.0,
                    'finger_curls': {
                        'thumb': c_thumb,
                        'index': c_index,
                        'middle': c_mid,
                        'ring': c_ring,
                        'pinky': c_pinky,
                    },
                    'gesture': demo_gesture,
                    'action': demo_action,
                    'is_pinching': False,
                    'orientation': (0.1 * math.sin(demo_t), 0.2 * math.cos(demo_t), 0.0),
                    'handedness': "Right"
                }

            # 2. Render Hand Cyber Skeleton & Glowing Nodes
            if hand_data is not None:
                tracker.draw_hand_mesh(frame, hand_data)

            # 3. Update 3D Robot Kinematics & Posture
            robot.update_from_hand(hand_data, w, h, mode_name=modes[mode_idx])

            # 4. Render 3D Robot into Scene
            robot.render(frame, cam_fx=750.0, cam_fy=750.0, wireframe_mode=wireframe_mode)

            # 5. Draw Dynamic Marionette Tension Strings
            strings.draw(frame, hand_data, robot.screen_anchors)

            # 6. Render Sci-Fi HUD Overlay
            if show_hud:
                cam_label = f"CAM:{current_cam_id}" if not is_synthetic else "DEMO"
                hud.draw(frame, hand_data, robot, mode_name=f"{modes[mode_idx]} | {cam_label}")

            # 7. Display Window
            cv2.imshow("Hand-Detect 3D Robot Puppet Controller", frame)

            # 8. Handle Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: # Q or ESC
                print("[INFO] Exiting application cleanly...")
                break
            elif key in [ord('n'), ord('N'), 9]: # N or TAB: Switch camera
                if not is_synthetic:
                    next_id = 0 if current_cam_id == 1 else 1
                    print(f"[CAMERA] Attempting switch to camera index {next_id}...")
                    new_cap = cv2.VideoCapture(next_id, cv2.CAP_AVFOUNDATION)
                    if not new_cap.isOpened():
                        new_cap = cv2.VideoCapture(next_id)
                    if new_cap.isOpened():
                        cap.release()
                        cap = new_cap
                        current_cam_id = next_id
                        print(f"[CAMERA] Switched to camera index {current_cam_id}")
                    else:
                        print(f"[CAMERA] Camera index {next_id} could not be opened.")
            elif key in [ord('1'), ord('j'), ord('J')]:
                robot.trigger_action("JUMP")
            elif key in [ord('2'), ord('d'), ord('D')]:
                robot.trigger_action("DANCE")
            elif key in [ord('3'), ord('k'), ord('K')]:
                robot.trigger_action("RUN")
            elif key in [ord('4'), ord('f'), ord('F')]:
                robot.trigger_action("FIGHT")
            elif key in [ord('5'), ord('p'), ord('P')]:
                robot.trigger_action("MARIONETTE")
            elif key in [ord('6'), ord('m'), ord('M')]:
                robot.trigger_action("MIMIC")
            elif key in [ord('7'), ord('s'), ord('S')]:
                robot.trigger_action("SLASH")
            elif key in [ord('l'), ord('L')]:
                robot.toggle_action_lock()
            elif key == ord(' '): # Space: Cycle mode
                mode_idx = (mode_idx + 1) % len(modes)
                robot.trigger_action(modes[mode_idx])
                print(f"[MODE] Switched to: {modes[mode_idx]}")
            elif key in [ord('w'), ord('W')]:
                wireframe_mode = not wireframe_mode
                print(f"[VIEW] Wireframe mode: {wireframe_mode}")
            elif key in [ord('r'), ord('R')]:
                robot.pos = np.array([-210.0, 10.0, 520.0], dtype=np.float32)
                robot.rot = np.array([0.05, 0.48, -0.02], dtype=np.float32)
                robot.action_mode = "MARIONETTE"
                robot.action_time = 0.0
                print("[POSE] Robot position, rotation & action reset.")
            elif key in [ord('c'), ord('C')]:
                mirror_mode = not mirror_mode
                print(f"[CAMERA] Mirror mode: {mirror_mode}")
            elif key in [ord('h'), ord('H')]:
                show_hud = not show_hud
                print(f"[HUD] HUD visible: {show_hud}")

    except KeyboardInterrupt:
        print("[INFO] Stopped by user interrupt.")
    finally:
        if cap is not None:
            cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[INFO] All resources released successfully.")


if __name__ == "__main__":
    main()
