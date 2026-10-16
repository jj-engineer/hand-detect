"""
Windows pre-flight for TITAN-9: packages, MediaPipe hand model, OpenGL 4.1 and the laptop camera.

Called by setup_windows.ps1; can also be run by hand:
    .venv-win\\Scripts\\python tools\\windows_selfcheck.py [--imports-only] [--no-camera] [--camera N]

Exit code: 0 = all good | 1 = a package is missing/broken | 2 = no OpenGL 4.1 window | 3 = only the camera failed
(code 3 is not fatal: the game still starts and can be played with the keyboard).
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED = ("numpy", "scipy", "cv2", "mediapipe", "moderngl", "glfw", "PIL")


def check_imports():
    bad = []
    for name in REQUIRED:
        try:
            mod = __import__(name)
            print("[PACKAGES] %-10s %s" % (name, getattr(mod, "__version__", "")))
        except Exception as e:
            bad.append(name)
            print("[PACKAGES] %-10s FAILED: %s: %s" % (name, type(e).__name__, e))
    return not bad


def check_audio():
    try:
        import sounddevice as sd
        sd.query_devices(kind="output")
        print("[AUDIO]    OK")
    except Exception as e:
        print("[AUDIO]    no sound output (%s) - the game will run silent" % e)


def check_mediapipe():
    try:
        import numpy as np
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1)
        hands.process(np.zeros((96, 96, 3), np.uint8))          # forces the model files to load
        hands.close()
        print("[HANDS]    MediaPipe hand model OK")
        return True
    except Exception as e:
        print("[HANDS]    FAILED: %s: %s" % (type(e).__name__, e))
        print("           (if your Windows user name or the folder path has non-English letters, move the project to e.g. C:\\TITAN9)")
        return False


def check_graphics():
    import glfw
    import moderngl
    if not glfw.init():
        print("[GRAPHICS] FAILED: glfw.init() - no display available (Remote Desktop / virtual machine?)")
        return False
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        win = glfw.create_window(64, 64, "probe", None, None)
        if not win:
            glfw.default_window_hints()                           # what *does* the driver offer? (for the log)
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            win2 = glfw.create_window(64, 64, "probe", None, None)
            if win2:
                glfw.make_context_current(win2)
                info = moderngl.create_context().info
                print("[GRAPHICS] FAILED: need OpenGL 4.1, driver offers %s on %s" % (info["GL_VERSION"], info["GL_RENDERER"]))
                glfw.destroy_window(win2)
            else:
                print("[GRAPHICS] FAILED: could not create any OpenGL window")
            print("           -> update the graphics driver (Intel / NVIDIA / AMD) from Windows Update or the maker's website")
            return False
        glfw.make_context_current(win)
        info = moderngl.create_context().info
        print("[GRAPHICS] OK  %s | OpenGL %s" % (info["GL_RENDERER"], info["GL_VERSION"].split()[0]))
        glfw.destroy_window(win)
        return True
    finally:
        glfw.terminate()


def check_camera(index):
    from titan.hand_input import HandInput
    t0 = time.time()
    cap, idx = HandInput(camera=index)._open_camera()
    if cap is None:
        print("[CAMERA]   FAILED: no working camera found")
        return False
    ok, frame = cap.read()
    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "?"
    h, w = (frame.shape[:2] if ok and frame is not None else (0, 0))
    cap.release()
    print("[CAMERA]   OK  index %d via %s, %dx%d, opened in %.1fs" % (idx, backend, w, h, time.time() - t0))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports-only", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    if not check_imports():
        return 1
    if args.imports_only:
        return 0
    check_audio()
    if not check_mediapipe():
        return 1
    if not check_graphics():
        return 2
    if args.no_camera:
        return 0
    return 0 if check_camera(args.camera) else 3


if __name__ == "__main__":
    sys.exit(main())
