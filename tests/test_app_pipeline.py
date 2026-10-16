"""Headless integration test: simulated hands -> intents -> actions, through the real App (no webcam needed)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import glfw
from titan.app import App
from titan.hand_input import HandState


class FakeHand:
    def __init__(self):
        self.s = HandState()
        self.s.camera_ok = True
        self.error = None
    def latest(self):
        return self.s
    def stop(self):
        pass
    def show(self, gesture, palm=(0.5, 0.5), curls=None, tilt=0.0, present=True):
        s = HandState()
        s.camera_ok, s.present, s.gesture, s.raw_gesture = True, present, gesture, gesture
        s.palm = np.array(palm, np.float32)
        s.curls = np.array(curls if curls is not None else [0.2] * 5, np.float32)
        s.index_tilt = tilt
        s.frame_id = self.s.frame_id + 1
        s.frame = np.zeros((240, 320, 3), np.uint8)
        self.s = s


def make_app():
    args = types.SimpleNamespace(size=[960, 600], fullscreen=False, no_vsync=True, quality="medium", demo=False, mute=True,
                                 no_music=True, camera=None, no_camera=True, shot=None, frames=0, no_reveal=True)
    app = App(args)
    app.fixed_dt = 1 / 60
    app.hand = FakeHand()
    app.clock = 1.0      # skip the start-up grace period
    return app


def run(app, n):
    for _ in range(n):
        app.step()


def main():
    app = make_app()
    h = app.hand
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(("PASS " if cond else "FAIL ") + name)

    run(app, 20)
    check("idle with no hand", app.mecha.status == "IDLE")

    # open palm at the right of the screen -> robot walks toward screen-right
    h.show("OPEN", palm=(0.85, 0.52))
    x0 = app.mecha.pos.copy()
    run(app, 150)
    check("open palm drives robot (moved %.1f units)" % float(np.linalg.norm(app.mecha.pos - x0)), np.linalg.norm(app.mecha.pos - x0) > 1.5)
    check("drive target marker shown", app.drive_target is not None)
    h.show("OPEN", palm=(0.5, 0.52), curls=[0.9, 0.1, 0.9, 0.9, 0.1])
    run(app, 60)
    cur = app.mecha.curl
    check("robot fingers mirror the hand (curls %s)" % np.round(cur, 2), cur[0] > 0.6 and cur[1] < 0.35 and cur[2] > 0.6)

    h.show("FIST"); run(app, 30)
    check("FIST -> combo", app.mecha.status == "COMBO")
    h.show("OPEN"); run(app, 120)
    check("combo ends after release", app.mecha.status in ("IDLE", "MOVE"))

    h.show("PEACE"); run(app, 20)
    check("PEACE -> jump slam", app.mecha.status == "JUMPSLAM")
    run(app, 260)

    h.show("POINT", tilt=0.3); run(app, 60)
    check("POINT -> laser", app.mecha.status == "LASER")
    h.show("OPEN"); run(app, 90)

    h.show("THREE"); run(app, 40)
    check("THREE -> shield", app.mecha.status == "SHIELD")
    h.show("OPEN"); run(app, 90)

    h.show("THUMBS_UP"); run(app, 20)
    check("THUMBS_UP -> dash", app.mecha.status == "DASH")
    run(app, 150)

    h.show("OPEN"); run(app, 10)
    h.show("ROCK"); run(app, 20)
    check("ROCK -> power-up", app.mecha.status == "POWERUP")
    run(app, 220)
    check("overdrive active after power-up", app.mecha.overdrive > 0)

    az0 = app.cam.t_az
    h.show("PINCH", palm=(0.5, 0.5)); run(app, 3)
    for i in range(20):
        h.show("PINCH", palm=(0.5 + 0.01 * i, 0.5)); run(app, 1)
    check("PINCH + move orbits camera (%.0f deg)" % (app.cam.t_az - az0), abs(app.cam.t_az - az0) > 30)

    # keyboard
    h.show("OPEN", present=False); run(app, 30)
    app.keys.add(glfw.KEY_2); run(app, 20)
    check("key 2 -> combo", app.mecha.status == "COMBO")
    app.keys.discard(glfw.KEY_2); run(app, 120)
    app.pressed.add(glfw.KEY_SPACE); run(app, 10)
    check("space -> jump", app.mecha.status == "JUMPSLAM")
    run(app, 260)
    for k in (glfw.KEY_F, glfw.KEY_B, glfw.KEY_L, glfw.KEY_R, glfw.KEY_T, glfw.KEY_C, glfw.KEY_O):
        app._on_press(k)
    run(app, 5)
    check("view / colour hotkeys run without error", True)

    bad = [n for n, ok in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(bad), len(results)))
    app.close()
    sys.exit(1 if bad else 0)


main()
