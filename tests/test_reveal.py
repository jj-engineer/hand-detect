"""Reveal workflow with simulated hands: empty stage -> open hand -> parts float -> fist -> assemble -> boot -> live."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import glfw
from titan.app import App
from titan.hand_input import HandState
from titan import reveal as RV


class FakeHand:
    def __init__(self):
        self.s = HandState(); self.s.camera_ok = True; self.error = None
    def latest(self): return self.s
    def stop(self): pass
    def show(self, gesture, palm=(0.5, 0.5), present=True):
        s = HandState(); s.camera_ok, s.present, s.gesture, s.raw_gesture = True, present, gesture, gesture
        s.palm = np.array(palm, np.float32); s.curls = np.full(5, 0.2, np.float32)
        s.frame_id = self.s.frame_id + 1; s.frame = np.zeros((240, 320, 3), np.uint8); self.s = s


def main():
    args = types.SimpleNamespace(size=[960, 600], fullscreen=False, no_vsync=True, quality="medium", demo=False, mute=True,
                                 no_music=True, camera=None, no_camera=True, shot=None, frames=0, no_reveal=False)
    app = App(args)
    app.fixed_dt = 1 / 30
    app.hand = FakeHand()
    h = app.hand
    res = []
    def check(name, cond):
        res.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)
    def run(n):
        for _ in range(n): app.step()
    rv = app.reveal

    run(40)
    check("starts on an empty stage (robot hidden)", rv.state == RV.WAIT and not rv.mecha_visible())
    h.show("PEACE"); run(30)
    check("other gestures do nothing while waiting", rv.state == RV.WAIT)
    h.show("OPEN", present=False); run(10)
    h.show("OPEN"); run(6)
    check("open hand needs a moment (debounce)", rv.state == RV.WAIT)
    run(10)
    check("held open hand -> components pop up", rv.state == RV.PARTS and rv.mecha_visible())
    run(15)
    check("parts appear progressively", 0 < int(rv.popped.sum()) < len(RV.GROUPS))
    h.show("FIST"); run(12)
    check("early fist is queued until all parts are out", rv.state == RV.PARTS and rv.pending_assemble)
    run(120)
    check("all %d components floated in, then assembly started" % len(RV.GROUPS), bool(rv.popped.all()) and rv.state in (RV.ASSEMBLE, RV.BOOT))
    run(140)
    check("every component clamped onto the body", rv.state in (RV.BOOT, RV.LIVE) and bool(rv.arrived.all()))
    check("controls are locked while the fist is still closed", app.control_locked or rv.state == RV.BOOT)
    run(90)
    check("boot-up finished -> live", rv.state == RV.LIVE)
    h.show("FIST"); run(20)
    check("still-closed fist does not trigger the combo", app.mecha.status != "COMBO")
    h.show("OPEN"); run(5)
    h.show("FIST"); run(20)
    check("after opening the hand, a fist starts the katana combo", app.mecha.status == "COMBO")
    h.show("OPEN"); run(120)

    # keyboard path + replay
    app.restart_reveal()
    run(10)
    check("X replays the reveal (empty stage again)", rv.state == RV.WAIT and not rv.mecha_visible())
    h.show("OPEN", present=False)
    app._on_press(glfw.KEY_ENTER); run(150)
    check("ENTER pops the parts", rv.state == RV.PARTS and rv.popped.all())
    app._on_press(glfw.KEY_ENTER); run(300)
    check("ENTER assembles -> live", rv.state == RV.LIVE)
    # the skin override during the show must be finite and place the pieces
    ok = True
    app.restart_reveal(); h.show("OPEN"); run(40)
    for _ in range(80):
        app.step(); ok &= rv.skin is None or bool(np.isfinite(rv.skin).all())
    check("skin matrices stay finite throughout", ok)
    print("\n%d/%d passed" % (sum(res), len(res)))
    app.close()
    sys.exit(0 if all(res) else 1)


main()
