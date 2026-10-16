"""Soak test: run the scripted demo (every action) for two full loops with hotkeys; fail on any exception / NaN."""
import os, sys, types, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, glfw
from titan.app import App


def main():
    args = types.SimpleNamespace(size=[800, 500], fullscreen=False, no_vsync=True, quality="medium", demo=True, mute=True,
                                 no_music=True, camera=None, no_camera=True, shot=None, frames=0, no_reveal=False)
    app = App(args)
    app.fixed_dt = 1 / 30
    seen = set()
    t0 = time.time()
    for i in range(2 * 44 * 30 + 16 * 30):
        if i == 700: app._on_press(glfw.KEY_C)
        if i == 1000: app._on_press(glfw.KEY_Q)
        if i == 1300: app._on_press(glfw.KEY_F)
        if i == 1600: app._on_press(glfw.KEY_B)
        if i == 1900: app._on_press(glfw.KEY_T); app._on_press(glfw.KEY_H)
        if i == 2200: app._on_press(glfw.KEY_H); app._on_press(glfw.KEY_C)
        app.step()
        m = app.mecha
        assert np.isfinite(m.skin).all(), "NaN in skin at frame %d" % i
        assert np.isfinite(m.pos).all()
        assert abs(m.pos[0]) < 30 and abs(m.pos[2]) < 30, "left the arena"
        seen.add(m.status)
    print("statuses seen:", sorted(seen))
    print("targets destroyed:", app.targets.score, "| particles alive:", app.fx.p.n, "| %.1fs wall" % (time.time() - t0))
    need = {"IDLE", "MOVE", "JUMPSLAM", "COMBO", "DASH", "LASER", "SHIELD", "POWERUP"}
    missing = need - seen
    print("MISSING:", missing if missing else "none")
    app.close()
    sys.exit(1 if missing else 0)


main()
