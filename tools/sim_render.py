"""Run the real simulation headless and render frames to a contact sheet.

usage: python tools/sim_render.py ACTION out.png [az el dist]
actions: jump combo dash laser shield power walk run
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import moderngl
from PIL import Image

from titan.mecha_model import build_mecha
from titan.renderer import Renderer
from titan.animation import Mecha, Intent
from titan.fx import FX
from titan.camera import OrbitCamera
from titan.scene import build_frame

SCRIPTS = {
    # name: (intent factory(t), duration, sample times)
    "jump": (lambda t: Intent(trig_jump=(t < 0.05)), 3.0, [0.2, 0.45, 0.7, 1.0, 1.25, 1.5, 1.66, 1.8, 2.2, 2.8]),
    "combo": (lambda t: Intent(hold_combo=t < 6.9), 8.5, [0.36, 0.60, 1.30, 1.50, 2.06, 2.28, 2.98, 3.62, 4.00, 5.52, 5.78, 7.9]),
    "combo2": (lambda t: Intent(hold_combo=t < 6.9), 8.5, [0.30, 0.62, 1.50, 5.78]),
    "dash": (lambda t: Intent(trig_dash=(t < 0.05)), 1.8, [0.1, 0.2, 0.35, 0.6, 0.85, 1.1, 1.3, 1.7]),
    "laser": (lambda t: Intent(hold_laser=t < 2.4, aim_pitch=0.1), 3.0, [0.2, 0.4, 0.7, 0.95, 1.3, 2.0, 2.45, 2.8]),
    "shield": (lambda t: Intent(hold_shield=t < 2.5), 3.0, [0.15, 0.4, 0.8, 1.4, 1.8, 2.2]),
    "power": (lambda t: Intent(trig_power=(t < 0.05)), 3.2, [0.3, 0.7, 1.2, 1.45, 1.6, 1.9, 2.4, 3.0]),
    "idle": (lambda t: Intent(), 2.0, [1.0, 1.3, 1.6, 1.9]),
    "walk": (lambda t: Intent(move_target=(0, 8)), 3.4, [1.7, 1.9, 2.1, 2.3, 2.5, 2.7]),
    "run": (lambda t: Intent(move_target=(0, 40), sprint=True), 3.4, [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]),
}


def main():
    name = sys.argv[1]
    out = sys.argv[2]
    az, el, dist = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (32, 10, 12)
    W, H = 800, 600
    ctx = moderngl.create_standalone_context()
    fbo = ctx.framebuffer([ctx.texture((W, H), 4)])
    r = Renderer(ctx, W, H, out_fbo=fbo)
    r.set_mecha(build_mecha())
    fx = FX()
    m = Mecha(fx)
    cam = OrbitCamera()
    cam.t_az = cam.az = az
    cam.t_el = cam.el = el
    cam.t_dist = cam.dist = dist
    fn, dur, samples = SCRIPTS[name]
    dt = 1 / 60
    t = 0.0
    imgs = []
    idx = 0
    now = 0.0
    while t < dur and idx < len(samples):
        m.update(dt, fn(t))
        fx.update(dt)
        cam.update(dt, np.array([m.pos[0], m.height, m.pos[2]], np.float32), fx, now, wide=m.camera_wide())
        t += dt
        now += dt
        if t >= samples[idx]:
            fd = build_frame(m, fx, cam, W / H, now)
            r.render(fd)
            img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
            imgs.append(img.copy())
            print("t=%.2f  %s  pos=%s h=%.2f" % (t, m.status, np.round(m.pos, 1), m.height))
            idx += 1
    cols = 3 if len(imgs) > 4 else 2
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for i, im in enumerate(imgs):
        sheet[(i // cols) * H:(i // cols + 1) * H, (i % cols) * W:(i % cols + 1) * W] = im
    Image.fromarray(sheet).save(out)
    print("saved", out)


main()
