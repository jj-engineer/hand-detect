"""Offscreen still renders of the mecha for visual QA.

usage: python tools/still.py out.png [view ...]
views: front back left right threeq top head hand chest foot legs
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import moderngl
from PIL import Image

from titan.mecha_model import build_mecha
from titan.renderer import Renderer, FrameData
from titan.skeleton import Pose, forward_kinematics, BI
from titan.xform import look_at, perspective

VIEWS = {
    # name: (azimuth deg, elevation deg, distance, target xyz, fov)
    "front": (0, 8, 11.5, (0, 2.05, 0), 36),
    "back": (180, 8, 11.5, (0, 2.05, 0), 36),
    "left": (90, 8, 11.5, (0, 2.05, 0), 36),
    "right": (-90, 8, 11.5, (0, 2.05, 0), 36),
    "threeq": (35, 12, 10.5, (0, 2.05, 0), 36),
    "threeqb": (-145, 14, 10.5, (0, 2.05, 0), 36),
    "top": (20, 70, 12, (0, 1.5, 0), 36),
    "head": (22, 6, 3.4, (0, 3.62, 0), 36),
    "hand": (40, 5, 2.6, (0.9, 1.0, 0.0), 36),
    "headf": (0, 4, 3.4, (0, 3.62, 0), 36),
    "heads": (90, 3, 3.6, (0, 3.62, 0), 36),
    "headb": (155, 8, 3.8, (0, 3.62, 0), 36),
    "chest": (18, 8, 5.2, (0, 2.85, 0), 36),
    "foot": (38, 12, 4.2, (0.4, 0.5, 0.0), 36),
    "legs": (28, 8, 6.5, (0, 1.0, 0), 36),
}


def camera(view, aspect):
    az, el, dist, tgt, fov = VIEWS[view]
    az, el = math.radians(az), math.radians(el)
    tgt = np.array(tgt, np.float32)
    eye = tgt + dist * np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)], np.float32)
    return eye, look_at(eye, tgt), perspective(math.radians(fov), aspect, 0.1, 300.0)


def render(names, pose=None, W=1280, H=800, out="still.png", theme="ONI", extra=None, cols=None):
    ctx = moderngl.create_standalone_context()
    fbo = ctx.framebuffer([ctx.texture((W, H), 4)])
    r = Renderer(ctx, W, H, out_fbo=fbo, scale=1.0)
    t0 = time.time()
    md = build_mecha()
    print("mesh build %.2fs  verts=%d tris=%d" % (time.time() - t0, len(md.pos), len(md.idx)))
    r.set_mecha(md)
    if pose is None:
        pose = Pose()
        pose.rot[BI['uarmL'], 2] = -0.20
        pose.rot[BI['uarmR'], 2] = 0.20
        pose.rot[BI['farmL'], 0] = -0.12
        pose.rot[BI['farmR'], 0] = -0.12
    world, skin = forward_kinematics(pose)
    imgs = []
    for n in names:
        eye, V, P = camera(n, W / H)
        fd = FrameData(view=V, proj=P, eye=eye, bones=skin, theme=theme, focus=np.asarray(pose.root_pos, np.float32))
        if extra:
            extra(fd, world)
        for _ in range(2):
            r.render(fd)
        img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
        imgs.append(img)
    if len(imgs) == 1:
        sheet = imgs[0]
    else:
        cols = cols or (2 if len(imgs) <= 4 else 3)
        rows = (len(imgs) + cols - 1) // cols
        sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
        for i, im in enumerate(imgs):
            sheet[(i // cols) * H:(i // cols + 1) * H, (i % cols) * W:(i % cols + 1) * W] = im
    Image.fromarray(sheet).save(out)
    print("saved", out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "still.png"
    names = sys.argv[2:] or ["front"]
    W, H = (1280, 800) if len(names) < 3 else (960, 640)
    render(names, out=out, W=W, H=H)
