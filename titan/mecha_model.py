"""
TITAN-9 "ONI" — samurai-armoured combat mecha with a yakuza oni face.

All geometry is authored in world-space rest coordinates (A-pose, facing +Z, Y up) for the *unstretched*
skeleton; the builder lifts the upper body and stretches the legs (see skeleton.py).
Left-side parts (x > 0) are mirrored to produce the right side.

Design language: black-lacquer lamellar armour (do / sode / kusazuri / haidate / suneate), blood-red lacing and
sash, gold fittings, a fanged oni menpo under a horned kabuto, glowing red irezumi lines, and a katana + saya.
"""

import math

import numpy as np

from . import geometry as G
from .geometry import loft, lathe, cylinder, sphere, torus, rbox, extrude, tube, bezier, merge, compute_normals, Mesh
from .materials import (ARMOR_A, ARMOR_B, ARMOR_C, GUNMETAL, CHROME, ACCENT, RUBBER, GLOW_A, GLOW_B, VISOR,
                        CARBON, STEEL, LAMP, HOT)
from .skeleton import BI, NB, LIFT, LEG_STRETCH, HIP_Y
from .xform import translate as T, rot_x as RX, rot_y as RY, rot_z as RZ, scale as SC

F32 = np.float32
PI = math.pi

# katana constants (shared with animation / sword code)
KAT_GRIP = np.array([-0.90, 0.84, 0.0], F32)          # grip centre in the right hand (rest coords)
from .skeleton import SHEATH_GRIP as _SG
SHEATH_GRIP = np.array(_SG, F32)                     # hilt position on the left hip
SHEATH_DIR = np.array([0.20, -0.30, -0.93], F32)
SHEATH_DIR /= np.linalg.norm(SHEATH_DIR)
BLADE_LEN = 1.72
BLADE_START = 0.27


class MechaData:
    pass


class Builder:
    def __init__(self):
        self.items = []
        self.xf = None
        self.base = None

    def _place(self, mesh, M):
        if M is not None:
            mesh = mesh.transformed(M)
        if self.xf is not None:
            mesh = mesh.transformed(self.xf)
        return mesh

    def add(self, mesh, bone, mat, M=None):
        self.items.append((self._place(mesh, M), BI[bone], mat))

    def lr(self, mesh, bone, mat, M=None):
        """Add a left-side mesh (world coords, x>0) plus its mirrored right-side twin."""
        mesh = self._place(mesh, M)
        bl, br = (BI[bone + "L"], BI[bone + "R"]) if bone + "L" in BI else (BI[bone], BI[bone])
        self.items.append((mesh, bl, mat))
        self.items.append((mesh.mirrored_x(), br, mat))

    def finish(self):
        pos, nrm, mid, bone, idx = [], [], [], [], []
        off = 0
        for mesh, b, m in self.items:
            n = len(mesh.pos)
            pos.append(mesh.pos)
            nrm.append(mesh.nrm)
            mid.append(np.full(n, m, F32))
            bone.append(np.full(n, b, F32))
            idx.append(mesh.idx + off)
            off += n
        md = MechaData()
        md.pos = np.concatenate(pos).astype(F32)
        md.nrm = np.concatenate(nrm).astype(F32)
        md.mid = np.concatenate(mid).astype(F32)
        md.bone = np.concatenate(bone).astype(F32)
        md.idx = np.concatenate(idx).astype(np.uint32)
        return md


# ------------------------------------------------------------------ helpers
def frame(p0, p1):
    p0 = np.array(p0, F32)
    v = np.array(p1, F32) - p0
    L = float(np.linalg.norm(v))
    y = v / L
    ref = np.array([0, 0, 1], F32) if abs(y[2]) < 0.9 else np.array([1, 0, 0], F32)
    x = np.cross(ref, y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    M = np.eye(4, dtype=F32)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, p0
    return M, L


def seg(p0, p1, w, d, r=0.012, k=2):
    M, L = frame(p0, p1)
    return rbox(w, L, d, r, k).transformed(M @ T(0, L / 2, 0))


def rod(p0, p1, r, bevel=0.0, n=18):
    M, L = frame(p0, p1)
    return cylinder(r, L, n, bevel).transformed(M)


def bl(secs, cx=0.0, cz=0.0, n=5.5, seg_n=40, **kw):
    out = []
    for s in secs:
        out.append((s[0], s[1], s[2], s[3] if len(s) > 3 else n, cx, cz))
    return loft(out, seg=seg_n, **kw)


def ext(poly, depth, bevel=0.015, steps=3):
    return extrude(poly, depth, bevel, steps)


def axis_x_cyl(r, length, bevel=0.01, n=22):
    return cylinder(r, length, n, bevel, y0=-length / 2).transformed(RZ(PI / 2))


def bolt(r=0.016, h=0.014):
    return cylinder(r, h, 8, h * 0.3).transformed(RX(PI / 2))


def fang(length, r, curve=0.0):
    """Cone along +Y (base at origin). Optional curve bends the tip toward +Z."""
    m = cylinder(r, length, 8, 0.0, r_top=r * 0.06)
    if curve:
        p = m.pos.copy()
        p[:, 2] += curve * (p[:, 1] / length) ** 2 * length
        m = Mesh(p, compute_normals(p, m.idx), m.idx)
    return m


def blade_mesh(length=BLADE_LEN, width=0.078, thick=0.015, curve=0.11, seg=40, edge_glow=False):
    """Katana blade along +Z (from 0 to length), edge toward -Y, tip curving toward +Y (the back)."""
    secs = []
    for i in range(seg + 1):
        u = i / seg
        w = width * (1.0 - 0.25 * u) * 0.5
        if u > 0.94:
            w *= 1.0 - (u - 0.94) / 0.06 * 0.9
        if edge_glow:
            secs.append((u * length, thick * 0.5, w, 4.0, 0.0, 0.0))
        else:
            th = thick * 0.5 * (1.0 - 0.35 * u)
            secs.append((u * length, th, w, 4.5, 0.0, w * 0.15))
    m = loft(secs, seg=14, cap0=True, cap1=True)
    m = m.transformed(RX(PI / 2))                       # Y (length) -> Z, Z (width) -> -Y
    p = m.pos.copy()
    p[:, 1] += curve * (p[:, 2] / length) ** 2
    if edge_glow:
        p[:, 1] -= 0.024                                  # thin glowing line near the cutting edge
    return Mesh(p, compute_normals(p, m.idx), m.idx)


# ------------------------------------------------------------------ model
def build_mecha():
    B = Builder()
    B.base = T(0, LIFT, 0)          # the whole upper body rides up on the longer legs
    B.xf = B.base
    _pelvis(B)
    _spine(B)
    _torso(B)
    _backpack(B)
    _head(B)
    _shoulders(B)
    _arms(B)
    _hands(B)
    _legs(B)
    _katana(B)
    return B.finish()


# ------------------------------------------------------------------ pelvis: obi sash + kusazuri skirt
def _pelvis(B):
    B.add(loft([(1.50, 0.40, 0.24, 4), (1.62, 0.50, 0.29, 4), (1.78, 0.54, 0.31, 4), (1.98, 0.47, 0.27, 4)], seg=36),
          "pelvis", GUNMETAL)
    # red obi sash with gold kamon buckle and hanging tails
    B.add(bl([(1.935, 0.505, 0.305), (1.97, 0.55, 0.345), (2.05, 0.55, 0.345), (2.075, 0.505, 0.305)], n=5.0),
          "pelvis", ARMOR_B)
    B.add(cylinder(0.105, 0.04, 28, 0.01).transformed(RX(PI / 2)), "pelvis", ACCENT, T(0, 2.005, 0.335))
    B.add(cylinder(0.055, 0.03, 24, 0.006).transformed(RX(PI / 2)), "pelvis", GLOW_A, T(0, 2.005, 0.37))
    for k, ang in enumerate((-0.12, 0.10)):
        B.add(ext([(-0.05, 0.0), (0.05, 0.0), (0.045, -0.36), (-0.045, -0.30)], 0.03, 0.008), "pelvis", ARMOR_B,
              T(-0.18 + 0.06 * k, 1.95, 0.355) @ RZ(ang) @ RX(-0.06))
    # kusazuri: two rows of hanging lacquered skirt plates around the hips (left-hip slot left open for the saya)
    for row, (y_top, drop, R_x, R_z, shift) in enumerate(((1.945, 0.44, 0.58, 0.40, 0.0), (1.80, 0.36, 0.62, 0.43, 0.5))):
        step = 24.0
        for a in range(-7, 8):
            thd = a * step + shift * step
            if abs(thd) > 178 or 18 <= thd <= 112:
                continue
            t = math.radians(thd)
            x, z = R_x * math.sin(t), R_z * math.cos(t)
            w = 0.098 if row == 0 else 0.104
            plate = ext([(-w, 0.0), (w, 0.0), (w * 0.9, -drop), (-w * 0.9, -drop)], 0.035, 0.009)
            M = T(x, y_top, z) @ RY(t) @ RX(-0.22 - 0.05 * row)
            B.add(plate, "pelvis", ARMOR_A, M)
            B.add(rbox(w * 1.7, 0.016, 0.03, 0.004, 1), "pelvis", ARMOR_B, M @ T(0, -drop + 0.03, 0.012))
            B.add(rbox(0.014, drop * 0.82, 0.014, 0.004, 1), "pelvis", ARMOR_B, M @ T(0, -drop * 0.5, 0.02))
            if row == 0:
                B.add(bolt(0.011, 0.01), "pelvis", ACCENT, M @ T(0, -0.04, 0.02))
    # hip ball joints
    B.lr(sphere(0.215, 26, 14), "pelvis", GUNMETAL, T(0.40, 1.72, 0))
    B.lr(torus(0.205, 0.022, 28, 8), "pelvis", ACCENT, T(0.40, 1.72, 0))


def _spine(B):
    secs = []
    for i in range(8):
        y = 1.97 + i * 0.036
        r = 0.30 if i % 2 == 0 else 0.345
        secs.append((y, r, r * 0.74, 2.6))
    B.add(loft(secs, seg=30), "spine", RUBBER)
    for sx in (-1, 1):
        B.add(rod((sx * 0.21, 1.99, 0.19), (sx * 0.21, 2.20, 0.19), 0.058, 0.01), "spine", GUNMETAL)
        B.add(rod((sx * 0.21, 2.12, 0.19), (sx * 0.21, 2.42, 0.19), 0.032), "chest", CHROME)
        B.add(rod((sx * 0.14, 1.99, -0.17), (sx * 0.14, 2.22, -0.17), 0.05, 0.008), "spine", GUNMETAL)
        pts = bezier((sx * 0.30, 2.0, 0.10), (sx * 0.40, 2.1, 0.2), (sx * 0.38, 2.2, 0.22), (sx * 0.30, 2.30, 0.12), 14)
        B.add(tube(pts, 0.018, 8), "spine", RUBBER)
    B.add(rod((0, 1.99, -0.22), (0, 2.36, -0.22), 0.04), "chest", CHROME)


# ------------------------------------------------------------------ torso: lamellar do cuirass
def _torso_dims(i):
    return 0.50 + 0.058 * i, 0.375 + 0.021 * i


def _torso_z(x, y):
    """Front surface z of the cuirass at (x, y) (superellipse n=6) for lacing that hugs the shell."""
    i = float(np.clip((y - 2.40) / 0.1, 0, 5.99))
    i0 = int(i)
    hx0, hz0 = _torso_dims(i0)
    q = min(abs(x) / hx0, 0.999)
    return hz0 * (1.0 - q ** 6) ** (1.0 / 6.0)


def _torso(B):
    B.add(loft([(2.20, 0.38, 0.27, 4), (2.40, 0.58, 0.38, 4), (2.80, 0.70, 0.40, 4), (3.00, 0.50, 0.30, 4)], seg=36),
          "chest", GUNMETAL)
    rows = 6
    for i in range(rows):
        y0 = 2.40 + 0.1 * i
        hx, hz = _torso_dims(i)
        k = 0.986
        B.add(bl([(y0, hx * k, hz * k), (y0 + 0.014, hx, hz), (y0 + 0.088, hx, hz), (y0 + 0.1, hx * k, hz * k)], n=6.0, seg_n=52),
              "chest", ARMOR_A)
        B.add(bl([(y0 - 0.006, hx + 0.006, hz + 0.006), (y0 + 0.010, hx + 0.006, hz + 0.006)], n=6.0, seg_n=52), "chest", ARMOR_B)
    # red lacing cords running down the front and back of the cuirass
    for sgn in (1, -1):
        for xo in (-0.46, -0.30, -0.15, 0.15, 0.30, 0.46):
            pts = []
            for y in np.linspace(2.42, 3.00, 12):
                hx, _ = _torso_dims(int(np.clip((y - 2.40) / 0.1, 0, 5)))
                x = xo * hx / 0.50
                pts.append((x, y, sgn * (_torso_z(x, y) + 0.014)))
            B.add(tube(np.array(pts), 0.012, 6), "chest", ARMOR_B)
    # muna-ita breast plate with gold border + reactor
    hexp = [(-0.42, -0.26), (0.42, -0.26), (0.54, 0.02), (0.40, 0.30), (-0.40, 0.30), (-0.54, 0.02)]
    B.add(ext([(x * 1.07, y * 1.07) for x, y in hexp], 0.05, 0.012), "chest", ACCENT, T(0, 2.74, 0.505))
    B.add(ext(hexp, 0.075, 0.02), "chest", ARMOR_A, T(0, 2.74, 0.53))
    zc = 0.585
    B.add(cylinder(0.255, 0.07, 36, 0.012).transformed(RX(PI / 2)), "chest", GUNMETAL, T(0, 2.68, zc - 0.05))
    B.add(torus(0.212, 0.034, 40, 10).transformed(RX(PI / 2)), "chest", ACCENT, T(0, 2.68, zc + 0.03))
    B.add(cylinder(0.168, 0.035, 36, 0.006).transformed(RX(PI / 2)), "chest", GLOW_A, T(0, 2.68, zc + 0.0))
    B.add(cylinder(0.085, 0.02, 28, 0.004).transformed(RX(PI / 2)), "chest", LAMP, T(0, 2.68, zc + 0.03))
    for a in range(8):
        ar = a * PI / 4
        B.add(ext([(-0.03, 0.0), (0.03, 0.0), (0.0, 0.07)], 0.03, 0.006), "chest", ACCENT,
              T(0.245 * math.cos(ar), 2.68 + 0.245 * math.sin(ar), zc + 0.015) @ RZ(ar - PI / 2))
    # shoulder cords (watagami) and standing collar
    B.lr(ext([(0.08, 0.0), (0.34, 0.0), (0.50, -0.34), (0.22, -0.34)], 0.05, 0.012), "chest", ARMOR_B,
         T(0.0, 3.04, 0.30) @ RX(-0.35))
    B.lr(ext([(0.08, 0.0), (0.34, 0.0), (0.50, -0.34), (0.22, -0.34)], 0.05, 0.012), "chest", ARMOR_B,
         T(0.0, 3.04, -0.30) @ RX(0.35) @ RY(PI))
    B.add(bl([(2.98, 0.33, 0.25), (3.08, 0.37, 0.28), (3.22, 0.34, 0.25)], n=4.0), "chest", ARMOR_A)
    B.add(bl([(3.0, 0.30, 0.22), (3.12, 0.33, 0.245), (3.21, 0.305, 0.22)], n=4.0), "chest", ARMOR_B)
    B.add(torus(0.335, 0.016, 36, 8), "chest", ACCENT, T(0, 3.215, 0.0))
    # back: gold clan roundel
    B.add(cylinder(0.20, 0.04, 32, 0.008).transformed(RX(-PI / 2)), "chest", ACCENT, T(0, 2.72, -0.50))
    B.add(cylinder(0.11, 0.05, 28, 0.008).transformed(RX(-PI / 2)), "chest", GLOW_A, T(0, 2.72, -0.53))


def _backpack(B):
    B.add(rbox(0.80, 0.62, 0.30, 0.05, 3), "chest", ARMOR_A, T(0, 2.68, -0.68))
    B.add(rbox(0.50, 0.34, 0.06, 0.02, 2), "chest", GUNMETAL, T(0, 2.72, -0.84))
    for i in range(4):
        B.add(rbox(0.40, 0.014, 0.05, 0.004, 1), "chest", GLOW_B, T(0, 2.60 + i * 0.07, -0.875))
    for sx in (-1, 1):
        B.add(cylinder(0.165, 0.58, 28, 0.03), "chest", GUNMETAL, T(sx * 0.30, 2.30, -0.68))
        B.add(torus(0.165, 0.016, 28, 8), "chest", ACCENT, T(sx * 0.30, 2.56, -0.68))
        B.add(torus(0.165, 0.016, 28, 8), "chest", ACCENT, T(sx * 0.30, 2.40, -0.68))
    prof = [(0.0, 2.27), (0.115, 2.26), (0.175, 2.05), (0.205, 2.05), (0.208, 2.07), (0.17, 2.29), (0.0, 2.30)]
    for sx, name in ((1, "nozL"), (-1, "nozR")):
        B.add(lathe(prof, 28, crease_deg=50), name, STEEL, T(sx * 0.30, 0.0, -0.68))
        B.add(cylinder(0.113, 0.012, 24, 0.0), name, HOT, T(sx * 0.30, 2.245, -0.68))
        B.add(torus(0.195, 0.014, 28, 8), name, ACCENT, T(sx * 0.30, 2.08, -0.68))
    # wings: slender swept oni blades with glowing red slits (bone "wing")
    B.lr(cylinder(0.05, 0.16, 18, 0.01).transformed(RX(PI / 2)), "wing", GUNMETAL, T(0.40, 2.86, -0.72))
    Mw = T(0.40, 2.86, -0.75)
    B.lr(ext([(0.0, -0.05), (0.10, -0.05), (0.66, 0.92), (0.58, 1.34), (0.30, 0.86), (0.06, 0.46)], 0.04, 0.01), "wing", ARMOR_A, Mw)
    B.lr(ext([(0.04, -0.09), (0.16, -0.09), (0.52, 0.34), (0.30, 0.40)], 0.05, 0.01), "wing", ARMOR_B, Mw)
    B.lr(ext([(0.16, 0.42), (0.30, 0.54), (0.50, 0.80), (0.48, 0.88), (0.28, 0.68), (0.14, 0.50)], 0.052, 0.006), "wing", GLOW_A, Mw)
    B.lr(ext([(0.62, 0.95), (0.66, 0.92), (0.60, 1.36), (0.57, 1.32)], 0.056, 0.004), "wing", ACCENT, Mw)


# ------------------------------------------------------------------ head: kabuto + fanged oni menpo
_BOWL_Y = [3.30, 3.38, 3.52, 3.65, 3.76, 3.82, 3.84]
_BOWL_HX = [0.30, 0.32, 0.31, 0.26, 0.17, 0.08, 0.03]
_BOWL_HZ = [0.31, 0.33, 0.32, 0.28, 0.19, 0.09, 0.035]


def _bowl_z(x, y):
    """Front surface z of the helmet bowl at (x, y), for decals hugging the shell."""
    a = float(np.interp(y, _BOWL_Y, _BOWL_HX))
    c = float(np.interp(y, _BOWL_Y, _BOWL_HZ))
    q = min(abs(x) / max(a, 1e-3), 0.999)
    return 0.02 + c * (1.0 - q ** 2.5) ** (1.0 / 2.5)


def _glow_line(B, pts, r=0.0055, mat=GLOW_A):
    B.add(tube(np.asarray(pts), r, 8), "head", mat)


def _head(B):
    # neck gorget
    B.add(loft([(2.98, 0.17, 0.17, 2.4), (3.14, 0.15, 0.15, 2.4)], seg=24), "neck", GUNMETAL)
    for i in range(3):
        B.add(torus(0.155, 0.02, 24, 8), "neck", RUBBER, T(0, 3.02 + i * 0.045, 0))
    for sx in (-1, 1):
        pts = bezier((sx * 0.12, 2.99, -0.10), (sx * 0.20, 3.05, -0.22), (sx * 0.16, 3.2, -0.2), (sx * 0.12, 3.27, -0.12), 14)
        B.add(tube(pts, 0.022, 10), "neck", RUBBER)
    B.xf = B.base @ T(0, 3.12, 0.02) @ SC(1.24) @ T(0, -3.12, -0.02)
    H = "head"
    # kabuto bowl + gold ribs
    B.add(bl([(3.30, 0.30, 0.31, 2.6), (3.38, 0.32, 0.33, 2.6), (3.52, 0.31, 0.32, 2.5), (3.65, 0.26, 0.28, 2.4),
              (3.76, 0.17, 0.19, 2.3), (3.82, 0.08, 0.09, 2.2), (3.84, 0.03, 0.035, 2.2)], cz=0.02, n=2.5, seg_n=40), H, ARMOR_A)
    for k in range(-3, 4):
        th = k * 0.30
        pts = []
        for y in np.linspace(3.36, 3.82, 12):
            r_x = float(np.interp(y, _BOWL_Y, _BOWL_HX)) + 0.006
            r_z = float(np.interp(y, _BOWL_Y, _BOWL_HZ)) + 0.006
            pts.append((r_x * math.sin(th), y, 0.02 + r_z * math.cos(th)))
        B.add(tube(np.array(pts), 0.0085, 6), H, ACCENT)
    # visor brim + angry brow plates
    B.add(ext([(-0.30, 0.0), (0.30, 0.0), (0.27, 0.24), (0.09, 0.33), (-0.09, 0.33), (-0.27, 0.24)], 0.03, 0.008), H, ARMOR_A,
          T(0, 3.60, 0.15) @ RX(PI / 2 - 0.30))
    B.lr(ext([(0.02, 3.465), (0.30, 3.515), (0.33, 3.615), (0.02, 3.535)], 0.075, 0.012), H, ARMOR_A, T(0, 0, 0.300))
    B.lr(ext([(0.02, 3.460), (0.30, 3.508), (0.325, 3.52), (0.02, 3.47)], 0.082, 0.004), H, ACCENT, T(0, 0, 0.300))
    # eye sockets + glowing slit eyes (inner corner low = scowl)
    B.add(rbox(0.56, 0.12, 0.14, 0.02, 2), H, VISOR, T(0, 3.43, 0.24))
    B.lr(ext([(0.035, 3.402), (0.225, 3.438), (0.245, 3.472), (0.035, 3.438)], 0.04, 0.006), H, GLOW_A, T(0, 0, 0.318))
    B.lr(ext([(0.05, 3.414), (0.19, 3.440), (0.20, 3.452), (0.05, 3.430)], 0.05, 0.004), H, LAMP, T(0, 0, 0.318))
    # oni menpo (red lacquer lower mask)
    B.add(ext([(-0.13, 3.14), (0.13, 3.14), (0.18, 3.31), (0.0, 3.38), (-0.18, 3.31)], 0.07, 0.016), H, ARMOR_B, T(0, 0, 0.275))
    B.lr(ext([(0.10, 3.16), (0.30, 3.22), (0.34, 3.36), (0.20, 3.41), (0.12, 3.33)], 0.06, 0.014), H, ARMOR_B,
         T(0.0, 0, 0.255) @ T(0.12, 0, 0) @ RY(0.55) @ T(-0.12, 0, 0))
    B.add(ext([(-0.10, 3.095), (0.10, 3.095), (0.14, 3.18), (-0.14, 3.18)], 0.07, 0.014), H, ARMOR_B, T(0, 0, 0.252))
    B.add(ext([(-0.045, 3.32), (0.045, 3.32), (0.0, 3.50)], 0.06, 0.012), H, ARMOR_B, T(0, 0, 0.292))     # nose ridge
    B.add(rbox(0.32, 0.075, 0.07, 0.014, 2), H, VISOR, T(0, 3.253, 0.30))                                  # mouth cavity
    # fangs: upper row, lower row and big curved canines
    for k in range(-3, 4):
        L = 0.040 if abs(k) < 3 else 0.088
        B.add(fang(L, 0.014 if abs(k) < 3 else 0.019).transformed(RX(PI)), H, CHROME, T(k * 0.040, 3.288, 0.322))
    for k in range(-2, 3):
        L = 0.036 if abs(k) < 2 else 0.078
        B.add(fang(L, 0.013 if abs(k) < 2 else 0.018, curve=0.25), H, CHROME, T(k * 0.045, 3.222, 0.322))
    for sx in (-1, 1):
        B.add(fang(0.13, 0.022, curve=0.2).transformed(RZ(sx * -1.15) @ RX(0.2)), H, ACCENT, T(sx * 0.315, 3.28, 0.15))   # cheek tusks
        B.add(seg((sx * 0.03, 3.315, 0.340), (sx * 0.17, 3.285, 0.315), 0.03, 0.026, 0.008, 1), H, ACCENT)     # moustache ridge
    B.add(fang(0.16, 0.034, curve=-0.35).transformed(RX(PI - 0.35)), H, ACCENT, T(0, 3.13, 0.285))          # chin spike
    # glowing irezumi lines: forehead and cheeks
    zb = lambda x, y: _bowl_z(x, y) + 0.010
    for xs, ys, xe, ye in ((0.0, 3.80, 0.0, 3.60), (0.07, 3.76, 0.15, 3.58), (-0.07, 3.76, -0.15, 3.58)):
        pts = []
        for u in np.linspace(0, 1, 10):
            x, y = xs + (xe - xs) * u, ys + (ye - ys) * u
            pts.append((x, y, zb(x, y)))
        _glow_line(B, pts)
    for sx in (-1, 1):
        for dy in (0.0, -0.032, -0.064):
            pts = bezier((sx * 0.14, 3.335 + dy, 0.318), (sx * 0.24, 3.31 + dy, 0.30), (sx * 0.31, 3.27 + dy * 0.9, 0.20),
                         (sx * 0.34, 3.23 + dy * 0.8, 0.07), 12)
            _glow_line(B, pts, 0.0048)
        pts = bezier((sx * 0.04, 3.48, 0.318), (sx * 0.12, 3.545, 0.32), (sx * 0.22, 3.60, 0.30), (sx * 0.29, 3.64, 0.25), 10)
        _glow_line(B, pts, 0.0045)
    # horns: two long swept front horns (gold), two short rear horns (red)
    for sx in (-1, 1):
        front = bezier((sx * 0.10, 3.66, 0.25), (sx * 0.16, 3.80, 0.31), (sx * 0.34, 3.98, 0.29), (sx * 0.66, 4.26, 0.10), 28)
        rear = bezier((sx * 0.13, 3.70, -0.03), (sx * 0.19, 3.88, -0.18), (sx * 0.30, 4.00, -0.38), (sx * 0.38, 4.04, -0.62), 22)
        B.add(tube(front, np.linspace(0.066, 0.004, len(front)), 10), H, ACCENT)
        B.add(tube(rear, np.linspace(0.034, 0.004, len(rear)), 10), H, ARMOR_B)
        B.add(cylinder(0.078, 0.07, 18, 0.01).transformed(RZ(sx * -0.45)), H, ARMOR_A, T(sx * 0.10, 3.62, 0.25))
        for u in (0.12, 0.30):
            i = int(u * 27)
            B.add(torus(0.066 - 0.05 * u, 0.013, 14, 6).transformed(RZ(sx * -0.9)), H, ARMOR_B, T(*front[i]))
    B.add(ext([(-0.028, 3.80), (0.028, 3.80), (0.0, 4.22)], 0.022, 0.005), H, ACCENT, T(0, 0, 0.17))        # centre maedate blade
    # shikoro neck guard: three flared rows of lacquered plates around the back and sides
    for layer in range(3):
        R = 0.355 + 0.05 * layer
        y_top = 3.305 - 0.085 * layer
        for k in range(-6, 7):
            th = math.radians(180 + k * 19.0 + (9.5 if layer % 2 else 0))
            if abs(math.degrees(th) - 180) > 112:
                continue
            w = 0.082
            plate = ext([(-w, 0.0), (w, 0.0), (w * 1.1, -0.12), (-w * 1.1, -0.12)], 0.03, 0.008)
            M = T(R * math.sin(th), y_top, 0.02 + R * math.cos(th)) @ RY(th) @ RX(-0.30)
            B.add(plate, H, ARMOR_A, M)
            B.add(rbox(w * 2.0, 0.014, 0.03, 0.004, 1), H, ARMOR_B, M @ T(0, -0.115, 0.01))
    # fukigaeshi side flaps with gold trim
    B.lr(ext([(0.0, 0.0), (0.19, 0.05), (0.23, 0.26), (0.07, 0.32), (-0.02, 0.15)], 0.04, 0.01), H, ARMOR_A,
         T(0.26, 3.46, 0.21) @ RY(0.95) @ RZ(-0.15))
    B.lr(ext([(0.0, 0.0), (0.205, 0.055), (0.245, 0.28), (0.075, 0.34), (-0.025, 0.155)], 0.03, 0.008), H, ACCENT,
         T(0.26, 3.455, 0.205) @ RY(0.95) @ RZ(-0.15))
    # ears / sensors
    B.lr(axis_x_cyl(0.085, 0.10, 0.012), H, GUNMETAL, T(0.325, 3.40, 0.02))
    B.lr(axis_x_cyl(0.035, 0.12, 0.004), H, GLOW_B, T(0.325, 3.40, 0.02))
    B.xf = B.base


# ------------------------------------------------------------------ shoulders: sode (layered plates)
def _shoulders(B):
    P = "pad"
    cx = 0.98
    for i in range(5):
        y = 3.16 - 0.082 * i
        w = 0.56 + 0.030 * i
        d = 0.60 + 0.012 * i
        tilt = -0.10 - 0.045 * i
        M = T(cx + 0.012 * i, y, 0.0) @ RZ(tilt)
        B.lr(rbox(w, 0.075, d, 0.018, 2), P, ARMOR_A, M)
        B.lr(rbox(w * 1.02, 0.014, d * 1.01, 0.005, 1), P, ARMOR_B, M @ T(0, -0.038, 0))        # red edge lacing
        if i == 0:
            for (sx_, sz_, bw, bd) in ((0, 1, w * 1.03, 0.014), (0, -1, w * 1.03, 0.014), (1, 0, 0.014, d * 1.03), (-1, 0, 0.014, d * 1.03)):
                B.lr(rbox(bw, 0.016, bd, 0.004, 1), P, ACCENT, M @ T(sx_ * w * 0.5, 0.036, sz_ * d * 0.5))   # gold border
    for z in (0.31, -0.31):
        for x in (0.80, 0.93, 1.06, 1.19):
            B.lr(seg((x, 3.17, z), (x + 0.14, 2.78, z), 0.026, 0.014, 0.004, 1), P, ARMOR_B)
    B.lr(cylinder(0.075, 0.03, 24, 0.006), P, ACCENT, T(cx, 3.205, 0.0))
    B.lr(cylinder(0.04, 0.03, 20, 0.004), P, GLOW_A, T(cx, 3.215, 0.0))
    B.lr(sphere(0.245, 26, 14), "uarm", GUNMETAL, T(0.88, 2.72, 0))


# ------------------------------------------------------------------ arms: segmented black + gold + red
def _arms(B):
    B.lr(cylinder(0.15, 0.78, 24, 0.01), "uarm", GUNMETAL, T(0.89, 1.98, 0))
    for k, (y0, y1) in enumerate(((2.60, 2.42), (2.40, 2.22), (2.20, 2.04))):
        hx = 0.22 - 0.004 * k
        B.lr(bl([(y0, hx, hx + 0.01), (y0 - 0.01, hx + 0.02, hx + 0.03), (y1 + 0.01, hx + 0.02, hx + 0.03), (y1, hx, hx + 0.01)],
                cx=0.89, n=5.0), "uarm", ARMOR_A)
        B.lr(bl([(y1 + 0.016, hx + 0.026, hx + 0.036), (y1 + 0.004, hx + 0.026, hx + 0.036)], cx=0.89, n=5.0), "uarm", ARMOR_B)
    B.lr(torus(0.245, 0.012, 28, 6), "uarm", ACCENT, T(0.89, 2.61, 0.0))
    B.lr(axis_x_cyl(0.165, 0.46, 0.02), "uarm", GUNMETAL, T(0.92, 1.98, -0.02))
    B.lr(axis_x_cyl(0.10, 0.54, 0.01), "uarm", ACCENT, T(0.92, 1.98, -0.02))
    # forearm: kote bracer with gold stripe + red cuff
    B.lr(cylinder(0.15, 0.74, 24, 0.01), "farm", GUNMETAL, T(0.93, 1.24, -0.02))
    B.lr(bl([(1.93, 0.20, 0.21), (1.84, 0.255, 0.27), (1.60, 0.285, 0.31), (1.44, 0.26, 0.28), (1.30, 0.21, 0.22)], cx=0.93, cz=-0.02, n=5.0),
         "farm", ARMOR_A)
    B.lr(bl([(1.34, 0.222, 0.232), (1.31, 0.222, 0.232)], cx=0.93, cz=-0.02, n=5.0), "farm", ARMOR_B)
    B.lr(bl([(1.46, 0.272, 0.292), (1.50, 0.274, 0.294)], cx=0.93, cz=-0.02, n=5.0), "farm", ACCENT)
    B.lr(ext([(-0.17, 0.30), (0.17, 0.30), (0.21, -0.28), (-0.21, -0.28)], 0.05, 0.012), "farm", ARMOR_A,
         T(0.93 + 0.28, 1.62, -0.02) @ RZ(-0.04) @ RY(-PI / 2))
    B.lr(ext([(-0.025, 0.28), (0.025, 0.28), (0.025, -0.26), (-0.025, -0.26)], 0.07, 0.008), "farm", ACCENT,
         T(0.93 + 0.28, 1.62, -0.02) @ RZ(-0.04) @ RY(-PI / 2))
    B.lr(ext([(-0.16, 0.0), (0.16, 0.0), (0.12, 0.17), (-0.12, 0.17)], 0.055, 0.012), "farm", ARMOR_A,
         T(0.93, 1.90, 0.255) @ RX(0.55))
    B.lr(rod((0.66, 1.86, 0.10), (0.67, 1.45, 0.10), 0.03), "farm", CHROME)
    B.lr(torus(0.205, 0.03, 30, 8), "farm", GUNMETAL, T(0.93, 1.27, -0.02))


def _hands(B):
    B.lr(sphere(0.125, 20, 12), "hand", GUNMETAL, T(0.94, 1.15, 0))
    B.lr(rbox(0.15, 0.27, 0.28, 0.03, 2), "hand", GUNMETAL, T(0.94, 1.01, 0))
    B.lr(rbox(0.055, 0.21, 0.25, 0.02, 2), "hand", ARMOR_A, T(1.03, 1.03, 0))
    B.lr(rbox(0.058, 0.04, 0.26, 0.01, 1), "hand", ACCENT, T(1.03, 0.93, 0))
    B.lr(axis_x_cyl(0.065, 0.026, 0.004), "hand", GLOW_A, T(0.862, 1.02, 0))
    B.lr(axis_x_cyl(0.085, 0.03, 0.006), "hand", ACCENT, T(0.872, 1.02, 0))
    lens = {"i": (0.16, 0.14), "m": (0.17, 0.15), "r": (0.16, 0.14), "p": (0.125, 0.11)}
    for f, z in (("i", 0.095), ("m", 0.032), ("r", -0.032), ("p", -0.095)):
        la, lb = lens[f]
        wd = 0.05 if f != "p" else 0.044
        B.lr(seg((0.94, 0.93, z), (0.94, 0.93 - la, z), wd, 0.046, 0.016), "f%sa" % f, GUNMETAL)
        B.lr(sphere(0.027, 14, 8), "f%sa" % f, STEEL, T(0.94, 0.93, z))
        B.lr(seg((0.94, 0.765, z), (0.94, 0.765 - lb, z), wd * 0.95, 0.042, 0.015), "f%sb" % f, ARMOR_A)
        B.lr(rbox(wd * 0.9, 0.03, 0.04, 0.012, 1), "f%sb" % f, RUBBER, T(0.94, 0.765 - lb, z))
    B.lr(seg((0.90, 1.08, 0.13), (0.90, 0.985, 0.185), 0.056, 0.052, 0.018), "tha", GUNMETAL)
    B.lr(sphere(0.03, 14, 8), "tha", STEEL, T(0.90, 1.08, 0.13))
    B.lr(seg((0.90, 0.975, 0.19), (0.90, 0.885, 0.245), 0.05, 0.046, 0.016), "thb", ARMOR_A)


# ------------------------------------------------------------------ legs: haidate / hizayoroi / suneate
def _legs(B):
    B.xf = T(0, LIFT + HIP_Y, 0) @ np.diag([1.0, LEG_STRETCH, 1.0, 1.0]).astype(F32) @ T(0, -HIP_Y, 0)
    B.lr(cylinder(0.18, 0.78, 24, 0.01), "thigh", GUNMETAL, T(0.40, 0.96, 0))
    # haidate: stacked thigh plates with lacing
    for k in range(5):
        y0 = 1.64 - 0.13 * k
        hx = 0.215 + 0.011 * k
        hz = 0.235 + 0.012 * k
        B.lr(bl([(y0 - 0.12, hx * 0.99, hz * 0.99), (y0 - 0.108, hx, hz), (y0 - 0.012, hx, hz), (y0, hx * 0.99, hz * 0.99)],
                cx=0.40, cz=0.02, n=5.0), "thigh", ARMOR_A)
        B.lr(bl([(y0 - 0.126, hx + 0.005, hz + 0.005), (y0 - 0.112, hx + 0.005, hz + 0.005)], cx=0.40, cz=0.02, n=5.0), "thigh", ARMOR_B)
    B.lr(ext([(-0.16, 0.30), (0.16, 0.30), (0.19, -0.26), (-0.19, -0.26)], 0.04, 0.01), "thigh", ARMOR_A,
         T(0.40 + 0.265, 1.31, 0.02) @ RY(-PI / 2))
    B.lr(rbox(0.04, 0.5, 0.012, 0.004, 1), "thigh", ACCENT, T(0.40, 1.31, 0.30))
    B.lr(rod((0.40, 1.58, -0.27), (0.40, 1.04, -0.25), 0.036), "thigh", CHROME)
    B.lr(rod((0.40, 1.58, -0.27), (0.40, 1.34, -0.265), 0.055, 0.006), "thigh", GUNMETAL)
    B.lr(axis_x_cyl(0.16, 0.46, 0.02), "thigh", GUNMETAL, T(0.40, 0.95, 0))
    B.lr(axis_x_cyl(0.095, 0.54, 0.01), "thigh", ACCENT, T(0.40, 0.95, 0))
    # knee guard: black dome with a gold ring
    kn = sphere(0.19, 26, 14).transformed(np.diag([1.0, 0.95, 0.78, 1.0]).astype(F32))
    B.lr(kn, "shin", ARMOR_A, T(0.40, 0.98, 0.20))
    B.lr(torus(0.17, 0.016, 28, 7).transformed(RX(PI / 2)), "shin", ACCENT, T(0.40, 0.98, 0.33))
    B.lr(cylinder(0.05, 0.03, 20, 0.006).transformed(RX(PI / 2)), "shin", GLOW_A, T(0.40, 0.98, 0.345))
    # suneate shin guard
    B.lr(cylinder(0.15, 0.70, 24, 0.01), "shin", GUNMETAL, T(0.40, 0.26, 0))
    B.lr(bl([(0.88, 0.20, 0.22), (0.76, 0.25, 0.29), (0.46, 0.245, 0.29), (0.31, 0.18, 0.21), (0.26, 0.17, 0.19)], cx=0.40, cz=0.03, n=5.0),
         "shin", ARMOR_A)
    B.lr(ext([(-0.16, 0.30), (0.16, 0.30), (0.12, -0.30), (-0.12, -0.30)], 0.05, 0.012), "shin", ARMOR_A, T(0.40, 0.62, 0.325) @ RX(0.06))
    B.lr(rbox(0.035, 0.58, 0.014, 0.004, 1), "shin", ACCENT, T(0.40, 0.62, 0.355))
    for y in (0.78, 0.46):
        B.lr(bl([(y, 0.256, 0.30), (y + 0.03, 0.258, 0.302)], cx=0.40, cz=0.03, n=5.0), "shin", ARMOR_B)
    B.lr(bl([(0.84, 0.14, 0.11), (0.62, 0.18, 0.14), (0.40, 0.16, 0.12)], cx=0.40, cz=-0.30, n=3.5), "shin", ARMOR_A)
    B.lr(cylinder(0.105, 0.12, 22, 0.01, r_top=0.075), "shin", STEEL, T(0.40, 0.31, -0.30))
    B.lr(cylinder(0.062, 0.012, 20, 0.0), "shin", HOT, T(0.40, 0.305, -0.30))
    B.lr(axis_x_cyl(0.13, 0.40, 0.02), "shin", GUNMETAL, T(0.40, 0.24, 0.0))
    # foot (not stretched: it still stands on the floor)
    B.xf = None
    B.lr(ext([(-0.30, 0.04), (0.50, 0.04), (0.58, 0.10), (0.50, 0.16), (0.30, 0.22), (0.08, 0.30), (-0.18, 0.30), (-0.32, 0.18)],
             0.34, 0.03), "foot", ARMOR_A, T(0.40, 0, 0) @ RY(-PI / 2))
    B.lr(ext([(0.34, 0.05), (0.52, 0.05), (0.60, 0.105), (0.52, 0.165), (0.34, 0.205)], 0.37, 0.02), "foot", ARMOR_B,
         T(0.40, 0, 0) @ RY(-PI / 2))
    B.lr(rbox(0.36, 0.045, 0.92, 0.015, 2), "foot", RUBBER, T(0.40, 0.024, 0.14))
    B.lr(rbox(0.26, 0.12, 0.13, 0.02, 2), "foot", ARMOR_A, T(0.40, 0.14, -0.285))
    B.lr(rbox(0.22, 0.05, 0.2, 0.015, 2), "foot", GUNMETAL, T(0.40, 0.31, 0.04))
    B.lr(rbox(0.30, 0.012, 0.05, 0.004, 1), "foot", ACCENT, T(0.40, 0.255, 0.30))
    B.xf = B.base


# ------------------------------------------------------------------ katana, saya
OVAL = np.diag([1.0, 0.82, 1.0, 1.0]).astype(F32)


def _katana_parts(include_blade):
    """Parts of the katana in local coords: origin = grip centre, +Z along the blade, edge toward -Y."""
    parts = []
    parts.append((cylinder(0.034, 0.46, 20, 0.006, y0=-0.23).transformed(RX(PI / 2)).transformed(OVAL), ARMOR_A))   # grip core
    for k in range(9):                                                                                      # red ito wrap
        ring = torus(0.034, 0.0075, 18, 5).transformed(RX(PI / 2)).transformed(OVAL).transformed(T(0, 0, -0.19 + k * 0.047))
        parts.append((ring, ARMOR_B))
    parts.append((cylinder(0.092, 0.016, 28, 0.004).transformed(RX(PI / 2)).transformed(T(0, 0, 0.235)), ACCENT))  # tsuba
    parts.append((cylinder(0.058, 0.04, 20, 0.008).transformed(RX(PI / 2)).transformed(T(0, 0, 0.215)), ACCENT))   # habaki collar
    parts.append((sphere(0.046, 16, 8).transformed(np.diag([1.0, 0.82, 0.8, 1.0]).astype(F32)).transformed(T(0, 0, -0.24)), ACCENT))
    if include_blade:
        parts.append((blade_mesh().transformed(T(0, 0, BLADE_START)), CHROME))
        parts.append((blade_mesh(edge_glow=True, width=0.02, thick=0.017, curve=0.11).transformed(T(0, 0, BLADE_START + 0.02)), GLOW_A))
    return parts


def _katana(B):
    B.xf = B.base
    # in the right hand
    for mesh, mat in _katana_parts(True):
        B.add(mesh, "kat", mat, T(*KAT_GRIP))
    # on the left hip (hilt only; the blade lives inside the saya)
    z = SHEATH_DIR
    up = np.array([0, 1, 0], F32)
    ya = -(up - np.dot(up, z) * z)
    ya /= np.linalg.norm(ya)
    xa = np.cross(ya, z)
    R = np.eye(4, dtype=F32)
    R[:3, 0], R[:3, 1], R[:3, 2] = xa, ya, z
    Msh = T(*SHEATH_GRIP) @ R
    for mesh, mat in _katana_parts(False):
        B.add(mesh, "katS", mat, Msh)
    # saya (scabbard): glossy black lacquer with gold fittings and a red sageo cord
    p0 = SHEATH_GRIP + SHEATH_DIR * 0.27
    p1 = SHEATH_GRIP + SHEATH_DIR * (0.27 + 1.74)
    M, L = frame(p0, p1)
    secs = [(0.0, 0.050, 0.060, 3.0), (0.04, 0.046, 0.054, 3.0), (L * 0.5, 0.040, 0.048, 3.0), (L - 0.05, 0.036, 0.044, 3.0), (L, 0.016, 0.02, 3.0)]
    B.add(loft(secs, seg=18).transformed(M), "pelvis", ARMOR_A)
    for yy, rr in ((0.0, 0.07), (0.30, 0.06), (L - 0.06, 0.05)):
        B.add(torus(rr * 0.82, 0.013, 18, 6).transformed(M @ T(0, yy, 0)), "pelvis", ACCENT)
    B.add(cylinder(0.026, 0.07, 14, 0.006, r_top=0.014).transformed(M @ T(0, L - 0.02, 0)), "pelvis", ACCENT)
    sag = bezier(SHEATH_GRIP + SHEATH_DIR * 0.45 + np.array([0.0, 0.0, 0.04]),
                 SHEATH_GRIP + SHEATH_DIR * 0.5 + np.array([-0.08, -0.25, 0.1]),
                 SHEATH_GRIP + SHEATH_DIR * 0.6 + np.array([-0.15, -0.4, 0.16]),
                 SHEATH_GRIP + SHEATH_DIR * 0.7 + np.array([-0.12, -0.55, 0.08]), 14)
    B.add(tube(sag, 0.012, 6), "pelvis", ARMOR_B)
