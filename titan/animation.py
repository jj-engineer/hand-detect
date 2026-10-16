"""
Pose engine, locomotion and the six power actions of the TITAN mecha.

Conventions (bone-local, bones hang along -Y for limbs):
  arm / leg rx < 0  -> swings forward & up       rx > 0 -> swings backward
  forearm rx < 0    -> elbow flexion              shin rx > 0   -> knee flexion
  left-side rz > 0  -> abduction (outward);  right side is mirrored (rz, ry negated)
  spine / head rx > 0 -> bows forward;  ry > 0 -> yaws toward +X (robot's left)
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .skeleton import BI, NB, Pose, forward_kinematics, world_point, world_dir, LOCAL, PARENT, REST
from . import ik
from .mecha_model import SHEATH_DIR as _SDIR

F32 = np.float32
PI = math.pi

# ------------------------------------------------------------------ pose vectors
EXTRAS = ["y", "fwd", "pitch", "roll", "yaw", "cL", "cR", "wopen", "nozt", "thr", "glow", "ground"]
XI = {n: NB * 3 + i for i, n in enumerate(EXTRAS)}
VLEN = NB * 3 + len(EXTRAS)
SYM = {"uarm", "farm", "hand", "pad", "thigh", "shin", "foot", "wing", "noz"}


def _zero():
    v = np.zeros(VLEN, F32)
    v[XI["ground"]] = 1.0
    return v


def _set_rot(v, bone, val):
    v[BI[bone] * 3:BI[bone] * 3 + 3] = val


def mk(base=None, **kw):
    v = (STAND if base is None else base).copy()
    for k, val in kw.items():
        if k in XI:
            v[XI[k]] = float(val)
        elif k in BI:
            _set_rot(v, k, val)
        elif k in SYM:
            _set_rot(v, k + "L", (val[0], val[1], val[2]))
            _set_rot(v, k + "R", (val[0], -val[1], -val[2]))
        else:
            raise KeyError(k)
    return v


STAND = None
STAND = mk(_zero(),
           spine=(0.04, 0, 0), neck=(-0.04, 0, 0),
           uarm=(-0.05, 0, 0.18), farm=(-0.32, 0, 0), thigh=(-0.06, 0, 0.05), shin=(0.12, 0, 0),
           wing=(0, 0, 0.32), cL=0.25, cR=0.25)


# ------------------------------------------------------------------ easing + timelines
def e_lin(x): return x
def e_io(x): return x * x * (3 - 2 * x)
def e_out(x): return 1 - (1 - x) ** 3
def e_in(x): return x ** 3
def e_snap(x): return 1 - (1 - x) ** 5
def e_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


EASE = {"lin": e_lin, "io": e_io, "out": e_out, "in": e_in, "snap": e_snap, "back": e_back}


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def smooth(a, b, x):
    t = clamp((x - a) / (b - a + 1e-9))
    return t * t * (3 - 2 * t)


class Timeline:
    def __init__(self, keys, events=()):
        self.keys = sorted(keys, key=lambda k: k[0])
        self.events = sorted(events)
        self.dur = self.keys[-1][0]

    def sample(self, t):
        t = clamp(t, 0.0, self.dur)
        ks = self.keys
        for i in range(len(ks) - 1):
            if t <= ks[i + 1][0]:
                t0, v0, _ = ks[i]
                t1, v1, ease = ks[i + 1]
                x = (t - t0) / max(t1 - t0, 1e-6)
                return v0 + (v1 - v0) * EASE[ease](x)
        return ks[-1][1].copy()


def K(t, ease="io", **kw):
    return (t, mk(**kw), ease)


# ------------------------------------------------------------------ authored poses
GUARD = dict(
    ground=1, pitch=0.06, spine=(0.14, 0.0, 0.0), chest=(0.0, -0.12, 0.0), neck=(-0.10, 0, 0), head=(0.06, 0.0, 0.0),
    uarmL=(-0.75, 0.0, 0.30), farmL=(-1.95, 0.0, 0.0), uarmR=(-0.60, 0.0, -0.22), farmR=(-2.0, 0.0, 0.0),
    thighL=(-0.42, 0.0, 0.20), shinL=(0.62, 0, 0), thighR=(0.12, 0.0, -0.24), shinR=(0.50, 0, 0),
    cL=1.0, cR=1.0, wing=(0, 0, 0.1), noz=(0.0, 0, 0))

# --- punches (each begins and ends on GUARD so strikes chain seamlessly)
JAB = Timeline([
    K(0.00, **GUARD),
    K(0.09, "out", **{**GUARD, "chest": (0, 0.12, 0), "fwd": -0.06, "uarmL": (-0.55, 0, 0.30), "farmL": (-2.2, 0, 0)}),
    K(0.19, "snap", **{**GUARD, "chest": (0.0, -0.55, 0), "spine": (0.2, -0.2, 0), "pelvis": (0, -0.14, 0), "fwd": 0.38,
                       "uarmL": (-1.50, 0, 0.10), "farmL": (-0.10, 0, 0), "head": (0.05, 0.3, 0),
                       "thighL": (-0.62, 0, 0.18), "shinL": (0.82, 0, 0)}),
    K(0.30, "io", **{**GUARD, "chest": (0.0, -0.5, 0), "spine": (0.2, -0.18, 0), "pelvis": (0, -0.12, 0), "fwd": 0.34,
                     "uarmL": (-1.46, 0, 0.10), "farmL": (-0.12, 0, 0), "thighL": (-0.62, 0, 0.18), "shinL": (0.82, 0, 0)}),
    K(0.52, "io", **GUARD),
], events=[(0.19, "hitL")])

CROSS = Timeline([
    K(0.00, **GUARD),
    K(0.11, "out", **{**GUARD, "chest": (0, -0.34, 0), "pelvis": (0, -0.1, 0), "fwd": -0.04, "uarmR": (-0.35, 0, -0.2), "farmR": (-2.25, 0, 0)}),
    K(0.23, "snap", **{**GUARD, "chest": (0.0, 0.75, 0), "spine": (0.24, 0.25, 0), "pelvis": (0, 0.30, 0), "fwd": 0.55,
                       "uarmR": (-1.52, 0, -0.08), "farmR": (-0.08, 0, 0), "head": (0.05, -0.35, 0),
                       "thighL": (-0.55, 0, 0.2), "thighR": (0.34, 0, -0.26), "shinR": (0.42, 0, 0), "shinL": (0.72, 0, 0)}),
    K(0.34, "io", **{**GUARD, "chest": (0.0, 0.68, 0), "spine": (0.24, 0.22, 0), "pelvis": (0, 0.26, 0), "fwd": 0.5,
                     "uarmR": (-1.48, 0, -0.08), "farmR": (-0.1, 0, 0), "thighR": (0.34, 0, -0.26)}),
    K(0.58, "io", **GUARD),
], events=[(0.23, "hitR")])

HOOK = Timeline([
    K(0.00, **GUARD),
    K(0.14, "out", **{**GUARD, "chest": (0, 0.55, 0), "pelvis": (0, 0.25, 0), "fwd": -0.05, "spine": (0.14, 0.1, 0),
                      "uarmL": (-1.05, 0.0, 1.30), "farmL": (-1.45, 0, 0)}),
    K(0.27, "snap", **{**GUARD, "chest": (0, -1.0, 0), "spine": (0.16, -0.3, 0), "pelvis": (0, -0.35, 0), "fwd": 0.32,
                       "uarmL": (-1.30, 0.0, 0.05), "farmL": (-1.50, 0, 0), "head": (0.05, 0.4, 0),
                       "thighL": (-0.55, 0, 0.2), "shinL": (0.75, 0, 0)}),
    K(0.36, "io", **{**GUARD, "chest": (0, -0.92, 0), "spine": (0.16, -0.28, 0), "pelvis": (0, -0.3, 0), "fwd": 0.28,
                     "uarmL": (-1.28, 0.0, 0.05), "farmL": (-1.50, 0, 0)}),
    K(0.60, "io", **GUARD),
], events=[(0.27, "hitL")])

UPPER = Timeline([
    K(0.00, **GUARD),
    K(0.15, "out", **{**GUARD, "chest": (0.1, 0.35, 0), "pelvis": (0.12, 0.2, 0), "spine": (0.38, 0.1, 0),
                      "thighL": (-0.62, 0, 0.2), "shinL": (0.95, 0, 0), "thighR": (0.2, 0, -0.26), "shinR": (0.8, 0, 0),
                      "uarmR": (-0.15, 0, -0.18), "farmR": (-1.95, 0, 0), "fwd": -0.05}),
    K(0.29, "snap", **{**GUARD, "chest": (-0.12, 0.6, 0), "pelvis": (-0.05, 0.28, 0), "spine": (0.0, 0.2, 0), "fwd": 0.42,
                       "thighL": (-0.58, 0, 0.2), "shinL": (0.7, 0, 0), "thighR": (0.3, 0, -0.26), "shinR": (0.3, 0, 0),
                       "uarmR": (-1.45, 0, -0.1), "farmR": (-1.30, 0, 0), "head": (-0.12, -0.2, 0), "y": 0.08}),
    K(0.40, "io", **{**GUARD, "chest": (-0.1, 0.55, 0), "pelvis": (-0.05, 0.25, 0), "spine": (0.0, 0.18, 0), "fwd": 0.38,
                     "uarmR": (-1.4, 0, -0.1), "farmR": (-1.35, 0, 0), "y": 0.04}),
    K(0.64, "io", **GUARD),
], events=[(0.29, "hitR")])

SLAM = Timeline([
    K(0.00, **GUARD),
    K(0.30, "out", **{**GUARD, "pitch": -0.02, "chest": (-0.3, 0, 0), "spine": (-0.15, 0, 0), "neck": (-0.2, 0, 0), "y": 0.12,
                      "uarmL": (-2.85, 0, 0.22), "uarmR": (-2.85, 0, -0.22), "farmL": (-0.55, 0, 0), "farmR": (-0.55, 0, 0),
                      "thighL": (-0.2, 0, 0.22), "thighR": (-0.15, 0, -0.22), "shinL": (0.2, 0, 0), "shinR": (0.15, 0, 0),
                      "head": (-0.15, 0, 0), "fwd": -0.1}),
    K(0.44, "in", **{**GUARD, "pitch": 0.12, "chest": (0.75, 0, 0), "spine": (0.5, 0, 0), "neck": (0.1, 0, 0), "fwd": 0.35,
                     "uarmL": (-0.95, 0, 0.18), "uarmR": (-0.95, 0, -0.18), "farmL": (-0.35, 0, 0), "farmR": (-0.35, 0, 0),
                     "thighL": (-0.8, 0, 0.26), "thighR": (-0.7, 0, -0.26), "shinL": (1.1, 0, 0), "shinR": (1.0, 0, 0),
                     "head": (0.15, 0, 0)}),
    K(0.75, "io", **{**GUARD, "pitch": 0.12, "chest": (0.7, 0, 0), "spine": (0.48, 0, 0), "fwd": 0.32,
                     "uarmL": (-0.95, 0, 0.18), "uarmR": (-0.95, 0, -0.18), "farmL": (-0.35, 0, 0), "farmR": (-0.35, 0, 0),
                     "thighL": (-0.8, 0, 0.26), "thighR": (-0.7, 0, -0.26), "shinL": (1.1, 0, 0), "shinR": (1.0, 0, 0)}),
    K(1.10, "io", **GUARD),
], events=[(0.44, "slam")])

# --- jump slam
CROUCH = dict(spine=(0.40, 0, 0), chest=(0.1, 0, 0), neck=(-0.3, 0, 0), pitch=0.10, uarm=(0.9, 0, 0.25), farm=(-0.4, 0, 0),
              thigh=(-1.05, 0, 0.18), shin=(1.75, 0, 0), wing=(0, 0, -0.2), noz=(0.1, 0, 0), cL=1.0, cR=1.0, thr=0.3, glow=0.4)
JUMP = Timeline([
    K(0.00),
    K(0.42, "io", **CROUCH),
    K(0.52, "snap", **{**CROUCH, "spine": (0.0, 0, 0), "chest": (-0.1, 0, 0), "thigh": (-0.15, 0, 0.1), "shin": (0.25, 0, 0),
                       "uarm": (-2.3, 0, 0.25), "farm": (-0.3, 0, 0), "wing": (0, 0, -0.55), "thr": 1.0, "noz": (0.25, 0, 0),
                       "ground": 0, "neck": (-0.1, 0, 0), "pitch": -0.05, "glow": 1.0}),
    K(0.95, "out", **{**CROUCH, "spine": (-0.05, 0, 0), "thigh": (-0.95, 0, 0.2), "shin": (1.35, 0, 0), "uarm": (-2.6, 0, 0.3),
                      "farm": (-0.4, 0, 0), "wing": (0, 0, -0.55), "thr": 0.8, "noz": (0.3, 0, 0), "ground": 0, "pitch": -0.1,
                      "neck": (0.0, 0, 0), "glow": 1.0}),
    K(1.22, "io", **{**CROUCH, "spine": (0.1, 0, 0), "thigh": (-0.6, 0, 0.2), "shin": (0.9, 0, 0), "uarmL": (0.7, 0, 0.3),
                     "uarmR": (-2.9, 0, -0.1), "farmL": (-0.3, 0, 0), "farmR": (-0.2, 0, 0), "wing": (0, 0, -0.5), "thr": 0.9,
                     "noz": (0.4, 0, 0), "ground": 0, "pitch": 0.55, "neck": (-0.4, 0, 0), "glow": 1.0}),
    K(1.55, "in", **{**CROUCH, "spine": (0.15, 0, 0), "chest": (0.1, 0, 0), "thighL": (0.25, 0, 0.2), "thighR": (0.4, 0, -0.2),
                     "shin": (0.15, 0, 0), "uarmL": (0.8, 0, 0.3), "uarmR": (-2.95, 0, -0.1), "farmL": (-0.3, 0, 0),
                     "farmR": (-0.15, 0, 0), "wing": (0, 0, -0.4), "thr": 1.0, "noz": (0.7, 0, 0), "ground": 0,
                     "pitch": 1.0, "neck": (-0.7, 0, 0), "glow": 1.0}),
    # superhero landing: left knee near the floor, right fist planted
    K(1.68, "snap", **{**CROUCH, "spine": (0.55, 0, 0), "chest": (0.2, 0.1, 0), "neck": (-0.55, 0, 0), "pitch": 0.28,
                       "thighL": (-1.75, 0, 0.12), "shinL": (2.25, 0, 0), "footL": (0.75, 0, 0),
                       "thighR": (-0.2, 0, -0.60), "shinR": (0.6, 0, 0),
                       "uarmL": (0.5, 0, 0.45), "farmL": (-0.3, 0, 0), "uarmR": (-0.55, 0, -0.30), "farmR": (-0.15, 0, 0),
                       "wing": (0, 0, -0.5), "thr": 0.4, "glow": 1.0, "noz": (0.2, 0, 0), "y": 0.0}),
    K(2.25, "io", **{**CROUCH, "spine": (0.55, 0, 0), "chest": (0.2, 0.1, 0), "neck": (-0.55, 0, 0), "pitch": 0.28,
                     "thighL": (-1.75, 0, 0.12), "shinL": (2.25, 0, 0), "footL": (0.75, 0, 0), "thighR": (-0.2, 0, -0.60),
                     "shinR": (0.6, 0, 0), "uarmL": (0.5, 0, 0.45), "farmL": (-0.3, 0, 0), "uarmR": (-0.55, 0, -0.30),
                     "farmR": (-0.15, 0, 0), "wing": (0, 0, -0.5), "thr": 0.0, "glow": 0.6}),
    K(2.9, "io"),
], events=[(0.50, "launch"), (1.66, "land")])

# --- jet dash
DASH = Timeline([
    K(0.00),
    K(0.14, "out", **dict(spine=(0.3, 0, 0), pitch=0.15, thigh=(-0.3, 0, 0.1), shin=(0.5, 0, 0), uarm=(0.4, 0, 0.2),
                          wing=(0, 0, -0.3), noz=(0.3, 0, 0), thr=0.8, cL=1, cR=1)),
    K(0.30, "io", **dict(spine=(0.15, 0, 0), chest=(0.15, 0, 0), pitch=1.12, neck=(-0.9, 0, 0), thigh=(0.55, 0, 0.12),
                         shin=(0.6, 0, 0), uarm=(0.75, 0, 0.12), farm=(-0.15, 0, 0), wing=(0, 0, -0.6), noz=(0.05, 0, 0), thr=1.0,
                         ground=0, cL=1, cR=1, glow=1.0, y=0.55)),
    K(0.78, "lin", **dict(spine=(0.15, 0, 0), chest=(0.15, 0, 0), pitch=1.12, neck=(-0.9, 0, 0), thigh=(0.6, 0, 0.12),
                          shin=(0.5, 0, 0), uarm=(0.8, 0, 0.12), farm=(-0.15, 0, 0), wing=(0, 0, -0.6), noz=(0.05, 0, 0), thr=1.0,
                          ground=0, cL=1, cR=1, glow=1.0, y=0.6)),
    K(1.05, "io", **dict(spine=(0.3, 0, 0), pitch=0.3, neck=(-0.3, 0, 0), thigh=(-0.55, 0, 0.2), shin=(0.8, 0, 0), uarm=(0.4, 0, 0.3),
                         farm=(-0.4, 0, 0), wing=(0, 0, -0.3), noz=(0.6, 0, 0), thr=0.4, ground=1, cL=1, cR=1, glow=0.5, fwd=0.0)),
    K(1.55, "io"),
], events=[(0.10, "ignite"), (1.0, "skid")])

# --- laser aiming pose (left arm fires)
LASER_AIM = dict(ground=1, pitch=0.0, spine=(0.0, 0.0, 0.0), chest=(-0.1, 0.35, 0), neck=(-0.05, -0.3, 0), head=(0.0, -0.2, 0),
                 uarmL=(-1.50, 0.0, 0.06), farmL=(-0.06, -1.50, 0.0), handL=(0.0, 0.0, 0.0),
                 uarmR=(-0.3, 0, -0.35), farmR=(-0.8, 0, 0),
                 thighL=(-0.35, 0, 0.30), shinL=(0.5, 0, 0), thighR=(0.25, 0, -0.35), shinR=(0.45, 0, 0),
                 cL=0.0, cR=0.6, wing=(0, 0, -0.35), glow=0.0)
LASER = Timeline([
    K(0.00),
    K(0.35, "out", **LASER_AIM),
    K(0.90, "io", **{**LASER_AIM, "glow": 1.0, "pitch": -0.05, "spine": (-0.06, 0, 0)}),
])

SHIELD_POSE = dict(ground=1, pitch=0.0, spine=(0.1, 0, 0), chest=(0.0, 0, 0), neck=(-0.05, 0, 0),
                   uarm=(-0.85, 0.0, 0.18), farm=(-1.55, 0.0, 0.0), hand=(0, 0, 0),
                   thighL=(-0.35, 0, 0.26), shinL=(0.55, 0, 0), thighR=(-0.1, 0, -0.30), shinR=(0.45, 0, 0),
                   cL=0.0, cR=0.0, wing=(0, 0, -0.3), glow=0.6)
SHIELD = Timeline([K(0.00), K(0.32, "out", **SHIELD_POSE), K(0.60, "io", **{**SHIELD_POSE, "glow": 0.8})])

# --- power-up / overdrive
POWER = Timeline([
    K(0.00),
    K(0.50, "io", **dict(spine=(0.45, 0, 0), chest=(0.15, 0, 0), neck=(-0.3, 0, 0), pitch=0.1, uarm=(0.35, 0, 0.55), farm=(-0.5, 0, 0),
                         thigh=(-0.95, 0, 0.25), shin=(1.65, 0, 0), cL=1, cR=1, glow=0.3)),
    K(1.25, "lin", **dict(spine=(0.5, 0, 0), chest=(0.18, 0, 0), neck=(-0.35, 0, 0), pitch=0.12, uarm=(0.4, 0, 0.58), farm=(-0.55, 0, 0),
                          thigh=(-1.0, 0, 0.27), shin=(1.7, 0, 0), cL=1, cR=1, glow=0.8, wing=(0, 0, 0.0))),
    K(1.50, "snap", **dict(spine=(-0.2, 0, 0), chest=(-0.28, 0, 0), neck=(-0.5, 0, 0), pitch=-0.05, uarm=(-0.15, 0, 1.35),
                           farm=(-0.25, 0, 0), thigh=(-0.4, 0, 0.45), shin=(0.7, 0, 0), cL=1, cR=1, glow=1.0, wing=(0, 0, -0.6),
                           y=0.05)),
    K(2.30, "io", **dict(spine=(-0.14, 0, 0), chest=(-0.22, 0, 0), neck=(-0.45, 0, 0), pitch=-0.04, uarm=(-0.12, 0, 1.25),
                         farm=(-0.28, 0, 0), thigh=(-0.4, 0, 0.45), shin=(0.7, 0, 0), cL=1, cR=1, glow=1.0, wing=(0, 0, -0.6))),
    K(2.90, "io", wing=(0, 0, -0.5), glow=0.6),
], events=[(0.45, "charge"), (1.48, "burst")])


# left arm raised and turned so the open hand faces the viewer
SHOW_POSE = {"uarmL": (-0.75, 0.0, 0.42), "farmL": (-1.45, 0.0, 0.0), "handL": (0.0, 0.0, 0.0)}


# --- boot-up (after the reveal assembly): slumped, powered-down body wakes up and stands
SLUMP = dict(spine=(0.40, 0, 0), chest=(0.12, 0, 0), neck=(0.55, 0, 0), head=(0.28, 0, 0), pitch=0.06,
             uarm=(0.14, 0, 0.07), farm=(-0.14, 0, 0), thigh=(-0.30, 0, 0.07), shin=(0.55, 0, 0),
             wing=(0, 0, 0.42), cL=0.1, cR=0.1, glow=0.0)
RISE1 = dict(spine=(-0.06, 0, 0), chest=(-0.14, 0, 0), neck=(-0.22, 0, 0), head=(-0.10, 0, 0), pitch=-0.02,
             uarm=(-0.30, 0, 0.62), farm=(-1.15, 0, 0), thigh=(-0.18, 0, 0.30), shin=(0.34, 0, 0),
             wing=(0, 0, -0.35), cL=1.0, cR=1.0, glow=1.0)
RISE2 = dict(spine=(0.02, 0, 0), chest=(-0.05, 0, 0), neck=(-0.10, 0, 0), head=(0.0, 0, 0), pitch=0.0,
             uarm=(-0.12, 0, 0.36), farm=(-0.7, 0, 0), thigh=(-0.10, 0, 0.16), shin=(0.20, 0, 0),
             wing=(0, 0, 0.12), cL=0.7, cR=0.7, glow=0.5)
BOOTUP = Timeline([
    K(0.00, **SLUMP), K(0.50, "lin", **SLUMP), K(1.15, "back", **RISE1), K(1.75, "io", **RISE2), K(2.40, "io"),
], events=[(0.55, "ignite")])


def mirror_pv(v):
    """Mirror a pose vector left<->right."""
    o = v.copy()
    for n in list(BI):
        if n.endswith("L") and n[:-1] + "R" in BI:
            a, b = BI[n], BI[n[:-1] + "R"]
            ra, rb = v[a * 3:a * 3 + 3].copy(), v[b * 3:b * 3 + 3].copy()
            o[a * 3:a * 3 + 3] = (rb[0], -rb[1], -rb[2])
            o[b * 3:b * 3 + 3] = (ra[0], -ra[1], -ra[2])
    for n in ("spine", "chest", "neck", "head", "pelvis"):
        i = BI[n]
        o[i * 3 + 1] = -v[i * 3 + 1]
        o[i * 3 + 2] = -v[i * 3 + 2]
    o[XI["cL"]], o[XI["cR"]] = v[XI["cR"]], v[XI["cL"]]
    o[XI["yaw"]] = -v[XI["yaw"]]
    o[XI["roll"]] = -v[XI["roll"]]
    return o


# katana hilt frame (rest): z along the blade (backwards inside the saya), edge up
_z = np.asarray(_SDIR, np.float64)
_ya = -(np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], _z) * _z)
_ya /= np.linalg.norm(_ya)
R_HILT0 = np.column_stack([np.cross(_ya, _z), _ya, _z])
GRIP_OFF = {"R": np.array([0.04, -0.34, 0.0]), "L": np.array([-0.04, -0.34, 0.0])}   # grip centre relative to the wrist bone
SPRING_W = {"spine": 18.0, "chest": 15.0, "neck": 14.0, "head": 11.0, "pad": 20.0, "uarm": 26.0, "farm": 28.0,
            "hand": 30.0, "wing": 14.0, "noz": 22.0}


# ------------------------------------------------------------------ intents
@dataclass
class Intent:
    move_target: object = None         # (x, z) world or None
    move_dir: object = None            # (x, z) world, |v| <= 1
    turn: float = 0.0                  # -1..1 rotate in place
    hold_combo: bool = False
    hold_laser: bool = False
    hold_shield: bool = False
    trig_jump: bool = False
    trig_dash: bool = False
    trig_power: bool = False
    curls: object = None               # np.array(5) thumb..pinky 0..1 or None
    aim_yaw: float = 0.0               # radians offset while laser is held
    aim_pitch: float = 0.0
    sprint: bool = False


# contact points used for automatic floor grounding: (bone, rest-local offset)
CONTACTS = [("footL", (0, -0.22, -0.30)), ("footL", (0, -0.22, 0.55)), ("footR", (0, -0.22, -0.30)), ("footR", (0, -0.22, 0.55)),
            ("shinL", (0, -0.04, 0.40)), ("shinR", (0, -0.04, 0.40)), ("handL", (0, -0.55, 0.0)), ("handR", (0, -0.55, 0.0))]
ATTACH = {
    "palmL": ("handL", (-0.078, -0.16, 0.0)), "palmR": ("handR", (0.078, -0.16, 0.0)),
    "fistL": ("handL", (0.0, -0.38, 0.0)), "fistR": ("handR", (0.0, -0.38, 0.0)),
    "nozL": ("nozL", (0.0, -0.24, 0.0)), "nozR": ("nozR", (0.0, -0.24, 0.0)),
    "calfL": ("shinL", (0.0, -0.74, -0.30)), "calfR": ("shinR", (0.0, -0.74, -0.30)),
    "reactor": ("chest", (0.0, 0.43, 0.56)), "head": ("head", (0.0, 0.3, 0.1)),
    "footL": ("footL", (0.0, -0.22, 0.1)), "footR": ("footR", (0.0, -0.22, 0.1)),
    "eyeL": ("head", (0.10, 0.30, 0.34)), "eyeR": ("head", (-0.10, 0.30, 0.34)),
    "chest": ("chest", (0.0, 0.4, 0.0)),
    "bladeTip": ("kat", (0.0, 0.05, 1.99)), "bladeMid": ("kat", (0.0, 0.03, 1.12)), "bladeBase": ("kat", (0.0, 0.0, 0.40)),
}


# ------------------------------------------------------------------ actions
class Action:
    hold = False
    blend_in = 0.12
    blend_out = 0.28

    def __init__(self, m):
        self.m = m
        self.t = 0.0
        self.w = 0.0
        self.finishing = False
        self.done = False
        self.pv = STAND.copy()
        self._ev = set()

    def release(self):
        self.finishing = True

    def _envelope(self, dt):
        if self.finishing:
            self.w = max(0.0, self.w - dt / self.blend_out)
            if self.w <= 0.0:
                self.done = True
        else:
            self.w = min(1.0, self.w + dt / self.blend_in)

    def _events(self, tl, t):
        for te, name in tl.events:
            if t >= te and name not in self._ev:
                self._ev.add(name)
                self.event(name)

    def event(self, name):
        pass

    def update(self, dt, intent):
        raise NotImplementedError

    def locks_move(self):
        return True


class OneShot(Action):
    TL = None

    def update(self, dt, intent):
        self.t += dt
        self.pv = self.TL.sample(self.t)
        self._events(self.TL, self.t)
        self.motion(dt, intent)
        if self.t >= self.TL.dur and not self.finishing:
            self.finishing = True
            self.blend_out = 0.18
        self._envelope(dt)

    def motion(self, dt, intent):
        pass


class JumpSlam(OneShot):
    TL = JUMP
    T_LAUNCH, T_APEX, T_DIVE, T_LAND = 0.52, 0.95, 1.22, 1.68
    H = 4.6
    DIST = 4.2

    def __init__(self, m):
        super().__init__(m)
        self.start = m.pos.copy()
        self.dir = np.array([math.sin(m.yaw), 0, math.cos(m.yaw)], F32)

    def event(self, name):
        m = self.m
        if name == "launch":
            m.fx.dust(m.pos, n=70, speed=7.0, radius=0.5, size=0.7)
            m.fx.ring(m.pos, r0=0.6, speed=11.0, thick=0.3, col=(0.5, 0.9, 1.0), intensity=3.0, life=0.7)
            m.fx.sparks(m.pos + np.array([0, 0.1, 0], F32), n=50, speed=9)
            m.fx.impact_feel(shake=0.45, flash=0.25, aberr=0.4, fov=0.5, col=(0.7, 0.95, 1.0))
            m.sfx("jump")
        elif name == "land":
            p = m.pos.copy()
            p[1] = 0
            m.fx.ring(p, r0=0.5, speed=15.0, thick=0.5, col=(1.0, 0.9, 0.6), intensity=3.4, life=0.9, fill=0.16)
            m.fx.ring(p, r0=0.3, speed=9.0, thick=0.35, col=(0.4, 0.9, 1.0), intensity=4.0, life=1.2)
            m.fx.dust(p, n=160, speed=11.0, radius=0.6, size=1.1, life=2.0)
            m.fx.sparks(p + np.array([0, 0.15, 0], F32), n=110, speed=15, life=1.0)
            m.fx.burst_ball(p + np.array([0, 0.4, 0], F32), size=2.4, col=(2.2, 1.6, 1.0), life=0.25)
            m.fx.impact_feel(shake=1.0, stop=0.09, slow=0.55, slow_scale=0.4, flash=0.55, aberr=1.0, fov=1.2, col=(1.0, 0.92, 0.8))
            m.sfx("slam")
            m.hit(p, 4.2, 3.0)

    def motion(self, dt, intent):
        t, m = self.t, self.m
        # horizontal travel
        s = clamp((t - self.T_LAUNCH) / (self.T_LAND - self.T_LAUNCH))
        k = s * s * (3 - 2 * s) * 0.4 + s * 0.6
        m.pos[0] = self.start[0] + self.dir[0] * self.DIST * k
        m.pos[2] = self.start[2] + self.dir[2] * self.DIST * k
        # vertical arc
        y = 0.0
        if self.T_LAUNCH <= t < self.T_APEX:
            y = self.H * e_out(clamp((t - self.T_LAUNCH) / (self.T_APEX - self.T_LAUNCH)))
        elif self.T_APEX <= t < self.T_DIVE:
            y = self.H + 0.12 * math.sin((t - self.T_APEX) / (self.T_DIVE - self.T_APEX) * PI)
        elif self.T_DIVE <= t < self.T_LAND:
            y = self.H * (1.0 - e_in(clamp((t - self.T_DIVE) / (self.T_LAND - self.T_DIVE))))
        self.pv[XI["y"]] = y
        # airborne ground weight (smooth on both ends)
        g = 1.0 - smooth(self.T_LAUNCH - 0.04, self.T_LAUNCH + 0.10, t)
        g = max(g, smooth(self.T_LAND - 0.10, self.T_LAND - 0.01, t))
        self.pv[XI["ground"]] = g
        self.m.thrust = self.pv[XI["thr"]]
        if self.T_LAUNCH < t < self.T_LAND:
            m.fx.shake(0.01)
        m.vel[:] = 0


class BootUp(OneShot):
    """Powered-down slump (held while frozen) -> reactor ignites -> the mecha rises to its feet."""
    TL = BOOTUP
    blend_in = 0.01

    def __init__(self, m):
        super().__init__(m)
        self.frozen = False
        self.w = 1.0

    def event(self, name):
        if name == "ignite":
            m = self.m
            m.emis_target = 1.0
            m.charge_boost = 1.0
            m.fx.shake(0.4)

    def update(self, dt, intent):
        if self.frozen:
            self.pv = self.TL.sample(0.0)
            self.w = 1.0
            self.m.charge_target = 0.0
            self.m.thrust = 0.0
            return
        super().update(dt, intent)
        self.m.charge_target = max(self.m.charge_target, 1.0 if self.t < 1.4 else 0.4)


class Dash(OneShot):
    TL = DASH
    DIST = 9.0

    def __init__(self, m):
        super().__init__(m)
        self.start = m.pos.copy()
        self.dir = np.array([math.sin(m.yaw), 0, math.cos(m.yaw)], F32)
        self.last = 0.0

    def event(self, name):
        m = self.m
        if name == "ignite":
            m.fx.ring(m.pos, r0=0.5, speed=10.0, thick=0.3, col=(0.6, 0.95, 1.0), intensity=3.5, life=0.6)
            m.fx.dust(m.pos, n=60, speed=8.0, radius=0.4, size=0.7)
            m.fx.impact_feel(shake=0.4, flash=0.3, aberr=0.6, fov=0.9, col=(0.7, 0.95, 1.0))
            m.sfx("dash")
        elif name == "skid":
            m.sfx("skid")

    def motion(self, dt, intent):
        t, m = self.t, self.m
        # smooth distance profile: fast launch, skid at the end
        s = clamp((t - 0.12) / 0.95)
        prog = 1 - (1 - s) ** 2.2
        d = self.DIST * prog
        m.pos[0] = self.start[0] + self.dir[0] * d
        m.pos[2] = self.start[2] + self.dir[2] * d
        v = d - self.last
        self.last = d
        speed = v / max(dt, 1e-4)
        m.thrust = self.pv[XI["thr"]]
        if 0.2 < t < 1.1:
            m.fx.shake(0.012)
            # speed streaks trailing behind
            back = -self.dir
            for sgn in (-1, 1):
                pos = m.pos + np.array([0, 1.2 + np.random.rand() * 2.0, 0], F32) + np.array([-self.dir[2], 0, self.dir[0]], F32) * sgn * (0.6 + np.random.rand() * 2.0)
                m.fx.glow(pos, vel=back * 14.0, size=0.08, size_end=0.02, life=0.35, col=(0.6, 0.9, 1.0), alpha=0.8)
        if t > 0.95:
            # skid dust at the feet
            fl = m.attach("footL")
            fr = m.attach("footR")
            for f in (fl, fr):
                m.fx.dust(np.array([f[0], 0.02, f[2]], F32), n=2, speed=2.0, radius=0.1, size=0.5, life=1.0)
        m.vel[:] = 0


# ------------------------------------------------------------------ katana moves
class SK:
    """Sword key: hilt position H (chest-local), blade yaw/pitch (chest-local, yaw + = toward the robot's left), edge roll."""

    def __init__(self, t, H, yaw, pitch, roll=0.0, two=0.0, w=1.0, ease="io"):
        self.t, self.H, self.yaw, self.pitch, self.roll, self.two, self.w, self.ease = t, np.array(H, np.float64), yaw, pitch, roll, two, w, ease


def blade_dir(yaw, pitch):
    return np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], np.float64)


class SwordTimeline:
    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k.t)
        self.dur = self.keys[-1].t

    def sample(self, t):
        ks = self.keys
        t = clamp(t, 0.0, self.dur)
        for i in range(len(ks) - 1):
            if t <= ks[i + 1].t:
                a, b = ks[i], ks[i + 1]
                x = EASE[b.ease]((t - a.t) / max(b.t - a.t, 1e-6))
                lerp = lambda p, q: p + (q - p) * x
                yaw, pit = lerp(a.yaw, b.yaw), lerp(a.pitch, b.pitch)
                return dict(H=lerp(a.H, b.H), D=blade_dir(yaw, pit), roll=lerp(a.roll, b.roll), two=lerp(a.two, b.two), w=lerp(a.w, b.w))
        k = ks[-1]
        return dict(H=k.H.copy(), D=blade_dir(k.yaw, k.pitch), roll=k.roll, two=k.two, w=k.w)


class Move:
    def __init__(self, name, body, sword, events=(), trail=(), spin=None):
        self.name, self.body, self.sword = name, body, sword
        self.events = sorted(events)
        self.trail = trail                 # [(t0, t1)] windows where the blade leaves a glowing streak
        self.spin = spin                   # (t0, t1) yaw spin of 2*pi
        self.dur = max(body.dur, sword.dur)


HILT_C = (0.34, -0.18, 0.36)             # sheathed hilt, chest-local
SHEATH_YAW, SHEATH_PITCH = 2.93, -0.30
GUARD_H = (-0.30, 0.06, 0.86)
ARMS_TUCK = dict(uarmL=(-0.30, 0, 0.32), farmL=(-1.1, 0, 0), uarmR=(-0.3, 0, -0.32), farmR=(-1.1, 0, 0))
KAMAE = dict(ground=1, pitch=0.05, spine=(0.10, 0, 0), chest=(0, 0.12, 0), neck=(-0.08, 0, 0), head=(0.02, 0, 0), pelvis=(0, 0.08, 0),
             thighR=(-0.46, 0.0, -0.20), shinR=(0.56, 0, 0), thighL=(0.12, 0.0, 0.22), shinL=(0.44, 0, 0),
             cL=1.0, cR=1.0, wing=(0, 0, 0.15), **ARMS_TUCK)


def kb(**kw):
    return {**KAMAE, **kw}


def _k(t, ease="io", **kw):
    return K(t, ease, **kb(**kw))


KAMAE_SK = SK(0.0, GUARD_H, 0.0, 0.22, two=1.0)

# 1 - quick-draw horizontal cut (iaido), starting from the idle hand-on-hilt pose
DRAW = Move("draw",
    Timeline([
        K(0.00, **{**kb(), "ground": 1, "thighR": (-0.10, 0, -0.10), "shinR": (0.22, 0, 0), "thighL": (0.05, 0, 0.16), "shinL": (0.28, 0, 0),
                   "chest": (0, 0.0, 0), "pelvis": (0, 0, 0), "spine": (0.05, 0, 0)}),
        K(0.26, "io", **kb(spine=(0.20, 0, 0), chest=(0, 0.45, 0), pelvis=(0, 0.28, 0), fwd=-0.06, thighR=(-0.20, 0, -0.22), shinR=(0.70, 0, 0),
                           thighL=(0.20, 0, 0.24), shinL=(0.66, 0, 0), head=(0.08, -0.15, 0))),
        K(0.46, "out", **kb(spine=(0.14, 0, 0), chest=(0, 0.10, 0), pelvis=(0, 0.04, 0), fwd=0.28, thighR=(-0.80, 0, -0.24), shinR=(0.95, 0, 0),
                            thighL=(0.28, 0, 0.26), shinL=(0.55, 0, 0))),
        K(0.62, "snap", **kb(spine=(0.20, 0, 0), chest=(0, -0.78, 0), pelvis=(0, -0.42, 0), fwd=0.48, thighR=(-0.86, 0, -0.26), shinR=(1.0, 0, 0),
                             thighL=(0.34, 0, 0.28), shinL=(0.50, 0, 0), head=(0.04, 0.35, 0))),
        K(0.82, "io", **kb(spine=(0.18, 0, 0), chest=(0, -0.70, 0), pelvis=(0, -0.38, 0), fwd=0.44, thighR=(-0.86, 0, -0.26), shinR=(1.0, 0, 0),
                           thighL=(0.34, 0, 0.28), shinL=(0.50, 0, 0), head=(0.04, 0.30, 0))),
        K(1.10, "io", **kb()),
    ]),
    SwordTimeline([
        SK(0.00, HILT_C, SHEATH_YAW, SHEATH_PITCH, roll=0.0, two=0.0),
        SK(0.26, (0.42, -0.10, 0.50), SHEATH_YAW, SHEATH_PITCH, two=0.0),
        SK(0.46, (0.15, 0.02, 0.88), 1.35, -0.12, two=0.0),
        SK(0.62, (-0.62, 0.22, 0.72), -1.30, 0.02, roll=0.15, two=0.0),
        SK(0.82, (-0.60, 0.26, 0.66), -1.40, 0.10, roll=0.15, two=0.0),
        SK(1.10, GUARD_H, 0.0, 0.22, two=1.0),
    ]),
    events=[(0.28, "draw"), (0.50, "swing"), (0.61, "hit")], trail=[(0.46, 0.80)])

# 2 - diagonal downward cut (kesa-giri)
KESA = Move("kesa",
    Timeline([_k(0.00), _k(0.22, "out", chest=(-0.10, -0.50, 0), pelvis=(0, -0.26, 0), spine=(-0.08, 0, 0), fwd=-0.10, thighR=(-0.36, 0, -0.20),
                           shinR=(0.50, 0, 0)),
              _k(0.40, "snap", chest=(0.12, 0.62, 0), pelvis=(0, 0.34, 0), spine=(0.30, 0, 0), fwd=0.50, thighR=(-0.82, 0, -0.24), shinR=(1.00, 0, 0),
                 thighL=(0.30, 0, 0.26), shinL=(0.50, 0, 0), head=(0.06, -0.20, 0)),
              _k(0.56, "io", chest=(0.10, 0.55, 0), pelvis=(0, 0.30, 0), spine=(0.28, 0, 0), fwd=0.46, thighR=(-0.82, 0, -0.24), shinR=(1.0, 0, 0),
                 thighL=(0.30, 0, 0.26), shinL=(0.50, 0, 0)),
              _k(0.80, "io")]),
    SwordTimeline([SK(0.00, GUARD_H, 0.0, 0.22, two=1.0),
                   SK(0.22, (-0.52, 1.02, 0.28), -0.25, 1.25, roll=0.2, two=1.0),
                   SK(0.40, (0.08, -0.22, 0.92), 0.55, -0.50, roll=0.1, two=1.0),
                   SK(0.56, (0.10, -0.24, 0.90), 0.60, -0.55, two=1.0),
                   SK(0.80, GUARD_H, 0.0, 0.22, two=1.0)]),
    events=[(0.34, "swing"), (0.40, "hit")], trail=[(0.30, 0.52)])

# 3 - rising reverse cut
REVERSE = Move("reverse",
    Timeline([_k(0.00), _k(0.18, "out", chest=(0.10, 0.55, 0), pelvis=(0, 0.28, 0), spine=(0.22, 0, 0), fwd=-0.06, thighR=(-0.30, 0, -0.22), shinR=(0.80, 0, 0)),
              _k(0.38, "snap", chest=(-0.10, -0.60, 0), pelvis=(0, -0.32, 0), spine=(-0.06, 0, 0), fwd=0.34, thighR=(-0.70, 0, -0.24), shinR=(0.78, 0, 0),
                 thighL=(0.25, 0, 0.25), shinL=(0.40, 0, 0), head=(-0.10, 0.2, 0)),
              _k(0.52, "io", chest=(-0.08, -0.52, 0), pelvis=(0, -0.28, 0), spine=(-0.04, 0, 0), fwd=0.30, thighR=(-0.70, 0, -0.24), shinR=(0.78, 0, 0)),
              _k(0.74, "io")]),
    SwordTimeline([SK(0.00, GUARD_H, 0.0, 0.22, two=1.0),
                   SK(0.18, (0.28, -0.30, 0.74), 0.65, -0.58, two=1.0),
                   SK(0.38, (-0.60, 0.86, 0.52), -0.55, 0.98, roll=0.45, two=1.0),
                   SK(0.52, (-0.58, 0.90, 0.48), -0.60, 1.05, two=1.0),
                   SK(0.74, GUARD_H, 0.0, 0.22, two=1.0)]),
    events=[(0.30, "swing"), (0.37, "hit")], trail=[(0.26, 0.48)])

# 4 - front kick (sword held in guard)
KICKM = Move("kick",
    Timeline([_k(0.00),
              _k(0.20, "out", spine=(-0.12, 0, 0), pelvis=(0, 0, 0.10), thighR=(-1.35, 0, -0.16), shinR=(1.65, 0, 0), footR=(-1.45, 0, 0), thighL=(0.10, 0, 0.14),
                 shinL=(0.55, 0, 0), fwd=-0.05),
              _k(0.36, "snap", spine=(-0.22, 0, 0), chest=(0, -0.12, 0), pelvis=(-0.08, 0.0, 0.10), thighR=(-1.55, 0, -0.14), shinR=(0.10, 0, 0),
                 footR=(-1.40, 0, 0), thighL=(0.14, 0, 0.14), shinL=(0.60, 0, 0), fwd=0.15),
              _k(0.50, "io", spine=(-0.14, 0, 0), thighR=(-1.35, 0, -0.16), shinR=(1.3, 0, 0), footR=(-1.3, 0, 0), thighL=(0.12, 0, 0.14), shinL=(0.5, 0, 0)),
              _k(0.74, "io")]),
    SwordTimeline([SK(0.00, GUARD_H, 0.0, 0.22, two=1.0), SK(0.36, (-0.30, 0.10, 0.80), 0.1, 0.30, two=1.0), SK(0.74, GUARD_H, 0.0, 0.22, two=1.0)]),
    events=[(0.34, "kick")])

# 5 - spinning cut (full 360 degrees with the blade out)
SPINM = Move("spin",
    Timeline([_k(0.00),
              _k(0.20, "out", chest=(0, -0.55, 0), pelvis=(0, -0.35, 0), spine=(0.20, 0, 0), thighR=(-0.30, 0, -0.30), shinR=(0.80, 0, 0),
                 thighL=(0.20, 0, 0.30), shinL=(0.80, 0, 0), fwd=-0.05),
              _k(0.64, "lin", chest=(0, 0.55, 0), pelvis=(0, 0.30, 0), spine=(0.12, 0, 0), thighR=(-0.34, 0, -0.30), shinR=(0.70, 0, 0),
                 thighL=(0.24, 0, 0.30), shinL=(0.70, 0, 0), y=0.06),
              _k(0.84, "io", chest=(0, 0.1, 0), pelvis=(0, 0.04, 0)),
              _k(1.02, "io")]),
    SwordTimeline([SK(0.00, GUARD_H, 0.0, 0.22, two=1.0),
                   SK(0.20, (-0.58, 0.28, 0.62), -0.9, 0.05, two=0.0),
                   SK(0.64, (-0.62, 0.26, 0.70), -1.45, 0.02, two=0.0),
                   SK(0.84, (-0.50, 0.20, 0.78), -0.5, 0.15, two=0.6),
                   SK(1.02, GUARD_H, 0.0, 0.22, two=1.0)]),
    events=[(0.28, "swing"), (0.36, "hit"), (0.48, "hit"), (0.60, "hit")], trail=[(0.26, 0.66)], spin=(0.22, 0.64))

# 6 - overhead finishing cut with a ground shockwave
OVERHEAD = Move("overhead",
    Timeline([_k(0.00),
              _k(0.36, "out", spine=(-0.26, 0, 0), chest=(-0.14, 0, 0), neck=(-0.2, 0, 0), pitch=-0.03, fwd=-0.16, thighR=(-0.40, 0, -0.22), shinR=(0.72, 0, 0),
                 thighL=(0.18, 0, 0.24), shinL=(0.66, 0, 0), head=(-0.15, 0, 0)),
              _k(0.56, "snap", spine=(0.50, 0, 0), chest=(0.22, 0, 0), neck=(0.10, 0, 0), pitch=0.15, fwd=0.74, thighR=(-1.08, 0, -0.26), shinR=(1.30, 0, 0),
                 thighL=(0.40, 0, 0.28), shinL=(0.28, 0, 0), head=(0.12, 0, 0)),
              _k(0.86, "io", spine=(0.46, 0, 0), chest=(0.20, 0, 0), pitch=0.14, fwd=0.70, thighR=(-1.06, 0, -0.26), shinR=(1.28, 0, 0),
                 thighL=(0.40, 0, 0.28), shinL=(0.28, 0, 0)),
              _k(1.28, "io")]),
    SwordTimeline([SK(0.00, GUARD_H, 0.0, 0.22, two=1.0),
                   SK(0.36, (-0.14, 1.16, 0.26), 0.0, 1.50, two=1.0),
                   SK(0.56, (-0.10, -0.12, 1.08), 0.0, -0.28, two=1.0),
                   SK(0.86, (-0.10, -0.12, 1.06), 0.0, -0.30, two=1.0),
                   SK(1.28, GUARD_H, 0.0, 0.22, two=1.0)]),
    events=[(0.50, "swing"), (0.55, "slam")], trail=[(0.44, 0.62)])

# - sheathing: flick the blood off, return the blade to the saya
SHEATHE = Move("sheathe",
    Timeline([_k(0.00),
              _k(0.22, "out", chest=(0, -0.45, 0), pelvis=(0, -0.2, 0), spine=(0.08, 0, 0), thighR=(-0.30, 0, -0.22), shinR=(0.55, 0, 0)),
              _k(0.52, "io", chest=(0, 0.10, 0), pelvis=(0, 0.04, 0), spine=(0.10, 0, 0), thighR=(-0.14, 0, -0.14), shinR=(0.35, 0, 0), thighL=(0.05, 0, 0.16), shinL=(0.30, 0, 0)),
              K(0.90, "io", **{**kb(), "thighR": (-0.10, 0, -0.10), "shinR": (0.22, 0, 0), "thighL": (0.05, 0, 0.16), "shinL": (0.28, 0, 0),
                               "chest": (0, 0, 0), "pelvis": (0, 0, 0), "spine": (0.05, 0, 0)})]),
    SwordTimeline([SK(0.00, GUARD_H, 0.0, 0.22, two=1.0),
                   SK(0.22, (-0.42, 0.52, 0.58), -0.95, 0.65, roll=0.3, two=0.0),
                   SK(0.52, (0.18, 0.0, 0.72), 2.45, -0.18, two=0.0),
                   SK(0.78, HILT_C, SHEATH_YAW, SHEATH_PITCH, two=0.0),
                   SK(0.90, HILT_C, SHEATH_YAW, SHEATH_PITCH, two=0.0)]),
    events=[(0.24, "swing"), (0.80, "sheathe")], trail=[(0.20, 0.40)])


class Combo(Action):
    """Hold the fist: draw the katana and chain cuts, a kick, a spin and an overhead finisher; release to sheathe."""
    hold = True
    blend_in = 0.15
    blend_out = 0.30
    LOOP = [KESA, REVERSE, KICKM, SPINM, KESA, OVERHEAD]

    def __init__(self, m):
        super().__init__(m)
        self.holding = True
        self.queue = [DRAW]
        self.loop_i = 0
        self.move = DRAW
        self.ts = 0.0
        self.prev_seg = None
        self.sheathing = False

    def release(self):
        self.holding = False

    def _next(self):
        if self.sheathing:
            self.finishing = True
            return
        if not self.holding:
            self.sheathing = True
            self.move = SHEATHE
        else:
            self.move = self.LOOP[self.loop_i % len(self.LOOP)]
            self.loop_i += 1
        self.ts = 0.0
        self._ev.clear()

    def event(self, name):
        m = self.m
        if name == "draw":
            m.sword_drawn = True
            m.sfx("sword_draw")
            tip = m.attach("bladeBase")
            m.fx.sparks(tip, n=26, speed=7.0, col=(2.4, 1.2, 0.5), life=0.5)
            m.fx.burst_ball(tip, size=0.5, col=(1.8, 0.9, 0.4), life=0.12)
            m.fx.shake(0.12)
        elif name == "sheathe":
            m.sword_drawn = False
            m.sfx("sheathe")
            m.fx.sparks(m.attach("bladeBase"), n=12, speed=4.0, col=(2.0, 1.2, 0.6), life=0.3)
        elif name == "swing":
            m.sfx("sword_swing", pitch=0.9 + 0.2 * np.random.rand())
        elif name == "hit":
            tip, mid = m.attach("bladeTip"), m.attach("bladeMid")
            for p in (tip, mid):
                m.hit(p, 1.3, 1.1)
            m.fx.sparks(tip, n=34, speed=9.0, col=(2.6, 1.0, 0.4), life=0.5)
            m.fx.burst_ball(tip, size=0.7, col=(2.0, 0.8, 0.35), life=0.12)
            m.fx.impact_feel(shake=0.30, stop=0.045, flash=0.10, aberr=0.35)
            m.sfx("sword_hit", pitch=0.9 + 0.25 * np.random.rand())
        elif name == "kick":
            p = m.attach("footR")
            m.hit(p, 1.6, 1.3)
            m.fx.sparks(p, n=40, speed=8.0, col=(2.4, 1.4, 0.6), life=0.5)
            m.fx.burst_ball(p, size=0.8, col=(2.0, 1.1, 0.5), life=0.14)
            m.fx.impact_feel(shake=0.35, stop=0.05, flash=0.12, aberr=0.4)
            m.sfx("kick")
        elif name == "slam":
            tip = m.attach("bladeTip")
            f = m.forward
            p = np.array([tip[0], 0.0, tip[2]], F32)
            m.fx.ring(p, r0=0.4, speed=14.0, thick=0.45, col=(1.0, 0.25, 0.1), intensity=3.6, life=0.9, fill=0.14)
            m.fx.ring(p, r0=0.2, speed=8.0, thick=0.3, col=(1.0, 0.75, 0.3), intensity=3.0, life=1.0)
            m.fx.dust(p, n=100, speed=9.0, radius=0.5, size=1.0, life=1.8)
            m.fx.sparks(p + np.array([0, 0.1, 0], F32), n=100, speed=13, col=(2.8, 1.0, 0.4), life=0.9)
            m.fx.burst_ball(p + np.array([0, 0.3, 0], F32), size=2.2, col=(2.6, 1.0, 0.4), life=0.24)
            m.fx.impact_feel(shake=0.9, stop=0.08, slow=0.45, slow_scale=0.45, flash=0.4, aberr=0.8, fov=1.0, col=(1.0, 0.55, 0.35))
            m.sfx("slam")
            m.hit(p, 3.4, 2.2)
            self.wave = dict(pos=p.copy(), dir=f.copy(), t=0.0)

    def update(self, dt, intent):
        m = self.m
        self.t += dt
        mv = self.move
        self.ts += dt
        self.pv = mv.body.sample(self.ts)
        sw = mv.sword.sample(self.ts)
        sw["w"] *= min(1.0, self.w * 1.4)
        m.sword_ik = sw
        if mv.spin is not None:
            t0, t1 = mv.spin
            m.spin_yaw = 2 * PI * e_io((self.ts - t0) / (t1 - t0))
        for te, name in mv.events:
            if self.ts >= te and (mv.name, name, te) not in self._ev:
                self._ev.add((mv.name, name, te))
                self.event(name)
        # glowing blade streak while cutting: fill the surface swept by the blade since the last frame
        trail_on = any(a <= self.ts <= b for a, b in mv.trail)
        if trail_on and m.sword_drawn and m.world is not None:
            tip, mid = m.attach("bladeTip"), m.attach("bladeMid")
            cur = (mid.copy(), tip.copy())
            prev = self.prev_seg if self.prev_seg is not None else cur
            for u in np.linspace(0.0, 1.0, 5):
                for v in np.linspace(0.25, 1.0, 8):
                    p0 = prev[0] + (prev[1] - prev[0]) * v
                    p1 = cur[0] + (cur[1] - cur[0]) * v
                    m.fx.glow(p0 + (p1 - p0) * u, size=0.16, size_end=0.02, life=0.24, col=(1.0, 0.24, 0.08), alpha=0.30)
            self.prev_seg = cur
        else:
            self.prev_seg = None
        # moving slash-wave projectile from the overhead finisher
        wv = getattr(self, "wave", None)
        if wv is not None:
            wv["t"] += dt
            wv["pos"] = wv["pos"] + wv["dir"] * 16.0 * dt
            if wv["t"] > 0.7:
                self.wave = None
            else:
                a = 1.0 - wv["t"] / 0.7
                side = np.array([wv["dir"][2], 0, -wv["dir"][0]], F32)
                for k in np.linspace(-1.3, 1.3, 9):
                    p = wv["pos"] + side * k + np.array([0, 0.25 + 0.5 * (1 - abs(k) / 1.4), 0], F32)
                    m.fx.glow(p, size=0.34 * a, size_end=0.05, life=0.18, col=(1.0, 0.2, 0.08), alpha=0.55 * a)
                m.hit(wv["pos"], 1.8, 0.35 * dt * 60 / 60)
        if self.ts >= mv.dur:
            self._next()
        # draw / sheathe bookkeeping: sheathing keeps the sword visible until its event; a skipped event must not strand the blade
        if self.finishing and m.sword_drawn:
            m.sword_drawn = False
        self._envelope(dt)
        m.thrust = 0.0


class Laser(Action):
    hold = True
    blend_in = 0.25
    blend_out = 0.3

    def __init__(self, m):
        super().__init__(m)
        self.holding = True
        self.beam = 0.0
        self.charge_snd = False
        self.fired = False
        self.aim_yaw = 0.0
        self.aim_pitch = 0.0
        self.dir = np.array([0, 0, 1], F32)
        self.hit_pt = None
        self.spark_t = 0.0

    def release(self):
        self.holding = False

    def update(self, dt, intent):
        m = self.m
        self.t += dt
        pv = LASER.sample(self.t)
        # vertical aim tilts the firing arm
        self.aim_pitch += (clamp(intent.aim_pitch, -0.4, 0.5) - self.aim_pitch) * min(1, dt * 8)
        pv[BI["uarmL"] * 3] += -self.aim_pitch * 0.9
        self.pv = pv
        # turn the whole body toward the aim
        self.aim_yaw += (clamp(intent.aim_yaw, -1.1, 1.1) - self.aim_yaw) * min(1, dt * 7)
        m.yaw += self.aim_yaw * min(1, dt * 2.2) * 1.0
        glow = pv[XI["glow"]]
        m.charge_target = max(m.charge_target, glow)
        if self.t > 0.30 and not self.charge_snd:
            self.charge_snd = True
            m.sfx("laser_charge")
        palm = m.attach("palmL")
        if 0.30 < self.t < 0.85 and self.holding:
            # charge-up: particles spiral into the palm
            k = (self.t - 0.3) / 0.55
            for _ in range(3):
                ang = np.random.rand() * 2 * PI
                r = 0.9 * (1 - k) + 0.1
                off = np.array([math.cos(ang) * r, math.sin(ang) * r, 0], F32)
                f = np.array([math.sin(m.yaw), 0, math.cos(m.yaw)], F32)
                side = np.array([math.cos(m.yaw), 0, -math.sin(m.yaw)], F32)
                pos = palm + f * 0.35 + side * off[0] + np.array([0, off[1], 0], F32)
                m.fx.glow(pos, vel=(palm - pos) * 5.0, size=0.10, size_end=0.02, life=0.2, col=(0.4, 1.0, 1.0), alpha=1.0)
            m.fx.glow(palm, size=0.25 + 0.5 * k, size_end=0.1, life=0.06, col=(0.5, 1.0, 1.0), alpha=1.0)
        # beam
        if self.holding and self.t > 0.88:
            if not self.fired:
                self.fired = True
                m.sfx("laser_fire")
                m.fx.impact_feel(shake=0.5, flash=0.3, aberr=0.6, fov=0.6, col=(0.6, 1.0, 1.0))
            self.beam = min(1.0, self.beam + dt * 10)
        else:
            self.beam = max(0.0, self.beam - dt * 6)
        yaw = m.yaw
        bp = self.aim_pitch * 0.7
        self.dir = np.array([math.sin(yaw) * math.cos(bp), math.sin(bp), math.cos(yaw) * math.cos(bp)], F32)
        self.origin = palm + self.dir * 0.12
        # intersect floor or run to max range
        length = 38.0
        self.hit_pt = None
        if self.dir[1] < -0.01:
            tf = -self.origin[1] / self.dir[1]
            if 0 < tf < length:
                length = tf
                self.hit_pt = self.origin + self.dir * tf
        # target hits along the ray
        th = m.ray_hit(self.origin, self.dir, length)
        if th is not None and self.beam > 0.5:
            length = th[0]
            self.hit_pt = th[1]
        self.length = length
        if self.beam > 0.3 and self.hit_pt is not None:
            self.spark_t += dt
            if self.spark_t > 0.03:
                self.spark_t = 0.0
                m.fx.sparks(self.hit_pt, n=7, speed=6.0, life=0.5, col=(0.6, 2.0, 2.4), up=0.8)
                m.fx.glow(self.hit_pt, size=0.5, size_end=0.8, life=0.08, col=(0.6, 1.0, 1.0), alpha=0.9)
                m.fx.smoke(self.hit_pt, n=1, size=0.3, life=0.8, alpha=0.2)
        if self.beam > 0.3:
            m.fx.shake(0.02 * self.beam)
            m.hit_beam(self.origin, self.dir, length, dt)
        if not self.holding and self.beam <= 0.0:
            self.finishing = True
        if not self.holding and not self.finishing:
            self.finishing = True
        self._envelope(dt)
        m.beam = self

    def beams(self):
        """Beam draw records."""
        if self.beam < 0.02:
            return []
        t = self.m.fx.t
        w = 0.15 * self.beam * (1.0 + 0.10 * math.sin(t * 60.0))
        p0 = self.origin
        p1 = self.origin + self.dir * self.length
        return [dict(p0=p0, p1=p1, w0=w, w1=w, col=(0.20, 0.85, 1.0), core=(1.2, 1.4, 1.4), int=1.5 * self.beam, mode=0),
                dict(p0=p0, p1=p1, w0=w * 2.8, w1=w * 2.8, col=(0.1, 0.5, 1.0), core=(0.15, 0.4, 0.9), int=0.35 * self.beam, mode=0)]


class Shield(Action):
    hold = True
    blend_in = 0.2
    blend_out = 0.3

    def __init__(self, m):
        super().__init__(m)
        self.holding = True
        self.s = 0.0
        self.next_hit = 0.9
        m.sfx("shield_on")

    def release(self):
        self.holding = False

    def update(self, dt, intent):
        m = self.m
        self.t += dt
        self.pv = SHIELD.sample(self.t)
        m.charge_target = max(m.charge_target, 0.6)
        if self.holding:
            self.s = min(1.0, self.s + dt * 3.2)
            self.next_hit -= dt
            if self.next_hit <= 0:
                self.next_hit = 0.8 + np.random.rand() * 1.1
                d = np.array([np.random.randn() * 0.7, np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.9 + 0.4], F32)
                f = np.array([math.sin(m.yaw), 0, math.cos(m.yaw)], F32)
                side = np.array([math.cos(m.yaw), 0, -math.sin(m.yaw)], F32)
                dw = f * d[2] + side * d[0] + np.array([0, d[1], 0], F32)
                dw /= np.linalg.norm(dw)
                m.fx.shield_hit(dw)
                p = self.center() + dw * 2.4
                m.fx.sparks(p, n=22, speed=5.0, life=0.4, col=(0.5, 1.6, 2.4))
                m.fx.glow(p, size=0.5, size_end=1.0, life=0.15, col=(0.5, 1.0, 1.0), alpha=1.0)
                m.fx.shake(0.12)
                m.sfx("shield_hit", pitch=0.9 + np.random.rand() * 0.3)
        else:
            self.s = max(0.0, self.s - dt * 4.0)
            if self.s <= 0:
                self.finishing = True
        self._envelope(dt)
        m.shield = self

    def center(self):
        m = self.m
        f = np.array([math.sin(m.yaw), 0, math.cos(m.yaw)], F32)
        return np.array([m.pos[0], 2.1 + max(m.pos[1], 0), m.pos[2]], F32) + f * 0.2

    def draw(self):
        if self.s < 0.01:
            return None
        k = e_back(self.s)
        pulse = 1.0 + 0.04 * math.sin(self.m.fx.t * 9)
        hits = np.zeros((4, 4), F32)
        hits[:, 3] = 9.0
        for i, (d, a) in enumerate(self.m.fx.shield_hits[-4:]):
            hits[i, :3] = d
            hits[i, 3] = a
        # world-space directions need to be in the sphere's local frame (identity rotation) -> same
        return dict(center=self.center(), radius=2.75 * k * pulse, col=(0.25, 0.85, 1.0), int=0.55 * min(1.0, self.s * 1.3), hits=hits)


class PowerUp(OneShot):
    TL = POWER
    blend_in = 0.2

    def event(self, name):
        m = self.m
        if name == "charge":
            m.sfx("powerup")
        elif name == "burst":
            p = m.pos.copy()
            p[1] = 0
            for i, (sp, th) in enumerate(((10.0, 0.45), (16.0, 0.35), (22.0, 0.25))):
                m.fx.ring(p, r0=0.4, speed=sp, thick=th, col=(1.0, 0.75, 0.3) if i != 1 else (0.4, 0.9, 1.0), intensity=3.2, life=1.1 + i * 0.2, fill=0.1)
            m.fx.dust(p, n=120, speed=8.0, radius=0.5, size=1.0, life=2.0)
            m.fx.sparks(p + np.array([0, 0.3, 0], F32), n=140, speed=14, life=1.2, up=1.0)
            m.fx.impact_feel(shake=0.9, flash=0.8, aberr=1.2, fov=1.3, slow=0.45, slow_scale=0.5, col=(1.0, 0.9, 0.6))
            m.overdrive = 14.0
            m.sfx("powerup_burst")

    def motion(self, dt, intent):
        m, t = self.m, self.t
        m.charge_target = max(m.charge_target, self.pv[XI["glow"]])
        if t < 1.5:
            # energy spiralling up the body
            k = t / 1.5
            for _ in range(2):
                a = np.random.rand() * 2 * PI
                r = 1.3 * (1 - 0.5 * k)
                pos = np.array([m.pos[0] + math.cos(a) * r, 0.05, m.pos[2] + math.sin(a) * r], F32)
                m.fx.glow(pos, vel=(0, 3.5 + 3.0 * k, 0), size=0.14, size_end=0.03, life=0.7, col=(1.0, 0.7, 0.25), alpha=1.0)
            m.fx.embers(m.pos, n=2, radius=1.2)
            m.fx.shake(0.01 + 0.015 * k)
        m.thrust = 0.2 if t < 1.4 else 0.0
        m.vel[:] = 0


# ------------------------------------------------------------------ the mecha
class Mecha:
    VMAX_WALK = 2.6
    VMAX_RUN = 6.5

    def __init__(self, fx, audio=None):
        self.fx = fx
        self.audio = audio
        self.pos = np.zeros(3, F32)
        self.vel = np.zeros(3, F32)
        self.yaw = 0.0
        self.speed = 0.0
        self.phase = 0.0
        self.action = None
        self.overdrive = 0.0
        self.charge = 0.0
        self.charge_target = 0.0
        self.thrust = 0.0
        self.time = 0.0
        self.curl = np.full(5, 0.25, F32)
        self.curl_l = 0.25
        self.curl_r = 0.25
        self.pose = Pose()
        self.world = None
        self.skin = None
        self.beam = None
        self.shield = None
        self.cooldown = {}
        self.last_step = [0.0, 0.0]
        self.show_w = 0.0
        self.show_hand = False
        self.emis_scale = 1.0
        self.emis_target = 1.0
        self.charge_boost = 0.0
        self.gphase = 0.0
        self.gait_amt = 0.0
        self.gait_Y = 0.0
        self.sword_drawn = False
        self.sword_ik = None
        self.spin_yaw = 0.0
        self.hilt_w = 0.0
        self.look_world = None
        self.look_yaw = 0.0
        self.look_pitch = 0.0
        self.frozen_pose = False
        self.on_hit = None
        self.on_ray = None
        self.on_beam = None
        self.status = "IDLE"
        self.gait_pv = STAND.copy()
        self.last_pv = STAND.copy()
        self.xfade_pv = STAND.copy()
        self.xfade = 0.0
        self.step_side = 0
        self.energy = 1.0
        self.update(0.0, Intent())

    # ---- hooks into the world (targets etc.)
    def hit(self, p, radius, power):
        if self.on_hit:
            self.on_hit(np.asarray(p, F32), radius, power)

    def ray_hit(self, origin, direction, length):
        return self.on_ray(origin, direction, length) if self.on_ray else None

    def hit_beam(self, origin, direction, length, dt):
        if self.on_beam:
            self.on_beam(origin, direction, length, dt)

    def sfx(self, name, **kw):
        if self.audio:
            self.audio.play(name, **kw)

    def attach(self, name):
        bone, off = ATTACH[name]
        return world_point(self.world, bone, off)

    def attach_dir(self, name, d):
        bone, _ = ATTACH[name]
        return world_dir(self.world, bone, d)

    @property
    def forward(self):
        return np.array([math.sin(self.yaw), 0, math.cos(self.yaw)], F32)

    # ---- action management
    def _try_start(self, intent):
        ready = lambda n: self.cooldown.get(n, 0.0) <= 0.0
        trig = []
        if intent.trig_power and ready("power"):
            trig.append((PowerUp, "power", 3.0))
        if intent.trig_jump and ready("jump"):
            trig.append((JumpSlam, "jump", 0.6))
        if intent.trig_dash and ready("dash"):
            trig.append((Dash, "dash", 0.5))
        holds = []
        if intent.hold_shield:
            holds.append((Shield, "shield", 0.0))
        if intent.hold_laser:
            holds.append((Laser, "laser", 0.0))
        if intent.hold_combo:
            holds.append((Combo, "combo", 0.0))
        a = self.action
        if a is not None and not a.done:
            can_interrupt = a.hold and bool(trig)                      # power moves cancel held actions at once
            can_replace = a.hold and a.finishing and a.w < 0.4 and bool(holds)
            if not (can_interrupt or can_replace):
                return
        order = trig + holds
        if order:
            cls, name, cd = order[0]
            self.xfade_pv = self.last_pv.copy()
            self.xfade = 1.0
            if isinstance(self.action, Combo):
                self.sword_drawn = False
            self.action = cls(self)
            tl = getattr(self.action, "TL", None)
            self.cooldown[name] = cd + (tl.dur if tl is not None else 0.0)

    def update(self, dt, intent):
        self.time += dt
        for k in list(self.cooldown):
            self.cooldown[k] = max(0.0, self.cooldown[k] - dt)
        self.overdrive = max(0.0, self.overdrive - dt)
        self.beam = None
        self.shield = None
        self.sword_ik = None
        self.spin_yaw = 0.0
        self.thrust = 0.0
        self.frozen_pose = isinstance(self.action, BootUp) and self.action.frozen
        self.charge_target = 0.12 + (0.45 if self.overdrive > 0 else 0.0)
        # ---- actions
        a = self.action
        if a is not None:
            if a.hold and not a.finishing:
                held = {"Combo": intent.hold_combo, "Laser": intent.hold_laser, "Shield": intent.hold_shield}[type(a).__name__]
                if not held:
                    a.release()
        self._try_start(intent)
        a = self.action
        if a is not None and a.done:
            self.action = a = None
        # ---- base locomotion (when free or blending)
        free = a is None
        self._locomotion(dt, intent, free)
        self.show_hand = free and intent.curls is not None
        base = self._base_pose(dt, intent)
        if a is not None:
            a.update(dt, intent)
            w = a.w
            pv = base + (a.pv - base) * w
            self.status = type(a).__name__.upper()
            if a.done:
                self.action = None
        else:
            pv = base
            self.status = "SPRINT" if self.speed > 4.0 else "MOVE" if self.speed > 0.4 else "IDLE"
        # cross-fade from the previous pose when an action starts or is interrupted
        if self.xfade > 0.0:
            pv = pv + (self.xfade_pv - pv) * self.xfade
            self.xfade = max(0.0, self.xfade - dt / 0.20)
        self.last_pv = pv.copy()
        # charge glow smoothing
        self.charge += (self.charge_target - self.charge) * min(1.0, dt * 6)
        self.emis_scale += (self.emis_target - self.emis_scale) * min(1.0, dt * 2.5)
        self._build_pose(pv, dt, intent)
        self._spawn_fx(dt)

    # ---- locomotion
    def _locomotion(self, dt, intent, free):
        vdir = np.zeros(3, F32)
        target_speed = 0.0
        if free:
            if intent.move_target is not None:
                d = np.array([intent.move_target[0] - self.pos[0], 0, intent.move_target[1] - self.pos[2]], F32)
                dist = float(np.linalg.norm(d))
                if dist > 0.45:
                    vdir = d / dist
                    target_speed = min(self.VMAX_RUN if (dist > 4.5 or intent.sprint) else self.VMAX_WALK + 0.4, 0.9 + dist * 1.1)
            elif intent.move_dir is not None:
                mv = np.array([intent.move_dir[0], 0, intent.move_dir[1]], F32)
                n = float(np.linalg.norm(mv))
                if n > 0.05:
                    vdir = mv / n
                    target_speed = (self.VMAX_RUN if intent.sprint else self.VMAX_WALK) * min(1.0, n)
            if self.overdrive > 0:
                target_speed *= 1.35
        k = min(1.0, dt * (5.0 if target_speed > self.speed else 7.0))
        self.speed += (target_speed - self.speed) * k
        if free:
            if self.speed > 0.25 and np.linalg.norm(vdir) > 0.1:
                goal = math.atan2(vdir[0], vdir[2])
                diff = (goal - self.yaw + PI) % (2 * PI) - PI
                self.yaw += clamp(diff, -7.0 * dt, 7.0 * dt) if abs(diff) > 0.02 else diff
            elif abs(intent.turn) > 0.05:
                self.yaw += intent.turn * 2.4 * dt
            if self.speed > 0.02:
                f = self.forward
                self.pos += f * self.speed * dt
                self.vel = f * self.speed
        r = float(np.hypot(self.pos[0], self.pos[2]))
        if r > 26.0:
            self.pos[0] *= 26.0 / r
            self.pos[2] *= 26.0 / r
        # gait clock: cadence rises with speed; foot-planted via IK
        run = smooth(3.2, 5.2, self.speed)
        cyc = 1.15 + (0.60 - 1.15) * run
        prev = self.gphase
        if self.speed > 0.12 and dt > 0:
            self.gphase = (self.gphase + dt / cyc) % 1.0
            if self.speed > 0.8:
                if self.gphase < prev:
                    self._footstep("L")
                if prev < 0.5 <= self.gphase:
                    self._footstep("R")
        self.gait_amt += (smooth(0.12, 0.9, self.speed) - self.gait_amt) * min(1.0, dt * 9.0)

    def _gait_params(self):
        sp = self.speed
        run = smooth(3.2, 5.2, sp)
        sf = 0.62 + (0.40 - 0.62) * run
        cyc = 1.15 + (0.60 - 1.15) * run
        return run, sf, cyc

    def _gait_leg_targets(self):
        """Per-leg ankle targets in the root frame: (lateral, forward, lift, foot pitch) + pelvis height + flight hover."""
        sp = self.speed
        run, sf, cyc = self._gait_params()
        L = max(sp, 0.0) * sf * cyc                      # distance a planted foot travels back relative to the body
        H = 0.20 + 0.32 * run
        out = {}
        for side, off, lat in (("L", 0.0, 1.0), ("R", 0.5, -1.0)):
            ph = (self.gphase + off) % 1.0
            if ph < sf:                                   # stance: foot fixed in the world
                u = ph / sf
                z = L * (0.5 - u)
                lift = 0.0
                if u < 0.18:
                    pitch = -0.22 * (1.0 - u / 0.18)      # heel strike (toes up)
                elif u > 0.70:
                    k = (u - 0.70) / 0.30
                    pitch = 0.55 * k                      # heel lifts, toe pushes off
                    lift = 0.07 * k
                else:
                    pitch = 0.0
            else:                                         # swing
                u = (ph - sf) / (1.0 - sf)
                e = 0.5 - 0.5 * math.cos(PI * u)
                z = -L * 0.5 + L * e
                lift = H * (math.sin(PI * u) ** 0.85) + 0.07 * (1.0 - u) ** 2
                pitch = 0.55 * (1.0 - e) - 0.22 * e + 0.18 * math.sin(PI * u)
            out[side] = (lat * (0.40 - 0.07 * run), z, lift, pitch)
        crouch = 0.05 + 0.07 * run
        A = 0.022 + 0.02 * run
        Y = -crouch - A + A * math.cos(4 * PI * (self.gphase - sf / 2))
        hover = 0.0
        if run > 0.3:
            m = self.gphase % 0.5
            if sf < 0.5 and sf <= m < 0.5:
                hover = 0.18 * run * math.sin(PI * (m - sf) / (0.5 - sf))
        return out, Y, hover

    def _base_pose(self, dt, intent):
        t = self.time
        pv = STAND.copy()
        # idle breathing + weight shift
        br = math.sin(t * 1.9)
        pv[BI["chest"] * 3] += 0.012 * br
        pv[BI["spine"] * 3] += 0.01 * br
        pv[BI["pelvis"] * 3 + 2] += 0.018 * math.sin(t * 0.9)
        pv[BI["uarmL"] * 3 + 2] += 0.015 * br
        pv[BI["uarmR"] * 3 + 2] -= 0.015 * br
        pv[XI["wopen"]] = 0.0
        sp = self.speed
        gw = self.gait_amt
        self.gait_Y = 0.0
        if gw > 0.01:
            amp = clamp(sp / 2.2)
            run = smooth(3.2, 5.2, sp)
            c, s = math.cos(2 * PI * self.gphase), math.sin(2 * PI * self.gphase)
            A_arm = (0.38 + 0.36 * run) * amp
            for side, sg in (("L", 1.0), ("R", -1.0)):
                cs = c * sg                                   # left arm is back when the left leg is forward (phase 0)
                pv[BI["uarm" + side] * 3] = A_arm * cs - 0.06 - 0.28 * run
                pv[BI["farm" + side] * 3] = -0.35 - 0.95 * run - 0.30 * max(0.0, -cs) * amp
                pv[BI["uarm" + side] * 3 + 2] = sg * -1.0 * -0.16 + (0.0)
            pv[BI["pelvis"] * 3 + 1] = -0.13 * c * amp
            pv[BI["spine"] * 3 + 1] = 0.07 * c * amp
            pv[BI["chest"] * 3 + 1] = 0.10 * c * amp
            pv[BI["pelvis"] * 3 + 2] = 0.045 * s * amp
            pv[BI["chest"] * 3 + 2] = -0.03 * s * amp
            pv[XI["pitch"]] = 0.04 + 0.22 * run
            pv[BI["spine"] * 3] += 0.04 + 0.12 * run
            pv[BI["neck"] * 3] -= 0.12 * run
            pv[XI["thr"]] = 0.5 * run
            pv[XI["nozt"]] = 0.9 * run
            pv[XI["wopen"]] = 0.7 * run
            pv[XI["cL"]] = pv[XI["cR"]] = 0.5 + 0.5 * run
            # legs are IK driven: pelvis height and (flight) hover come from the gait
            targets, Y, hover = self._gait_leg_targets()
            self.gait_Y = Y
            pv[XI["y"]] = gw * (Y + hover)
            pv[XI["ground"]] = 1.0 - gw
        # tracked hand: the robot raises its left hand to mirror your fingers
        tgt = (1.0 - smooth(0.8, 2.4, sp)) if self.show_hand else 0.0
        self.show_w += (tgt - self.show_w) * min(1.0, dt * 4.0)
        if self.show_w > 0.001:
            w_ = self.show_w
            for bone, val in SHOW_POSE.items():
                i = BI[bone] * 3
                for ax in range(3):
                    pv[i + ax] += (val[ax] - pv[i + ax]) * w_
        # idle: right hand rests on the katana hilt, left hand steadies the saya
        free = self.action is None
        hilt_t = 1.0 if (free and not self.sword_drawn and sp < 0.7 and not self.frozen_pose) else 0.0
        self.hilt_w += (hilt_t - self.hilt_w) * min(1.0, dt * 3.0)
        # overdrive: wings stay open, glow up
        if self.overdrive > 0:
            pv[XI["wopen"]] = max(pv[XI["wopen"]], 0.55)
        return pv

    def _footstep(self, side):
        if self.world is None:
            return
        f = self.attach("foot" + side)
        run = self.speed > 3.6
        self.fx.dust(np.array([f[0], 0.02, f[2]], F32), n=6 if run else 3, speed=1.6, radius=0.2, size=0.35, life=0.9)
        self.fx.shake(0.04 if not run else 0.07)
        self.sfx("step", vol=0.5 + 0.4 * clamp(self.speed / 5.0), pitch=0.9 + np.random.rand() * 0.2)
        if run:
            self.fx.sparks(np.array([f[0], 0.05, f[2]], F32), n=5, speed=3.0, life=0.3)

    # ---- IK helpers -----------------------------------------------------------------
    def _apply_springs(self, rot, dt):
        """Second-order follow-through on torso, head, shoulders and arms (never on IK-driven legs)."""
        if not hasattr(self, "_sp_idx"):
            idx, w = [], []
            for name, om in SPRING_W.items():
                for b in BI:
                    base = b[:-1] if b[-1] in "LR" and b[:-1] in SPRING_W else b
                    if base == name:
                        for ax in range(3):
                            idx.append(BI[b] * 3 + ax)
                            w.append(om)
            self._sp_idx = np.array(idx)
            self._sp_w = np.array(w, np.float64)
            self._sp_x = rot[self._sp_idx].astype(np.float64)
            self._sp_v = np.zeros(len(idx), np.float64)
        tgt = rot[self._sp_idx].astype(np.float64)
        if dt <= 0:
            self._sp_x[:] = tgt
            return
        n = max(1, int(math.ceil(dt / 0.008)))
        h = dt / n
        zeta = 0.80
        w = self._sp_w
        for _ in range(n):
            a = w * w * (tgt - self._sp_x) - 2 * zeta * w * self._sp_v
            self._sp_v += a * h
            self._sp_x += self._sp_v * h
        # snap if wildly off (e.g. after a pose reset) to avoid long swings
        if np.max(np.abs(self._sp_x - tgt)) > 2.0:
            self._sp_x[:] = tgt
            self._sp_v[:] = 0
        rot[self._sp_idx] = self._sp_x.astype(rot.dtype)

    def _blend_rot(self, pose, bone, euler, w):
        i = BI[bone]
        cur = pose.rot[i].astype(np.float64)
        e = np.asarray(euler, np.float64)
        d = (e - cur + PI) % (2 * PI) - PI                    # shortest way round
        pose.rot[i] = (cur + d * w).astype(F32)

    def _gait_ik(self, pose, base_pos, gw):
        targets, Y, hover = self._gait_leg_targets()
        pose.root_pos[:] = (base_pos[0], 0.0, base_pos[2])
        world, _ = forward_kinematics(pose)
        yaw = pose.root_yaw
        Ry_ = ik.Ry(yaw)
        origin = np.array([base_pos[0], 0.0, base_pos[2]], np.float64)
        for side, sg in (("L", 1.0), ("R", -1.0)):
            lat, z, lift, pitch = targets[side]
            T_w = origin + Ry_ @ np.array([lat, 0.22 + lift - Y, z])
            thigh = BI["thigh" + side]
            P = world[PARENT[thigh]][:3, :3].astype(np.float64)
            S = world[thigh][:3, 3].astype(np.float64)
            o1 = LOCAL[BI["shin" + side]].astype(np.float64)
            o2 = LOCAL[BI["foot" + side]].astype(np.float64)
            pole = Ry_ @ np.array([sg * 0.10, 0.06, 1.0])
            e1, kappa, ok = ik.solve_limb(P, S, o1, o2, T_w, pole, +1, (0.0, 2.3))
            R1 = P @ ik.euler_R(*e1)
            R2 = R1 @ ik.Rx(kappa)
            R3 = R2.T @ (Ry_ @ ik.Rx(pitch))
            self._blend_rot(pose, "thigh" + side, e1, gw)
            self._blend_rot(pose, "shin" + side, (kappa, 0.0, 0.0), gw)
            self._blend_rot(pose, "foot" + side, ik.R_to_euler(R3), gw)

    def _arm_ik(self, pose, world, side, wrist_world, R_des, pole_local, w):
        """Drive one arm so its wrist reaches wrist_world with hand orientation R_des (all world space)."""
        chest = world[BI["chest"]]
        P = chest[:3, :3].astype(np.float64)
        up = BI["uarm" + side]
        S = world[up][:3, 3].astype(np.float64)
        o1 = LOCAL[BI["farm" + side]].astype(np.float64)
        o2 = LOCAL[BI["hand" + side]].astype(np.float64)
        pole = P @ np.asarray(pole_local, np.float64)
        e1, kappa, ok = ik.solve_limb(P, S, o1, o2, wrist_world, pole, -1, (-2.6, 0.0))
        R1 = P @ ik.euler_R(*e1)
        R2 = R1 @ ik.Rx(kappa)
        R3 = R2.T @ R_des
        self._blend_rot(pose, "uarm" + side, e1, w)
        self._blend_rot(pose, "farm" + side, (kappa, 0.0, 0.0), w)
        self._blend_rot(pose, "hand" + side, ik.R_to_euler(R3), w)

    def _grip_frame(self, H, D, roll):
        """World rotation of a hand gripping a sword: Z along the blade, edge direction rotated by roll."""
        D = np.asarray(D, np.float64)
        D = D / np.linalg.norm(D)
        down = np.array([0.0, -1.0, 0.0])
        e0 = down - D * float(np.dot(down, D))
        if np.linalg.norm(e0) < 1e-6:
            e0 = np.array([0.0, 0.0, 1.0])
        e0 /= np.linalg.norm(e0)
        Ed = ik.axis_angle_R(D, roll) @ e0
        Y = -Ed
        X = np.cross(Y, D)
        return np.column_stack([X, Y, D])

    def _sword_overlay(self, pose, world):
        """Arm IK for sword work: either an action-driven grip or the idle hand-on-hilt pose."""
        chest = world[BI["chest"]].astype(np.float64)
        Pc = chest[:3, :3]
        sw = self.sword_ik
        if sw is not None and sw["w"] > 0.002:
            H = chest[:3, :3] @ np.asarray(sw["H"], np.float64) + chest[:3, 3]
            D = Pc @ np.asarray(sw["D"], np.float64)
            R = self._grip_frame(H, D, sw.get("roll", 0.0))
            self._arm_ik(pose, world, "R", H - R @ GRIP_OFF["R"], R, (-0.55, -0.60, -0.45), sw["w"])
            if sw.get("two", 0.0) > 0.002:
                G2 = H - (R[:, 2]) * sw.get("grip2", 0.30)
                self._arm_ik(pose, world, "L", G2 - R @ GRIP_OFF["L"], R, (0.55, -0.60, -0.45), sw["two"] * sw["w"])
            return
        hw = self.hilt_w
        if hw > 0.002:
            ks = world[BI["katS"]].astype(np.float64)
            Rh = ks[:3, :3] @ R_HILT0
            H = ks[:3, 3]
            self._arm_ik(pose, world, "R", H - Rh @ GRIP_OFF["R"], Rh, (-0.55, -0.60, -0.45), hw)
            wl = hw * (1.0 - self.show_w)
            if wl > 0.002:
                G2 = H + Rh[:, 2] * 0.56
                self._arm_ik(pose, world, "L", G2 - Rh @ GRIP_OFF["L"], Rh, (0.55, -0.60, -0.45), wl)

    # ---- pose assembly
    def _build_pose(self, pv, dt, intent):
        pose = self.pose
        rot = pv[:NB * 3].copy()          # flat: index with bone*3 + axis
        ex = lambda n: float(pv[XI[n]])
        a = self.action
        aw = a.w if a is not None else 0.0
        sword = self.sword_ik
        gripR = max(sword["w"] if sword else 0.0, self.hilt_w)
        gripL = max((sword["two"] * sword["w"]) if sword else 0.0, self.hilt_w * (1.0 - self.show_w))
        # fingers: tracked hand when free, otherwise action-driven fists
        tracked = a is None and intent.curls is not None
        k = min(1.0, dt * 14.0)
        if tracked:
            self.curl += (np.asarray(intent.curls, F32) - self.curl) * k
            cL = cR = self.curl
            self.curl_l = self.curl_r = float(self.curl[1:].mean())
        else:
            self.curl_l += (ex("cL") - self.curl_l) * k
            self.curl_r += (ex("cR") - self.curl_r) * k
            self.curl[:] = 0.5 * (self.curl_l + self.curl_r)
            cL = np.full(5, self.curl_l, F32)
            cR = np.full(5, self.curl_r, F32)
        cR = cR + (0.95 - cR) * gripR
        cL = cL + (0.70 - cL) * gripL if gripL > 0 else cL
        self._apply_fingers(rot, cL, "L")
        self._apply_fingers(rot, cR, "R")
        # wing binders: authored rz plus a continuous "open" amount from locomotion / overdrive
        wo = ex("wopen") * 0.95
        flutter = 0.02 * math.sin(self.time * 1.3)
        rot[BI["wingL"] * 3 + 2] += -wo + flutter
        rot[BI["wingR"] * 3 + 2] += wo - flutter
        noz = ex("nozt")
        for side in ("L", "R"):
            rot[BI["noz" + side] * 3] += noz
        # head tracks the look-at target (camera / player) with a natural lag
        self._head_look(rot, dt, aw)
        # shoulder pads follow the arms partially
        for side in ("L", "R"):
            ua = rot[BI["uarm" + side] * 3:BI["uarm" + side] * 3 + 3]
            rot[BI["pad" + side] * 3] = 0.30 * ua[0]
            rot[BI["pad" + side] * 3 + 2] = 0.40 * ua[2]
        # root-level pitch/roll go on the pelvis
        rot[BI["pelvis"] * 3] += ex("pitch")
        rot[BI["pelvis"] * 3 + 2] += ex("roll")
        self._apply_springs(rot, dt)
        pose.rot[:] = rot.reshape(NB, 3)
        pose.root_yaw = self.yaw + ex("yaw") + self.spin_yaw
        pose.root_pitch = 0.0
        pose.root_roll = 0.0
        off = self.forward * ex("fwd")
        base_pos = self.pos + np.array([off[0], 0, off[2]], F32)
        g = ex("ground")
        # ---- foot flattening for the authored (FK) legs
        for side in ("L", "R"):
            chain = rot[BI["pelvis"] * 3] + rot[BI["thigh" + side] * 3] + rot[BI["shin" + side] * 3]
            fi = BI["foot" + side] * 3
            pose.rot[BI["foot" + side], 0] = rot[fi] - g * chain
        # ---- IK layers: gait legs, sword / hilt arms
        gw = self.gait_amt * (1.0 - aw)
        need_arms = (sword is not None and sword["w"] > 0.002) or self.hilt_w > 0.002
        if gw > 0.01:
            self._gait_ik(pose, base_pos, gw)
        if need_arms:
            pose.root_pos[:] = (base_pos[0], 0.0, base_pos[2])
            w1, _ = forward_kinematics(pose)
            self._sword_overlay(pose, w1)
            for side in ("L", "R"):                       # pads follow the IK-driven arms too
                ua = pose.rot[BI["uarm" + side]]
                pose.rot[BI["pad" + side], 0] = 0.30 * ua[0]
                pose.rot[BI["pad" + side], 2] = 0.40 * ua[2]
        pose.root_pos[:] = (base_pos[0], 0.0, base_pos[2])
        world, _ = forward_kinematics(pose)
        low = 1e9
        for bone, off_ in CONTACTS:
            low = min(low, float(world_point(world, bone, off_)[1]))
        y_ground = -low
        y = max(ex("y") + g * y_ground, y_ground)
        self.height = y
        pose.root_pos[:] = (base_pos[0], y, base_pos[2])
        self.world, self.skin = forward_kinematics(pose)
        # the katana is either in the right hand or sheathed on the hip
        Z = np.zeros((4, 4), F32)
        Z[3, 3] = 1.0
        self.skin[BI["katS" if self.sword_drawn else "kat"]] = Z
        self.thrust_now = clamp(self.thrust if self.action is not None else ex("thr"))
        self.thrust = 0.0 if self.action is None else self.thrust

    def _head_look(self, rot, dt, aw):
        tgt = self.look_world
        wgt = 0.0 if (tgt is None or self.frozen_pose or self.world is None) else (1.0 - 0.75 * aw)
        want_y = want_p = 0.0
        if wgt > 0:
            hp = world_point(self.world, "head", (0, 0, 0))
            v = np.asarray(tgt, np.float64) - hp
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            lx = v[0] * cy - v[2] * sy                        # component toward the robot's left
            lz = v[0] * sy + v[2] * cy                        # component along its heading
            want_y = max(-1.0, min(1.0, math.atan2(lx, lz)))
            want_p = max(-0.35, min(0.35, math.atan2(v[1], math.hypot(lx, lz))))
        kk = min(1.0, dt * 3.5)
        self.look_yaw += (want_y * wgt - self.look_yaw) * kk
        self.look_pitch += (want_p * wgt - self.look_pitch) * kk
        rot[BI["neck"] * 3 + 1] += 0.42 * self.look_yaw
        rot[BI["head"] * 3 + 1] += 0.58 * self.look_yaw
        rot[BI["neck"] * 3] -= 0.40 * self.look_pitch
        rot[BI["head"] * 3] -= 0.60 * self.look_pitch

    def _apply_fingers(self, rot, curls, side):
        sg = -1.0 if side == "L" else 1.0
        # curls: thumb, index, middle, ring, pinky
        for k, f in enumerate(("i", "m", "r", "p")):
            c = float(curls[k + 1])
            rot[BI["f%sa%s" % (f, side)] * 3 + 2] = sg * c * 1.35
            rot[BI["f%sb%s" % (f, side)] * 3 + 2] = sg * c * 1.45
        ct = float(curls[0])
        rot[BI["tha" + side] * 3 + 2] = sg * ct * 0.7
        rot[BI["thb" + side] * 3 + 2] = sg * ct * 0.9

    # ---- per-frame FX around the body
    def _spawn_fx(self, dt):
        fx = self.fx
        w = self.world
        thr = clamp(self.thrust_now)
        self.flames = []
        if thr > 0.05 or self.overdrive > 0:
            level = max(thr, 0.0)
            for side in ("L", "R"):
                exit_p = self.attach("noz" + side)
                d = -self.attach_dir("noz" + side, (0, 1, 0))      # exhaust direction (down in bone frame)
                d = d / (np.linalg.norm(d) + 1e-6)
                length = (0.5 + 3.2 * level) * (0.9 + 0.2 * np.random.rand())
                width = 0.20 + 0.16 * level
                self.flames.append(dict(p0=exit_p, p1=exit_p + d * length, w0=width, w1=width * 0.15, col=(0.25, 0.65, 1.0),
                                        core=(1.3, 1.5, 1.7), int=0.9 + 1.0 * level, mode=1))
                if level > 0.25:
                    fx.glow(exit_p + d * 0.2, size=0.35 + 0.4 * level, size_end=0.1, life=0.06, col=(0.5, 0.8, 1.0), alpha=0.55)
                    fx.glow(exit_p + d * length * 0.8, vel=d * 2.0, size=0.2, size_end=0.04, life=0.25, col=(0.35, 0.65, 1.0), alpha=0.4)
                    if np.random.rand() < 0.6:
                        fx.sparks(exit_p + d * length * 0.9, n=1, speed=3.0, life=0.4, col=(1.2, 1.4, 2.0), size=0.04)
                if level > 0.5 and exit_p[1] < 4.0:
                    ray = exit_p + d * length
                    if ray[1] < 0.3:
                        fx.dust(np.array([ray[0], 0.03, ray[2]], F32), n=2, speed=4.0, radius=0.25, size=0.6, life=0.9)
            if level > 0.5:
                for side in ("L", "R"):
                    c = self.attach("calf" + side)
                    d = np.array([0, -1, -0.35], F32)
                    d = (self.world[BI["shin" + side]][:3, :3] @ d)
                    d /= np.linalg.norm(d) + 1e-6
                    self.flames.append(dict(p0=c, p1=c + d * (0.3 + 1.1 * level), w0=0.10, w1=0.02, col=(0.25, 0.65, 1.0),
                                            core=(1.5, 1.7, 2.0), int=1.3, mode=1))
        # overdrive aura
        if self.overdrive > 0 and np.random.rand() < 0.8:
            a = np.random.rand() * 2 * PI
            r = 0.5 + np.random.rand() * 0.5
            p = np.array([self.pos[0] + math.cos(a) * r, 0.2 + np.random.rand() * 3.2 + max(self.height, 0), self.pos[2] + math.sin(a) * r], F32)
            fx.glow(p, vel=(0, 1.6, 0), size=0.10, size_end=0.02, life=0.8, col=(1.0, 0.65, 0.2), alpha=0.9)

    # ---- outputs for the renderer
    def beams(self):
        out = list(self.flames)
        if isinstance(self.action, Laser) or self.beam is not None:
            b = self.beam or (self.action if isinstance(self.action, Laser) else None)
            if b is not None:
                out += b.beams()
        return out

    def shield_draw(self):
        a = self.action
        return a.draw() if isinstance(a, Shield) else None

    def camera_wide(self):
        """Extra camera distance requested by the current action so big moves stay in frame."""
        n = type(self.action).__name__ if self.action is not None else ""
        return {"JumpSlam": 6.0, "Dash": 5.5, "PowerUp": 2.5, "Laser": 2.0, "Combo": 0.8}.get(n, 0.0) + (1.5 if self.speed > 4.0 else 0.0)

    def begin_powerdown(self):
        """Hold the body in a slumped, dark, powered-down pose (used while the parts are floating / assembling)."""
        self.action = BootUp(self)
        self.action.frozen = True
        self.emis_target = 0.05
        self.emis_scale = 0.05

    def begin_boot(self):
        if not isinstance(self.action, BootUp):
            self.action = BootUp(self)
        self.action.frozen = False
        self.action.t = 0.0
        self.action._ev.clear()
        self.emis_target = 1.0

    def emissive_boost(self):
        return self.emis_scale * (1.0 + (0.6 if self.overdrive > 0 else 0.0))

    def power_level(self):
        return clamp(self.overdrive / 3.0) if self.overdrive > 0 else 0.0
