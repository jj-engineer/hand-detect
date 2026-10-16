"""
Hand-gated reveal sequence.

  WAIT      empty stage (no robot).  open hand  ->
  PARTS     the robot's components pop up one by one and float in an exploded view.  closed hand  ->
  ASSEMBLE  every component flies in and clamps onto the (powered-down) body with clanks and sparks  ->
  BOOT      reactor ignites, eyes light, the mecha stands up  ->
  LIVE      normal control.
"""

import math

import numpy as np

from .skeleton import BI, NB
from .xform import translate as T

F32 = np.float32
PI = math.pi
WAIT, PARTS, ASSEMBLE, BOOT, LIVE = range(5)
NAMES = ["WAIT", "PARTS", "ASSEMBLE", "BOOT", "LIVE"]


def _side_bones(side):
    b = ["hand" + side, "tha" + side, "thb" + side]
    for f in "imrp":
        b += ["f%sa%s" % (f, side), "f%sb%s" % (f, side)]
    return b


# (name, leader bone, bones, assemble rank)
GROUPS = [
    ("torso", "chest", ["chest"], 0),
    ("waist", "pelvis", ["pelvis", "spine"], 1),
    ("thighL", "thighL", ["thighL"], 2), ("thighR", "thighR", ["thighR"], 2),
    ("shinL", "shinL", ["shinL"], 3), ("shinR", "shinR", ["shinR"], 3),
    ("footL", "footL", ["footL"], 4), ("footR", "footR", ["footR"], 4),
    ("padL", "padL", ["padL"], 3), ("padR", "padR", ["padR"], 3),
    ("uarmL", "uarmL", ["uarmL"], 3), ("uarmR", "uarmR", ["uarmR"], 3),
    ("farmL", "farmL", ["farmL"], 4), ("farmR", "farmR", ["farmR"], 4),
    ("handL", "handL", _side_bones("L"), 5), ("handR", "handR", _side_bones("R"), 5),
    ("wingL", "wingL", ["wingL"], 5), ("wingR", "wingR", ["wingR"], 5),
    ("nozL", "nozL", ["nozL"], 5), ("nozR", "nozR", ["nozR"], 5),
    ("head", "head", ["neck", "head"], 6),
    ("hilt", "katS", ["katS"], 6),
]
POP_ORDER = ["torso", "waist", "head", "padL", "padR", "uarmL", "uarmR", "farmL", "farmR", "handL", "handR",
             "thighL", "thighR", "shinL", "shinR", "footL", "footR", "wingL", "wingR", "nozL", "nozR", "hilt"]

POP_GAP = 0.15
POP_DUR = 0.55
ASM_DUR = 1.0
ASM_GAP = 0.20


def e_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def e_back(x):
    x = min(max(x, 0.0), 1.0)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], F32)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)


def mat_to_quat(R):
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], F32)


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + (q1 - q0) * t
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


class Reveal:
    def __init__(self, md, fx, audio=None):
        self.fx = fx
        self.audio = audio
        self.rng = np.random.default_rng(21)
        self.state = WAIT
        self.t = 0.0
        self.clock = 0.0
        self.skin = None
        self.bone_fx = None
        n = len(GROUPS)
        self.C = np.zeros((n, 3), F32)
        bones = md.bone.astype(int)
        for i, (name, leader, bl, rank) in enumerate(GROUPS):
            mask = np.isin(bones, [BI[b] for b in bl])
            pts = md.pos[mask]
            self.C[i] = (pts.min(0) + pts.max(0)) * 0.5
        # exploded layout: fan the anatomy outward, separating left/right, lifting everything off the floor
        self.P = np.zeros((n, 3), F32)
        for i, (name, _, _, _) in enumerate(GROUPS):
            c = self.C[i]
            side = 0.0 if abs(c[0]) < 0.12 else math.copysign(1.0, c[0])
            zoff = (0.55 if i % 2 else -0.35)
            self.P[i] = (side * 0.95 + 2.15 * c[0], 0.65 + 1.28 * c[1], 1.5 * c[2] + zoff)
        self.idx = {g[0]: i for i, g in enumerate(GROUPS)}
        self.pop_time = np.full(n, 1e9, F32)
        self.phase = self.rng.random(n).astype(F32) * 6.28
        self.spin0 = self.rng.random(n).astype(F32) * 6.28
        self.spin_w = (0.35 + 0.35 * self.rng.random(n)).astype(F32) * np.where(self.rng.random(n) > 0.5, 1, -1)
        self.cur_p = self.P.copy()
        self.cur_R = [np.eye(3, dtype=F32) for _ in range(n)]
        self.cur_s = np.ones(n, F32)
        self.p0 = self.P.copy()
        self.q0 = [mat_to_quat(np.eye(3)) for _ in range(n)]
        self.arrived = np.zeros(n, bool)
        self.arrive_t = np.full(n, 1e9, F32)
        self.delay = np.zeros(n, F32)
        self.popped = np.zeros(n, bool)
        self.boot_started = False
        self.ring_t = 0.0
        self.pending_assemble = False
        self.hum_on = False
        self.flash_boot = 0.0

    # ------------------------------------------------------------ transitions
    def trigger_parts(self, mecha=None):
        if self.state != WAIT:
            return
        self.state, self.t = PARTS, 0.0
        order = {n: k for k, n in enumerate(POP_ORDER)}
        for i, (name, _, _, _) in enumerate(GROUPS):
            self.pop_time[i] = 0.35 + order[name] * POP_GAP
        self.popped[:] = False
        if mecha is not None:
            mecha.begin_powerdown()

    def parts_ready(self):
        return self.state == PARTS and bool(self.popped.all())

    def trigger_assemble(self):
        if self.state != PARTS:
            return
        if not self.parts_ready():
            self.pending_assemble = True
            return
        self.state, self.t = ASSEMBLE, 0.0
        self.pending_assemble = False
        self.p0 = self.cur_p.copy()
        self.q0 = [mat_to_quat(R) for R in self.cur_R]
        self.arrived[:] = False
        self.arrive_t[:] = 1e9
        for i, (name, _, _, rank) in enumerate(GROUPS):
            self.delay[i] = rank * ASM_GAP + 0.05 * self.rng.random()
        if self.audio:
            self.audio.play("assemble", vol=0.9)
            self.audio.loop("float_hum", 0.0)
            self.hum_on = False

    def skip(self):
        self.state = LIVE
        self.skin = None
        self.bone_fx = None

    def reset(self, mecha=None):
        self.state, self.t = WAIT, 0.0
        self.pop_time[:] = 1e9
        self.popped[:] = False
        self.arrived[:] = False
        self.boot_started = False
        self.pending_assemble = False
        if self.audio:
            self.audio.loop("float_hum", 0.0)
        self.hum_on = False

    # ------------------------------------------------------------ per frame
    def active(self):
        return self.state != LIVE

    def mecha_visible(self):
        return self.state != WAIT

    def camera_wide(self):
        return {WAIT: 5.0, PARTS: 8.0, ASSEMBLE: 6.0 - 2.0 * e_io(self.t / (ASM_GAP * 6 + ASM_DUR)),
                BOOT: 4.0 * (1 - e_io(self.t / 1.8)), LIVE: 0.0}[self.state]

    def camera_ybias(self):
        return {WAIT: 0.4, PARTS: 1.15, ASSEMBLE: 0.7 * (1 - e_io(self.t / 3.0)), BOOT: 0.0, LIVE: 0.0}[self.state]

    def prompt(self):
        if self.state == WAIT:
            return "SHOW YOUR OPEN HAND", "hold an open palm in front of the camera   [ENTER]"
        if self.state == PARTS:
            if self.parts_ready() or self.pending_assemble:
                return "CLOSE YOUR HAND TO ASSEMBLE", "make a fist   [ENTER]"
            return "COMPONENTS ONLINE", "materialising..."
        if self.state == ASSEMBLE:
            return "ASSEMBLING", ""
        if self.state == BOOT:
            return "SYSTEM ONLINE", ""
        return None

    def update(self, dt, mecha):
        """Advance the sequence; produces self.skin / self.bone_fx while active."""
        self.clock += dt
        self.t += dt
        fx, aud = self.fx, self.audio
        if self.state == WAIT:
            self.skin, self.bone_fx = None, None
            self._ring(dt, 2.2)
            return
        if self.state == LIVE:
            self.skin, self.bone_fx = None, None
            return
        n = len(GROUPS)
        Z = np.zeros((4, 4), F32)
        Z[3, 3] = 1.0
        skin = np.tile(Z, (NB, 1, 1))
        bfx = np.zeros((NB, 2), F32)
        live = mecha.skin

        if self.state == PARTS:
            self._ring(dt, 1.6)
            if not self.hum_on and aud:
                aud.loop("float_hum", 0.28)
                self.hum_on = True
            for i, (name, leader, bl, rank) in enumerate(GROUPS):
                age = self.t - float(self.pop_time[i])
                if age < 0:
                    continue
                if not self.popped[i]:
                    self.popped[i] = True
                    self._pop_fx(i)
                e = e_back(age / POP_DUR)
                s = 0.15 + 0.85 * e
                bob = 0.09 * math.sin(1.3 * self.clock + self.phase[i])
                pos = self.P[i] + np.array([0, bob, 0], F32)
                spin = self.spin0[i] + self.spin_w[i] * self.clock
                R = _rot_y(spin) @ _rot_x(0.12 * math.sin(0.8 * self.clock + self.phase[i])) @ _rot_z(0.10 * math.sin(1.1 * self.clock + 2 * self.phase[i]))
                self.cur_p[i], self.cur_R[i], self.cur_s[i] = pos, R, s
                M = np.eye(4, dtype=F32)
                M[:3, :3] = R * s
                M[:3, 3] = pos - (R * s) @ self.C[i]
                diss = 1.0 - min(age / 0.5, 1.0)
                glow = max(0.10 + 0.05 * math.sin(2 * self.clock + self.phase[i]), 1.6 * (1 - min(age / 1.0, 1.0)))
                for b in bl:
                    skin[BI[b]] = M
                    bfx[BI[b]] = (glow, diss)
                if np.random.rand() < 0.04:
                    fx.glow(pos + np.random.randn(3).astype(F32) * 0.3, vel=(0, 0.6, 0), size=0.08, size_end=0.02, life=0.9,
                            col=(1.0, 0.25, 0.1), alpha=0.9)
            if self.pending_assemble and self.parts_ready():
                self.trigger_assemble()
        elif self.state == ASSEMBLE:
            all_arrived = True
            for i, (name, leader, bl, rank) in enumerate(GROUPS):
                ta = self.t - float(self.delay[i])
                u = min(max(ta / ASM_DUR, 0.0), 1.0)
                e = e_io(u)
                L = live[BI[leader]]
                c = self.C[i]
                p1 = (L @ np.append(c, 1.0))[:3]
                q1 = mat_to_quat(L[:3, :3])
                pos = self.p0[i] + (p1 - self.p0[i]) * e
                arc = math.sin(PI * e) * 1.3
                pos = pos + np.array([0, 0.5 * arc, 1.0 * arc], F32)
                R = quat_to_mat(slerp(self.q0[i], q1, e))
                self.cur_p[i], self.cur_R[i] = pos, R
                M = np.eye(4, dtype=F32)
                M[:3, :3] = R
                M[:3, 3] = pos - R @ c
                if u >= 1.0 and not self.arrived[i]:
                    self.arrived[i] = True
                    self.arrive_t[i] = self.t
                    self._clank_fx(i, p1)
                if not self.arrived[i]:
                    all_arrived = False
                since = self.t - float(self.arrive_t[i])
                glow = 0.10 * (1 - e) + (1.3 * math.exp(-since * 5.0) if self.arrived[i] else 0.35 * math.sin(PI * e))
                for b in bl:
                    skin[BI[b]] = M if u < 1.0 else live[BI[b]]
                    bfx[BI[b]] = (glow, 0.0)
                if 0.0 < u < 1.0 and np.random.rand() < 0.5:
                    fx.glow(pos, vel=(0, 0, 0), size=0.16, size_end=0.02, life=0.35, col=(1.0, 0.35, 0.12), alpha=0.8)
            if all_arrived and self.t > 0.5:
                self.state, self.t = BOOT, 0.0
                self._boot_fx(mecha)
        if self.state == BOOT:
            self.skin, self.bone_fx = None, None
            if self.t > 2.3:
                self.state = LIVE
            return
        self.skin, self.bone_fx = skin, bfx

    # ------------------------------------------------------------ effects
    def _ring(self, dt, period):
        self.ring_t -= dt
        if self.ring_t <= 0:
            self.ring_t = period
            self.fx.ring(np.array([0, 0, 0], F32), r0=0.8, speed=3.4, thick=0.22, col=(1.0, 0.18, 0.08), intensity=1.3, life=2.4)

    def _pop_fx(self, i):
        p = self.P[i]
        fx = self.fx
        fx.sparks(p, n=34, speed=5.0, col=(2.6, 0.8, 0.3), life=0.8, grav=2.0, up=0.2)
        fx.burst_ball(p, size=1.6, col=(2.4, 0.5, 0.2), life=0.3)
        fx.ring(np.array([p[0], 0.0, p[2]], F32), r0=0.2, speed=3.0, thick=0.12, col=(1.0, 0.25, 0.1), intensity=1.4, life=0.7)
        fx.shake(0.05)
        if self.audio:
            self.audio.play("pop", vol=0.8, pitch=0.85 + 0.35 * self.rng.random(), pan=float(np.clip(p[0] * 0.25, -0.8, 0.8)))

    def _clank_fx(self, i, p):
        fx = self.fx
        fx.sparks(p, n=40, speed=7.0, col=(2.8, 1.8, 0.7), life=0.6, grav=10.0)
        fx.burst_ball(p, size=1.1, col=(2.6, 1.4, 0.6), life=0.2)
        fx.shake(0.10)
        if self.audio:
            self.audio.play("clank", vol=0.85, pitch=0.8 + 0.5 * self.rng.random(), pan=float(np.clip(p[0] * 0.25, -0.8, 0.8)))

    def _boot_fx(self, mecha):
        fx = self.fx
        c = np.array([mecha.pos[0], 0.0, mecha.pos[2]], F32)
        fx.ring(c, r0=0.4, speed=12.0, thick=0.4, col=(1.0, 0.25, 0.1), intensity=3.2, life=1.2, fill=0.12)
        fx.ring(c, r0=0.3, speed=7.0, thick=0.3, col=(1.0, 0.7, 0.3), intensity=3.0, life=1.4)
        fx.dust(c, n=90, speed=7.0, radius=0.5, size=0.9, life=1.8)
        fx.impact_feel(shake=0.8, flash=0.55, aberr=0.9, fov=1.0, slow=0.35, slow_scale=0.5, col=(1.0, 0.55, 0.3))
        if self.audio:
            self.audio.play("boot", vol=1.0)
        mecha.begin_boot()
