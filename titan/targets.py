"""Floating training drones that react to punches, slams and the laser beam."""

import math

import numpy as np

from . import geometry as G
from .materials import ARMOR_A, ARMOR_B, ARMOR_C, GUNMETAL, CHROME, ACCENT, GLOW_B, HOT, RUBBER
from .mecha_model import Builder
from .skeleton import BI
from .xform import translate as T, rot_x as RX, rot_y as RY, rot_z as RZ

F32 = np.float32
rng = np.random.default_rng(5)


def build_drone():
    """Compact hover-drone mesh authored around the origin (all vertices on bone 0)."""
    B = Builder()
    R = "root"
    B.add(G.sphere(0.42, 28, 16), R, ARMOR_C)
    B.add(G.torus(0.62, 0.05, 40, 10), R, ACCENT)
    B.add(G.torus(0.62, 0.02, 40, 6), R, GLOW_B, T(0, 0.075, 0))
    B.add(G.cylinder(0.20, 0.10, 24, 0.02).transformed(RX(np.pi / 2)), R, GUNMETAL, T(0, 0, 0.36))
    B.add(G.cylinder(0.14, 0.04, 24, 0.0).transformed(RX(np.pi / 2)), R, GLOW_B, T(0, 0, 0.44))
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 4
        B.add(G.extrude([(-0.07, 0.0), (0.07, 0.0), (0.04, 0.34), (-0.04, 0.34)], 0.03, 0.006), R, ARMOR_A,
              T(math.cos(a) * 0.30, 0.18, math.sin(a) * 0.30) @ RY(-a + np.pi / 2) @ RX(-0.6))
    B.add(G.cylinder(0.12, 0.08, 20, 0.01), R, HOT, T(0, -0.50, 0))
    B.add(G.cylinder(0.05, 0.10, 12, 0.0), R, CHROME, T(0, 0.40, 0))
    B.add(G.rbox(0.3, 0.05, 0.12, 0.012, 1), R, ARMOR_B, T(0, 0.16, -0.30))
    md = B.finish()
    md.bone[:] = 0.0
    return md


class Target:
    def __init__(self, pos):
        self.pos = np.array(pos, F32)
        self.vel = np.zeros(3, F32)
        self.home = self.pos.copy()
        self.hp = 3.0
        self.flash = 0.0
        self.alive = True
        self.respawn = 0.0
        self.phase = rng.random() * 6.28
        self.yaw = rng.random() * 6.28
        self.spawn_t = 0.0


class Targets:
    RADIUS = 0.85

    def __init__(self, fx, audio=None, count=5):
        self.fx = fx
        self.audio = audio
        self.items = []
        self.score = 0
        self.time = 0.0
        for _ in range(count):
            self.items.append(self._spawn(None))

    def _spawn(self, around):
        ang = rng.random() * 2 * math.pi
        r = 4.5 + rng.random() * 6.0
        cx, cz = (0.0, 0.0) if around is None else (around[0], around[2])
        pos = (cx + math.cos(ang) * r, 1.3 + rng.random() * 2.4, cz + math.sin(ang) * r)
        t = Target(pos)
        t.spawn_t = 0.0
        return t

    def update(self, dt, player_pos):
        self.time += dt
        for t in self.items:
            if not t.alive:
                t.respawn -= dt
                if t.respawn <= 0:
                    n = self._spawn(player_pos)
                    t.__dict__.update(n.__dict__)
                    self.fx.ring(np.array([t.pos[0], 0.0, t.pos[2]], F32), r0=0.3, speed=4.0, thick=0.15, col=(1.0, 0.3, 0.2), intensity=1.5, life=0.6)
                continue
            t.flash = max(0.0, t.flash - dt * 4.0)
            t.spawn_t += dt
            t.phase += dt
            t.yaw += dt * 1.2
            # gentle patrol around home + recover from knockback
            hover = t.home + np.array([math.sin(t.phase * 0.5) * 0.9, math.sin(t.phase * 1.3) * 0.25, math.cos(t.phase * 0.4) * 0.9], F32)
            t.vel += (hover - t.pos) * 3.0 * dt
            t.vel *= math.exp(-2.2 * dt)
            t.pos += t.vel * dt

    # ---- queries / damage
    def _damage(self, t, amount, knock_dir=None):
        t.hp -= amount
        t.flash = 1.0
        if knock_dir is not None:
            t.vel += knock_dir * 6.0
        if t.hp <= 0 and t.alive:
            self._explode(t)

    def _explode(self, t):
        t.alive = False
        t.respawn = 2.5
        self.score += 1
        fx = self.fx
        p = t.pos.copy()
        fx.burst_ball(p, size=3.4, col=(3.0, 1.5, 0.6), life=0.35)
        fx.sparks(p, n=90, speed=11.0, life=1.0, col=(2.8, 1.5, 0.5), up=0.6)
        fx.smoke(p, n=14, vel=(0, 0.6, 0), size=0.8, life=2.0, col=(0.25, 0.25, 0.28), alpha=0.5, spread=1.4)
        fx.ring(np.array([p[0], 0.02, p[2]], F32), r0=0.3, speed=9.0, thick=0.25, col=(1.0, 0.55, 0.2), intensity=2.2, life=0.7)
        fx.impact_feel(shake=0.35, flash=0.12, aberr=0.3)
        if self.audio:
            self.audio.play("explode", vol=0.9)

    def hit_sphere(self, point, radius, power):
        for t in self.items:
            if not t.alive:
                continue
            d = t.pos - point
            dist = float(np.linalg.norm(d))
            if dist < radius + self.RADIUS:
                self._damage(t, 1.5 * power, d / (dist + 1e-6))
                self.fx.sparks(t.pos, n=18, speed=7.0, life=0.5)
                if self.audio and t.alive:
                    self.audio.play("hit_small", vol=0.8)

    def ray(self, origin, direction, length):
        best = None
        for t in self.items:
            if not t.alive:
                continue
            oc = t.pos - origin
            tp = float(np.dot(oc, direction))
            if tp < 0 or tp > length:
                continue
            d2 = float(np.dot(oc, oc) - tp * tp)
            if d2 < (self.RADIUS * 0.9) ** 2:
                tt = tp - math.sqrt(max((self.RADIUS * 0.9) ** 2 - d2, 0.0))
                if best is None or tt < best[0]:
                    best = (tt, origin + direction * tt, t)
        return best

    def beam_damage(self, origin, direction, length, dt):
        b = self.ray(origin, direction, length)
        if b is not None:
            self._damage(b[2], 2.6 * dt)

    # ---- render data
    def props(self):
        out = []
        for t in self.items:
            if not t.alive:
                continue
            s = min(1.0, t.spawn_t * 3.0)
            M = T(*t.pos) @ RY(t.yaw) @ np.diag([s, s, s, 1.0]).astype(F32)
            out.append(dict(model=M, flash=t.flash))
        return out
