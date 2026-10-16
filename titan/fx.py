"""Visual effects: vectorised particle system, shockwave rings, beams, screen shake / flash / hit-stop."""

import math
import numpy as np

F32 = np.float32
rng = np.random.default_rng(7)


class Particles:
    """Fixed-capacity SoA particle pool. kind: 0 glow (additive), 1 smoke (alpha), 2 spark (hot core)."""

    def __init__(self, cap=9000):
        self.cap = cap
        self.n = 0
        self.pos = np.zeros((cap, 3), F32)
        self.vel = np.zeros((cap, 3), F32)
        self.age = np.zeros(cap, F32)
        self.life = np.ones(cap, F32)
        self.s0 = np.zeros(cap, F32)
        self.s1 = np.zeros(cap, F32)
        self.c0 = np.zeros((cap, 4), F32)
        self.c1 = np.zeros((cap, 4), F32)
        self.kind = np.zeros(cap, F32)
        self.grav = np.zeros(cap, F32)
        self.drag = np.zeros(cap, F32)
        self.bounce = np.zeros(cap, F32)

    def emit(self, n, pos, vel, life, size, size_end, c0, c1=None, kind=0, grav=0.0, drag=0.0, bounce=0.0):
        n = int(n)
        if n <= 0:
            return
        n = min(n, self.cap - self.n)
        if n <= 0:
            return
        i0, i1 = self.n, self.n + n

        def bc(v, shape):
            a = np.asarray(v, F32)
            return np.broadcast_to(a, (n,) + shape).copy() if a.shape == shape else a[:n]

        self.pos[i0:i1] = bc(pos, (3,))
        self.vel[i0:i1] = bc(vel, (3,))
        self.life[i0:i1] = np.asarray(life, F32) if np.ndim(life) == 0 else np.asarray(life, F32)[:n]
        self.age[i0:i1] = 0.0
        self.s0[i0:i1] = size if np.ndim(size) == 0 else np.asarray(size, F32)[:n]
        self.s1[i0:i1] = size_end if np.ndim(size_end) == 0 else np.asarray(size_end, F32)[:n]
        c0 = np.asarray(c0, F32)
        self.c0[i0:i1] = np.broadcast_to(c0, (n, 4)) if c0.ndim == 1 else c0[:n]
        c1 = c0 * np.array([1, 1, 1, 0], F32) if c1 is None else np.asarray(c1, F32)
        self.c1[i0:i1] = np.broadcast_to(c1, (n, 4)) if c1.ndim == 1 else c1[:n]
        self.kind[i0:i1] = kind
        self.grav[i0:i1] = grav
        self.drag[i0:i1] = drag
        self.bounce[i0:i1] = bounce
        self.n = i1

    def update(self, dt):
        n = self.n
        if n == 0:
            return
        s = slice(0, n)
        self.age[s] += dt
        d = np.exp(-self.drag[s] * dt)[:, None]
        self.vel[s] *= d
        self.vel[s, 1] -= self.grav[s] * dt
        self.pos[s] += self.vel[s] * dt
        below = (self.pos[s, 1] < 0.01) & (self.bounce[s] > 0)
        if below.any():
            idx = np.nonzero(below)[0]
            self.pos[idx, 1] = 0.01
            self.vel[idx, 1] *= -self.bounce[idx]
            self.vel[idx, 0] *= 0.7
            self.vel[idx, 2] *= 0.7
        alive = self.age[s] < self.life[s]
        if not alive.all():
            keep = np.nonzero(alive)[0]
            m = len(keep)
            for a in (self.pos, self.vel, self.age, self.life, self.s0, self.s1, self.c0, self.c1, self.kind,
                      self.grav, self.drag, self.bounce):
                a[:m] = a[keep]
            self.n = m

    def sprites(self):
        n = self.n
        if n == 0:
            return None
        t = (self.age[:n] / self.life[:n])[:, None]
        col = self.c0[:n] + (self.c1[:n] - self.c0[:n]) * t
        size = self.s0[:n] + (self.s1[:n] - self.s0[:n]) * t[:, 0]
        out = np.empty((n, 9), F32)
        out[:, 0:3] = self.pos[:n]
        out[:, 3] = size
        out[:, 4:8] = col
        out[:, 8] = self.kind[:n]
        return out


def _unit(n, up_bias=0.0):
    v = rng.normal(size=(n, 3)).astype(F32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    v[:, 1] = np.abs(v[:, 1]) * (1 - up_bias) + up_bias * np.abs(v[:, 1]) + 0.0
    return v


class FX:
    def __init__(self):
        self.p = Particles()
        self.rings = []        # persistent floor rings
        self.beams = []        # per-frame beams (rebuilt each frame by the owner)
        self.trauma = 0.0
        self.flash = 0.0
        self.flash_col = (1.0, 1.0, 1.0)
        self.hitstop = 0.0
        self.slowmo = 0.0
        self.slow_scale = 1.0
        self.aberr = 0.0
        self.fov_kick = 0.0
        self.t = 0.0
        self.shield_hits = []  # [dir(3), age]

    # --------------------------------------------------------- frame update
    def update(self, dt, sim_dt=None):
        """dt: real frame time (UI-feel timers); sim_dt: simulation time (particles), slowed by hit-stop / slow-mo."""
        sim_dt = dt if sim_dt is None else sim_dt
        self.t += dt
        self.p.update(sim_dt)
        for r in self.rings:
            r["age"] += sim_dt
        self.rings = [r for r in self.rings if r["age"] < r["life"]]
        self.trauma = max(0.0, self.trauma - dt * 1.6)
        self.flash = max(0.0, self.flash - dt * 4.5)
        self.aberr = max(0.0, self.aberr - dt * 1.5)
        self.fov_kick *= math.exp(-dt * 5.0)
        self.hitstop = max(0.0, self.hitstop - dt)
        self.slowmo = max(0.0, self.slowmo - dt)
        self.shield_hits = [[d, a + dt] for d, a in self.shield_hits if a + dt < 1.3]

    def time_scale(self):
        if self.hitstop > 0:
            return 0.06
        if self.slowmo > 0:
            return self.slow_scale
        return 1.0

    def shake(self, amount):
        self.trauma = min(1.0, self.trauma + amount)

    def do_flash(self, amount, col=(1, 1, 1)):
        self.flash = max(self.flash, amount)
        self.flash_col = col

    def impact_feel(self, shake=0.3, stop=0.0, slow=0.0, slow_scale=0.35, flash=0.0, aberr=0.0, fov=0.0, col=(1, 1, 1)):
        self.shake(shake)
        if stop:
            self.hitstop = max(self.hitstop, stop)
        if slow:
            self.slowmo = max(self.slowmo, slow)
            self.slow_scale = slow_scale
        if flash:
            self.do_flash(flash, col)
        self.aberr = max(self.aberr, aberr)
        self.fov_kick = max(self.fov_kick, fov)

    # --------------------------------------------------------- emitters
    def ring(self, center, r0=0.3, speed=9.0, thick=0.25, col=(0.3, 0.9, 1.0), intensity=3.0, life=0.8, fill=0.0):
        self.rings.append(dict(center=np.array(center, F32), r0=r0, speed=speed, thick=thick, col=col,
                               int=intensity, life=life, age=0.0, fill=fill))

    def ring_draw_list(self):
        out = []
        for r in self.rings:
            t = r["age"] / r["life"]
            rad = r["r0"] + r["speed"] * r["age"] * (1.0 - 0.45 * t)
            out.append(dict(center=r["center"], radius=rad, thick=r["thick"] * (1 + 1.2 * t), col=r["col"],
                            int=r["int"] * (1 - t) ** 1.5, fill=r["fill"] * (1 - t)))
        return out

    def sparks(self, pos, n=40, speed=8.0, col=(2.5, 1.6, 0.6), life=0.7, size=0.05, up=0.5, grav=14.0):
        v = _unit(n)
        v[:, 1] = np.abs(v[:, 1]) * (0.4 + up)
        sp = (speed * (0.35 + 0.65 * rng.random(n))).astype(F32)[:, None]
        c0 = np.array([*col, 1.0], F32)
        self.p.emit(n, np.asarray(pos, F32), v * sp, life * (0.5 + 0.7 * rng.random(n)).astype(F32), size, size * 0.2,
                    c0, c0 * np.array([1, 0.5, 0.3, 0], F32), kind=2, grav=grav, drag=0.6, bounce=0.35)

    def dust(self, center, n=40, speed=5.0, radius=0.3, col=(0.35, 0.37, 0.42), size=0.5, life=1.4, up=0.5):
        ang = rng.random(n) * 2 * math.pi
        d = np.stack([np.cos(ang), np.zeros(n), np.sin(ang)], 1).astype(F32)
        pos = np.asarray(center, F32) + d * radius + np.array([0, 0.18, 0], F32)
        vel = d * speed * (0.5 + rng.random(n))[:, None] + np.array([0, up, 0], F32) * rng.random(n)[:, None] * 2
        self.p.emit(n, pos, vel.astype(F32), life * (0.6 + 0.8 * rng.random(n)).astype(F32), size * 0.5,
                    size * (1.6 + rng.random(n)).astype(F32), np.array([*col, 0.34], F32),
                    np.array([*col, 0.0], F32), kind=1, drag=2.2, grav=-0.3)

    def smoke(self, pos, n=6, vel=(0, 1, 0), size=0.3, life=1.2, col=(0.25, 0.27, 0.3), alpha=0.4, spread=0.6):
        v = (np.asarray(vel, F32) + rng.normal(size=(n, 3)).astype(F32) * spread)
        self.p.emit(n, np.asarray(pos, F32) + rng.normal(size=(n, 3)).astype(F32) * 0.05, v, life, size * 0.5, size * 2.5,
                    np.array([*col, alpha], F32), np.array([*col, 0.0], F32), kind=1, drag=1.2)

    def glow(self, pos, vel=(0, 0, 0), n=1, size=0.5, life=0.3, col=(1.0, 0.7, 0.3), alpha=1.0, size_end=0.0, drag=0.0,
             grav=0.0):
        self.p.emit(n, np.asarray(pos, F32), np.asarray(vel, F32), life, size, size_end,
                    np.array([*col, alpha], F32), np.array([*col, 0.0], F32), kind=0, drag=drag, grav=grav)

    def embers(self, center, n=8, radius=1.2, col=(2.4, 1.2, 0.35), up=3.0):
        ang = rng.random(n) * 2 * math.pi
        r = radius * np.sqrt(rng.random(n))
        pos = np.stack([center[0] + np.cos(ang) * r, np.full(n, 0.05), center[2] + np.sin(ang) * r], 1).astype(F32)
        vel = np.stack([rng.normal(size=n) * 0.3, up * (0.6 + rng.random(n)), rng.normal(size=n) * 0.3], 1).astype(F32)
        c0 = np.array([*col, 1.0], F32)
        self.p.emit(n, pos, vel, (1.0 + rng.random(n) * 1.2).astype(F32), 0.05, 0.01, c0, c0 * np.array([1, 0.4, 0.2, 0], F32),
                    kind=2, drag=0.4)

    def burst_ball(self, pos, size=1.5, col=(3.0, 2.2, 1.2), life=0.25):
        self.glow(pos, size=size * 0.4, size_end=size, life=life, col=col, alpha=1.0)

    def shield_hit(self, direction):
        d = np.asarray(direction, F32)
        d /= np.linalg.norm(d) + 1e-6
        self.shield_hits.append([d, 0.0])

    # --------------------------------------------------------- outputs
    def sprites(self):
        return self.p.sprites()

    def camera_shake(self, t):
        """(dx, dy, dz, roll) offsets from current trauma."""
        a = self.trauma ** 2
        if a < 1e-4:
            return 0.0, 0.0, 0.0, 0.0
        f = t * 28.0
        return (a * 0.18 * math.sin(f * 1.3 + 1.0), a * 0.15 * math.sin(f * 1.7 + 2.0),
                a * 0.10 * math.sin(f * 1.1 + 4.0), a * 0.03 * math.sin(f * 2.1))
