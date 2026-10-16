"""Orbit camera with smoothing, view presets (front/back/sides/top), follow target, auto turntable and shake."""

import math
import numpy as np

from .xform import look_at, perspective

F32 = np.float32

PRESETS = {
    "FRONT": (0.0, 10.0, 11.5),
    "BACK": (180.0, 10.0, 11.5),
    "LEFT": (90.0, 10.0, 11.5),
    "RIGHT": (-90.0, 10.0, 11.5),
    "TOP": (0.0, 78.0, 15.0),
    "3/4 FRONT": (38.0, 14.0, 10.8),
    "3/4 BACK": (-142.0, 14.0, 10.8),
}
PRESET_ORDER = ["FRONT", "3/4 FRONT", "LEFT", "3/4 BACK", "BACK", "RIGHT", "TOP"]


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class OrbitCamera:
    def __init__(self):
        self.az, self.el, self.dist = 28.0, 13.0, 11.5          # smoothed (deg, deg, units)
        self.t_az, self.t_el, self.t_dist = 28.0, 13.0, 11.5     # targets
        self.target = np.array([0, 2.05, 0], F32)
        self.fov = 36.0
        self.auto_orbit = False
        self.auto_speed = 14.0
        self.label = "3/4 FRONT"
        self.az_vel = 0.0
        self.follow_dead = 2.2

    # ---- control
    def orbit(self, d_az, d_el):
        self.t_az += d_az
        self.t_el = float(np.clip(self.t_el + d_el, -4.0, 85.0))
        self.label = "ORBIT"
        self.auto_orbit = False

    def zoom(self, factor):
        self.t_dist = float(np.clip(self.t_dist * factor, 3.2, 28.0))

    def preset(self, name, heading_deg=0.0):
        az, el, dist = PRESETS[name]
        # nearest equivalent azimuth so the camera takes the short way round
        goal = az + heading_deg
        self.t_az = self.t_az + _wrap(goal - self.t_az)
        self.t_el, self.t_dist = el, dist
        self.label = name
        self.auto_orbit = False

    def cycle_preset(self, step, heading_deg=0.0):
        try:
            i = PRESET_ORDER.index(self.label)
        except ValueError:
            i = 0
        self.preset(PRESET_ORDER[(i + step) % len(PRESET_ORDER)], heading_deg)

    def toggle_auto(self):
        self.auto_orbit = not self.auto_orbit
        self.label = "TURNTABLE" if self.auto_orbit else self.label

    # ---- per-frame
    def update(self, dt, focus, fx=None, now=0.0, wide=0.0, ybias=0.0):
        if self.auto_orbit:
            self.t_az += self.auto_speed * dt
        k = 1.0 - math.exp(-dt * 7.0)
        self.az += (self.t_az - self.az) * k
        self.el += (self.t_el - self.el) * k
        self.dist += (self.t_dist - self.dist) * (1.0 - math.exp(-dt * 5.0))
        h = max(float(focus[1]), 0.0)
        # follow the robot only once it leaves a dead zone, so absolute hand-targeting stays stable near the centre
        dx, dz = focus[0] - self.target[0], focus[2] - self.target[2]
        dist = math.hypot(dx, dz)
        self.robot_c = np.array([focus[0], 2.05 + max(float(focus[1]), 0.0), focus[2]], F32)
        if dist > self.follow_dead:
            k_xz = 1.0 - math.exp(-dt * (3.0 + min(dist - self.follow_dead, 6.0)))
            ex = (dist - self.follow_dead) / dist
            self.target[0] += dx * ex * k_xz
            self.target[2] += dz * ex * k_xz
        self.yb = getattr(self, "yb", 0.0)
        self.yb += (ybias - self.yb) * (1.0 - math.exp(-dt * 2.0))
        gy = 2.05 + 0.85 * h + self.yb
        self.target[1] += (gy - self.target[1]) * (1.0 - math.exp(-dt * (4.0 + 5.0 * min(h / 2.0, 1.0))))
        self.h_s = getattr(self, "h_s", 0.0)
        self.h_s += (h - self.h_s) * (1.0 - math.exp(-dt * 3.0))
        self.wide = getattr(self, "wide", 0.0)
        self.wide += (wide - self.wide) * (1.0 - math.exp(-dt * 2.2))
        self.shake_off = fx.camera_shake(now) if fx is not None else (0, 0, 0, 0)
        self.kick = fx.fov_kick if fx is not None else 0.0

    def matrices(self, aspect):
        az, el = math.radians(self.az), math.radians(self.el)
        d = self.dist + 0.9 * getattr(self, "h_s", 0.0) + getattr(self, "wide", 0.0)
        off = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)], F32)
        sx, sy, sz, roll = getattr(self, "shake_off", (0, 0, 0, 0))
        tgt = self.target + np.array([sx, sy, sz], F32)
        eye = tgt + off * d
        # distance guard: never let the robot get closer than ~9 units to the lens
        rc = getattr(self, "robot_c", None)
        if rc is not None:
            need = max(0.0, 9.2 - float(np.linalg.norm(rc - eye)))
            self.guard = getattr(self, "guard", 0.0)
            self.guard += (need - self.guard) * 0.12
            eye = tgt + off * (d + self.guard)
        up = np.array([math.sin(roll), math.cos(roll), 0.0], F32)
        self._eye = eye
        V = look_at(eye, tgt, up)
        fov = self.fov - getattr(self, "kick", 0.0) * 6.0
        P = perspective(math.radians(fov), aspect, 0.1, 400.0)
        return eye, V, P

    def eye_pos(self):
        """World position of the lens (what the robot's head tracks)."""
        return getattr(self, "_eye", np.array([0, 3, 10], F32))

    def right_vector(self):
        az = math.radians(self.az)
        return np.array([math.cos(az), 0.0, -math.sin(az)], F32)

    def forward_flat(self):
        """Ground-plane unit vector pointing away from the camera."""
        az = math.radians(self.az)
        return np.array([-math.sin(az), 0.0, -math.cos(az)], F32)
