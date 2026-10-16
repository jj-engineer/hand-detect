"""Small 4x4 matrix toolkit (column-vector convention: p' = M @ p)."""

import numpy as np

F32 = np.float32


def ident():
    return np.eye(4, dtype=F32)


def translate(x, y=None, z=None):
    if y is None:
        x, y, z = x
    m = np.eye(4, dtype=F32)
    m[:3, 3] = (x, y, z)
    return m


def scale(sx, sy=None, sz=None):
    if sy is None:
        sy = sz = sx
    return np.diag([sx, sy, sz, 1.0]).astype(F32)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=F32)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=F32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=F32)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def euler(rx, ry, rz):
    """Rotation applied in Y (twist) -> X (pitch) -> Z (roll) order: R = Ry @ Rx @ Rz."""
    return rot_y(ry) @ rot_x(rx) @ rot_z(rz)


def rot_axis(axis, a):
    axis = np.asarray(axis, F32)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    m = np.eye(4, dtype=F32)
    m[:3, :3] = [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]
    return m


def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=F32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4, dtype=F32)
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = -2.0 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m


def normalize(v):
    v = np.asarray(v, F32)
    return v / (np.linalg.norm(v) + 1e-9)


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, F32)
    f = normalize(np.asarray(target, F32) - eye)
    s = normalize(np.cross(f, np.asarray(up, F32)))
    u = np.cross(s, f)
    m = np.eye(4, dtype=F32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def to_gl(m):
    """Column-major bytes for a GLSL mat4 uniform."""
    return np.ascontiguousarray(np.asarray(m, F32).T).tobytes()


def mats_to_gl(ms):
    """Column-major bytes for a GLSL mat4[] uniform from an (N,4,4) array."""
    return np.ascontiguousarray(np.asarray(ms, F32).transpose(0, 2, 1)).tobytes()


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def reflect_y():
    m = np.eye(4, dtype=F32)
    m[1, 1] = -1.0
    return m
