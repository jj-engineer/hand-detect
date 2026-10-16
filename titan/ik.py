"""
Analytic two-bone IK for the mecha's legs and arms (float64 numpy).

Bone convention (see skeleton.py): world(bone) = world(parent) @ T(local) @ E(rx, ry, rz),  E = Ry @ Rx @ Rz.
A limb is   bone1 (hip / shoulder)  --o1-->  bone2 (knee / elbow)  --o2-->  end (ankle / wrist)
with offsets o1, o2 expressed in the bone-1 / bone-2 frames at rest.  The joint-2 hinge is a pure rx rotation.
"""

import math

import numpy as np


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def euler_R(rx, ry, rz):
    return Ry(ry) @ Rx(rx) @ Rz(rz)


def R_to_euler(R):
    """Inverse of euler_R: returns (rx, ry, rz)."""
    sb = -R[1, 2]
    b = math.asin(max(-1.0, min(1.0, sb)))
    if abs(R[1, 2]) < 0.999999:
        a = math.atan2(R[0, 2], R[2, 2])
        c = math.atan2(R[1, 0], R[1, 1])
    else:
        a = math.atan2(-R[2, 0], R[0, 0])
        c = 0.0
    return b, a, c


def axis_angle_R(axis, ang):
    axis = np.asarray(axis, np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def align_R(a, b):
    """Minimal rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return axis_angle_R(np.cross(a, ref), math.pi)
    return axis_angle_R(v, math.atan2(s, c))


def solve_limb(P, S, o1, o2, T, pole, kappa_sign, kappa_range):
    """
    P  : 3x3 world rotation of the parent of bone 1.
    S  : world position of bone-1's origin (hip / shoulder).
    o1 : offset of joint 2 in bone-1's frame (rest),  o2 : offset of the end point in bone-2's frame (rest).
    T  : world target for the end point.
    pole : world-space direction the middle joint (knee / elbow) should point toward.
    kappa_sign : +1 (knee: flexion rx2 >= 0) or -1 (elbow: rx2 <= 0).
    Returns (euler1, kappa, reached) where euler1 = (rx, ry, rz) of bone 1 and the hinge rotation of bone 2 is (kappa, 0, 0).
    """
    o1 = np.asarray(o1, np.float64)
    o2 = np.asarray(o2, np.float64)
    L1, L2 = np.linalg.norm(o1), np.linalg.norm(o2)
    d = P.T @ (np.asarray(T, np.float64) - np.asarray(S, np.float64))       # target in the parent frame
    dist = float(np.linalg.norm(d))
    lo, hi = abs(L1 - L2) + 1e-3, (L1 + L2) * 0.9995
    reached = lo <= dist <= hi
    dist_c = min(max(dist, lo), hi)
    if dist < 1e-6:
        d = np.array([0.0, -1.0, 0.0])
        dist = 1.0
    d_hat = d / dist
    d_vec = d_hat * dist_c
    # hinge angle from |o1 + Rx(k) o2| = dist
    A = o1[1] * o2[1] + o1[2] * o2[2]
    B = -o1[1] * o2[2] + o1[2] * o2[1]
    c = (dist_c ** 2 - L1 ** 2 - L2 ** 2) * 0.5 - o1[0] * o2[0]
    Rm = math.hypot(A, B)
    phi = math.atan2(B, A)
    q = max(-1.0, min(1.0, c / max(Rm, 1e-9)))
    off = math.acos(q)
    sols = [phi + off, phi - off]
    sols = [((k + math.pi) % (2 * math.pi)) - math.pi for k in sols]
    good = [k for k in sols if kappa_sign * k >= -1e-6]
    kappa = min(good, key=abs) if good else min(sols, key=lambda k: abs(k))
    kappa = max(kappa_range[0], min(kappa_range[1], kappa))
    w = o1 + Rx(kappa) @ o2                                                    # end point in bone-1's frame
    Ra = align_R(w / np.linalg.norm(w), d_hat)
    e0 = Ra @ o1                                                               # joint 2 direction after the minimal rotation
    pole_p = P.T @ np.asarray(pole, np.float64)                                 # world -> parent frame
    # twist about the reach axis so the middle joint points toward the pole
    pp = pole_p - d_hat * float(np.dot(pole_p, d_hat))
    ee = e0 - d_hat * float(np.dot(e0, d_hat))
    if np.linalg.norm(pp) > 1e-6 and np.linalg.norm(ee) > 1e-6:
        pp /= np.linalg.norm(pp)
        ee /= np.linalg.norm(ee)
        ang = math.atan2(float(np.dot(d_hat, np.cross(ee, pp))), float(np.dot(ee, pp)))
        R1 = axis_angle_R(d_hat, ang) @ Ra
    else:
        R1 = Ra
    return R_to_euler(R1), kappa, reached


def limb_fk(P, S, o1, o2, e1, kappa):
    """Forward kinematics for testing: returns (joint2 world pos, end world pos, bone-2 world rotation)."""
    R1 = P @ euler_R(*e1)
    j2 = np.asarray(S, np.float64) + R1 @ np.asarray(o1, np.float64)
    R2 = R1 @ Rx(kappa)
    end = j2 + R2 @ np.asarray(o2, np.float64)
    return j2, end, R2
