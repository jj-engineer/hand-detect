"""Mecha skeleton: bone table, rest pose, forward kinematics."""

import numpy as np

from .xform import translate, euler

F32 = np.float32

BONES = []        # (name, parent_index, rest_world_position)
BI = {}           # name -> index

# proportions: legs are 12% longer; everything above the legs rides up so the feet still touch the floor
LIFT = 0.18
LEG_STRETCH = 1.12
HIP_Y = 1.72


def leg_stretch_y(y):
    """Rest-pose y of a point on the (stretched) leg, authored for the original unstretched skeleton."""
    return LIFT + HIP_Y + (y - HIP_Y) * LEG_STRETCH


def _adjust(name, pos):
    p = np.array(pos, F32)
    if name == "root":
        return p
    if name.startswith("shin") or name.startswith("foot"):
        p[1] = leg_stretch_y(p[1])
    else:
        p[1] += LIFT
    return p


def _b(name, parent, pos):
    BI[name] = len(BONES)
    BONES.append((name, BI[parent] if parent is not None else -1, _adjust(name, pos)))


_b("root", None, (0, 0, 0))
_b("pelvis", "root", (0, 1.75, 0))
_b("spine", "pelvis", (0, 1.97, 0))
_b("chest", "spine", (0, 2.22, 0))
_b("neck", "chest", (0, 3.00, 0.02))
_b("head", "neck", (0, 3.12, 0.02))
for _s, _side in ((1, "L"), (-1, "R")):
    _b("pad" + _side, "chest", (_s * 0.88, 2.84, 0))
    _b("uarm" + _side, "chest", (_s * 0.88, 2.76, 0))
    _b("farm" + _side, "uarm" + _side, (_s * 0.92, 1.98, -0.02))
    _b("hand" + _side, "farm" + _side, (_s * 0.94, 1.18, 0))
    for _f, _z in (("i", 0.095), ("m", 0.032), ("r", -0.032), ("p", -0.095)):
        _b("f%sa%s" % (_f, _side), "hand" + _side, (_s * 0.94, 0.93, _z))
        _b("f%sb%s" % (_f, _side), "f%sa%s" % (_f, _side), (_s * 0.94, 0.77, _z))
    _b("tha" + _side, "hand" + _side, (_s * 0.90, 1.08, 0.13))
    _b("thb" + _side, "tha" + _side, (_s * 0.90, 0.98, 0.19))
    _b("thigh" + _side, "pelvis", (_s * 0.40, 1.72, 0))
    _b("shin" + _side, "thigh" + _side, (_s * 0.40, 0.95, 0))
    _b("foot" + _side, "shin" + _side, (_s * 0.40, 0.22, 0.02))
    _b("wing" + _side, "chest", (_s * 0.40, 2.86, -0.70))
    _b("noz" + _side, "chest", (_s * 0.30, 2.28, -0.64))
_b("kat", "handR", (-0.90, 0.84, 0.0))       # katana held in the right hand (grip centre, blade along +Z)
SHEATH_GRIP = (0.34, 2.04, 0.36)
_b("katS", "pelvis", SHEATH_GRIP)      # sheathed katana hilt on the left hip

NB = len(BONES)
PARENT = np.array([b[1] for b in BONES])
REST = np.array([b[2] for b in BONES], F32)
LOCAL = REST.copy()
for _i in range(NB):
    if PARENT[_i] >= 0:
        LOCAL[_i] = REST[_i] - REST[PARENT[_i]]
REST_INV = np.array([translate(-REST[i]) for i in range(NB)], F32)
_LOCAL_T = np.array([translate(LOCAL[i]) for i in range(NB)], F32)


class Pose:
    """Per-bone euler rotations (rx, ry, rz) plus root translation/yaw."""

    def __init__(self):
        self.rot = np.zeros((NB, 3), F32)
        self.root_pos = np.zeros(3, F32)
        self.root_yaw = 0.0
        self.root_pitch = 0.0
        self.root_roll = 0.0

    def copy(self):
        p = Pose()
        p.rot = self.rot.copy()
        p.root_pos = self.root_pos.copy()
        p.root_yaw, p.root_pitch, p.root_roll = self.root_yaw, self.root_pitch, self.root_roll
        return p


def forward_kinematics(pose, extra=None):
    """Returns (world, skin) arrays of shape (NB,4,4)."""
    world = np.zeros((NB, 4, 4), F32)
    for i in range(NB):
        r = pose.rot[i]
        if i == 0:
            m = (translate(pose.root_pos) @ euler(pose.root_pitch, pose.root_yaw, pose.root_roll))
        else:
            m = world[PARENT[i]] @ _LOCAL_T[i] @ euler(r[0], r[1], r[2])
        world[i] = m
    skin = world @ REST_INV
    return world, skin


def world_point(world, bone, local_offset=(0, 0, 0)):
    """World position of a point given in rest-world coordinates relative to the bone origin."""
    i = BI[bone] if isinstance(bone, str) else bone
    p = np.append(np.asarray(local_offset, F32), 1.0)
    return (world[i] @ p)[:3]


def world_dir(world, bone, local_dir):
    i = BI[bone] if isinstance(bone, str) else bone
    return world[i][:3, :3] @ np.asarray(local_dir, F32)
