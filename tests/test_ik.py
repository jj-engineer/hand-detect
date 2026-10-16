"""Round-trip test of the two-bone IK against forward kinematics, using the mecha's real limb offsets."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from titan.ik import *
from titan.skeleton import LOCAL, BI

rng = np.random.default_rng(1)
worst = {}
for name, (b1, b2, b3, sign, rk) in {
        "arm L": ("uarmL", "farmL", "handL", -1, (-2.6, 0.0)),
        "arm R": ("uarmR", "farmR", "handR", -1, (-2.6, 0.0)),
        "leg L": ("thighL", "shinL", "footL", 1, (0.0, 2.4)),
        "leg R": ("thighR", "shinR", "footR", 1, (0.0, 2.4))}.items():
    o1, o2 = LOCAL[BI[b2]].astype(float), LOCAL[BI[b3]].astype(float)
    errs, bad = [], 0
    for _ in range(400):
        P = euler_R(rng.uniform(-0.3, 0.3), rng.uniform(-1, 1), rng.uniform(-0.2, 0.2))
        S = rng.uniform(-1, 1, 3)
        if sign == -1:
            e1 = (rng.uniform(-2.4, 0.6), rng.uniform(-0.8, 0.8), rng.uniform(-0.2, 1.4) * (1 if "L" in name else -1))
            k = rng.uniform(-2.2, -0.05)
        else:
            e1 = (rng.uniform(-1.4, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.5) * (1 if "L" in name else -1))
            k = rng.uniform(0.05, 2.0)
        j2, end, _ = limb_fk(P, S, o1, o2, e1, k)
        pole = (j2 - S)                       # elbow direction from the true pose
        got_e1, got_k, reached = solve_limb(P, S, o1, o2, end, pole, sign, rk)
        j2b, endb, _ = limb_fk(P, S, o1, o2, got_e1, got_k)
        err = float(np.linalg.norm(endb - end))
        # the middle joint must also land where the pole asked (same hinge plane => same elbow direction)
        e_true = (j2 - S) / np.linalg.norm(j2 - S)
        e_got = (j2b - S) / np.linalg.norm(j2b - S)
        err = max(err, float(np.linalg.norm(e_true - e_got)) * 0.05)
        errs.append(err)
        if err > 5e-3:
            bad += 1
    worst[name] = (max(errs), float(np.mean(errs)), bad)
    print("%-6s max err %.5f  mean %.6f  >5mm: %d/400" % (name, *worst[name]))
ok = all(v[2] == 0 and v[0] < 5e-3 for v in worst.values())
# unreachable targets must clamp gracefully (no NaN)
o1, o2 = LOCAL[BI["farmL"]].astype(float), LOCAL[BI["handL"]].astype(float)
e1, k, reached = solve_limb(np.eye(3), np.zeros(3), o1, o2, np.array([5.0, 0.0, 0.0]), np.array([0, -1, 0.0]), -1, (-2.6, 0.0))
print("unreachable ->", np.round(e1, 3), round(k, 3), "reached:", reached, "finite:", bool(np.isfinite(e1).all()))
ok = ok and np.isfinite(e1).all() and not reached
# euler round trip
for _ in range(200):
    a = rng.uniform(-1.2, 1.2, 3)
    R = euler_R(*a)
    assert np.allclose(euler_R(*R_to_euler(R)), R, atol=1e-9)
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
