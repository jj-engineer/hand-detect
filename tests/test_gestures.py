"""Gesture classifier tests on a synthetic anatomical hand (no camera needed)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from titan.hand_input import finger_curls, classify


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_hand(curl, thumb_out=1.0, thumb_up=False, pinch=False, scale=1.0):
    """curl: 4 values index..pinky 0..1; hand faces camera, fingers up (+y negative in image => we use y-up here)."""
    w = np.zeros((21, 3))
    mcp_x = {5: 0.030, 9: 0.010, 13: -0.010, 17: -0.030}
    lens = {5: (0.045, 0.025, 0.022), 9: (0.048, 0.028, 0.024), 13: (0.044, 0.026, 0.023), 17: (0.035, 0.020, 0.019)}
    for k, base in enumerate((5, 9, 13, 17)):
        c = curl[k]
        p = np.array([mcp_x[base], 0.09, 0.0])
        w[base] = p
        d = np.array([0.0, 1.0, 0.0])
        ang_each = (0.5 * c * 1.5, c * 1.75, c * 1.2)     # MCP, PIP, DIP bend (rad) towards -z
        ang = 0.0
        for j, L in enumerate(lens[base]):
            ang += ang_each[j]
            d = np.array([0.0, math.cos(ang), -math.sin(ang)])
            p = p + d * L
            w[base + 1 + j] = p
    # thumb
    cmc = np.array([0.040, 0.020, 0.0]); w[1] = cmc
    ext_dir = np.array([0.85, 0.5, 0.0]) if not thumb_up else np.array([0.40, 0.92, 0.0])
    ext_dir /= np.linalg.norm(ext_dir)
    segs = (0.040, 0.036, 0.030)
    p = cmc
    for j, L in enumerate(segs):
        # tucked thumb folds over the knuckles (pointing at the index MCP, towards the palm)
        tuck_dir = (np.array([0.030, 0.085, -0.02]) - p)
        tuck_dir = tuck_dir / (np.linalg.norm(tuck_dir) + 1e-9)
        d = ext_dir * thumb_out + tuck_dir * (1 - thumb_out)
        d /= np.linalg.norm(d)
        p = p + d * min(L, np.linalg.norm(np.array([0.030, 0.085, -0.02]) - p) * (0.9 if thumb_out < 0.5 else 10))
        w[2 + j] = p
    if pinch:
        w[4] = w[8] + np.array([0.004, -0.002, 0.0])
    img = w[:, :2] * np.array([1, -1]) * 3.0 + 0.5       # image coords: y down
    img = np.concatenate([img, np.zeros((21, 1))], 1)
    return w * scale, img


CASES = {
    "OPEN":      dict(curl=(0, 0, 0, 0), thumb_out=1.0),
    "FIST":      dict(curl=(1, 1, 1, 1), thumb_out=0.0),
    "PEACE":     dict(curl=(0, 0, 1, 1), thumb_out=0.0),
    "POINT":     dict(curl=(0, 1, 1, 1), thumb_out=0.0),
    "ROCK":      dict(curl=(0, 1, 1, 0), thumb_out=0.0),
    "THREE":     dict(curl=(0, 0, 0, 1), thumb_out=0.0),
    "THUMBS_UP": dict(curl=(1, 1, 1, 1), thumb_out=1.0, thumb_up=True),
    "PINCH":     dict(curl=(0.55, 0, 0, 0), thumb_out=0.6, pinch=True),
}


def main():
    ok = True
    for name, kw in CASES.items():
        for jitter in (0.0, 0.08, -0.08):
            kw2 = dict(kw)
            kw2["curl"] = tuple(float(np.clip(c + (jitter if c in (0, 1) else 0) * (1 if c < 0.5 else -1), 0, 1)) for c in kw["curl"])
            w, img = make_hand(**kw2)
            curls = finger_curls(w)
            got = classify(curls, w, img)
            good = got == name
            ok &= good
            print("%-10s jitter=%+.2f -> %-10s curls=%s %s" % (name, jitter, got, np.round(curls, 2), "" if good else "  <-- MISMATCH"))
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


main()
