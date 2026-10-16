"""
Threaded webcam hand tracking (MediaPipe) with rotation-invariant finger-curl estimation,
debounced gesture recognition and One-Euro smoothing.

Gestures:  OPEN  FIST  PEACE  POINT  ROCK  THUMBS_UP  THREE  PINCH  (UNKNOWN = in-between pose)
"""

import math
import os
import sys
import threading
import time
import warnings

if sys.platform.startswith("win"):
    # MSMF's hardware-transform path makes some Windows webcams take ~30 s to open; must be set before cv2 loads
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _a(dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t, x):
        x = np.asarray(x, np.float32)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        ad = self._a(dt, self.dc)
        self.dx = ad * dx + (1 - ad) * self.dx
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = cutoff * 2 * math.pi * dt
        a = a / (a + 1.0)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()

    def reset(self):
        self.x = None


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return math.acos(max(-1.0, min(1.0, c)))


def finger_curls(w):
    """w: (21,3) metric world landmarks. Returns np.array(5) thumb..pinky, 0=straight 1=fully curled."""
    out = np.zeros(5, np.float32)
    palm_w = float(np.linalg.norm(w[5] - w[17])) + 1e-6
    # thumb: how far the tip has moved across the palm
    d = float(np.linalg.norm(w[4] - w[5])) / palm_w
    ang = _angle(w[3] - w[2], w[4] - w[3]) + _angle(w[2] - w[1], w[3] - w[2])
    ext = np.clip((d - 0.50) / 0.75, 0.0, 1.0)
    out[0] = float(np.clip(1.0 - ext * 0.8 - (1.0 - min(ang / 1.2, 1.0)) * 0.2, 0.0, 1.0))
    for k, (name, (a, b, c, e)) in enumerate(FINGERS.items()):
        ang0 = _angle(w[a] - w[0], w[b] - w[a])
        ang1 = _angle(w[b] - w[a], w[c] - w[b])
        ang2 = _angle(w[c] - w[b], w[e] - w[c])
        out[k + 1] = float(np.clip((0.5 * ang0 + ang1 + 0.8 * ang2) / 3.1, 0.0, 1.0))
    return out


def classify(curls, w, img):
    """Return gesture name for the instantaneous pose."""
    t, i, m, r, p = curls
    palm_w = float(np.linalg.norm(w[5] - w[17])) + 1e-6
    pinch = float(np.linalg.norm(w[4] - w[8])) / palm_w
    ext = lambda c: c < 0.38
    cur = lambda c: c > 0.58
    others_ext = ext(m) and (r < 0.5) and (p < 0.55)
    if pinch < 0.38 and others_ext:
        return "PINCH"
    # thumbs-up needs strong evidence: tip well clear of the knuckles and the thumb pointing up in the image
    thumb_ratio = float(np.linalg.norm(w[4] - w[5])) / palm_w
    tv = img[4][:2] - img[2][:2]
    thumb_up = thumb_ratio > 0.85 and tv[1] < 0 and abs(tv[0]) < 1.3 * abs(tv[1])
    if cur(i) and cur(m) and cur(r) and cur(p):
        return "THUMBS_UP" if thumb_up else "FIST"
    if ext(i) and cur(m) and cur(r) and cur(p):
        return "POINT"
    if ext(i) and ext(m) and cur(r) and cur(p):
        return "PEACE"
    if ext(i) and cur(m) and cur(r) and ext(p):
        return "ROCK"
    if ext(i) and ext(m) and ext(r) and cur(p):
        return "THREE"
    if ext(i) and ext(m) and ext(r) and ext(p):
        return "OPEN"
    return "UNKNOWN"


class HandState:
    def __init__(self):
        self.present = False
        self.gesture = "NONE"
        self.raw_gesture = "NONE"
        self.curls = np.full(5, 0.25, np.float32)
        self.palm = np.array([0.5, 0.5], np.float32)       # normalised, mirrored image coords (0..1)
        self.palm_vel = np.zeros(2, np.float32)
        self.roll = 0.0
        self.index_tilt = 0.0
        self.landmarks = None                                # (21,3) image coords 0..1
        self.frame = None                                    # BGR mirrored preview
        self.fps = 0.0
        self.stamp = 0.0
        self.age_lost = 99.0
        self.camera_ok = False
        self.camera_name = ""
        self.frame_id = 0


class HandInput:
    """Background thread: capture -> MediaPipe -> filtered HandState (latest-value semantics)."""

    def __init__(self, camera=None, hold_frames=3, preview_size=(320, 240)):
        self.camera_index = camera
        self.hold_frames = hold_frames
        self.preview_size = preview_size
        self.state = HandState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    # ---- public
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest(self):
        with self._lock:
            return self.state

    # ---- camera
    @staticmethod
    def _backends():
        """Capture backends to try for this OS, most reliable first."""
        if sys.platform == "darwin":
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        return [cv2.CAP_V4L2, cv2.CAP_ANY]

    def _open_camera(self):
        win = sys.platform.startswith("win")
        # laptops expose the built-in webcam as index 0; on the dev Mac index 1 is the built-in one
        cands = [self.camera_index] if self.camera_index is not None else ([0, 1, 2] if win else [1, 0, 2])
        warmup = 30 if win else 8                       # DirectShow cams deliver black frames while auto-exposure settles
        best = None
        for idx in cands:
            for backend in self._backends():
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ok, frame = False, None
                for _ in range(warmup):
                    ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue                            # opened but no frames: try the next backend
                if float(np.mean(frame)) > 3.0:
                    return cap, idx
                if best is None:
                    best = (cap, idx)
                else:
                    cap.release()
                break                                   # this index delivers frames: move on to the next index
        return best if best else (None, -1)

    def _run(self):
        try:
            import mediapipe as mp
            hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                             min_detection_confidence=0.6, min_tracking_confidence=0.55)
        except Exception as e:                     # pragma: no cover
            self.error = "mediapipe: %s" % e
            return
        cap, idx = self._open_camera()
        if cap is None:
            if sys.platform.startswith("win"):
                self.error = ("no camera available (Windows Settings > Privacy & security > Camera: allow desktop apps; "
                              "close Zoom/Teams/Camera app that may be using it)")
            else:
                self.error = "no camera available (check macOS camera permission)"
            return
        st = self.state
        st.camera_ok = True
        st.camera_name = "camera %d" % idx
        palm_f = OneEuro(1.4, 0.05)
        curl_f = OneEuro(2.5, 0.2)
        lm_f = OneEuro(2.0, 0.08)
        roll_f = OneEuro(1.5, 0.05)
        pending, count, stable = "NONE", 0, "NONE"
        miss = 0
        t_prev = time.time()
        fps = 0.0
        t0 = time.time()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            now = time.time() - t0
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t_prev, 1e-3))
            t_prev = time.time()
            prev = cv2.resize(frame, self.preview_size)
            new = HandState()
            new.camera_ok = True
            new.camera_name = st.camera_name
            new.fps = fps
            new.frame = prev
            new.stamp = now
            new.frame_id = st.frame_id + 1
            if res.multi_hand_landmarks and res.multi_hand_world_landmarks:
                miss = 0
                lm = res.multi_hand_landmarks[0].landmark
                img = np.array([[p.x, p.y, p.z] for p in lm], np.float32)
                img = lm_f(now, img)
                wl = np.array([[p.x, p.y, p.z] for p in res.multi_hand_world_landmarks[0].landmark], np.float32)
                raw_curls = finger_curls(wl)
                curls = np.clip(curl_f(now, raw_curls), 0, 1)
                raw = classify(raw_curls, wl, img)
                if raw == pending:
                    count += 1
                else:
                    pending, count = raw, 1
                need = self.hold_frames if raw != "UNKNOWN" else self.hold_frames + 2
                if count >= need:
                    stable = raw
                palm_raw = img[[0, 5, 9, 13, 17], :2].mean(0)
                palm = palm_f(now, palm_raw)
                vec = img[9, :2] - img[0, :2]
                roll = float(roll_f(now, [math.atan2(vec[0], -vec[1])])[0])
                tilt_v = img[8, :2] - img[5, :2]
                new.present = True
                new.gesture = stable
                new.raw_gesture = raw
                new.curls = curls.astype(np.float32)
                new.palm = palm.astype(np.float32)
                new.palm_vel = (palm - st.palm).astype(np.float32) if st.present else np.zeros(2, np.float32)
                new.roll = roll
                new.index_tilt = math.atan2(tilt_v[0], -tilt_v[1])
                new.landmarks = img
                new.age_lost = 0.0
            else:
                miss += 1
                new.present = False
                new.age_lost = st.age_lost + 1.0 / 30.0 if not st.present else 1.0 / 30.0
                if miss > 8:
                    pending, count, stable = "NONE", 0, "NONE"
                    palm_f.reset()
                    curl_f.reset()
                    lm_f.reset()
                    roll_f.reset()
                new.gesture = stable if miss <= 8 else "NONE"
                new.curls = st.curls
                new.palm = st.palm
                new.landmarks = None
            with self._lock:
                self.state = new
            st = new
        cap.release()
        hands.close()


def draw_skeleton(frame, landmarks, color=(255, 220, 80)):
    """Draw hand skeleton onto a (small) BGR preview frame in place."""
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(frame, p, r, (255, 255, 255), -1, cv2.LINE_AA)
    return frame
