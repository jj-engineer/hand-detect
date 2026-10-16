"""
Procedural audio: every sound effect and the music loop are synthesised with numpy/scipy at start-up
(no asset files). Playback uses a small callback mixer on top of sounddevice.
"""

import threading
import numpy as np

try:
    from scipy.signal import butter, lfilter
except Exception:                                   # pragma: no cover
    butter = lfilter = None

SR = 44100
rng = np.random.default_rng(11)
F32 = np.float32


# ------------------------------------------------------------------ dsp helpers
def tt(d):
    return np.arange(int(SR * d)) / SR


def noise(d):
    return rng.uniform(-1, 1, int(SR * d))


def _f(kind, x, fc, order=2):
    if isinstance(fc, (tuple, list)):
        b, a = butter(order, [fc[0] / (SR / 2), min(fc[1] / (SR / 2), 0.99)], "band")
    else:
        b, a = butter(order, min(fc / (SR / 2), 0.99), kind)
    return lfilter(b, a, x)


def lp(x, fc, order=2): return _f("low", x, fc, order)
def hp(x, fc, order=2): return _f("high", x, fc, order)
def bp(x, lo, hi, order=2): return _f("band", x, (lo, hi), order)


def env_exp(n, rate):
    return np.exp(-np.arange(n) / SR * rate)


def sweep(f0, f1, d, curve="exp"):
    t = tt(d)
    k = t / d
    f = f0 * (f1 / f0) ** k if curve == "exp" else f0 + (f1 - f0) * k
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def saw(f, t):
    return 2.0 * ((t * f) % 1.0) - 1.0


def attack(n, a):
    e = np.ones(n)
    k = int(a * SR)
    e[:k] = np.linspace(0, 1, k)
    return e


def norm(x, peak=0.9):
    m = np.max(np.abs(x)) + 1e-9
    return (x / m * peak).astype(F32)


def _comb(x, d, g):
    """Feedback comb y[n] = x[n] + g*y[n-d], computed block-wise (fast, vectorised)."""
    y = x.copy()
    for i in range(d, len(y), d):
        j = min(i + d, len(y))
        y[i:j] += g * y[i - d:j - d]
    return y


def reverb(x, amount=0.25, sizes=(0.0297, 0.0371, 0.0411, 0.0437)):
    out = np.zeros_like(x)
    for sz in sizes:
        out += _comb(x, int(sz * SR), 0.72)
    return x + amount * out / len(sizes)


def loopable(x, fade=0.04):
    n = int(fade * SR)
    y = x[:-n].copy()
    y[:n] = y[:n] * np.linspace(0, 1, n) + x[-n:] * np.linspace(1, 0, n)
    return y


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


# ------------------------------------------------------------------ sound design
def make_sounds():
    S = {}
    # footstep: weighty thud + metal clank
    n = int(0.30 * SR)
    thud = sweep(95, 38, 0.30) * env_exp(n, 16)
    clank = bp(noise(0.30), 1300, 4200) * env_exp(n, 55) * 0.35
    S["step"] = norm(thud + clank, 0.8)
    # punch: whoosh then impact
    t = tt(0.55)
    whoosh = bp(noise(0.55), 500, 3500) * np.sin(np.clip(t / 0.16, 0, 1) * np.pi) * (t < 0.18) * 0.5
    imp = np.zeros_like(t)
    i0 = int(0.14 * SR)
    m = len(t) - i0
    imp[i0:] = sweep(170, 46, m / SR)[:m] * env_exp(m, 13) + hp(noise(m / SR), 1800)[:m] * env_exp(m, 110) * 0.7
    S["punch"] = norm(whoosh + imp, 0.95)
    # slam: sub boom + debris rumble + crash, reverberant
    n = int(2.2 * SR)
    boom = sweep(88, 24, 2.2) * env_exp(n, 3.2)
    rumble = lp(noise(2.2), 190) * env_exp(n, 2.4) * 2.2
    crash = bp(noise(2.2), 600, 7000) * env_exp(n, 9) * 0.45
    S["slam"] = norm(reverb(boom + rumble + crash, 0.5), 1.0)
    # jump: pneumatic whump + rising thruster burst
    n = int(1.0 * SR)
    t = tt(1.0)
    burst = lp(noise(1.0), 1400) * attack(n, 0.04) * env_exp(n, 2.4) * 1.2
    rise = hp(noise(1.0), 1500) * np.sin(np.clip(t / 0.5, 0, 1) * np.pi / 2) * env_exp(n, 4) * 0.25
    whump = sweep(130, 45, 1.0) * env_exp(n, 9)
    S["jump"] = norm(burst + rise + whump, 0.95)
    # dash: roaring thrusters
    n = int(1.3 * SR)
    t = tt(1.3)
    roar = bp(noise(1.3), 160, 2600) * attack(n, 0.05) * env_exp(n, 1.7) * 1.4
    whine = np.sin(2 * np.pi * np.cumsum(260 + 700 * np.clip(t / 0.6, 0, 1)) / SR) * env_exp(n, 3) * 0.2
    S["dash"] = norm(roar + whine + sweep(100, 40, 1.3) * env_exp(n, 7) * 0.7, 0.95)
    S["skid"] = norm(bp(noise(0.55), 1500, 6500) * env_exp(int(0.55 * SR), 5.0) * attack(int(0.55 * SR), 0.03), 0.5)
    # laser: charge, fire, hum loop
    n = int(0.65 * SR)
    t = tt(0.65)
    ch = sweep(180, 2000, 0.65) * (0.2 + 0.8 * t / 0.65) * (0.75 + 0.25 * np.sin(2 * np.pi * 30 * t))
    ch += hp(noise(0.65), 4000) * 0.08 * (t / 0.65)
    S["laser_charge"] = norm(ch, 0.8)
    n = int(0.6 * SR)
    z0 = sweep(2200, 240, 0.35)
    zap = np.zeros(n)
    zap[:len(z0)] = z0 * env_exp(len(z0), 7)
    zap += hp(noise(0.6), 2500)[:n] * env_exp(n, 13) * 0.5 + sweep(130, 48, 0.6)[:n] * env_exp(n, 5) * 0.9
    S["laser_fire"] = norm(zap, 0.95)
    t = tt(1.0)
    hum = (np.sin(2 * np.pi * 110 * t) + 0.55 * np.sin(2 * np.pi * 220 * t + 1) + 0.35 * saw(165, t) * 0.5
           + 0.25 * np.sin(2 * np.pi * 440 * t)) * (0.7 + 0.3 * np.sin(2 * np.pi * 40 * t))
    hum += bp(noise(1.0), 3000, 8000) * 0.05
    S["laser_loop"] = loopable(np.concatenate([hum, hum[:int(0.04 * SR)]]) * 0.5)
    # shield
    n = int(0.9 * SR)
    t = tt(0.9)
    sw = np.zeros(n)
    s6 = sweep(260, 1500, 0.6)
    sw[:len(s6)] = s6
    on = (sw * 0.8 + sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate((880, 1320, 1760))) * 0.1 * env_exp(n, 3))
    S["shield_on"] = norm(on * attack(n, 0.02) * env_exp(n, 1.2), 0.8)
    t = tt(1.04)
    sh = (np.sin(2 * np.pi * 120 * t) + np.sin(2 * np.pi * 125 * t) * 0.9 + 0.35 * np.sin(2 * np.pi * 360 * t)
          + 0.2 * np.sin(2 * np.pi * 750 * t)) * (0.75 + 0.25 * np.sin(2 * np.pi * 7 * t))
    S["shield_loop"] = loopable(sh * 0.3)
    n = int(0.7 * SR)
    t = tt(0.7)
    hit = (np.sin(2 * np.pi * 1250 * t) + 0.7 * np.sin(2 * np.pi * 1870 * t)) * env_exp(n, 9) * 0.5
    hit += sweep(150, 55, 0.7) * env_exp(n, 12) + bp(noise(0.7), 800, 5000) * env_exp(n, 40) * 0.4
    S["shield_hit"] = norm(hit, 0.8)
    # power up
    n = int(2.0 * SR)
    t = tt(2.0)
    k = np.clip(t / 1.8, 0, 1)
    f = 60 * (900 / 60) ** k
    ph = 2 * np.pi * np.cumsum(f) / SR
    pu = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)) * (0.15 + 0.85 * k ** 1.5)
    chord = sum(np.sin(2 * np.pi * midi(m) * t) for m in (57, 64, 69, 72)) * 0.12 * k ** 2
    S["powerup"] = norm((pu + chord) * attack(n, 0.1) * np.r_[np.ones(int(1.8 * SR)), np.linspace(1, 0, n - int(1.8 * SR))], 0.85)
    n = int(2.0 * SR)
    t = tt(2.0)
    bb = sweep(120, 28, 2.0) * env_exp(n, 3) * 1.2 + lp(noise(2.0), 260) * env_exp(n, 3) * 1.4
    shim = sum(np.sin(2 * np.pi * f * t) for f in (880, 1320, 1760, 2640)) * env_exp(n, 2.5) * 0.12
    S["powerup_burst"] = norm(reverb(bb + shim + bp(noise(2.0), 800, 6000) * env_exp(n, 7) * 0.4, 0.5), 1.0)
    # explosion
    n = int(1.4 * SR)
    ex = lp(noise(1.4), 1800) * env_exp(n, 4.5) * 1.1 + sweep(100, 28, 1.4) * env_exp(n, 4) + hp(noise(1.4), 2500) * env_exp(n, 22) * 0.4
    S["explode"] = norm(reverb(ex, 0.3), 1.0)
    # ui blip
    n = int(0.09 * SR)
    S["ui"] = norm(sweep(1400, 2100, 0.09) * env_exp(n, 30) * attack(n, 0.004), 0.4)
    S["hit_small"] = norm(bp(noise(0.25), 600, 5000) * env_exp(int(0.25 * SR), 38) + sweep(300, 90, 0.25) * env_exp(int(0.25 * SR), 20), 0.7)
    # ---- reveal sequence: parts pop in, float, assemble, boot
    n = int(0.55 * SR)
    t = tt(0.55)
    pop = np.zeros(n)
    c0 = sweep(320, 1900, 0.14)
    pop[:len(c0)] = c0 * env_exp(len(c0), 14) * 0.6
    sp = (np.sin(2 * np.pi * 2400 * t) + 0.7 * np.sin(2 * np.pi * 3700 * t)) * env_exp(n, 12) * 0.22
    sp += hp(noise(0.55), 4500) * env_exp(n, 40) * 0.25
    sp += sweep(130, 58, 0.55) * env_exp(n, 11) * 0.5
    S["pop"] = norm(pop + sp, 0.8)
    t = tt(2.04)
    hum = (np.sin(2 * np.pi * 110 * t) + 0.7 * np.sin(2 * np.pi * 165 * t) + 0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.3 * np.sin(2 * np.pi * 330 * t) + 0.12 * np.sin(2 * np.pi * 880 * t + 1.0) + 0.12 * np.sin(2 * np.pi * 884 * t))
    hum *= 0.75 + 0.25 * np.sin(2 * np.pi * 1.0 * t)
    hum += lp(noise(2.04), 700) * 0.12
    S["float_hum"] = loopable(hum / (np.max(np.abs(hum)) + 1e-9) * 0.35, 0.05)
    n = int(3.2 * SR)
    t = tt(3.2)
    k = np.clip(t / 2.8, 0, 1)
    f = 70 * (1500 / 70) ** k
    ph = 2 * np.pi * np.cumsum(f) / SR
    rise = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * (0.1 + 0.9 * k ** 1.6)
    wh = bp(noise(3.2), 250, 4200) * (0.05 + 0.95 * k ** 2) * 0.9
    tail = np.r_[np.ones(int(2.8 * SR)), np.linspace(1, 0, n - int(2.8 * SR))]
    S["assemble"] = norm((rise + wh) * attack(n, 0.1) * tail, 0.8)
    n = int(0.7 * SR)
    t = tt(0.7)
    cl = sum(np.sin(2 * np.pi * f * t) * env_exp(n, d) * a for f, d, a in
             ((520, 8, 1.0), (837, 11, 0.8), (1263, 15, 0.6), (1850, 19, 0.45), (2540, 25, 0.3)))
    cl += hp(noise(0.7), 1500) * env_exp(n, 70) * 0.6 + sweep(160, 60, 0.7) * env_exp(n, 14) * 0.9
    S["clank"] = norm(cl, 0.9)
    n = int(3.4 * SR)
    t = tt(3.4)
    bm = sweep(75, 26, 3.4) * env_exp(n, 2.2) * 1.2 + lp(noise(3.4), 220) * env_exp(n, 2.6) * 0.9
    k = np.clip(t / 1.2, 0, 1)
    fr = 110 * (880 / 110) ** k
    sy = np.sin(2 * np.pi * np.cumsum(fr) / SR) * (0.25 * (1 - k) + 0.05) * (t < 1.4)
    chord = sum(np.sin(2 * np.pi * midi(m) * t) for m in (45, 48, 52, 57)) * 0.14 * attack(n, 0.5) * env_exp(n, 0.9)
    zap = hp(noise(3.4), 3000) * env_exp(n, 9) * 0.35
    S["boot"] = norm(reverb(bm + sy + chord + zap, 0.45), 1.0)
    # ---- katana
    n = int(0.9 * SR)
    t = tt(0.9)
    nz = noise(0.9)
    sweepn = bp(nz, 1800, 9000) * np.sin(np.clip(t / 0.4, 0, 1) * np.pi * 0.5) * np.exp(-np.clip(t - 0.35, 0, 9) * 9) * 1.0
    ring = sum(np.sin(2 * np.pi * f * t) * env_exp(n, d) * a for f, d, a in ((3150, 5, 0.5), (4720, 6, 0.4), (6300, 8, 0.3)))
    S["sword_draw"] = norm(sweepn * 0.9 + ring * (t > 0.25) * 0.5, 0.8)
    n = int(0.5 * SR)
    t = tt(0.5)
    sw = bp(noise(0.5), 500, 6500) * np.sin(np.clip(t / 0.22, 0, 1) * np.pi) * (t < 0.24) * 0.8
    sw += sweep(220, 70, 0.5) * env_exp(n, 14) * 0.25
    S["sword_swing"] = norm(sw, 0.75)
    n = int(0.6 * SR)
    t = tt(0.6)
    sh = hp(noise(0.6), 2400) * env_exp(n, 45) * 0.8 + sweep(240, 70, 0.6) * env_exp(n, 15)
    sh += sum(np.sin(2 * np.pi * f * t) * env_exp(n, d) * a for f, d, a in ((2100, 18, 0.4), (3300, 24, 0.3)))
    S["sword_hit"] = norm(sh, 0.9)
    n = int(0.8 * SR)
    t = tt(0.8)
    sc = bp(noise(0.8), 1400, 5200) * np.sin(np.clip(t / 0.4, 0, 1) * np.pi) * (t < 0.42) * 0.5
    ck = np.zeros(n)
    i0 = int(0.43 * SR)
    m = n - i0
    ck[i0:] = hp(noise(m / SR), 2800)[:m] * env_exp(m, 90) + sweep(320, 140, m / SR)[:m] * env_exp(m, 40) * 0.8
    S["sheathe"] = norm(sc + ck, 0.8)
    n = int(0.45 * SR)
    kk = sweep(135, 48, 0.45) * env_exp(n, 12) + bp(noise(0.45), 300, 2600) * np.sin(np.clip(tt(0.45) / 0.14, 0, 1) * np.pi) * (tt(0.45) < 0.16) * 0.5
    S["kick"] = norm(kk, 0.9)
    # thruster loop
    t = tt(1.04)
    th = lp(noise(1.04), 750) * 0.9 + bp(noise(1.04), 120, 380) * 0.8
    S["thruster_loop"] = loopable(th / (np.max(np.abs(th)) + 1e-9) * 0.5)
    return {k: np.asarray(v, F32) for k, v in S.items()}


def make_music(bpm=92, bars=16):
    beat = 60.0 / bpm
    N = int(bars * 4 * beat * SR)
    out = np.zeros((N, 2))
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]       # Am F C G
    # pad
    for bar in range(bars):
        ch = chords[bar % 4]
        s0 = int(bar * 4 * beat * SR)
        d = 4 * beat + 1.2
        t = tt(d)
        sig = np.zeros_like(t)
        for note in ch + [ch[0] - 12]:
            for det in (-0.09, 0.0, 0.09):
                sig += saw(midi(note) * 2 ** (det / 12), t) * 0.5
        sig = lp(sig, 850) * attack(len(t), 0.9) * np.r_[np.ones(int(4 * beat * SR)), np.linspace(1, 0, len(t) - int(4 * beat * SR))]
        n = min(len(sig), N - s0)
        out[s0:s0 + n, 0] += sig[:n] * 0.040
        out[s0:s0 + n, 1] += sig[:n] * 0.040
    # bass pulse (8ths) and kick
    for bar in range(bars):
        root = chords[bar % 4][0] - 24
        for e in range(8):
            s0 = int((bar * 4 + e * 0.5) * beat * SR)
            d = 0.45 * beat
            t = tt(d)
            note = np.sin(2 * np.pi * midi(root) * t) + 0.4 * saw(midi(root), t)
            note = lp(note, 380) * env_exp(len(t), 7)
            n = min(len(note), N - s0)
            a = 0.16 if e % 2 == 0 else 0.10
            out[s0:s0 + n] += note[:n, None] * a
        if bar >= 1:
            for b in range(4):
                s0 = int((bar * 4 + b) * beat * SR)
                k = sweep(130, 42, 0.28) * env_exp(int(0.28 * SR), 15)
                n = min(len(k), N - s0)
                out[s0:s0 + n] += k[:n, None] * 0.20
                if b % 2 == 1 and bar >= 3:
                    sn = hp(noise(0.22), 1500) * env_exp(int(0.22 * SR), 22) * 0.07
                    s1 = s0
                    n = min(len(sn), N - s1)
                    out[s1:s1 + n] += sn[:n, None]
    # hats
    for bar in range(2, bars):
        for e in range(8):
            if e % 2 == 0:
                continue
            s0 = int((bar * 4 + e * 0.5) * beat * SR)
            h = hp(noise(0.05), 7000) * env_exp(int(0.05 * SR), 70) * 0.06
            n = min(len(h), N - s0)
            out[s0:s0 + n] += h[:n, None]
    # arpeggio (pentatonic) with ping-pong delay
    arp_notes = [69, 72, 76, 79, 81, 79, 76, 72]
    for bar in range(4, bars):
        for s16 in range(16):
            if (s16 + bar) % 3 == 2:
                continue
            m = arp_notes[(s16 + bar * 2) % len(arp_notes)] + (12 if (bar % 8) >= 4 and s16 % 4 == 0 else 0)
            s0 = int((bar * 4 + s16 * 0.25) * beat * SR)
            t = tt(0.4)
            w = (np.abs(((t * midi(m)) % 1.0) * 4 - 2) - 1) * env_exp(len(t), 9)
            w = lp(w, 3500)
            pan = 0.35 + 0.3 * ((s16 % 2))
            n = min(len(w), N - s0)
            out[s0:s0 + n, 0] += w[:n] * 0.030 * (1 - pan)
            out[s0:s0 + n, 1] += w[:n] * 0.030 * pan
            d1 = s0 + int(0.375 * beat * SR)
            n = min(len(w), max(N - d1, 0))
            if n > 0:
                out[d1:d1 + n, 0] += w[:n] * 0.013 * pan
                out[d1:d1 + n, 1] += w[:n] * 0.013 * (1 - pan)
    out[:, 0] = reverb(out[:, 0], 0.35)
    out[:, 1] = reverb(out[:, 1], 0.35, sizes=(0.0313, 0.0391, 0.0429, 0.0457))
    out = loopable_stereo(out)
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.85).astype(F32)


def loopable_stereo(x, fade=0.6):
    n = int(fade * SR)
    y = x[:-n].copy()
    y[:n] = y[:n] * np.linspace(0, 1, n)[:, None] + x[-n:] * np.linspace(1, 0, n)[:, None]
    return y


# ------------------------------------------------------------------ mixer
class Voice:
    def __init__(self, buf, vol, pitch, pan, loop, key=None):
        self.buf = buf
        self.vol = vol
        self.cur = 0.0 if loop else vol
        self.pitch = pitch
        self.pan = pan
        self.loop = loop
        self.pos = 0.0
        self.key = key
        self.done = False
        self.stop_when_silent = False


class Audio:
    def __init__(self, music=True, volume=0.8):
        self.available = False
        self.sounds = {}
        self.voices = []
        self.lock = threading.Lock()
        self.master = volume
        self.music_on = music
        self.music_vol = 0.55
        self.stream = None
        self._music_buf = None
        try:
            import sounddevice as sd
            self.sd = sd
            self.sounds = make_sounds()
            self.stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32", blocksize=512, callback=self._cb)
            self.stream.start()
            self.available = True
            threading.Thread(target=self._build_music, daemon=True).start()
        except Exception as e:                                  # pragma: no cover
            print("[AUDIO] disabled:", e)

    def _build_music(self):
        try:
            self._music_buf = make_music()
            if self.music_on:
                self._start_music()
        except Exception as e:                                  # pragma: no cover
            print("[AUDIO] music failed:", e)

    def _start_music(self):
        with self.lock:
            if self._music_buf is None or any(v.key == "music" for v in self.voices):
                return
            v = Voice(self._music_buf, self.music_vol, 1.0, 0.0, True, "music")
            self.voices.append(v)

    def toggle_music(self):
        self.music_on = not self.music_on
        if self.music_on:
            self._start_music()
        else:
            with self.lock:
                for v in self.voices:
                    if v.key == "music":
                        v.vol = 0.0
                        v.stop_when_silent = True
        return self.music_on

    # ---- api
    def play(self, name, vol=1.0, pitch=1.0, pan=0.0):
        if not self.available or name not in self.sounds:
            return
        with self.lock:
            if len(self.voices) > 40:
                return
            self.voices.append(Voice(self.sounds[name], vol, pitch, pan, False))

    def loop(self, name, vol=1.0, pitch=1.0):
        """Start or update a looping sound; vol 0 fades it out."""
        if not self.available or name not in self.sounds:
            return
        with self.lock:
            for v in self.voices:
                if v.key == name:
                    v.vol, v.pitch = vol, pitch
                    v.stop_when_silent = vol <= 0.001
                    return
            if vol > 0.001:
                self.voices.append(Voice(self.sounds[name], vol, pitch, 0.0, True, name))

    def _cb(self, outdata, frames, time_info, status):
        out = np.zeros((frames, 2), F32)
        with self.lock:
            voices = list(self.voices)
        ar = np.arange(frames)
        for v in voices:
            buf = v.buf
            n = len(buf)
            idx = v.pos + ar * v.pitch
            if v.loop:
                idx = np.mod(idx, n)
                v.pos = (v.pos + frames * v.pitch) % n
                i0 = idx.astype(np.int64)
                i1 = (i0 + 1) % n
            else:
                if v.pos >= n - 1:
                    v.done = True
                    continue
                v.pos += frames * v.pitch
                valid = idx < n - 1
                idx = np.where(valid, idx, 0)
                i0 = idx.astype(np.int64)
                i1 = np.minimum(i0 + 1, n - 1)
            fr = (idx - i0).astype(F32)
            if buf.ndim == 1:
                s = buf[i0] * (1 - fr) + buf[i1] * fr
                if not v.loop:
                    s = s * valid
                gl = 1.0 - max(v.pan, 0.0)
                gr = 1.0 + min(v.pan, 0.0)
                chunk = np.stack([s * gl, s * gr], 1)
            else:
                s = buf[i0] * (1 - fr[:, None]) + buf[i1] * fr[:, None]
                chunk = s
            v.cur += (v.vol - v.cur) * 0.12
            out += chunk * v.cur
            if v.stop_when_silent and v.cur < 0.002:
                v.done = True
        if any(v.done for v in voices):
            with self.lock:
                self.voices = [v for v in self.voices if not v.done]
        out *= self.master
        outdata[:] = np.tanh(out * 1.15) / 1.15

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
