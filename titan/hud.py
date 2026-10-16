"""
Glass-style HUD rendered with PIL into small per-panel textures (re-drawn only when content changes),
composited over the 3D scene by the renderer.
"""

import math
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .hand_input import draw_skeleton

FONT_AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
FONT_COND = "/System/Library/Fonts/Avenir Next Condensed.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"

# Non-macOS: bare filenames are resolved by Pillow in %WINDIR%\Fonts (Windows) or the usual system dirs (Linux)
FONT_FALLBACKS = {
    "bold": ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
    "heavy": ["seguibl.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
    "reg": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
    "med": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
    "cond": ["bahnschrift.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSansCondensed.ttf", "DejaVuSans.ttf"],
    "mono": ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"],
    "monob": ["consolab.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf"],
}

GUIDE = [
    # gesture, label, action, finger flags (thumb..pinky)
    ("OPEN", "OPEN PALM", "MOVE  (palm = target)", (1, 1, 1, 1, 1)),
    ("FIST", "FIST", "KATANA COMBO", (0, 0, 0, 0, 0)),
    ("PEACE", "PEACE", "JUMP SLAM", (0, 1, 1, 0, 0)),
    ("POINT", "POINT", "LASER BEAM", (0, 1, 0, 0, 0)),
    ("THREE", "THREE", "ENERGY SHIELD", (0, 1, 1, 1, 0)),
    ("THUMBS_UP", "THUMB UP", "JET DASH", (1, 0, 0, 0, 0)),
    ("ROCK", "ROCK", "POWER-UP", (0, 1, 0, 0, 1)),
    ("PINCH", "PINCH + MOVE", "ORBIT 360 CAMERA", (1, 1, 0, 0, 0)),
]


def _rgb(c):
    return tuple(int(max(0, min(1, v)) * 255) for v in c)


class Canvas:
    """Draw in logical pixels onto an S-times larger RGBA image."""

    def __init__(self, w, h, S):
        self.S = S
        self.w, self.h = w, h
        self.img = Image.new("RGBA", (int(w * S), int(h * S)), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self._fonts = {}

    def font(self, kind, size):
        key = (kind, size)
        if key not in self._fonts:
            path, idx = {"bold": (FONT_AVENIR, 2), "heavy": (FONT_AVENIR, 8), "reg": (FONT_AVENIR, 7), "med": (FONT_AVENIR, 5),
                         "cond": (FONT_COND, 2), "mono": (FONT_MONO, 0), "monob": (FONT_MONO, 1)}[kind]
            px = int(size * self.S)
            f = None
            try:
                f = ImageFont.truetype(path, px, index=idx)
            except Exception:
                for name in FONT_FALLBACKS[kind]:
                    try:
                        f = ImageFont.truetype(name, px)
                        break
                    except Exception:
                        continue
            if f is None:
                try:
                    f = ImageFont.load_default(px)
                except TypeError:                       # Pillow < 10.1 has no sized default font
                    f = ImageFont.load_default()
            self._fonts[key] = f
        return self._fonts[key]

    def rrect(self, x, y, w, h, r, fill=None, outline=None, width=1):
        S = self.S
        self.d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=r * S, fill=fill, outline=outline,
                                 width=max(1, int(width * S)))

    def line(self, pts, fill, width=1):
        S = self.S
        self.d.line([(px * S, py * S) for px, py in pts], fill=fill, width=max(1, int(width * S)), joint="curve")

    def ellipse(self, cx, cy, r, fill=None, outline=None, width=1):
        S = self.S
        self.d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=fill, outline=outline,
                       width=max(1, int(width * S)))

    def text(self, x, y, s, kind="reg", size=12, fill=(255, 255, 255, 255), anchor="ls", track=0.0):
        f = self.font(kind, size)
        S = self.S
        if track <= 0:
            self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)
            return
        # manual letter spacing; honour horizontal anchors by pre-measuring
        total = sum(self.d.textlength(ch, font=f) + track * S for ch in s) - track * S
        cx = x * S
        if anchor[0] == "m":
            cx -= total / 2
        elif anchor[0] == "r":
            cx -= total
        for ch in s:
            self.d.text((cx, y * S), ch, font=f, fill=fill, anchor="ls")
            cx += self.d.textlength(ch, font=f) + track * S

    def text_w(self, s, kind, size, track=0.0):
        f = self.font(kind, size)
        return (self.d.textlength(s, font=f) + track * self.S * len(s)) / self.S


def glass(c, w, h, accent, r=14, alpha=150):
    c.rrect(0, 0, w, h, r, fill=(6, 12, 20, alpha), outline=(*accent, 70), width=1)
    # top highlight
    c.line([(r, 1.5), (w - r, 1.5)], fill=(255, 255, 255, 38), width=1)
    # accent tick
    c.rrect(10, 0, 26, 2.2, 1, fill=(*accent, 230))


def hand_icon(c, cx, cy, s, flags, color, gesture=None):
    """Tiny finger-pose pictogram (thumb, index, middle, ring, pinky)."""
    col = color
    pw, ph = s * 0.60, s * 0.42
    px, py = cx - pw / 2, cy - ph * 0.10
    c.rrect(px, py, pw, ph, s * 0.12, fill=(*col[:3], 70), outline=col, width=1.4)
    if gesture == "PINCH":
        flags = (0, 0, 1, 1, 1)
    fw = pw / 4.5
    xs = [px + pw * (0.03 + 0.245 * i) for i in range(4)]
    for i, ext in enumerate(flags[1:]):
        L = s * (0.46 if i != 3 else 0.38) if ext else s * 0.10
        x = xs[i]
        c.rrect(x + fw * 0.12, py - L + 1, fw * 0.76, L + 2, fw * 0.38, fill=(*col[:3], 235 if ext else 130), outline=None)
    # thumb
    if gesture == "PINCH":
        c.ellipse(cx - pw * 0.28, py - s * 0.15, s * 0.16, outline=col, width=1.6)
        c.rrect(xs[0] - 2, py - s * 0.26, fw, s * 0.26, fw / 2, fill=(*col[:3], 235))
    elif flags[0]:
        if gesture == "THUMBS_UP":
            c.rrect(px - fw * 0.9, py - s * 0.40, fw * 1.1, s * 0.5, fw / 2, fill=(*col[:3], 235))
        else:
            c.line([(px + 1, py + ph * 0.55), (px - s * 0.30, py + ph * 0.05)], fill=(*col[:3], 235), width=fw * 0.9)
    else:
        c.rrect(px + 2, py + ph * 0.25, fw * 1.2, fw * 0.8, fw / 2, fill=(*col[:3], 130))


class Panel:
    def __init__(self, renderer, w, h, S):
        self.r = renderer
        self.w, self.h, self.S = w, h, S
        self.tex = renderer.make_texture(int(w * S), int(h * S))
        self.alpha = 1.0
        self.visible = True
        self.ready = False

    def canvas(self):
        return Canvas(self.w, self.h, self.S)

    def upload(self, canvas):
        self.tex.write(canvas.img.tobytes())
        self.ready = True

    def draw(self, x, y, S_fb, u=1.0):
        if self.visible and self.ready and self.alpha > 0.01:
            self.r.draw_hud(self.tex, x * S_fb, y * S_fb, self.w * S_fb * u, self.h * S_fb * u, self.alpha)


class Hud:
    def __init__(self, renderer, win_w, win_h, fb_w):
        self.r = renderer
        self.S = max(1.0, fb_w / float(win_w))        # retina factor
        self.ww, self.wh = win_w, win_h
        self.visible = True
        self.p_title = Panel(renderer, 330, 84, self.S)
        self.p_status = Panel(renderer, 252, 136, self.S)
        self.p_guide = Panel(renderer, 276, 44 + 8 * 50, self.S)
        self.p_cam = Panel(renderer, 296, 246, self.S)
        self.p_action = Panel(renderer, 560, 96, self.S)
        self.p_keys = Panel(renderer, 1100, 28, self.S)
        self.p_toast = Panel(renderer, 640, 70, self.S)
        self.p_prompt = Panel(renderer, 900, 150, self.S)
        self.live = True
        self._prompt_key = None
        self.p_hand = Panel(renderer, 296, 206, self.S)
        self.hand_tex = None
        self.accent = (60, 230, 255)
        self._t = {"title": 0, "status": 0, "action": 0, "cam": 0, "guide": 0, "keys": 0}
        self._last = {}
        self.toast_text = ""
        self.toast_t = 0.0
        self.toast_col = (255, 255, 255)
        self._toast_drawn = None
        self.cam_frame_id = -1

    def toast(self, text, col=(255, 255, 255), dur=1.6):
        self.toast_text, self.toast_t, self.toast_col = text, dur, col

    # ------------------------------------------------------------------ panels
    def _draw_title(self, theme_label, hud_col):
        c = self.p_title.canvas()
        a = self.accent
        glass(c, 330, 84, a)
        c.text(20, 42, "TITAN-9", "heavy", 30, (240, 248, 255, 255), track=3)
        c.text(21, 62, "HAND-COMMAND COMBAT MECHA", "cond", 11.5, (*a, 235), track=2.2)
        c.ellipse(24, 75, 3, fill=(*_rgb(hud_col), 255))
        c.text(34, 78.5, theme_label, "cond", 10.5, (190, 205, 220, 220), track=1.2)
        self.p_title.upload(c)

    def _draw_status(self, s):
        c = self.p_status.canvas()
        a = self.accent
        glass(c, 252, 136, a)
        rows = [("FPS", "%3d" % round(s["fps"]), None),
                ("VIEW", s["view"], None),
                ("HAND", "TRACKING" if s["hand"] else ("SEARCHING" if s["camera_ok"] else "KEYBOARD"), (120, 255, 170) if s["hand"] else (255, 200, 90)),
                ("AUDIO", ("MUSIC ON" if s["music"] else "SFX ONLY") if s["audio"] else "OFF", None),
                ("TARGETS", "%02d DOWN" % s["score"], None)]
        y = 30
        for k, v, col in rows:
            c.text(18, y, k, "cond", 11, (150, 170, 190, 230), track=2)
            c.text(234, y, v, "monob", 12, (*(col or (225, 238, 250)), 255), anchor="rs")
            y += 22
        self.p_status.upload(c)

    def _draw_guide(self, gesture, present):
        c = self.p_guide.canvas()
        a = self.accent
        W, H = self.p_guide.w, self.p_guide.h
        glass(c, W, H, a)
        c.text(18, 28, "FINGER COMMANDS", "cond", 12, (*a, 240), track=2.4)
        y0 = 40
        for i, (g, label, action, flags) in enumerate(GUIDE):
            y = y0 + i * 50
            active = present and g == gesture
            if active:
                c.rrect(8, y + 2, W - 16, 46, 10, fill=(*a, 52), outline=(*a, 200), width=1.2)
            col = (*a, 255) if active else (210, 222, 235, 235)
            hand_icon(c, 38, y + 31, 38, flags, col, g)
            c.text(72, y + 22, label, "bold", 13, (*col[:3], 255), track=0.8)
            c.text(72, y + 39, action, "cond", 11.5, (*(a if active else (150, 170, 190)), 235), track=0.6)
        self.p_guide.upload(c)

    def _draw_cam(self, state, gesture):
        c = self.p_cam.canvas()
        a = self.accent
        W, H = 296, 246
        glass(c, W, H, a, r=14, alpha=170)
        x0, y0, vw, vh = 8, 30, 280, 208
        c.text(16, 21, "OPTIC FEED", "cond", 11.5, (*a, 240), track=2.4)
        live = state is not None and state.frame is not None
        c.text(W - 14, 21, ("LIVE" if live else "NO SIGNAL"), "monob", 10.5, (120, 255, 170, 255) if live else (255, 120, 100, 255), anchor="rs")
        if live:
            fr = state.frame.copy()
            draw_skeleton(fr, state.landmarks, (255, 230, 90))
            fr = cv2.resize(fr, (int(vw * self.S), int(vh * self.S)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            # slight cyan grade + darken edges
            rgb = (rgb.astype(np.float32) * np.array([0.82, 0.95, 1.0], np.float32)).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(rgb).convert("RGBA")
            mask = Image.new("L", im.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius=int(9 * self.S), fill=255)
            c.img.paste(im, (int(x0 * self.S), int(y0 * self.S)), mask)
            # gesture badge
            label = gesture.replace("_", " ") if gesture not in ("NONE", "UNKNOWN") else ("NO HAND" if not state.present else "TRACKING")
            tw = c.text_w(label, "bold", 12.5, 1.2) + 20
            c.rrect(x0 + 8, y0 + vh - 32, tw, 24, 8, fill=(4, 10, 18, 190), outline=(*a, 200), width=1)
            c.text(x0 + 18, y0 + vh - 15, label, "bold", 12.5, (*a, 255), track=1.2)
            # reticle corners
            for (ax, ay, sx, sy) in ((x0 + 4, y0 + 4, 1, 1), (x0 + vw - 4, y0 + 4, -1, 1), (x0 + 4, y0 + vh - 4, 1, -1), (x0 + vw - 4, y0 + vh - 4, -1, -1)):
                c.line([(ax, ay + 12 * sy), (ax, ay), (ax + 12 * sx, ay)], fill=(*a, 220), width=1.4)
        else:
            c.rrect(x0, y0, vw, vh, 9, fill=(5, 9, 14, 220))
            c.text(W / 2, y0 + vh / 2 - 2, "KEYBOARD MODE", "bold", 14, (255, 200, 90, 255), anchor="ms", track=2)
            c.text(W / 2, y0 + vh / 2 + 18, "camera unavailable", "cond", 12, (150, 170, 190, 230), anchor="ms")
        self.p_cam.upload(c)

    def _draw_prompt(self, title, sub):
        c = self.p_prompt.canvas()
        W, H = 900, 150
        a = self.accent
        tw = c.text_w(title, "heavy", 40, 5.0)
        c.rrect(W / 2 - tw / 2 - 60, 18, tw + 120, 120, 16, fill=(4, 8, 14, 150))
        c.text(W / 2 - tw / 2, 62, title, "heavy", 40, (245, 248, 255, 255), track=5.0)
        c.line([(W / 2 - tw / 2 - 30, 78), (W / 2 + tw / 2 + 30, 78)], fill=(*a, 220), width=2)
        for dx in (-1, 1):
            c.line([(W / 2 + dx * (tw / 2 + 30), 70), (W / 2 + dx * (tw / 2 + 30), 86)], fill=(*a, 255), width=2)
        if sub:
            c.text(W / 2, 112, sub.upper(), "cond", 15, (235, 225, 215, 235), anchor="ms", track=3.0)
        self.p_prompt.upload(c)

    def _draw_hand_frame(self):
        c = self.p_hand.canvas()
        a = self.accent
        glass(c, 296, 206, a, r=14, alpha=170)
        c.text(16, 21, "ROBOT HAND  //  MIRROR", "cond", 11.5, (*a, 240), track=2.2)
        self.p_hand.upload(c)

    def _draw_action(self, s):
        c = self.p_action.canvas()
        a = self.accent
        W, H = 560, 96
        glass(c, W, H, a)
        c.text(22, 32, "SYSTEM", "cond", 11, (150, 170, 190, 230), track=2.4)
        name = s["action"]
        c.text(22, 68, name, "heavy", 30, (240, 248, 255, 255), track=2.5)
        # right: energy / overdrive
        bx, bw = 300, 238
        c.text(bx, 32, "OVERDRIVE", "cond", 11, (150, 170, 190, 230), track=2.4)
        od = s["overdrive"]
        c.rrect(bx, 40, bw, 10, 5, fill=(255, 255, 255, 28))
        if od > 0:
            c.rrect(bx, 40, max(10, bw * min(od / 14.0, 1.0)), 10, 5, fill=(255, 190, 70, 255))
            c.text(bx + bw, 32, "%.1fs" % od, "monob", 11, (255, 200, 90, 255), anchor="rs")
        else:
            c.text(bx + bw, 32, "READY  [6] / ROCK", "monob", 10.5, (150, 170, 190, 255), anchor="rs")
        c.text(bx, 66, "SPEED", "cond", 11, (150, 170, 190, 230), track=2.4)
        sp = min(s["speed"] / 6.5, 1.0)
        c.rrect(bx, 72, bw, 8, 4, fill=(255, 255, 255, 28))
        if sp > 0.02:
            c.rrect(bx, 72, max(8, bw * sp), 8, 4, fill=(*a, 255))
        self.p_action.upload(c)

    def _draw_keys(self):
        c = self.p_keys.canvas()
        txt = ("1-6 ACTIONS   WASD MOVE   DRAG / ARROWS ORBIT   F B L R T VIEWS   O TURNTABLE   C COLOR   "
               "M MUSIC   H HUD   P SHOT   Q QUALITY")
        c.text(550, 18, txt, "cond", 11.5, (170, 190, 210, 215), anchor="ms", track=0.9)
        self.p_keys.upload(c)

    def _draw_toast(self, text, col):
        c = self.p_toast.canvas()
        W, H = 640, 70
        tw = c.text_w(text, "heavy", 34, 4.0)
        c.text(W / 2 - tw / 2, 48, text, "heavy", 34, (*col, 255), track=4.0)
        c.line([(W / 2 - tw / 2 - 20, 58), (W / 2 + tw / 2 + 20, 58)], fill=(*col, 180), width=1.6)
        self.p_toast.upload(c)

    # ------------------------------------------------------------------ frame
    def update(self, dt, s, hand_state):
        """s: dict of status fields from the app."""
        if not self.visible:
            return
        now = time.time()
        a = _rgb(s["hud_col"])
        if a != self.accent:
            self.accent = a
            self._last.clear()
            self._t = {k: 0 for k in self._t}
            self.p_keys.ready = False
        if now - self._t["title"] > 1e8 or self._last.get("theme") != s["theme_label"]:
            self._last["theme"] = s["theme_label"]
            self._draw_title(s["theme_label"], s["hud_col"])
            self._t["title"] = 1e9 + now
        if now - self._t["status"] > 0.25:
            key = (round(s["fps"]), s["view"], s["hand"], s["music"], s["audio"], s["score"], s["camera_ok"])
            if key != self._last.get("status"):
                self._draw_status(s)
                self._last["status"] = key
            self._t["status"] = now
        g = s["gesture"]
        gk = (g, s["hand"])
        if gk != self._last.get("guide"):
            self._draw_guide(g, s["hand"])
            self._last["guide"] = gk
        if hand_state is not None and hand_state.frame_id != self.cam_frame_id or self._last.get("cam_ok") != s["camera_ok"]:
            if now - self._t["cam"] > 1 / 24.0:
                self._draw_cam(hand_state if s["camera_ok"] else None, g)
                self.cam_frame_id = hand_state.frame_id if hand_state is not None else -1
                self._last["cam_ok"] = s["camera_ok"]
                self._t["cam"] = now
        elif not s["camera_ok"] and self._last.get("cam_drawn") is None:
            self._draw_cam(None, g)
            self._last["cam_drawn"] = True
        if now - self._t["action"] > 0.1:
            key = (s["action"], round(s["overdrive"], 1), round(s["speed"] * 4))
            if key != self._last.get("action"):
                self._draw_action(s)
                self._last["action"] = key
            self._t["action"] = now
        self.live = bool(s.get("live", True))
        pr = s.get("prompt")
        key = (pr, a)
        if key != self._prompt_key:
            self._prompt_key = key
            if pr:
                self._draw_prompt(pr[0], pr[1])
        self.p_prompt.visible = bool(pr)
        self.p_prompt.alpha = 0.78 + 0.22 * math.sin(time.time() * 3.2) if pr else 0.0
        if not self.p_keys.ready:
            self._draw_keys()
        if not self.p_hand.ready or self._last.get('hand_acc') != a:
            self._draw_hand_frame()
            self._last['hand_acc'] = a
        # toast
        if self.toast_t > 0:
            self.toast_t -= dt
            if self._toast_drawn != (self.toast_text, self.toast_col):
                self._draw_toast(self.toast_text, self.toast_col)
                self._toast_drawn = (self.toast_text, self.toast_col)
            self.p_toast.alpha = max(0.0, min(1.0, self.toast_t * 3.0, (1.6 - self.toast_t) * 6.0 + 0.0 if self.toast_t > 1.3 else 1.0))
        else:
            self.p_toast.alpha = 0.0

    def draw(self, ww, wh, fb_scale):
        if not self.visible:
            return
        k = fb_scale
        u = max(0.55, min(1.3, min(ww / 1440.0, wh / 900.0)))     # UI scale follows the window size
        m = 22 * u
        P = self
        P.p_title.draw(m, m * 0.9, k, u)
        P.p_status.draw(ww - m - P.p_status.w * u, m * 0.9, k, u)
        if self.live:
            P.p_guide.draw(ww - m - P.p_guide.w * u, 170 * u, k, u)
        cam_y = wh - m - P.p_cam.h * u - 30 * u
        P.p_cam.draw(m, cam_y, k, u)
        hy = cam_y - 12 * u - P.p_hand.h * u
        if self.live:
            P.p_hand.draw(m, hy, k, u)
        if self.live and self.hand_tex is not None and P.p_hand.ready:
            self.r.draw_hud(self.hand_tex, (m + 8 * u) * k, (hy + 30 * u) * k, 280 * u * k, 168 * u * k, 1.0, flip=True)
        if self.live:
            P.p_action.draw((ww - P.p_action.w * u) / 2, wh - m - P.p_action.h * u - 32 * u, k, u)
        P.p_prompt.draw((ww - P.p_prompt.w * u) / 2, wh * 0.74, k, u)
        P.p_keys.draw((ww - P.p_keys.w * u) / 2, wh - 38 * u, k, u)
        P.p_toast.draw((ww - P.p_toast.w * u) / 2, 110 * u, k, u)
