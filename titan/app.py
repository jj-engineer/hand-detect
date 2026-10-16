"""
TITAN-9 Hand-Command Mecha — main application.

Window / input / hybrid hand control / 360 camera / simulation + render loop.
"""

import argparse
import math
import os
import sys
import time

import glfw
import moderngl
import numpy as np
from PIL import Image

from .animation import Mecha, Intent
from .audio import Audio
from .camera import OrbitCamera, PRESET_ORDER
from .fx import FX
from .hud import Hud
from .materials import THEME_ORDER, THEMES
from .mecha_model import build_mecha
from .renderer import Renderer, FrameData
from .reveal import Reveal, WAIT, PARTS, ASSEMBLE, BOOT, LIVE
from .scene import build_frame
from .targets import Targets, build_drone

F32 = np.float32


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Demo:
    """Scripted showcase (also used for automated screenshots when no webcam is available)."""

    SCRIPT = [
        (0.0, 3.0, dict(move_target=(3.0, 3.5))),
        (3.0, 3.1, dict(trig_jump=True)),
        (6.8, 13.2, dict(hold_combo=True)),
        (14.2, 14.3, dict(trig_dash=True)),
        (16.6, 20.0, dict(hold_laser=True)),
        (20.6, 23.4, dict(hold_shield=True)),
        (24.0, 24.1, dict(trig_power=True)),
        (28.5, 31.0, dict(move_target=(-4.5, -3.0))),
        (31.4, 31.5, dict(trig_jump=True)),
        (35.0, 41.5, dict(hold_combo=True)),
    ]
    LENGTH = 44.0

    def intent(self, t, mecha):
        t = t % self.LENGTH
        it = Intent()
        for a, b, kw in self.SCRIPT:
            if a <= t < b:
                for k, v in kw.items():
                    setattr(it, k, v)
        if 16.6 <= t < 20.0:
            it.aim_yaw = 0.7 * math.sin((t - 16.6) * 1.1)
            it.aim_pitch = 0.05
        return it


class App:
    def __init__(self, args):
        self.args = args
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, True)
        glfw.window_hint(glfw.SAMPLES, 0)
        ww, wh = args.size
        area = None
        if sys.platform.startswith("win") and not args.fullscreen:
            # Windows laptops are often 1366x768 / 1536x864: shrink to fit the usable desktop (taskbar + title bar excluded)
            pm = glfw.get_primary_monitor()
            if pm:
                ax, ay, aw, ah = glfw.get_monitor_workarea(pm)
                if aw > 0 and ah > 0:
                    ww, wh = min(ww, aw - 16), min(wh, ah - 48)
                    area = (ax, ay, aw, ah)
        mon = glfw.get_primary_monitor() if args.fullscreen else None
        self.win = glfw.create_window(ww, wh, "TITAN-9  //  Hand-Command Mecha", mon, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError(
                "could not create an OpenGL 4.1 window - update your graphics driver (Intel/NVIDIA/AMD) "
                "and make sure the app is not running through Remote Desktop or a virtual machine")
        if area is not None:
            ax, ay, aw, ah = area
            glfw.set_window_pos(self.win, ax + (aw - ww) // 2, max(ay + 40, ay + (ah - wh) // 2))
        glfw.make_context_current(self.win)
        glfw.swap_interval(0 if args.no_vsync else 1)
        self.ctx = moderngl.create_context()
        self.ww, self.wh = glfw.get_window_size(self.win)
        self.fw, self.fh = glfw.get_framebuffer_size(self.win)
        self.quality = args.quality
        scale = {"high": 1.0, "medium": 0.78, "low": 0.58}[self.quality]
        scale = min(scale, 2600.0 / max(self.fw, 1))
        self.renderer = Renderer(self.ctx, self.fw, self.fh, scale=scale, quality="low" if self.quality == "low" else "high")
        print("[GPU]", self.ctx.info["GL_RENDERER"], "| framebuffer %dx%d | internal scale %.2f" % (self.fw, self.fh, scale))
        t0 = time.time()
        md = build_mecha()
        self.renderer.set_mecha(md)
        self.renderer.set_props(build_drone())
        print("[MODEL] built in %.2fs (%d tris)" % (time.time() - t0, len(md.idx)))
        # secondary tiny renderer: live close-up of the robot's hand (mirrors your finger curls)
        self.hand_w, self.hand_h = int(280 * self.fw / float(self.ww)), int(168 * self.fw / float(self.ww))
        self.hand_fbo = self.ctx.framebuffer([self.ctx.texture((self.hand_w, self.hand_h), 4)])
        self.hand_renderer = Renderer(self.ctx, self.hand_w, self.hand_h, out_fbo=self.hand_fbo, scale=1.0, quality="low", shadow_size=1024)
        self.hand_renderer.set_mecha(md)
        self.hand_renderer.set_props(build_drone())
        self.fx = FX()
        self.audio = None if args.mute else Audio(music=not args.no_music)
        self.mecha = Mecha(self.fx, self.audio)
        self.reveal = Reveal(md, self.fx, self.audio)
        if getattr(args, "no_reveal", False):
            self.reveal.skip()
        self.g_hold = {}
        self.control_locked = False
        self.unlock_t = 0.0
        self.targets = Targets(self.fx, self.audio, count=5)
        self.mecha.on_hit = self.targets.hit_sphere
        self.mecha.on_ray = self.targets.ray
        self.mecha.on_beam = self.targets.beam_damage
        self.cam = OrbitCamera()
        self.hud = Hud(self.renderer, self.ww, self.wh, self.fw)
        self.hud.hand_tex = self.hand_fbo.color_attachments[0]
        self.theme_i = 0
        self.keys = set()
        self.pressed = set()
        self.dragging = False
        self.last_cursor = (0.0, 0.0)
        self.demo = Demo() if args.demo else None
        self.demo_t = 0.0
        self.hand = None
        self.hand_state = None
        self.g_prev = "NONE"
        self.pinch_prev = None
        self.drive_target = None
        self.fps = 60.0
        self.frame = 0
        self.last_time = time.time()
        self.clock = 0.0
        self.last_action = "IDLE"
        self.overdrive_prev = 0.0
        self.shot_path = args.shot
        self.shot_frames = args.frames
        self.audio_loops = {}
        self.fixed_dt = None
        self._install_callbacks()
        if not args.no_camera and not args.demo:
            from .hand_input import HandInput
            self.hand = HandInput(camera=args.camera)
            self.hand.start()
        if args.demo:
            self.cam.auto_orbit = True
            self.cam.label = "TURNTABLE"
            self.cam.auto_speed = 9.0

    # ------------------------------------------------------------------ callbacks
    def _install_callbacks(self):
        w = self.win
        glfw.set_key_callback(w, self._on_key)
        glfw.set_mouse_button_callback(w, self._on_mouse)
        glfw.set_cursor_pos_callback(w, self._on_cursor)
        glfw.set_scroll_callback(w, self._on_scroll)
        glfw.set_framebuffer_size_callback(w, self._on_resize)

    def _on_resize(self, win, fw, fh):
        if fw < 16 or fh < 16:
            return
        self.fw, self.fh = fw, fh
        self.ww, self.wh = glfw.get_window_size(win)
        scale = {"high": 1.0, "medium": 0.78, "low": 0.58}[self.quality]
        self.renderer.scale = min(scale, 2600.0 / fw)
        self.renderer.resize(fw, fh)
        self.hud = Hud(self.renderer, self.ww, self.wh, fw)
        self.hud.hand_tex = self.hand_fbo.color_attachments[0]

    def _on_key(self, win, key, sc, action, mods):
        if action == glfw.PRESS:
            self.keys.add(key)
            self.pressed.add(key)
            self._on_press(key)
        elif action == glfw.RELEASE:
            self.keys.discard(key)

    def _on_press(self, key):
        c = self.cam
        h = math.degrees(self.mecha.yaw)
        if key in (glfw.KEY_ESCAPE,):
            glfw.set_window_should_close(self.win, True)
        elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            if self.reveal.state == WAIT:
                self.reveal.trigger_parts(self.mecha)
            elif self.reveal.state == PARTS:
                self.reveal.trigger_assemble()
        elif key == glfw.KEY_X:
            self.restart_reveal()
        elif key == glfw.KEY_F:
            c.preset("FRONT", h)
            self.hud.toast("FRONT VIEW", (110, 235, 255), 0.9)
        elif key == glfw.KEY_B:
            c.preset("BACK", h)
            self.hud.toast("REAR VIEW", (110, 235, 255), 0.9)
        elif key == glfw.KEY_L:
            c.preset("LEFT", h)
            self.hud.toast("LEFT VIEW", (110, 235, 255), 0.9)
        elif key == glfw.KEY_R:
            c.preset("RIGHT", h)
            self.hud.toast("RIGHT VIEW", (110, 235, 255), 0.9)
        elif key == glfw.KEY_T:
            c.preset("TOP", h)
            self.hud.toast("TOP VIEW", (110, 235, 255), 0.9)
        elif key == glfw.KEY_V:
            c.cycle_preset(1, h)
            self.hud.toast(c.label + " VIEW", (110, 235, 255), 0.9)
        elif key == glfw.KEY_O:
            c.toggle_auto()
            self.hud.toast("360° TURNTABLE " + ("ON" if c.auto_orbit else "OFF"), (110, 235, 255), 1.0)
        elif key == glfw.KEY_C:
            self.theme_i = (self.theme_i + 1) % len(THEME_ORDER)
            t = THEMES[THEME_ORDER[self.theme_i]]
            self.hud.toast(t["label"].split("//")[0].strip() + " ARMOR", tuple(int(v * 255) for v in t["hud"]), 1.2)
        elif key == glfw.KEY_M:
            if self.audio and self.audio.available:
                on = self.audio.toggle_music()
                self.hud.toast("MUSIC " + ("ON" if on else "OFF"), (255, 255, 255), 0.9)
        elif key == glfw.KEY_H:
            self.hud.visible = not self.hud.visible
        elif key == glfw.KEY_P:
            self.save_shot()
        elif key == glfw.KEY_Q:
            order = ["high", "medium", "low"]
            self.quality = order[(order.index(self.quality) + 1) % 3]
            self._on_resize(self.win, self.fw, self.fh)
            self.renderer.quality = "low" if self.quality == "low" else "high"
            self.hud.toast("QUALITY: " + self.quality.upper(), (255, 255, 255), 1.0)
        elif key == glfw.KEY_EQUAL:
            c.zoom(0.85)
        elif key == glfw.KEY_MINUS:
            c.zoom(1.18)

    def _on_mouse(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS
            self.last_cursor = glfw.get_cursor_pos(win)

    def _on_cursor(self, win, x, y):
        if self.dragging:
            dx, dy = x - self.last_cursor[0], y - self.last_cursor[1]
            self.cam.orbit(-dx * 0.35, dy * 0.25)
            self.last_cursor = (x, y)

    def _on_scroll(self, win, dx, dy):
        self.cam.zoom(0.92 ** dy)

    # ------------------------------------------------------------------ input -> intent
    def _edge(self, key):
        if key in self.pressed:
            return True
        return False

    def restart_reveal(self):
        """Hide the robot again and replay the parts-reveal sequence."""
        self.mecha.action = None
        self.mecha.pos[:] = 0
        self.mecha.yaw = 0.0
        self.mecha.speed = 0.0
        self.mecha.emis_scale = self.mecha.emis_target = 1.0
        self.reveal.reset(self.mecha)
        self.g_hold.clear()
        self.control_locked = False

    def _reveal_triggers(self, dt):
        """Open hand -> parts pop up; fist -> assemble. (ENTER does the same from the keyboard.)"""
        rv = self.reveal
        hs = self.hand.latest() if self.hand is not None else None
        self.hand_state = hs
        g = hs.gesture if (hs is not None and hs.present) else "NONE"
        for k in list(self.g_hold):
            if k != g:
                self.g_hold[k] = 0.0
        self.g_hold[g] = self.g_hold.get(g, 0.0) + dt
        if rv.state == WAIT and self.g_hold.get("OPEN", 0.0) > 0.35:
            rv.trigger_parts(self.mecha)
        elif rv.state == PARTS and self.g_hold.get("FIST", 0.0) > 0.25:
            rv.trigger_assemble()
        if self.demo is not None:
            if rv.state == WAIT and self.clock > 1.6:
                rv.trigger_parts(self.mecha)
            elif rv.parts_ready() and self.clock > 1.6 + 4.6 + 1.2:
                rv.trigger_assemble()

    def make_intent(self, dt):
        k = self.keys
        cam = self.cam
        it = Intent()
        if self.demo is not None:
            return self.demo.intent(self.demo_t, self.mecha)
        if self.control_locked:
            hs0 = self.hand.latest() if self.hand is not None else None
            if hs0 is None or (not hs0.present and hs0.age_lost > 0.5) or (hs0.present and hs0.gesture != "FIST"):
                self.control_locked = False
        right, fwd = cam.right_vector(), cam.forward_flat()
        mv = np.zeros(2, F32)
        if glfw.KEY_W in k: mv += (fwd[0], fwd[2])
        if glfw.KEY_S in k: mv -= (fwd[0], fwd[2])
        if glfw.KEY_D in k: mv += (right[0], right[2])
        if glfw.KEY_A in k: mv -= (right[0], right[2])
        if np.linalg.norm(mv) > 0.05:
            it.move_dir = tuple(mv / max(1.0, float(np.linalg.norm(mv))))
        it.sprint = glfw.KEY_LEFT_SHIFT in k or glfw.KEY_RIGHT_SHIFT in k
        it.hold_combo = glfw.KEY_2 in k
        it.hold_laser = glfw.KEY_4 in k
        it.hold_shield = glfw.KEY_5 in k
        it.trig_jump = glfw.KEY_1 in self.pressed or glfw.KEY_SPACE in self.pressed
        it.trig_dash = glfw.KEY_3 in self.pressed
        it.trig_power = glfw.KEY_6 in self.pressed
        # arrows orbit continuously
        if glfw.KEY_LEFT in k: cam.orbit(-90 * dt, 0)
        if glfw.KEY_RIGHT in k: cam.orbit(90 * dt, 0)
        if glfw.KEY_UP in k: cam.orbit(0, 45 * dt)
        if glfw.KEY_DOWN in k: cam.orbit(0, -45 * dt)
        if it.hold_laser:
            facing = float(np.dot(self.mecha.forward, fwd))
            sgn = -1.0 if facing > 0 else 1.0
            if glfw.KEY_D in k: it.aim_yaw += sgn * -0.9
            if glfw.KEY_A in k: it.aim_yaw += sgn * 0.9
            if glfw.KEY_W in k: it.aim_pitch += 0.3
            if glfw.KEY_S in k: it.aim_pitch -= 0.2
            it.move_dir = None
        # ---- hand
        hs = self.hand.latest() if self.hand is not None else None
        self.hand_state = hs
        if hs is not None and hs.present and (self.clock > 0.5) and not self.control_locked:
            g = hs.gesture
            px, py = float(hs.palm[0]), float(hs.palm[1])
            it.curls = hs.curls
            if g in ("OPEN", "UNKNOWN"):
                c = cam.target
                tx = c[0] + right[0] * (px - 0.5) * 17.0 + fwd[0] * (0.52 - py) * 15.0
                tz = c[2] + right[2] * (px - 0.5) * 17.0 + fwd[2] * (0.52 - py) * 15.0
                r = math.hypot(tx, tz)
                if r > 24.0:
                    tx, tz = tx * 24.0 / r, tz * 24.0 / r
                it.move_target = (tx, tz)
                self.drive_target = np.array([tx, 0.0, tz], F32)
            else:
                self.drive_target = None
            if g == "FIST":
                it.hold_combo = True
            elif g == "POINT":
                it.hold_laser = True
                facing = float(np.dot(self.mecha.forward, fwd))
                sgn = -1.0 if facing > 0 else 1.0
                it.aim_yaw = sgn * float(np.clip(hs.index_tilt * 1.4, -1.1, 1.1))
                it.aim_pitch = float(np.clip((0.5 - py) * 1.2, -0.3, 0.5))
            elif g == "THREE":
                it.hold_shield = True
            if g != self.g_prev:
                if g == "PEACE": it.trig_jump = True
                elif g == "ROCK": it.trig_power = True
                elif g == "THUMBS_UP": it.trig_dash = True
                if g not in ("NONE",) and self.audio:
                    self.audio.play("ui", vol=0.5)
            if g == "PINCH":
                if self.pinch_prev is not None:
                    d = hs.palm - self.pinch_prev
                    cam.orbit(-float(d[0]) * 320.0, float(d[1]) * 90.0)
                self.pinch_prev = hs.palm.copy()
            else:
                self.pinch_prev = None
            self.g_prev = g
        else:
            self.drive_target = None
            self.pinch_prev = None
            if hs is not None and not hs.present and hs.age_lost > 0.5:
                self.g_prev = "NONE"
        return it

    # ------------------------------------------------------------------ misc
    def save_shot(self, path=None):
        data = self.ctx.screen.read(components=3, viewport=(0, 0, self.fw, self.fh))
        img = Image.frombytes("RGB", (self.fw, self.fh), data).transpose(Image.FLIP_TOP_BOTTOM)
        if path is None:
            os.makedirs("screenshots", exist_ok=True)
            path = "screenshots/titan_%s.png" % time.strftime("%Y%m%d_%H%M%S")
        img.save(path)
        print("[SHOT]", path)
        self.hud.toast("SCREENSHOT SAVED", (255, 255, 255), 1.0)

    def _audio_loops(self):
        a = self.audio
        if a is None or not a.available:
            return
        m = self.mecha
        lb = getattr(m.action, "beam", 0.0) if type(m.action).__name__ == "Laser" else 0.0
        a.loop("laser_loop", 0.5 * float(lb))
        sh = getattr(m.action, "s", 0.0) if type(m.action).__name__ == "Shield" else 0.0
        a.loop("shield_loop", 0.35 * float(sh))
        a.loop("thruster_loop", 0.45 * float(getattr(m, "thrust_now", 0.0)))

    def _announce(self):
        m = self.mecha
        name = m.status
        if name != self.last_action:
            msg = {"JUMPSLAM": "JUMP SLAM", "COMBO": "COMBO", "DASH": "JET DASH", "LASER": "LASER BEAM",
                   "SHIELD": "ENERGY SHIELD", "POWERUP": "POWER-UP"}.get(name)
            if msg:
                self.hud.toast(msg, (255, 214, 120), 1.1)
            self.last_action = name
        if m.overdrive > 0 and self.overdrive_prev <= 0:
            self.hud.toast("OVERDRIVE ENGAGED", (255, 190, 70), 2.0)
        self.overdrive_prev = m.overdrive

    # ------------------------------------------------------------------ main loop
    def step(self):
        """Advance one frame: input -> simulation -> render -> HUD."""
        glfw.poll_events()
        now = time.time()
        dt = min(now - self.last_time, 0.05)
        if self.fixed_dt is not None:
            dt = self.fixed_dt
        self.last_time = now
        self.clock += dt
        self.fps += (1.0 / max(dt, 1e-4) - self.fps) * 0.05
        rv = self.reveal
        prev_state = rv.state
        if rv.state != LIVE:
            self._reveal_triggers(dt)
        live = rv.state == LIVE
        if live and self.demo is not None:
            self.demo_t += dt
        it = self.make_intent(dt) if live else Intent()
        self.pressed.clear()
        ts = self.fx.time_scale()
        sdt = dt * ts
        self.mecha.look_world = self.cam.eye_pos() if live else None
        self.mecha.update(sdt, it)
        if rv.state != LIVE:
            rv.update(sdt, self.mecha)
        if prev_state != LIVE and rv.state == BOOT and prev_state != BOOT:
            self.control_locked = True
        if prev_state == BOOT and rv.state == LIVE:
            self.hud.toast("TITAN-9 ONLINE", (255, 190, 90), 2.0)
        self.fx.update(dt, sdt)
        if live:
            self.targets.update(sdt, self.mecha.pos)
        if rv.state in (WAIT, PARTS, ASSEMBLE) and not self.dragging:
            self.cam.t_az = 16.0 + 18.0 * math.sin(self.clock * 0.32)          # gentle sway keeps the exploded view readable
            self.cam.t_el = 9.0
            self.cam.auto_orbit = False
            self.cam.label = "3/4 FRONT"
        self.cam.update(dt, np.array([self.mecha.pos[0], self.mecha.height, self.mecha.pos[2]], F32), self.fx, self.clock,
                        wide=max(self.mecha.camera_wide(), rv.camera_wide()), ybias=rv.camera_ybias())
        self._audio_loops()
        self._announce()
        theme = THEME_ORDER[self.theme_i]
        fd = build_frame(self.mecha, self.fx, self.cam, self.fw / float(self.fh), self.clock, theme=theme,
                         target=self.drive_target, props=self.targets.props() if live else [])
        fd.show_mecha = rv.mecha_visible()
        if rv.skin is not None:
            fd.bones = rv.skin
            fd.bone_fx = rv.bone_fx
        self.renderer.render(fd)
        if self.hud.visible and live and self.frame % 2 == 0:
            self._render_hand_cam(fd, theme)
        hs = self.hand_state
        camera_ok = bool(hs.camera_ok) if hs is not None else False
        if self.hand is not None and self.hand.error and not getattr(self, "_err_shown", False):
            self._err_shown = True
            print("[HAND]", self.hand.error)
            self.hud.toast("NO CAMERA - KEYBOARD MODE", (255, 200, 90), 3.0)
        status = dict(
            fps=self.fps, view=self.cam.label, hand=bool(hs is not None and hs.present), camera_ok=camera_ok,
            audio=bool(self.audio and self.audio.available), music=bool(self.audio and self.audio.music_on),
            score=self.targets.score, gesture=(hs.gesture if hs is not None else "NONE"),
            action=self.mecha.status.replace("JUMPSLAM", "JUMP SLAM").replace("POWERUP", "POWER-UP"),
            overdrive=self.mecha.overdrive, speed=float(self.mecha.speed),
            hud_col=THEMES[theme]["hud"], theme_label=THEMES[theme]["label"], live=live, prompt=rv.prompt())
        self.hud.update(dt, status, hs)
        self.hud.draw(self.ww, self.wh, self.fw / float(self.ww))
        self.frame += 1

    def _render_hand_cam(self, fd, theme):
        """Close-up of the robot's left hand / gauntlet."""
        m = self.mecha
        hp = m.attach("fistL") * 0.55 + m.attach("palmL") * 0.45
        f = m.forward
        left = np.array([math.cos(m.yaw), 0.0, -math.sin(m.yaw)], F32)
        eye = hp + f * 1.25 + left * 0.55 + np.array([0, 0.45, 0], F32)
        from .xform import look_at, perspective
        V = look_at(eye, hp + np.array([0, 0.0, 0], F32), (0, 1, 0))
        P = perspective(math.radians(34), self.hand_w / float(self.hand_h), 0.05, 100.0)
        hd = FrameData(view=V, proj=P, eye=eye, bones=fd.bones, focus=fd.focus, time=fd.time, emis_boost=fd.emis_boost,
                       charge=fd.charge, power=fd.power, theme=fd.theme, props=[], sprites=None, beams=[], rings=[],
                       exposure=1.0, bloom=0.2)
        self.hand_renderer.render(hd)

    def run(self):
        try:
            while not glfw.window_should_close(self.win):
                self.step()
                if self.shot_path and self.frame >= self.shot_frames:
                    self.save_shot(self.shot_path)
                    break
                glfw.swap_buffers(self.win)
        finally:
            self.close()

    def close(self):
        if self.hand is not None:
            self.hand.stop()
        if self.audio:
            self.audio.close()
        glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description="TITAN-9: hand-controlled combat mecha")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: auto-detect)")
    ap.add_argument("--no-camera", action="store_true", help="keyboard/mouse only")
    ap.add_argument("--demo", action="store_true", help="scripted showcase (no camera needed)")
    ap.add_argument("--no-reveal", action="store_true", help="skip the parts-reveal and start with the robot built")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-vsync", action="store_true")
    ap.add_argument("--quality", choices=["high", "medium", "low"], default="high")
    ap.add_argument("--size", type=int, nargs=2, default=[1440, 900], metavar=("W", "H"))
    ap.add_argument("--shot", type=str, default=None, help="save a screenshot after --frames frames and exit")
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
