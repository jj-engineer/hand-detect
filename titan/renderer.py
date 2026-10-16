"""
GPU renderer (moderngl / OpenGL 4.1).

Pipeline per frame:
  shadow map -> planar reflection -> half-res normal/depth pre-pass -> SSAO (+blur)
  -> HDR scene (sky, floor, mecha, additive FX) -> bloom -> composite/tonemap -> FXAA -> HUD
"""

import math
from dataclasses import dataclass, field

import moderngl
import numpy as np

from . import shaders as S
from . import materials as MAT
from . import geometry as G
from .xform import (look_at, ortho, translate, to_gl, mats_to_gl, reflect_y)

F32 = np.float32


def _dir_from(az, el):
    return np.array([math.cos(az) * math.cos(el), math.sin(el), math.sin(az) * math.cos(el)], F32)


KEY_DIR = _dir_from(1.15, 0.80)
RIM1_DIR = _dir_from(-1.05, 0.35)
RIM2_DIR = _dir_from(-2.15, 0.35)


@dataclass
class FrameData:
    view: np.ndarray
    proj: np.ndarray
    eye: np.ndarray
    bones: np.ndarray                      # (N,4,4) skin matrices
    focus: np.ndarray = field(default_factory=lambda: np.zeros(3, F32))
    time: float = 0.0
    emis_boost: float = 1.0
    charge: float = 0.0
    power: float = 0.0
    wear: float = 1.0
    target: object = None                  # np.ndarray(3) or None
    sprites: object = None                 # (n, 9) [pos3, size, rgba4, kind]
    beams: list = field(default_factory=list)
    rings: list = field(default_factory=list)
    shield: object = None                  # dict(center, radius, col, int, hits)
    props: list = field(default_factory=list)   # [dict(model=4x4, flash=float)]
    bone_fx: object = None                 # (NB,2) per-bone (glow, dissolve) for the reveal sequence
    show_mecha: bool = True
    flash: float = 0.0
    flash_col: tuple = (1.0, 1.0, 1.0)
    aberr: float = 0.0
    exposure: float = 0.95
    bloom: float = 0.28
    theme: str = "ONI"


def _set(prog, **kw):
    for k, v in kw.items():
        try:
            u = prog[k]
        except KeyError:
            continue
        if isinstance(v, (bytes, bytearray)):
            u.write(v)
        elif isinstance(v, np.ndarray):
            u.value = tuple(v.astype(float).tolist()) if v.ndim else float(v)
        else:
            u.value = v


class Renderer:
    SHADOW_SIZE = 4096
    SHADOW_HALF = 8.5

    def __init__(self, ctx, width, height, out_fbo=None, scale=1.0, quality="high", shadow_size=None):
        self.ctx = ctx
        if shadow_size:
            self.SHADOW_SIZE = shadow_size
        self.quality = quality
        self.out_fbo = out_fbo
        self.scale = scale
        self.mecha = None
        self.palette_name = None
        self._build_programs()
        self._build_static()
        self.resize(width, height)

    # ------------------------------------------------------------------ setup
    def _prog(self, vs, fs):
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _build_programs(self):
        P = self._prog
        self.p_env = P(S.FS_VERT, S.ENVGEN_FS)
        self.p_bg = P(S.FS_VERT, S.BG_FS)
        self.p_mecha = P(S.MECHA_VS, S.MECHA_FS)
        self.p_depth = P(S.DEPTH_VS, S.DEPTH_FS)
        self.p_gbuf = P(S.GBUF_VS, S.GBUF_FS)
        self.p_floor = P(S.FLOOR_VS, S.FLOOR_FS)
        self.p_floor_g = P(S.FLOOR_VS, S.FLOOR_GBUF_FS)
        self.p_ssao = P(S.FS_VERT, S.SSAO_FS)
        self.p_aoblur = P(S.FS_VERT, S.AOBLUR_FS)
        self.p_bdown = P(S.FS_VERT, S.BLOOM_DOWN_FS)
        self.p_bup = P(S.FS_VERT, S.BLOOM_UP_FS)
        self.p_comp = P(S.FS_VERT, S.COMPOSITE_FS)
        self.p_fxaa = P(S.FS_VERT, S.FXAA_FS)
        self.p_hud = P(S.HUD_VS, S.HUD_FS)
        self.p_sprite = P(S.SPRITE_VS, S.SPRITE_FS)
        self.p_beam = P(S.BEAM_VS, S.BEAM_FS)
        self.p_ring = P(S.RING_VS, S.RING_FS)
        self.p_shield = P(S.SHIELD_VS, S.SHIELD_FS)
        self._fs_vao = {}

    def _build_static(self):
        ctx = self.ctx
        self.vbo_quad = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], F32).tobytes())
        fl = 90.0
        self.vbo_floor = ctx.buffer(np.array([-fl, -fl, fl, -fl, -fl, fl, fl, fl], F32).tobytes())
        self.vao_floor = ctx.vertex_array(self.p_floor, [(self.vbo_floor, "2f", "in_xz")])
        self.vao_floor_g = ctx.vertex_array(self.p_floor_g, [(self.vbo_floor, "2f", "in_xz")])
        self.vao_ring = ctx.vertex_array(self.p_ring, [(self.vbo_quad, "2f", "in_quad")])
        self.vao_beam = ctx.vertex_array(self.p_beam, [(self.vbo_quad, "2f", "in_quad")])
        self.vao_hud = ctx.vertex_array(self.p_hud, [(self.vbo_quad, "2f", "in_quad")])
        # sprites
        self.max_sprites = 12000
        self.vbo_spr = ctx.buffer(reserve=self.max_sprites * 9 * 4, dynamic=True)
        self.vao_spr = ctx.vertex_array(self.p_sprite, [
            (self.vbo_quad, "2f", "in_quad"),
            (self.vbo_spr, "3f 1f 4f 1f/i", "in_pos", "in_size", "in_col", "in_kind")])
        # shield sphere
        sph = G.sphere(1.0, 56, 28)
        inter = np.concatenate([sph.pos, sph.nrm], 1).astype(F32)
        self.vbo_sph = ctx.buffer(inter.tobytes())
        self.ibo_sph = ctx.buffer(sph.idx.astype(np.uint32).tobytes())
        self.vao_sph = ctx.vertex_array(self.p_shield, [(self.vbo_sph, "3f 3f", "in_pos", "in_nrm")], self.ibo_sph)
        # environment map (equirect HDR + mips)
        self.env_w, self.env_h = 1024, 512
        self.tex_env = ctx.texture((self.env_w, self.env_h), 4, dtype="f2")
        fbo = ctx.framebuffer([self.tex_env])
        fbo.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self._fs(self.p_env).render(moderngl.TRIANGLES, vertices=3)
        self.tex_env.build_mipmaps()
        self.tex_env.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tex_env.repeat_x = True
        self.tex_env.repeat_y = False
        self.env_lod = math.log2(max(self.env_w, self.env_h)) - 0.5
        self.irr_lod = math.log2(self.env_w) - 3.0
        fbo.release()
        # shadow map
        s = self.SHADOW_SIZE
        self.tex_shadow = ctx.depth_texture((s, s))
        self.tex_shadow.compare_func = "<="
        self.tex_shadow.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_shadow.repeat_x = self.tex_shadow.repeat_y = False
        self.fbo_shadow = ctx.framebuffer(depth_attachment=self.tex_shadow)

    def _fs(self, prog):
        v = self._fs_vao.get(prog.glo)
        if v is None:
            v = self.ctx.vertex_array(prog, [])
            self._fs_vao[prog.glo] = v
        return v

    def set_mecha(self, md):
        """md: object with pos, nrm, mid, bone, idx arrays."""
        ctx = self.ctx
        inter = np.concatenate([md.pos, md.nrm, md.mid[:, None], md.bone[:, None]], 1).astype(F32)
        self.vbo_mecha = ctx.buffer(inter.tobytes())
        self.ibo_mecha = ctx.buffer(np.ascontiguousarray(md.idx, np.uint32).tobytes())
        layout = "3f 3f 1f 1f"
        names = ("in_pos", "in_nrm", "in_mid", "in_bone")
        mk = lambda p: self.ctx.vertex_array(p, [(self.vbo_mecha, layout, *names)], self.ibo_mecha, skip_errors=True)
        self.vao_mecha = mk(self.p_mecha)
        self.vao_mecha_depth = mk(self.p_depth)
        self.vao_mecha_g = mk(self.p_gbuf)
        self.mecha = md

    def set_props(self, md):
        ctx = self.ctx
        inter = np.concatenate([md.pos, md.nrm, md.mid[:, None], md.bone[:, None]], 1).astype(F32)
        self.vbo_prop = ctx.buffer(inter.tobytes())
        self.ibo_prop = ctx.buffer(np.ascontiguousarray(md.idx, np.uint32).tobytes())
        mk = lambda p: ctx.vertex_array(p, [(self.vbo_prop, "3f 3f 1f 1f", "in_pos", "in_nrm", "in_mid", "in_bone")],
                                        self.ibo_prop, skip_errors=True)
        self.vao_prop = mk(self.p_mecha)
        self.vao_prop_depth = mk(self.p_depth)
        self.vao_prop_g = mk(self.p_gbuf)

    def _prop_bytes(self, props):
        out = []
        for pr in props:
            arr = np.tile(np.eye(4, dtype=F32), (64, 1, 1))
            arr[0] = pr["model"]
            out.append(mats_to_gl(arr))
        return out

    def set_theme(self, name):
        if name == self.palette_name:
            return
        albedo, params = MAT.theme_arrays(name)
        self._albedo_bytes = albedo.astype(F32).tobytes()
        self._params_bytes = params.astype(F32).tobytes()
        self.palette_name = name
        self.accent = np.array(MAT.THEMES[name]["floor"], F32)
        self.fx_col = np.array(MAT.THEMES[name]["glowA"], F32)

    # ------------------------------------------------------------------ targets
    def resize(self, width, height):
        ctx = self.ctx
        self.W, self.H = int(width), int(height)
        iw, ih = max(int(self.W * self.scale), 64), max(int(self.H * self.scale), 64)
        self.IW, self.IH = iw, ih
        hw, hh = max(iw // 2, 32), max(ih // 2, 32)
        self.HW, self.HH = hw, hh

        def tex_f(w, h, comps=4):
            t = ctx.texture((w, h), comps, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            return t

        for name in ("tex_hdr", "fbo_hdr", "tex_refl", "fbo_refl", "tex_gn", "tex_gd", "fbo_g", "tex_ao", "fbo_ao",
                     "tex_ao2", "fbo_ao2", "tex_ldr", "fbo_ldr"):
            o = getattr(self, name, None)
            if o is not None:
                o.release()
        self.tex_hdr = tex_f(iw, ih)
        self.fbo_hdr = ctx.framebuffer([self.tex_hdr], ctx.depth_renderbuffer((iw, ih)))
        self.tex_refl = tex_f(hw, hh)
        self.fbo_refl = ctx.framebuffer([self.tex_refl], ctx.depth_renderbuffer((hw, hh)))
        self.tex_gn = tex_f(hw, hh)
        self.tex_gd = ctx.depth_texture((hw, hh))
        self.tex_gd.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_gd.compare_func = ""
        self.fbo_g = ctx.framebuffer([self.tex_gn], self.tex_gd)
        self.tex_ao = ctx.texture((hw, hh), 1, dtype="f1")
        self.tex_ao.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_ao.repeat_x = self.tex_ao.repeat_y = False
        self.fbo_ao = ctx.framebuffer([self.tex_ao])
        self.tex_ao2 = ctx.texture((hw, hh), 1, dtype="f1")
        self.tex_ao2.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_ao2.repeat_x = self.tex_ao2.repeat_y = False
        self.fbo_ao2 = ctx.framebuffer([self.tex_ao2])
        self.tex_ldr = ctx.texture((iw, ih), 4, dtype="f1")
        self.tex_ldr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_ldr.repeat_x = self.tex_ldr.repeat_y = False
        self.fbo_ldr = ctx.framebuffer([self.tex_ldr])
        # bloom chain
        for t in getattr(self, "bloom", []):
            t[0].release()
            t[1].release()
        self.bloom = []
        w, h = iw, ih
        for _ in range(6):
            w, h = max(w // 2, 2), max(h // 2, 2)
            tx = tex_f(w, h)
            self.bloom.append((tx, ctx.framebuffer([tx]), w, h))

    # ------------------------------------------------------------------ helpers
    def _common_lights(self, prog, eye):
        _set(prog, u_cam=tuple(eye),
             u_env_lod=self.env_lod, u_irr_lod=self.irr_lod, u_env_int=1.0, u_diff_int=0.34,
             u_key_dir=tuple(KEY_DIR), u_key_col=(2.7, 2.5, 2.2),
             u_rim1_dir=tuple(RIM1_DIR), u_rim1_col=(0.55, 2.0, 2.8),
             u_rim2_dir=tuple(RIM2_DIR), u_rim2_col=(3.0, 0.85, 0.55),
             u_shadow_texel=2.0 * self.SHADOW_HALF / self.SHADOW_SIZE)
        self.tex_env.use(0)
        self.tex_shadow.use(1)
        _set(prog, u_env=0, u_shadow=1)

    def _draw_mecha(self, vao, prog, vp, bones_bytes):
        _set(prog, u_vp=vp)
        prog["u_bones"].write(bones_bytes)
        vao.render(moderngl.TRIANGLES)

    # ------------------------------------------------------------------ frame
    def render(self, fd: FrameData):
        ctx = self.ctx
        self.set_theme(fd.theme)
        P, V = fd.proj, fd.view
        VP = P @ V
        eye = np.asarray(fd.eye, F32)
        bones_bytes = mats_to_gl(fd.bones)
        fxa = np.zeros((64, 2), F32)
        if fd.bone_fx is not None:
            fxa[:len(fd.bone_fx)] = fd.bone_fx
        fx_bytes = np.ascontiguousarray(fxa).tobytes()
        if bones_bytes and len(fd.bones) > 64:
            raise ValueError("too many bones")
        # pad bones to 64
        if len(fd.bones) < 64:
            pad = np.tile(np.eye(4, dtype=F32), (64 - len(fd.bones), 1, 1))
            bones_bytes = mats_to_gl(np.concatenate([fd.bones, pad], 0))

        # -------- shadow map ------------------------------------------------
        S_ = self.SHADOW_HALF
        texel = 2 * S_ / self.SHADOW_SIZE
        focus = np.asarray(fd.focus, F32).copy()
        focus[1] = 1.5
        Lv = look_at(focus + KEY_DIR * 30.0, focus, (0, 1, 0))
        fl = Lv @ np.append(focus, 1.0)
        dx = round(fl[0] / texel) * texel - fl[0]
        dy = round(fl[1] / texel) * texel - fl[1]
        Lv = translate(dx, dy, 0.0) @ Lv
        SVP = ortho(-S_, S_, -S_, S_, 1.0, 70.0) @ Lv
        self.fbo_shadow.use()
        ctx.viewport = (0, 0, self.SHADOW_SIZE, self.SHADOW_SIZE)
        ctx.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
        show = self.mecha is not None and fd.show_mecha
        if show:
            self._draw_mecha(self.vao_mecha_depth, self.p_depth, to_gl(SVP), bones_bytes)
        prop_bytes = self._prop_bytes(fd.props) if (fd.props and hasattr(self, "vao_prop")) else []
        for pb in prop_bytes:
            self.p_depth["u_bones"].write(pb)
            self.vao_prop_depth.render(moderngl.TRIANGLES)

        shadow_vp_bytes = to_gl(SVP)

        def setup_mecha_prog(use_ao, vp, cam, clip):
            p = self.p_mecha
            self._common_lights(p, cam)
            self.tex_ao2.use(2)
            _set(p, u_vp=to_gl(vp), u_shadow_vp=shadow_vp_bytes, u_ao=2,
                 u_res=(float(self.IW), float(self.IH)) if not clip else (float(self.HW), float(self.HH)),
                 u_use_ao=1.0 if use_ao else 0.0, u_emis_boost=fd.emis_boost, u_clip_y=0.0 if clip else -100.0,
                 u_charge=fd.charge, u_wear=fd.wear, u_time=fd.time)
            p["u_albedo"].write(self._albedo_bytes)
            p["u_params"].write(self._params_bytes)
            p["u_bones"].write(bones_bytes)
            p["u_bone_fx"].write(fx_bytes)
            _set(p, u_fx_col=tuple(self.fx_col))

        # -------- planar reflection ------------------------------------------
        R = reflect_y()
        VPr = VP @ R
        eye_r = eye * np.array([1, -1, 1], F32)
        self.fbo_refl.use()
        ctx.viewport = (0, 0, self.HW, self.HH)
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        _set(self.p_bg, u_inv_vp=to_gl(np.linalg.inv(VPr)), u_cam=tuple(eye_r), u_time=fd.time, u_glow=1.0)
        self._fs(self.p_bg).render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)
        if show:
            ctx.front_face = "cw"
            ctx.enable(moderngl.CULL_FACE)
            setup_mecha_prog(False, VPr, eye_r, True)
            self.vao_mecha.render(moderngl.TRIANGLES)
            for pb, pr in zip(prop_bytes, fd.props):
                self.p_mecha["u_bones"].write(pb)
                _set(self.p_mecha, u_emis_boost=fd.emis_boost + 3.0 * pr["flash"])
                self.vao_prop.render(moderngl.TRIANGLES)
            ctx.front_face = "ccw"
        self._draw_fx(fd, VPr, eye, reflect=True)

        # -------- normal/depth pre-pass + SSAO ---------------------------------
        self.fbo_g.use()
        ctx.viewport = (0, 0, self.HW, self.HH)
        ctx.clear(0.5, 0.5, 1.0, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        _set(self.p_floor_g, u_vp=to_gl(VP), u_view=to_gl(V))
        self.vao_floor_g.render(moderngl.TRIANGLE_STRIP)
        if show:
            ctx.enable(moderngl.CULL_FACE)
            _set(self.p_gbuf, u_vp=to_gl(VP), u_view=to_gl(V), u_clip_y=-100.0)
            self.p_gbuf["u_bones"].write(bones_bytes)
            self.vao_mecha_g.render(moderngl.TRIANGLES)
            for pb in prop_bytes:
                self.p_gbuf["u_bones"].write(pb)
                self.vao_prop_g.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        use_ao = self.quality != "low"
        if use_ao:
            self.fbo_ao.use()
            ctx.viewport = (0, 0, self.HW, self.HH)
            self.tex_gd.use(0)
            self.tex_gn.use(1)
            _set(self.p_ssao, u_depth=0, u_normal=1, u_proj=to_gl(P), u_inv_proj=to_gl(np.linalg.inv(P)),
                 u_radius=0.45, u_strength=1.25)
            self._fs(self.p_ssao).render(moderngl.TRIANGLES, vertices=3)
            self.fbo_ao2.use()
            self.tex_ao.use(0)
            self.tex_gd.use(1)
            _set(self.p_aoblur, u_ao=0, u_depth=1, u_texel=(1.0 / self.HW, 1.0 / self.HH))
            self._fs(self.p_aoblur).render(moderngl.TRIANGLES, vertices=3)

        # -------- HDR scene -------------------------------------------------------
        self.fbo_hdr.use()
        ctx.viewport = (0, 0, self.IW, self.IH)
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        _set(self.p_bg, u_inv_vp=to_gl(np.linalg.inv(VP)), u_cam=tuple(eye), u_time=fd.time, u_glow=1.0)
        self._fs(self.p_bg).render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)
        # floor
        pf = self.p_floor
        self._common_lights(pf, eye)
        self.tex_refl.use(3)
        self.tex_ao2.use(2)
        tgt = fd.target if fd.target is not None else np.zeros(3, F32)
        _set(pf, u_vp=to_gl(VP), u_shadow_vp=shadow_vp_bytes, u_refl=3, u_ao=2,
             u_res=(float(self.IW), float(self.IH)), u_use_ao=1.0 if use_ao else 0.0,
             u_focus=tuple(focus), u_accent=tuple(self.accent), u_target=tuple(tgt),
             u_target_on=1.0 if fd.target is not None else 0.0, u_time=fd.time, u_power=fd.power)
        self.vao_floor.render(moderngl.TRIANGLE_STRIP)
        # mecha
        if show:
            ctx.enable(moderngl.CULL_FACE)
            setup_mecha_prog(use_ao, VP, eye, False)
            self.vao_mecha.render(moderngl.TRIANGLES)
            for pb, pr in zip(prop_bytes, fd.props):
                self.p_mecha["u_bones"].write(pb)
                _set(self.p_mecha, u_emis_boost=fd.emis_boost + 3.0 * pr["flash"])
                self.vao_prop.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.CULL_FACE)
        # additive FX
        self._draw_fx(fd, VP, eye, reflect=False)

        # -------- bloom ----------------------------------------------------------------
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        src = self.tex_hdr
        sw, sh = self.IW, self.IH
        for i, (tx, fb, w, h) in enumerate(self.bloom):
            fb.use()
            ctx.viewport = (0, 0, w, h)
            src.use(0)
            _set(self.p_bdown, u_src=0, u_texel=(1.0 / sw, 1.0 / sh), u_prefilter=1.0 if i == 0 else 0.0,
                 u_threshold=1.0)
            self._fs(self.p_bdown).render(moderngl.TRIANGLES, vertices=3)
            src, sw, sh = tx, w, h
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.bloom) - 2, -1, -1):
            tx, fb, w, h = self.bloom[i]
            lo = self.bloom[i + 1]
            fb.use()
            ctx.viewport = (0, 0, w, h)
            lo[0].use(0)
            _set(self.p_bup, u_src=0, u_texel=(1.0 / lo[2], 1.0 / lo[3]), u_weight=1.0)
            self._fs(self.p_bup).render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)

        # -------- composite -> FXAA -> screen -------------------------------------------------
        self.fbo_ldr.use()
        ctx.viewport = (0, 0, self.IW, self.IH)
        self.tex_hdr.use(0)
        self.bloom[0][0].use(1)
        _set(self.p_comp, u_hdr=0, u_bloom=1, u_bloom_amt=fd.bloom, u_exposure=fd.exposure, u_time=fd.time,
             u_aberr=fd.aberr, u_flash=fd.flash, u_flash_col=tuple(fd.flash_col), u_vignette=0.55)
        self._fs(self.p_comp).render(moderngl.TRIANGLES, vertices=3)
        out = self.out_fbo if self.out_fbo is not None else ctx.screen
        out.use()
        ctx.viewport = (0, 0, self.W, self.H)
        self.tex_ldr.use(0)
        _set(self.p_fxaa, u_src=0, u_texel=(1.0 / self.IW, 1.0 / self.IH))
        self._fs(self.p_fxaa).render(moderngl.TRIANGLES, vertices=3)

    # ------------------------------------------------------------------ FX
    def _draw_fx(self, fd, vp, eye, reflect):
        ctx = self.ctx
        has = (fd.sprites is not None and len(fd.sprites)) or fd.beams or fd.rings or fd.shield
        if not has:
            return
        fbo = self.fbo_refl if reflect else self.fbo_hdr
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        fbo.depth_mask = False
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        vpb = to_gl(vp)
        view = fd.view
        right = view[0, :3].astype(F32)
        up = view[1, :3].astype(F32)
        for r in fd.rings:
            _set(self.p_ring, u_vp=vpb, u_center=tuple(r["center"]), u_extent=r["radius"] + r["thick"] * 3.0,
                 u_radius=r["radius"], u_thick=r["thick"], u_col=tuple(r["col"]), u_int=r["int"],
                 u_fill=r.get("fill", 0.0), u_time=fd.time)
            self.vao_ring.render(moderngl.TRIANGLE_STRIP)
        for b in fd.beams:
            _set(self.p_beam, u_vp=vpb, u_p0=tuple(b["p0"]), u_p1=tuple(b["p1"]), u_w0=b["w0"], u_w1=b["w1"],
                 u_cam=tuple(eye), u_col=tuple(b["col"]), u_core=tuple(b["core"]), u_int=b["int"],
                 u_time=fd.time, u_mode=float(b.get("mode", 0)),
                 u_len=float(np.linalg.norm(np.asarray(b["p1"]) - np.asarray(b["p0"]))))
            self.vao_beam.render(moderngl.TRIANGLE_STRIP)
        if fd.shield is not None:
            sh = fd.shield
            m = np.eye(4, dtype=F32)
            m[:3, :3] *= sh["radius"]
            m[:3, 3] = sh["center"]
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            _set(self.p_shield, u_vp=vpb, u_model=to_gl(m), u_cam=tuple(eye), u_col=tuple(sh["col"]),
                 u_int=sh["int"], u_time=fd.time)
            hits = np.zeros((4, 4), F32)
            hits[:, 3] = 9.0
            if sh.get("hits") is not None:
                for i, h in enumerate(sh["hits"][:4]):
                    hits[i] = h
            self.p_shield["u_hit"].write(hits.tobytes())
            self.vao_sph.render(moderngl.TRIANGLES)
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        if fd.sprites is not None and len(fd.sprites):
            n = min(len(fd.sprites), self.max_sprites)
            data = np.ascontiguousarray(fd.sprites[:n], F32)
            self.vbo_spr.write(data.tobytes())
            self.tex_gd.use(6)
            _set(self.p_sprite, u_vp=vpb, u_right=right, u_up=up, u_depth=6, u_res=(float(self.IW), float(self.IH)),
                 u_nf=(0.1, 400.0), u_soft=0.0 if reflect else 1.0)
            self.vao_spr.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        fbo.depth_mask = True
        ctx.disable(moderngl.BLEND)

    # ------------------------------------------------------------------ HUD
    def make_texture(self, w, h):
        t = self.ctx.texture((w, h), 4)
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t

    def draw_hud(self, tex, x, y, w, h, alpha=1.0, flip=False):
        """x,y,w,h in framebuffer pixels, origin top-left."""
        ctx = self.ctx
        out = self.out_fbo if self.out_fbo is not None else ctx.screen
        out.use()
        ctx.viewport = (0, 0, self.W, self.H)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.DEPTH_TEST)
        tex.use(0)
        nx = x / self.W * 2 - 1
        ny = 1 - (y + h) / self.H * 2
        _set(self.p_hud, u_rect=(nx, ny, w / self.W * 2, h / self.H * 2), u_tex=0, u_alpha=alpha, u_flip=1.0 if flip else 0.0)
        self.vao_hud.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
