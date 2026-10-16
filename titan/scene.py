"""Glue: assemble a renderer FrameData from the mecha, FX and camera state."""

import numpy as np

from .renderer import FrameData


def build_frame(mecha, fx, cam, aspect, now, theme="ONI", target=None, extra_sprites=None, exposure=0.95, bloom=0.28, props=None):
    eye, V, P = cam.matrices(aspect)
    spr = fx.sprites()
    if extra_sprites is not None and len(extra_sprites):
        spr = extra_sprites if spr is None else np.concatenate([spr, extra_sprites], 0)
    beams = mecha.beams()
    fd = FrameData(
        view=V, proj=P, eye=eye, bones=mecha.skin,
        focus=np.array([mecha.pos[0], 0.0, mecha.pos[2]], np.float32),
        time=now, emis_boost=mecha.emissive_boost(), charge=mecha.charge, power=mecha.power_level(),
        target=target, sprites=spr, beams=beams, rings=fx.ring_draw_list(), shield=mecha.shield_draw(),
        flash=fx.flash, flash_col=fx.flash_col, aberr=fx.aberr, theme=theme, exposure=exposure, bloom=bloom,
        props=props or [],
    )
    return fd
