"""Material slots and colour themes. The mecha mesh stores only a material id per vertex."""

import numpy as np

# material slot ids
ARMOR_A, ARMOR_B, ARMOR_C, GUNMETAL, CHROME, ACCENT, RUBBER, GLOW_A, GLOW_B, VISOR, CARBON, STEEL, LAMP, HOT = range(14)
N_MATERIALS = 16

# per-slot (metal, rough, emissive, coat) - theme independent except where noted
_PARAMS = {
    ARMOR_A: (0.00, 0.34, 0.0, 0.85),
    ARMOR_B: (0.00, 0.30, 0.0, 0.90),
    ARMOR_C: (0.05, 0.40, 0.0, 0.70),
    GUNMETAL: (1.00, 0.38, 0.0, 0.0),
    CHROME: (1.00, 0.10, 0.0, 0.0),
    ACCENT: (1.00, 0.24, 0.0, 0.0),
    RUBBER: (0.00, 0.85, 0.0, 0.0),
    GLOW_A: (0.00, 0.30, 2.6, 0.0),
    GLOW_B: (0.00, 0.30, 2.4, 0.0),
    VISOR: (0.20, 0.06, 0.0, 0.6),
    CARBON: (0.55, 0.30, 0.0, 0.4),
    STEEL: (1.00, 0.28, 0.0, 0.0),
    LAMP: (0.00, 0.30, 3.5, 0.0),
    HOT: (0.00, 0.30, 4.0, 0.0),
}

THEMES = {
    "ONI": dict(
        label="ONI // BLACK-BLOOD-GOLD",
        A=(0.030, 0.032, 0.040), B=(0.42, 0.008, 0.012), C=(0.085, 0.088, 0.096),
        accent=(1.00, 0.70, 0.20), glowA=(1.00, 0.08, 0.04), glowB=(1.00, 0.42, 0.06),
        floor=(0.95, 0.12, 0.07), hud=(1.00, 0.30, 0.20),
    ),
    "SHOGUN": dict(
        label="SHOGUN // WHITE-CRIMSON-GOLD",
        A=(0.60, 0.61, 0.64), B=(0.42, 0.008, 0.012), C=(0.08, 0.082, 0.09),
        accent=(1.00, 0.72, 0.24), glowA=(1.00, 0.10, 0.05), glowB=(1.00, 0.50, 0.10),
        floor=(0.12, 0.60, 0.85), hud=(0.30, 0.90, 1.00),
    ),
    "NIGHT": dict(
        label="NIGHT // STEALTH-NEON",
        A=(0.012, 0.012, 0.015), B=(0.16, 0.004, 0.006), C=(0.05, 0.052, 0.058),
        accent=(0.55, 0.06, 0.05), glowA=(1.00, 0.06, 0.12), glowB=(1.00, 0.50, 0.10),
        floor=(0.85, 0.06, 0.18), hud=(1.00, 0.35, 0.45),
    ),
}
THEME_ORDER = ["ONI", "SHOGUN", "NIGHT"]


def theme_arrays(name):
    t = THEMES[name]
    albedo = np.zeros((N_MATERIALS, 3), np.float32)
    params = np.zeros((N_MATERIALS, 4), np.float32)
    for k, v in _PARAMS.items():
        params[k] = v
    albedo[ARMOR_A] = t["A"]
    albedo[ARMOR_B] = t["B"]
    albedo[ARMOR_C] = t["C"]
    albedo[GUNMETAL] = (0.085, 0.09, 0.10)
    albedo[CHROME] = (0.92, 0.93, 0.95)
    albedo[ACCENT] = t["accent"]
    albedo[RUBBER] = (0.012, 0.012, 0.014)
    albedo[GLOW_A] = t["glowA"]
    albedo[GLOW_B] = t["glowB"]
    albedo[VISOR] = (0.012, 0.016, 0.022)
    albedo[CARBON] = (0.03, 0.032, 0.036)
    albedo[STEEL] = (0.62, 0.64, 0.67)
    albedo[LAMP] = (1.0, 0.96, 0.9)
    albedo[HOT] = (1.0, 0.55, 0.18)
    if name == "ONI":                     # deep glossy lacquer
        params[ARMOR_A] = (0.0, 0.27, 0.0, 0.55)
        params[ARMOR_B] = (0.0, 0.20, 0.0, 0.85)
    elif name == "NIGHT":                 # matte stealth coating
        params[ARMOR_A] = (0.25, 0.46, 0.0, 0.45)
        params[ARMOR_B] = (0.0, 0.40, 0.0, 0.5)
    return albedo, params
