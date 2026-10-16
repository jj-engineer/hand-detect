"""
Procedural hard-surface geometry kit.

Everything returns a `Mesh` (positions, smooth/analytic normals, triangle indices).
Primitives are built around the origin; lofts/lathes run along +Y, extrusions along Z.
"""

import numpy as np

F32 = np.float32


class Mesh:
    def __init__(self, pos, nrm, idx):
        self.pos = np.ascontiguousarray(pos, F32)
        self.nrm = np.ascontiguousarray(nrm, F32)
        self.idx = np.ascontiguousarray(idx, np.uint32).reshape(-1, 3)

    def transformed(self, M):
        M = np.asarray(M, F32)
        R = M[:3, :3]
        pos = self.pos @ R.T + M[:3, 3]
        try:
            N = np.linalg.inv(R).T
        except np.linalg.LinAlgError:
            N = R
        nrm = self.nrm @ N.T
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
        idx = self.idx[:, ::-1] if np.linalg.det(R) < 0 else self.idx
        return Mesh(pos, nrm, idx)

    def mirrored_x(self):
        s = np.array([-1, 1, 1], F32)
        return Mesh(self.pos * s, self.nrm * s, self.idx[:, ::-1])


def merge(meshes):
    meshes = [m for m in meshes if m is not None and len(m.pos)]
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    pos, nrm, idx, off = [], [], [], 0
    for m in meshes:
        pos.append(m.pos)
        nrm.append(m.nrm)
        idx.append(m.idx + off)
        off += len(m.pos)
    return Mesh(np.concatenate(pos), np.concatenate(nrm), np.concatenate(idx))


def compute_normals(pos, idx):
    pos = np.asarray(pos, np.float64)
    v0, v1, v2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area weighted
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, idx[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 1.0, 0.0]))
    return n.astype(F32)


# --------------------------------------------------------------------------- loft
def _superellipse(n, seg):
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / n
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e


def loft(sections, seg=28, cap0=True, cap1=True):
    """
    Sweep superellipse cross-sections along Y.
    section = (y, half_x, half_z, exponent[, cx, cz]); exponent 2 = ellipse, 4+ = rounded box.
    """
    if sections[0][0] > sections[-1][0]:      # keep rings ascending so faces point outward
        sections = list(reversed(sections))
        cap0, cap1 = cap1, cap0
    R = len(sections)
    rings = []
    for sec in sections:
        y, hx, hz, n = sec[:4]
        cx = sec[4] if len(sec) > 4 else 0.0
        cz = sec[5] if len(sec) > 5 else 0.0
        ux, uz = _superellipse(n, seg)
        rings.append(np.stack([cx + hx * ux, np.full(seg, y), cz + hz * uz], 1))
    pos = np.concatenate(rings, 0)
    i = np.arange(seg)
    j = (i + 1) % seg
    faces = []
    for r in range(R - 1):
        a, b = r * seg + i, r * seg + j
        c, d = (r + 1) * seg + i, (r + 1) * seg + j
        faces.append(np.stack([a, c, b], 1))
        faces.append(np.stack([b, c, d], 1))
    idx = np.concatenate(faces, 0)
    nrm = compute_normals(pos, idx)
    meshes = [Mesh(pos, nrm, idx)]

    def cap(ring, up):
        c = ring.mean(0)
        if np.abs(ring - c).max() < 1e-4:
            return None
        p = np.concatenate([ring, c[None]], 0)
        k = len(ring)
        t = []
        for q in range(k):
            q2 = (q + 1) % k
            t.append([k, q2, q] if up else [k, q, q2])
        n = np.tile(np.array([0, 1 if up else -1, 0], F32), (k + 1, 1))
        return Mesh(p, n, np.array(t))

    if cap0:
        meshes.append(cap(rings[0], False))
    if cap1:
        meshes.append(cap(rings[-1], True))
    return merge(meshes)


# --------------------------------------------------------------------------- lathe
def lathe(profile, seg=36, crease_deg=38.0):
    """Revolve a (radius, y) profile (bottom -> top, outside) around Y. Auto-creases sharp corners."""
    prof = np.asarray(profile, np.float64)
    K = len(prof)
    segn = np.zeros((K - 1, 2))
    valid = np.zeros(K - 1, bool)
    for k in range(K - 1):
        dr, dy = prof[k + 1] - prof[k]
        L = np.hypot(dr, dy)
        if L > 1e-9:
            segn[k] = (dy / L, -dr / L)
            valid[k] = True
    cos_crease = np.cos(np.radians(crease_deg))
    in_e, out_e = {}, {}
    entries = []  # (r, y, nr, ny)
    for k in range(K):
        n_in = segn[k - 1] if k > 0 and valid[k - 1] else None
        n_out = segn[k] if k < K - 1 and valid[k] else None
        r, y = prof[k]
        if n_in is not None and n_out is not None:
            if np.dot(n_in, n_out) < cos_crease:
                entries.append((r, y, *n_in))
                in_e[k] = len(entries) - 1
                entries.append((r, y, *n_out))
                out_e[k] = len(entries) - 1
            else:
                n = n_in + n_out
                n = n / (np.linalg.norm(n) + 1e-12)
                entries.append((r, y, *n))
                in_e[k] = out_e[k] = len(entries) - 1
        else:
            n = n_in if n_in is not None else n_out
            if n is None:
                n = np.array([1.0, 0.0])
            entries.append((r, y, *n))
            in_e[k] = out_e[k] = len(entries) - 1
    phi = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    cp, sp = np.cos(phi), np.sin(phi)
    pos, nrm = [], []
    for (r, y, nr, ny) in entries:
        pos.append(np.stack([r * cp, np.full(seg, y), r * sp], 1))
        nrm.append(np.stack([nr * cp, np.full(seg, ny), nr * sp], 1))
    pos = np.concatenate(pos)
    nrm = np.concatenate(nrm)
    i = np.arange(seg)
    j = (i + 1) % seg
    faces = []
    for k in range(K - 1):
        if not valid[k]:
            continue
        A, C = out_e[k], in_e[k + 1]
        a, b = A * seg + i, A * seg + j
        c, d = C * seg + i, C * seg + j
        faces.append(np.stack([a, c, b], 1))
        faces.append(np.stack([b, c, d], 1))
    return Mesh(pos, nrm, np.concatenate(faces, 0))


def cylinder(r, h, seg=28, bevel=0.0, r_top=None, y0=0.0):
    """Capped cylinder/cone along Y from y0 to y0+h with optional bevel on both rims."""
    rt = r if r_top is None else r_top
    b = min(bevel, r * 0.45, rt * 0.45, h * 0.45)
    p = [(0, y0)]
    if b > 1e-5:
        for a in np.linspace(0, np.pi / 2, 4):
            p.append((r - b + b * np.sin(a) * (r / max(r, 1e-6)), y0 + b - b * np.cos(a)))
        for a in np.linspace(0, np.pi / 2, 4):
            p.append((rt - b + b * np.cos(a), y0 + h - b + b * np.sin(a)))
    else:
        p += [(r, y0), (rt, y0 + h)]
    p.append((0, y0 + h))
    return lathe(p, seg)


def sphere(r, seg=28, rings=16):
    prof = [(r * np.sin(t), -r * np.cos(t)) for t in np.linspace(0, np.pi, rings + 1)]
    prof[0] = (0.0, -r)
    prof[-1] = (0.0, r)
    return lathe(prof, seg, crease_deg=89)


def torus(R, r, seg=36, seg_minor=12, squash=1.0):
    th = np.linspace(-np.pi / 2, 3 * np.pi / 2, seg_minor + 1)
    prof = [(R + r * np.cos(t), r * squash * np.sin(t)) for t in th[:-1]]
    prof.append(prof[0])
    return lathe(prof, seg, crease_deg=89)


# --------------------------------------------------------------------------- rounded box
def rbox(sx, sy, sz, r=0.02, k=3):
    """Rounded box centred on the origin with analytic normals (hard-surface bevels)."""
    h = np.array([sx, sy, sz], np.float64) / 2.0
    r = float(max(min(r, h.min() * 0.999), 1e-4))
    inner = h - r

    def axis_vals(hh):
        i = hh - r
        return np.concatenate([np.linspace(-hh, -i, k + 1), np.linspace(i, hh, k + 1)])

    vals = [axis_vals(h[a]) for a in range(3)]
    meshes = []
    for axis in range(3):
        a1, a2 = [x for x in range(3) if x != axis][:2]
        for sgn in (-1, 1):
            u, v = np.meshgrid(vals[a1], vals[a2], indexing="xy")
            q = np.zeros(u.shape + (3,))
            q[..., a1], q[..., a2], q[..., axis] = u, v, sgn * h[axis]
            pin = np.clip(q, -inner, inner)
            d = q - pin
            n = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
            p = pin + r * n
            H, W = u.shape
            gi = np.arange(H * W).reshape(H, W)
            tri = []
            a, b, c, e = gi[:-1, :-1], gi[:-1, 1:], gi[1:, 1:], gi[1:, :-1]
            tri.append(np.stack([a, b, c], -1).reshape(-1, 3))
            tri.append(np.stack([a, c, e], -1).reshape(-1, 3))
            tri = np.concatenate(tri)
            p2 = p.reshape(-1, 3)
            fn = np.cross(p2[tri[0, 1]] - p2[tri[0, 0]], p2[tri[0, 2]] - p2[tri[0, 0]])
            expect = np.zeros(3)
            expect[axis] = sgn
            if np.dot(fn, expect) < 0:
                tri = tri[:, ::-1]
            meshes.append(Mesh(p2, n.reshape(-1, 3), tri))
    return merge(meshes)


def box(sx, sy, sz):
    return rbox(sx, sy, sz, r=min(sx, sy, sz) * 0.04 + 0.002, k=1)


# --------------------------------------------------------------------------- extrusion
def _poly_ccw(P):
    P = np.asarray(P, np.float64)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area > 0 else P[::-1].copy()


def offset_poly(P, d):
    """Inward miter offset of a convex CCW polygon."""
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % n]
        e1 = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-12)
        e2 = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-12)
        n1 = np.array([-e1[1], e1[0]])
        n2 = np.array([-e2[1], e2[0]])
        m = (n1 + n2) / (1.0 + float(np.dot(n1, n2)) + 1e-9)
        out.append(p1 + d * m)
    return np.array(out)


def extrude(poly, depth, bevel=0.02, steps=3):
    """
    Extrude a convex polygon (XY plane) along Z, centred on z=0, with a rounded bevel.
    Each polygon edge keeps its own normals so the plate corners stay crisp.
    """
    P = _poly_ccw(poly)
    n = len(P)
    h = depth / 2.0
    b = min(bevel, h * 0.95)
    # polygon inradius guard
    c = P.mean(0)
    b = min(b, 0.45 * min(np.linalg.norm(P[i] - c) for i in range(n)))
    levels = []  # (inset, z, phi)
    for j in range(steps + 1):
        phi = (j / steps) * (np.pi / 2)
        levels.append((b * (1 - np.sin(phi)), (h - b) + b * np.cos(phi), phi, +1))
    back = [(d, -z, phi, -1) for (d, z, phi, _) in reversed(levels)]
    levels = levels + back
    # edge outward normals
    edge_out = []
    for i in range(n):
        e = P[(i + 1) % n] - P[i]
        e = e / (np.linalg.norm(e) + 1e-12)
        edge_out.append(np.array([e[1], -e[0]]))
    offs = {}
    for (d, z, phi, s) in levels:
        key = round(d, 8)
        if key not in offs:
            offs[key] = offset_poly(P, d + 0.0)
    meshes = []
    for i in range(n):
        i2 = (i + 1) % n
        pos, nrm = [], []
        for (d, z, phi, s) in levels:
            O = offs[round(d, 8)]
            no = edge_out[i]
            nz = np.cos(phi) * s
            nn = np.array([no[0] * np.sin(phi), no[1] * np.sin(phi), nz])
            for q in (i, i2):
                pos.append([O[q][0], O[q][1], z])
                nrm.append(nn)
        pos = np.array(pos)
        nrm = np.array(nrm)
        tri = []
        for L in range(len(levels) - 1):
            a, bb, cc, dd = 2 * L, 2 * L + 1, 2 * L + 2, 2 * L + 3
            tri += [[a, bb, dd], [a, dd, cc]]
        tri = np.array(tri)
        fn = np.cross(pos[tri[0, 1]] - pos[tri[0, 0]], pos[tri[0, 2]] - pos[tri[0, 0]])
        # tri[0] spans the front bevel start: expected normal ~ z + out; just test against out
        if np.dot(fn, np.array([edge_out[i][0], edge_out[i][1], 0.4])) < 0:
            tri = tri[:, ::-1]
        meshes.append(Mesh(pos, nrm, tri))
    Pin = offset_poly(P, b)
    for s in (+1, -1):
        pos = np.concatenate([Pin, Pin.mean(0, keepdims=True)])
        pos = np.concatenate([pos, np.full((len(pos), 1), s * h)], 1)
        nrm = np.tile(np.array([0, 0, s], F32), (len(pos), 1))
        k = len(Pin)
        tri = [[k, q, (q + 1) % k] if s > 0 else [k, (q + 1) % k, q] for q in range(k)]
        meshes.append(Mesh(pos, nrm, np.array(tri)))
    return merge(meshes)


def plate(points, depth, bevel=0.02, steps=3):
    return extrude(points, depth, bevel, steps)


# --------------------------------------------------------------------------- tubes
def bezier(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = [np.asarray(p, np.float64) for p in (p0, p1, p2, p3)]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def tube(points, radius, seg=10):
    pts = np.asarray(points, np.float64)
    N = len(pts)
    rad = np.full(N, radius) if np.isscalar(radius) else np.asarray(radius, np.float64)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    ref = np.array([0.0, 1.0, 0.0]) if abs(tang[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nrm0 = np.cross(tang[0], ref)
    nrm0 /= np.linalg.norm(nrm0)
    frames = []
    nn = nrm0
    for i in range(N):
        t = tang[i]
        nn = nn - np.dot(nn, t) * t
        nn /= np.linalg.norm(nn) + 1e-12
        bb = np.cross(t, nn)
        frames.append((nn, bb))
    phi = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    rings = []
    for i in range(N):
        nn, bb = frames[i]
        rings.append(pts[i] + rad[i] * (np.cos(phi)[:, None] * nn + np.sin(phi)[:, None] * bb))
    pos = np.concatenate(rings)
    ii = np.arange(seg)
    jj = (ii + 1) % seg
    faces = []
    for r in range(N - 1):
        a, b = r * seg + ii, r * seg + jj
        c, d = (r + 1) * seg + ii, (r + 1) * seg + jj
        faces.append(np.stack([a, b, c], 1))
        faces.append(np.stack([b, d, c], 1))
    idx = np.concatenate(faces)
    nrm = compute_normals(pos, idx)
    m = Mesh(pos, nrm, idx)
    # make sure winding is outward
    c = pos.mean(0)
    if np.mean(np.sum(m.nrm * (pos - c), 1)) < 0:
        m = Mesh(pos, -nrm, idx[:, ::-1])
    return m


def disc_ring(r_in, r_out, h, seg=32, bevel=0.0):
    """Flat annulus (washer / bezel) along Y."""
    b = min(bevel, (r_out - r_in) * 0.4, h * 0.4)
    if b > 1e-5:
        p = [(r_in, 0), (r_in + b, 0), (r_out - b, 0), (r_out, b), (r_out, h - b), (r_out - b, h),
             (r_in + b, h), (r_in, h - b), (r_in, b), (r_in + b, 0)]
        p = [(r_in + b, 0), (r_out - b, 0), (r_out, b), (r_out, h - b), (r_out - b, h), (r_in + b, h),
             (r_in, h - b), (r_in, b), (r_in + b, 0)]
    else:
        p = [(r_in, 0), (r_out, 0), (r_out, h), (r_in, h), (r_in, 0)]
    # annulus is a closed loop profile; orient so normals point outward
    return lathe(p, seg, crease_deg=45)
