"""
Synthetic training views for Ridgeline's skyline finder.

A view is rendered from eye level at a real place: open elevation data (AWS Terrain Tiles) for
the shape, Esri satellite imagery for the ground colours, true perspective and Earth curvature.
The exact skyline (top-most ground pixel in each column) and the distance of every pixel are
known.  `weather()` then turns one rendered view into many photos: haze by the physical
scattering formula I = J*t + A*(1 - t), t = exp(-distance / L), with visibilities from a few km
to clear air; clear, hazy, overcast and sunset skies; clouds; exposure, colour cast, blur,
sensor noise, JPEG, and black-and-white.
"""
import io
import math
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, 'ridgeline')
import oblique as ob          # noqa: E402
import search as se           # noqa: E402
import terrain as tr          # noqa: E402

R = tr.R_EARTH
K_REFR = tr.K_REFRACTION


def render_view(lat, lon, heading, fov, pitch_deg, W=900, H=600, eye=1.7, max_km=120.0, base='Satellite photo'):
    """Albedo image (H x W x 3, 0-1), distance (m, inf = sky) and skyline row per column."""
    A = se.AreaTerrain(lat, lon, 0.1, max_km + 1.0, se.BANDS_FINE)
    g = float(A.ground(np.array([lat]), np.array([lon]))[0])
    alt = g + eye
    f = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
    cy = H / 2.0
    cols = np.arange(W) + 0.5
    a_rel = np.arctan((cols - W / 2.0) / f)
    az = (heading + np.degrees(a_rel)) % 360.0
    K = 1400
    d = np.geomspace(8.0, max_km * 1000.0, K)
    la, lo = tr.destination(lat, lon, az[:, None], d[None, :])
    h = ob._dem(A, la, lo, np.broadcast_to(d, la.shape))
    drop = d ** 2 * (1.0 - K_REFR) / (2.0 * R)
    e = np.arctan2(h - alt - drop[None, :], d[None, :])
    p = math.radians(pitch_deg)
    y_ = np.sin(e)
    z_ = np.cos(e) * np.cos(a_rel[:, None])
    yc = y_ * math.cos(p) - z_ * math.sin(p)
    zc = y_ * math.sin(p) + z_ * math.cos(p)
    v = np.where(zc > 1e-6, cy - f * yc / np.maximum(zc, 1e-6), H * 4)
    M = np.minimum.accumulate(v, axis=1)
    rows = np.arange(H) + 0.5
    kstar = np.empty((H, W), np.int32)
    for i in range(W):
        kstar[:, i] = np.searchsorted(-M[i], -rows, side='left')
    ground = kstar < K
    ks = np.clip(kstar, 1, K - 1)
    ci = np.broadcast_to(np.arange(W)[None, :], ks.shape)
    m_prev = M[ci, ks - 1]
    v_k = v[ci, ks]
    t = np.clip((m_prev - rows[:, None]) / np.maximum(m_prev - v_k, 1e-6), 0.0, 1.0)
    dpx = np.where(ground, d[ks - 1] * (1 - t) + d[ks] * t, np.inf)
    lat_px = la[ci, ks - 1] * (1 - t) + la[ci, ks] * t
    lon_px = lo[ci, ks - 1] * (1 - t) + lo[ci, ks] * t
    rgb = np.zeros((H, W, 3), np.float32)
    for d0, d1, z, mt in ((0, 1500, 16, 40), (1500, 8000, 14, 48), (8000, 30000, 12, 40), (30000, 1e9, 10, 30)):
        msk = ground & (dpx >= d0) & (dpx < d1)
        if not msk.any():
            continue
        L, O = lat_px[msk], lon_px[msk]
        im = ob.Imagery(base, z, float(L.min()) - 0.002, float(L.max()) + 0.002, float(O.min()) - 0.002,
                        float(O.max()) + 0.002, max_tiles=mt)
        rgb[msk] = im.sample(L, O)
    has = ground.any(axis=0)
    top = np.where(has, ground.argmax(axis=0), H).astype(float)   # top-most ground row; H = all sky
    return dict(albedo=rgb, dist=dpx.astype(np.float32), ground=ground, skyline=top, size=(W, H))


def _fractal(rng, H, W, octaves=5, base=6):
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.normal(size=(max(2, int(n * H / W)) + 1, n + 1)).astype(np.float32)
        out += amp * np.asarray(Image.fromarray(g).resize((W, H), Image.BICUBIC))
        tot += amp
        amp *= 0.55
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-6)


SKIES = {   # (zenith, horizon) colours
    'clear': ((0.18, 0.36, 0.72), (0.70, 0.80, 0.92)),
    'hazy': ((0.45, 0.58, 0.78), (0.85, 0.88, 0.92)),
    'overcast': ((0.72, 0.74, 0.77), (0.86, 0.87, 0.88)),
    'white': ((0.90, 0.91, 0.92), (0.95, 0.95, 0.95)),
    'sunset': ((0.25, 0.30, 0.55), (0.98, 0.70, 0.45)),
    'dusk': ((0.10, 0.12, 0.28), (0.75, 0.45, 0.42)),
}


def weather(view, rng, kind=None, vis_km=None, allow_bw=True):
    """One photo-like image (PIL RGB) from a rendered view, plus the settings used."""
    alb, dist, ground = view['albedo'], view['dist'], view['ground']
    H, W = ground.shape
    kind = kind or str(rng.choice(list(SKIES), p=[0.28, 0.24, 0.17, 0.09, 0.12, 0.10]))
    zen, hor = (np.array(c, np.float32) for c in SKIES[kind])
    jitter = rng.normal(0, 0.035, 3).astype(np.float32)
    zen = np.clip(zen + jitter, 0, 1)
    hor = np.clip(hor + jitter * 0.6, 0, 1)
    yy = (np.arange(H)[:, None] / H).astype(np.float32)
    sk = view['skyline']
    hz_row = float(np.mean(np.where(sk < H, sk, H * 0.6))) / H
    tt = np.clip(yy / max(0.15, hz_row), 0, 1) ** rng.uniform(0.6, 1.6)
    sky = zen[None, None, :] * (1 - tt[..., None]) + hor[None, None, :] * tt[..., None]
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    if rng.random() < 0.35:                                   # sun glow
        sx, sy = rng.uniform(0, W), rng.uniform(-0.2 * H, hz_row * H)
        r = np.hypot(np.arange(W)[None, :] - sx, np.arange(H)[:, None] - sy) / W
        glow = rng.uniform(0.1, 0.45) * np.exp(-r / rng.uniform(0.08, 0.3))
        sky += glow[..., None] * np.array([1.0, 0.92, 0.75], np.float32)
    if kind != 'white' and rng.random() < 0.6:               # clouds
        cov = rng.uniform(0.05, 0.7)
        n = _fractal(rng, H, W)
        thr = np.quantile(n, 1 - cov)
        cm = np.clip((n - thr) * rng.uniform(1.0, 3.0), 0, 1)
        cm *= np.clip(1.15 - yy / max(0.2, hz_row) * rng.uniform(0.2, 0.9), 0, 1)
        ccol = np.array([0.95, 0.95, 0.96], np.float32) * rng.uniform(0.75, 1.05)
        if kind in ('sunset', 'dusk'):
            ccol = ccol * np.array([1.0, 0.75, 0.65], np.float32)
        sky = sky * (1 - cm[..., None]) + ccol * cm[..., None]
    land = alb * rng.uniform(0.6, 1.4) * (1 + rng.normal(0, 0.06, 3)).astype(np.float32)
    if kind in ('sunset', 'dusk'):
        land = land * rng.uniform(0.35, 0.8) * np.array([1.0, 0.85, 0.75], np.float32)
    if rng.random() < 0.15:                                   # bright land: snow, pale rock, over-exposure
        land = 1 - (1 - np.clip(land, 0, 1)) * rng.uniform(0.3, 0.7)
    if vis_km is None:
        vis_km = float(np.exp(rng.uniform(math.log(4.0), math.log(160.0))))
        if kind in ('hazy', 'overcast', 'white'):
            vis_km = min(vis_km, float(np.exp(rng.uniform(math.log(3.0), math.log(60.0)))))
    Lm = vis_km * 1000.0 / 3.0                                # visibility = distance where contrast is 5 %
    tmap = np.exp(-np.where(ground, dist, 0.0) / Lm)[..., None].astype(np.float32)
    air = hor * rng.uniform(0.92, 1.06)
    land = land * tmap + air * (1 - tmap)
    img = np.where(ground[..., None], land, sky)
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.5:
        s = rng.uniform(0.3, 1.3)
        img = np.stack([ndi.gaussian_filter(img[..., k], s) for k in range(3)], -1)
    img = img + rng.normal(0, rng.uniform(0.0, 0.025), img.shape)
    bw = allow_bw and rng.random() < 0.3
    if bw:
        wts = rng.dirichlet([3, 5, 2])
        img = np.repeat((img * wts).sum(-1, keepdims=True), 3, -1)
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    b = io.BytesIO()
    out.save(b, format='JPEG', quality=int(rng.integers(55, 96)))
    out = Image.open(io.BytesIO(b.getvalue())).convert('RGB')
    if bw:
        out = out.convert('L').convert('RGB')
    return out, dict(kind=kind, vis_km=vis_km, bw=bool(bw))
