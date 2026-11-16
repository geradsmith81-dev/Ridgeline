"""
Ridgeline - reading the skyline (sky / land boundary) from a colour or black-and-white photograph.

Method
------
1. The photo is turned upright (EXIF orientation) and scaled to a working width of about 900 px.
2. Each pixel gets a "looks like sky" probability from a small two-class Gaussian classifier.  It is
   first trained on the top rows (assumed sky) and the bottom rows (assumed land), then retrained on
   the two sides of the first skyline found, so it adapts to each picture: blue or grey sky, snow,
   sunset glow, or a black-and-white print (brightness and texture only).
3. For every column the cost of putting the boundary at row y is
       (land-looking pixels above y) + (sky-looking pixels below y) - w * (edge strength at y),
   and one connected line across the picture is found by dynamic programming with a penalty
   proportional to each vertical step (an exact L1 distance transform, so cliffs are allowed).
4. Columns where the line touches the top or bottom of the frame, or where the boundary is weak,
   are flagged as uncertain; the user can redraw or switch off columns in the app.

Angles
------
With the horizontal field of view F, the focal length in pixels is f = (W/2) / tan(F/2).  A skyline
pixel (x, y) is at horizontal angle  atan((x - cx) / f)  from the image centre and at elevation
atan((cy - y) / sqrt(f^2 + (x - cx)^2))  above the camera's optical-axis level.  The unknown camera
tilt (pitch) and a small roll are solved for during matching.
"""
import io
import math

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

try:                                   # iPhone HEIC/HEIF pictures
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

WORK_WIDTH = 900
DIAG_35MM = math.hypot(36.0, 24.0)       # 43.27 mm


# ------------------------------------------------------------------------------------------
# loading and EXIF
# ------------------------------------------------------------------------------------------
def _ratio(v):
    try:
        return float(v)
    except Exception:
        try:
            return float(v[0]) / float(v[1])
        except Exception:
            return None


def _dms(v, ref):
    try:
        d, m, s = (_ratio(x) for x in v)
        out = d + m / 60.0 + s / 3600.0
        if str(ref).upper().startswith(('S', 'W')):
            out = -out
        return out
    except Exception:
        return None


def load_image(data):
    """bytes or file-like -> (upright RGB PIL image, info dict with EXIF facts)."""
    if isinstance(data, (bytes, bytearray)):
        data = io.BytesIO(data)
    im = Image.open(data)
    info = exif_info(im)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    info['is_color'] = is_color_image(im)
    im = im.convert('RGB')
    info['width'], info['height'] = im.size
    return im, info


def exif_info(im):
    out = dict(focal_mm=None, focal35=None, make=None, model=None, lat=None, lon=None, alt=None,
               heading=None, heading_ref=None, taken=None)
    try:
        ex = im.getexif()
    except Exception:
        return out
    if not ex:
        return out
    out['make'] = (str(ex.get(0x010F) or '')).strip('\x00 ') or None
    out['model'] = (str(ex.get(0x0110) or '')).strip('\x00 ') or None
    try:
        sub = ex.get_ifd(0x8769)
    except Exception:
        sub = {}
    if sub.get(0x920A) is not None:
        out['focal_mm'] = _ratio(sub.get(0x920A))
    f35 = sub.get(0xA405)
    try:
        out['focal35'] = float(f35) if f35 not in (None, 0) else None
    except Exception:
        out['focal35'] = None
    out['taken'] = sub.get(0x9003) or ex.get(0x0132)
    try:
        g = ex.get_ifd(0x8825)
    except Exception:
        g = {}
    if g:
        if 2 in g and 4 in g:
            out['lat'] = _dms(g[2], g.get(1, 'N'))
            out['lon'] = _dms(g[4], g.get(3, 'E'))
        if 6 in g:
            a = _ratio(g[6])
            if a is not None:
                out['alt'] = -a if g.get(5) in (1, b'\x01') else a
        if 17 in g:
            out['heading'] = _ratio(g[17])
            out['heading_ref'] = str(g.get(16, 'T'))
    return out


def is_color_image(im):
    if im.mode in ('L', '1', 'LA', 'I', 'I;16', 'F'):
        return False
    a = np.asarray(im.convert('RGB').resize((96, 96)), dtype=np.float32)
    sat = a.max(axis=2) - a.min(axis=2)
    return bool(np.percentile(sat, 90) > 12.0)


def hfov_from_exif(info, w, h):
    """Horizontal field of view (deg) from the 35 mm-equivalent focal length (CIPA: by the diagonal)."""
    f35 = info.get('focal35')
    if not f35 or f35 <= 0:
        return None
    half_diag = math.atan(DIAG_35MM / 2.0 / f35)
    t = math.tan(half_diag) * w / math.hypot(w, h)
    return math.degrees(2.0 * math.atan(t))


def hfov_from_focal35(f35, w, h):
    return hfov_from_exif({'focal35': f35}, w, h)


# ------------------------------------------------------------------------------------------
# features and classifier
# ------------------------------------------------------------------------------------------
def _features(rgb, color):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum_s = ndi.gaussian_filter(lum, 1.0)
    gx = ndi.sobel(lum_s, axis=1)
    gy = ndi.sobel(lum_s, axis=0)
    gm = np.hypot(gx, gy)
    tex = ndi.gaussian_filter(gm, 3.0)
    feats = [lum_s, np.log1p(20 * tex)]
    if color:
        s = r + g + b + 1e-3
        feats += [ndi.gaussian_filter(b / s, 1.5), ndi.gaussian_filter((r - b) / s, 1.5),
                  ndi.gaussian_filter(rgb.max(2) - rgb.min(2), 1.5)]
    return np.stack(feats, axis=-1), gy, gm


class _Gauss2:
    """Two-class Gaussian (quadratic) classifier on a few features."""

    def fit(self, Xs, Xg):
        self.p = []
        for X in (Xs, Xg):
            mu = X.mean(0)
            C = np.cov(X.T) + np.eye(X.shape[1]) * 1e-3
            self.p.append((mu, np.linalg.inv(C), np.linalg.slogdet(C)[1]))
        return self

    def prob_sky(self, F):
        sh = F.shape[:-1]
        X = F.reshape(-1, F.shape[-1])
        ll = []
        for mu, Ci, ld in self.p:
            d = X - mu
            ll.append(-0.5 * (np.einsum('ij,jk,ik->i', d, Ci, d) + ld))
        z = np.clip(ll[0] - ll[1], -30, 30)
        return (1.0 / (1.0 + np.exp(-z))).reshape(sh)


# ------------------------------------------------------------------------------------------
# dynamic programming
# ------------------------------------------------------------------------------------------
def _l1_dt(prev, gamma):
    """out[y] = min_y' prev[y'] + gamma*|y - y'|, and the arg-min y'.  Exact, O(H)."""
    n = prev.size
    idx = np.arange(n)
    # from above (y' <= y)
    a = prev - gamma * idx
    am = np.minimum.accumulate(a)
    aa = np.maximum.accumulate(np.where(a <= am, idx, 0))
    f = am + gamma * idx
    # from below (y' >= y)
    c = (prev + gamma * idx)[::-1]
    cm = np.minimum.accumulate(c)
    ca = np.maximum.accumulate(np.where(c <= cm, idx, 0))
    b = cm[::-1] - gamma * idx
    barg = (n - 1 - ca)[::-1]
    use_f = f <= b
    return np.where(use_f, f, b), np.where(use_f, aa, barg)


def _dp(cost, gamma):
    H, W = cost.shape
    acc = cost[:, 0].copy()
    back = np.zeros((H, W), dtype=np.int32)
    for x in range(1, W):
        m, arg = _l1_dt(acc, gamma)
        back[:, x] = arg
        acc = m + cost[:, x]
    y = np.zeros(W, dtype=np.int32)
    y[-1] = int(np.argmin(acc))
    for x in range(W - 1, 0, -1):
        y[x - 1] = back[y[x], x]
    return y


def _boundary_cost(ps, edge, w_edge):
    H = ps.shape[0]
    land_above = np.vstack([np.zeros((1, ps.shape[1])), np.cumsum(1.0 - ps, axis=0)[:-1]])
    sky_below = np.cumsum(ps[::-1], axis=0)[::-1]
    return (land_above + sky_below) / H - w_edge * edge


def detect(im, color=None, smooth=1.0, edge_weight=1.0, iterations=3, work_width=WORK_WIDTH):
    """Find the skyline.

    Returns dict with x, y in ORIGINAL image pixels (one value per working column), valid mask,
    conf (0-1), sky probability (working size), work_size and size."""
    W0, H0 = im.size
    scale = min(1.0, work_width / float(W0))
    w, h = max(32, int(round(W0 * scale))), max(32, int(round(H0 * scale)))
    rgb = np.asarray(im.convert('RGB').resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    if color is None:
        color = is_color_image(im)
    F, gy, gm = _features(rgb, color)
    nf = F.shape[-1]
    flat = F.reshape(-1, nf)
    F = (F - flat.mean(0)) / (flat.std(0) + 1e-6)
    # edge strength: favour "brighter above, darker below", but accept either sign
    up = np.clip(-gy, 0, None)              # sobel gy > 0 means brighter below
    edge = 0.7 * up + 0.3 * gm
    edge = np.clip(edge / (np.percentile(edge, 99.5) + 1e-6), 0, 1.5)

    top = max(2, int(0.08 * h))
    bot = max(2, int(0.12 * h))
    Xs = F[:top].reshape(-1, nf)
    Xg = F[-bot:].reshape(-1, nf)
    gamma = 0.004 * smooth
    rng = np.random.default_rng(0)
    rows = np.arange(h)[:, None]
    for it in range(max(1, iterations)):
        ps = ndi.gaussian_filter(_Gauss2().fit(Xs, Xg).prob_sky(F), 1.0)
        y = _dp(_boundary_cost(ps, edge, 0.35 * edge_weight), gamma)
        m = max(3, int(0.03 * h))
        sky_m = rows < (y[None, :] - m)
        gnd_m = rows > (y[None, :] + m)
        if sky_m.sum() < 200 or gnd_m.sum() < 200:
            break
        Xs, Xg = F[sky_m], F[gnd_m]
        if Xs.shape[0] > 60000:
            Xs = Xs[rng.choice(Xs.shape[0], 60000, replace=False)]
        if Xg.shape[0] > 60000:
            Xg = Xg[rng.choice(Xg.shape[0], 60000, replace=False)]

    cols = np.arange(w)
    e_at = edge[np.clip(y, 0, h - 1), cols]
    m = max(3, int(0.02 * h))
    cps = np.vstack([np.zeros((1, w)), np.cumsum(ps, axis=0)])        # column prefix sums

    def band(lo, hi):
        lo = np.clip(lo, 0, h); hi = np.clip(hi, 0, h)
        n = np.maximum(hi - lo, 1)
        return (cps[hi, cols] - cps[lo, cols]) / n, hi > lo

    above, ok_a = band(y - 3 * m, y - m)
    below, ok_b = band(y + m, y + 3 * m)
    above = np.where(ok_a, above, 0.0)
    below = np.where(ok_b, below, 1.0)
    contrast = np.clip(above - below, 0, 1)
    conf = np.clip(0.6 * contrast + 0.4 * np.clip(e_at, 0, 1), 0, 1)
    valid = (y > 2 * m + 1) & (y < h - 2) & (contrast > 0.15) & (conf > 0.3)
    sx = W0 / float(w)
    sy = H0 / float(h)
    return dict(x=(cols + 0.5) * sx, y=(y + 0.5) * sy, valid=valid, conf=conf, sky=ps,
                work_size=(w, h), size=(W0, H0), color=bool(color))


# ------------------------------------------------------------------------------------------
# angles
# ------------------------------------------------------------------------------------------
def to_angles(x, y, size, hfov_deg):
    """Pixel skyline -> (theta deg, right of centre positive; elev deg above optical-axis level)."""
    W, H = size
    f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = np.asarray(x, float) - W / 2.0
    v = H / 2.0 - np.asarray(y, float)
    th = np.degrees(np.arctan2(u, f))
    el = np.degrees(np.arctan2(v, np.hypot(f, u)))
    return th, el


def resample_profile(th, el, valid, step=0.1):
    """Even angular grid of the skyline; stretches of invalid columns stay NaN."""
    th = np.asarray(th, float)
    el = np.where(np.asarray(valid, bool), el, np.nan)
    g = np.arange(th.min(), th.max() + 1e-9, step)
    ok = np.isfinite(el)
    if ok.sum() < 2:
        return g, np.full(g.size, np.nan)
    out = np.interp(g, th[ok], el[ok])
    i1 = np.clip(np.searchsorted(th, g), 0, th.size - 1)
    i0 = np.clip(i1 - 1, 0, th.size - 1)
    out[(~ok[i1]) & (~ok[i0])] = np.nan
    return g, out
