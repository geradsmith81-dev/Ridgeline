"""
Ridgeline - learned sky / land model for hard skylines (haze, pale rock against white sky,
faint far ridges behind darker near ones).

Each pixel gets 27 measures chosen to survive haze: brightness and colour, the "dark channel"
(haze and sky are bright in every colour), vertical edges at four scales divided by the local
spread of brightness (so a faint edge in a flat, hazy area counts as much as a strong edge in a
busy one), local contrast, texture, and how the pixel compares with everything above and below
it in its column.  A small neural network (27 -> 48 -> 24 -> 1) turns them into the probability
that the pixel is sky.  It was trained on synthetic eye-level views rendered from open elevation
data and satellite imagery at 32 mountain areas worldwide, each shown in many kinds of weather
(see training/ in the project).  Inference is plain numpy, so no extra libraries are needed.
"""
import os

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'assets', 'sky_model.npz')

FEATURE_NAMES = (
    'lum', 'lum_s1', 'blue_ratio', 'red_ratio', 'saturation', 'dark7', 'dark21',
    'gy_n1', 'gy_n2', 'gy_n4', 'gy_n8', 'gmag1', 'gmag4', 'lcn4', 'lcn16', 'tex3', 'tex8',
    'lum_vs_above', 'lum_vs_below', 'lum_vs_max_above', 'dark_vs_above', 'dark_vs_below',
    'blue_vs_above', 'blue_vs_below', 'tex_vs_above', 'row', 'col_lum_rank')


def _running_mean(a, from_top=True):
    """Mean of each column from the top down to the row before (or from the bottom up)."""
    if not from_top:
        return _running_mean(a[::-1], True)[::-1]
    c = np.cumsum(a, axis=0)
    n = np.arange(1, a.shape[0] + 1, dtype=np.float32)[:, None]
    m = c / n
    out = np.empty_like(m)
    out[0] = a[0]
    out[1:] = m[:-1]
    return out


def pixel_features(rgb):
    """rgb: H x W x 3 float (0-1) -> H x W x 27 float32 features."""
    rgb = np.asarray(rgb, np.float32)
    H, W, _ = rgb.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    L = 0.299 * r + 0.587 * g + 0.114 * b
    Ls = ndi.gaussian_filter(L, 1.0)
    s = r + g + b + 1e-3
    rb, rr = b / s, r / s
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / (mx + 1e-3)
    dark7 = ndi.minimum_filter(mn, 7)
    dark21 = ndi.minimum_filter(mn, 21)
    f = [L, Ls, rb, rr, sat, dark7, dark21]
    for sg in (1, 2, 4, 8):
        Lg = ndi.gaussian_filter(L, sg)
        gy = ndi.sobel(Lg, axis=0) / 8.0
        m1 = ndi.gaussian_filter(L, 3 * sg)
        sd = np.sqrt(np.maximum(ndi.gaussian_filter(L * L, 3 * sg) - m1 * m1, 0)) + 0.01
        f.append(np.clip(-gy / sd, -6, 6))                 # positive = brighter above
    g1x, g1y = ndi.sobel(Ls, axis=1) / 8.0, ndi.sobel(Ls, axis=0) / 8.0
    gm1 = np.hypot(g1x, g1y)
    L4 = ndi.gaussian_filter(L, 4)
    gm4 = np.hypot(ndi.sobel(L4, axis=1), ndi.sobel(L4, axis=0)) / 8.0
    f += [np.log(gm1 + 1e-3), np.log(gm4 + 1e-3)]
    for sg in (4, 16):
        m_ = ndi.gaussian_filter(L, sg)
        sd = np.sqrt(np.maximum(ndi.gaussian_filter(L * L, sg) - m_ * m_, 0)) + 0.01
        f.append(np.clip((Ls - m_) / sd, -6, 6))
    tex3 = np.log1p(50 * ndi.gaussian_filter(gm1, 3))
    tex8 = np.log1p(50 * ndi.gaussian_filter(gm1, 8))
    f += [tex3, tex8]
    f += [Ls - _running_mean(Ls), Ls - _running_mean(Ls, False), Ls - np.maximum.accumulate(Ls, axis=0),
          dark7 - _running_mean(dark7), dark7 - _running_mean(dark7, False),
          rb - _running_mean(rb), rb - _running_mean(rb, False), tex3 - _running_mean(tex3)]
    f.append(np.broadcast_to((np.arange(H, dtype=np.float32)[:, None] + 0.5) / H, (H, W)))
    rank = np.argsort(np.argsort(Ls, axis=0), axis=0).astype(np.float32) / max(1, H - 1)
    f.append(rank)
    return np.stack(f, -1).astype(np.float32)


class SkyNet:
    def __init__(self, path=MODEL_PATH):
        z = np.load(path)
        self.mu, self.sd = z['mu'], z['sd']
        self.layers = [(z[f'W{i}'], z[f'b{i}']) for i in range(1, 4)]
        self.version = str(z['version']) if 'version' in z else '?'

    def prob(self, rgb=None, feats=None):
        """Probability of sky per pixel (H x W)."""
        F = pixel_features(rgb) if feats is None else feats
        H, W, n = F.shape
        X = ((F.reshape(-1, n) - self.mu) / self.sd).astype(np.float32)
        out = np.empty(X.shape[0], np.float32)
        step = 200_000
        for s in range(0, X.shape[0], step):
            h = X[s:s + step]
            for i, (Wm, bm) in enumerate(self.layers):
                h = h @ Wm + bm
                if i < len(self.layers) - 1:
                    np.maximum(h, 0, out=h)
            out[s:s + step] = 1.0 / (1.0 + np.exp(-np.clip(h[:, 0], -30, 30)))
        return out.reshape(H, W)


_NET = None


def get_net():
    """The trained model, or None if the weights file is missing."""
    global _NET
    if _NET is None and os.path.exists(MODEL_PATH):
        try:
            _NET = SkyNet()
        except Exception:
            _NET = None
    return _NET
