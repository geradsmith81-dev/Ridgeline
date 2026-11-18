"""Add realistic, shaded clouds to the sky of existing synthetic views (land, haze and labels unchanged).

Real cumulus have bright sunlit tops and grey, textured undersides; stratus come in stretched layers;
storm cloud is dark grey.  The first training set only had flat white blobs, so the network learned
'grey textured patch = land'.  These copies teach it otherwise."""
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import _fractal   # noqa: E402


def _noise(rng, H, W, stretch=1.0):
    if stretch > 1.0:                                          # long horizontal layers
        w = max(16, int(W / stretch))
        n = _fractal(rng, H, w)
        return np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC))
    return _fractal(rng, H, W)


def add_clouds(img, ground, skyline, rng):
    """img: H x W x 3 float 0-1.  Returns new image and settings."""
    H, W, _ = img.shape
    sky_m = ~ground
    yy = (np.arange(H)[:, None] / H).astype(np.float32)
    sk = np.where(skyline < H, skyline, H * 0.6)
    hz = float(np.mean(sk)) / H
    kind = str(rng.choice(['cumulus', 'stratus', 'storm', 'broken'], p=[0.45, 0.2, 0.15, 0.2]))
    stretch = {'cumulus': 1.0, 'stratus': rng.uniform(3, 8), 'storm': rng.uniform(1, 2.5),
               'broken': rng.uniform(1.5, 3)}[kind]
    n = _noise(rng, H, W, stretch)
    cov = {'cumulus': rng.uniform(0.15, 0.6), 'stratus': rng.uniform(0.3, 0.9), 'storm': rng.uniform(0.5, 0.95),
           'broken': rng.uniform(0.3, 0.75)}[kind]
    thr = np.quantile(n, 1 - cov)
    cm = np.clip((n - thr) * rng.uniform(1.5, 5.0), 0, 1)
    # how far clouds come down towards the skyline (often right down to it)
    reach = rng.uniform(0.6, 1.15)
    cm *= np.clip((reach * hz - yy) / max(0.03, 0.15 * hz) + 0.5, 0, 1) if rng.random() < 0.5 else 1.0
    # density inside the cloud -> darker cores and bases
    dens = ndi.gaussian_filter(np.clip((n - thr) / (n.max() - thr + 1e-6), 0, 1), rng.uniform(1, 4))
    ns = ndi.gaussian_filter(n, rng.uniform(4, 9))
    lit = np.clip(0.5 + 2.5 * ndi.sobel(ns, axis=0) / (np.abs(ndi.sobel(ns, axis=0)).std() * 8 + 1e-6), 0, 1)
    bright = rng.uniform(0.85, 1.05)
    dark = {'cumulus': rng.uniform(0.25, 0.55), 'stratus': rng.uniform(0.15, 0.45),
            'storm': rng.uniform(0.45, 0.7), 'broken': rng.uniform(0.25, 0.55)}[kind]
    shade = bright * (1 - dark * dens) * (0.72 + 0.28 * lit)
    tex = ndi.gaussian_filter(rng.normal(size=(H, W)).astype(np.float32), rng.uniform(1.5, 3.5))
    shade = shade * (1 + rng.uniform(0.01, 0.04) * tex / (tex.std() + 1e-6))
    tint = np.array([1.0, 1.0, 1.0], np.float32) + rng.normal(0, 0.02, 3).astype(np.float32)
    if rng.random() < 0.3:                                     # bluish-grey bases
        tint = tint * np.array([0.93, 0.97, 1.05], np.float32)
    ccol = shade[..., None] * tint
    # clouds near the horizon fade into the sky colour behind them (they are far away and hazy)
    fade = np.clip((yy / max(0.1, hz)) ** 3, 0, 1) * rng.uniform(0.2, 0.8)
    out_sky = img * (1 - cm[..., None]) + ccol * cm[..., None]
    out_sky = out_sky * (1 - fade[..., None] * cm[..., None]) + img * (fade[..., None] * cm[..., None])
    out = np.where(sky_m[..., None], out_sky, img)
    # soften the composite at the skyline the way a lens would
    out = np.stack([ndi.gaussian_filter(out[..., k], 0.5) for k in range(3)], -1)
    return np.clip(out, 0, 1), dict(clouds=kind, cov=float(cov))


def augment_all(n_per_view=2, seed=7):
    rng = np.random.default_rng(seed)
    wlog = {(v, k): d for v, k, d in json.load(open('training/data/log.json'))}
    newlog = []
    for vp in sorted(glob.glob('training/data/*.npz')):
        z = np.load(vp)
        name = os.path.basename(vp)[:-4]
        jps = sorted(glob.glob(vp[:-4] + '_w[0-3].jpg'))
        for j in range(n_per_view):
            src = jps[int(rng.integers(len(jps)))]
            k_src = int(src[-5])
            im = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255.0
            bw = bool(wlog.get((name, k_src), {}).get('bw'))
            out, info = add_clouds(im, z['ground'], z['skyline'], rng)
            if bw:
                out = np.repeat(out.mean(-1, keepdims=True), 3, -1)
            pil = Image.fromarray((out * 255).astype(np.uint8))
            b = io.BytesIO()
            pil.save(b, format='JPEG', quality=int(rng.integers(60, 95)))
            k_new = 4 + j
            open(vp[:-4] + f'_w{k_new}.jpg', 'wb').write(b.getvalue())
            d = dict(wlog.get((name, k_src), {}))
            d.update(info, src=k_src)
            newlog.append([name, k_new, d])
    json.dump(newlog, open('training/data/log_clouds.json', 'w'))
    return len(newlog)
