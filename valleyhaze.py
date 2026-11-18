"""Altitude-dependent haze: valleys hazier than ridge tops (Smoky Mountains / Blue Ridge look).

Haze density falls off with height, beta(z) = beta0 * exp(-z / Hs) (z relative to the viewpoint, along
Earth's curved level surfaces), sometimes with a dense valley-haze layer below the viewpoint.  The optical
depth is integrated along each pixel's line of sight, so from a summit the far ridge tops - seen along
nearly level lines - stay darker than nearer valleys seen through the low haze.  Applied on top of the
existing views (I' = I * t + A * (1 - t)), so land, labels and skyline stay the same."""
import glob
import io
import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

R_EFF = 6371000.0 / 0.87          # Earth radius with normal refraction


def line_of_sight_depth(dist_m, el_rad, beta0, Hs, fog=None, n=28):
    """Optical depth from the camera to each ground pixel (arrays of the same shape)."""
    od = np.zeros_like(dist_m)
    for i in range(n):
        s = dist_m * (i + 0.5) / n
        z = s * np.sin(el_rad) + s * s / (2 * R_EFF)
        b = beta0 * np.exp(-np.clip(z, -3000, 6000) / Hs)
        if fog is not None:
            ztop, bf = fog
            b = b + bf / (1 + np.exp((z - ztop) / 60.0))       # soft-topped haze layer
        od += b * dist_m / n
    return od


def add_valley_haze(img, ground, dist_km, meta, rng):
    H, W, _ = img.shape
    f = (W / 2.0) / math.tan(math.radians(meta['fov']) / 2.0)
    rows = np.arange(H)[:, None] + 0.5
    el = np.radians(meta.get('pitch', 0.0)) + np.arctan((H / 2.0 - rows) / f)
    el = np.broadcast_to(el, (H, W))
    d = np.where(ground, dist_km, 0.0) * 1000.0
    v0 = float(np.exp(rng.uniform(math.log(15.0), math.log(120.0))))      # visibility at camera height, km
    Hs = rng.uniform(300.0, 1500.0)
    fog = None
    if rng.random() < 0.6:
        fog = (-rng.uniform(150.0, 1200.0), 3.0 / (1000.0 * rng.uniform(1.5, 8.0)))
    od = line_of_sight_depth(d, el, 3.0 / (v0 * 1000.0), Hs, fog)
    t = np.exp(-od)[..., None].astype(np.float32)
    sky = ~ground
    sk_rows = np.where(ground.any(0), ground.argmax(0), H)
    band = sky & (np.arange(H)[:, None] > (sk_rows[None, :] - 40))          # sky just above the skyline
    A = np.median(img[band], axis=0) if band.sum() > 50 else np.median(img[sky], axis=0)
    A = np.clip(A * rng.uniform(0.95, 1.05) + rng.normal(0, 0.015, 3), 0, 1).astype(np.float32)
    out = np.where(ground[..., None], img * t + A * (1 - t), img)
    out = np.stack([ndi.gaussian_filter(out[..., k], 0.4) for k in range(3)], -1)
    return np.clip(out, 0, 1), dict(valley_haze=True, v0_km=v0, Hs_m=Hs, fog_top_m=None if fog is None else fog[0])


def augment_all(n_per_view=2, seed=11):
    rng = np.random.default_rng(seed)
    logs = {}
    for f in ('log.json', 'log_clouds.json'):
        if os.path.exists('training/data/' + f):
            logs.update({(v, k): d for v, k, d in json.load(open('training/data/' + f))})
    newlog = []
    for vp in sorted(glob.glob('training/data/*.npz')):
        z = np.load(vp)
        name = os.path.basename(vp)[:-4]
        meta = json.loads(str(z['meta']))
        jps = sorted(glob.glob(vp[:-4] + '_w[0-5].jpg'))
        for j in range(n_per_view):
            src = jps[int(rng.integers(len(jps)))]
            k_src = int(src[-5])
            im = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255.0
            out, info = add_valley_haze(im, z['ground'], z['dist'], meta, rng)
            bw = bool(logs.get((name, k_src), {}).get('bw'))
            if bw:
                out = np.repeat(out.mean(-1, keepdims=True), 3, -1)
            b = io.BytesIO()
            Image.fromarray((out * 255).astype(np.uint8)).save(b, format='JPEG', quality=int(rng.integers(60, 95)))
            k_new = 6 + j
            open(vp[:-4] + f'_w{k_new}.jpg', 'wb').write(b.getvalue())
            d = dict(logs.get((name, k_src), {}))
            d.update(info, src=k_src)
            newlog.append([name, k_new, d])
    json.dump(newlog, open('training/data/log_valley.json', 'w'))
    return len(newlog)
