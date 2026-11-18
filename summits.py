"""Extra training views from summits and overlooks, looking out over layered lower ridges."""
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth                         # noqa: E402
import clouds                        # noqa: E402
import valleyhaze                    # noqa: E402
se = synth.se

SITES = {   # all well away from the held-out validation areas and the real test photos
    'mitchell': (35.765, -82.265), 'roan': (36.104, -82.122), 'grandfather': (36.100, -81.830),
    'rogers': (36.660, -81.545), 'catskills': (42.100, -74.300), 'whiteface': (44.366, -73.903),
    'brecon': (51.884, -3.437), 'vosges': (47.930, 7.100), 'bavaria': (47.600, 11.000),
    'japan_kiso': (35.790, 137.800), 'alaska_range': (63.700, -148.900), 'yukon_hills': (64.900, -147.300),
    'oregon_cascades': (44.200, -121.900), 'sierra_n': (39.300, -120.300),
}


def high_points(lat, lon, k=4, half=0.06, n=41, min_sep=0.02):
    la = lat + np.linspace(-half, half, n)
    lo = lon + np.linspace(-half, half, n) / math.cos(math.radians(lat))
    LA, LO = np.meshgrid(la, lo, indexing='ij')
    A = se.AreaTerrain(lat, lon, half * 111 * 1.5, 2.0, se.BANDS_FINE)
    g = A.ground(LA.ravel(), LO.ravel()).reshape(LA.shape)
    order = np.argsort(g.ravel())[::-1]
    pts = []
    for i in order:
        p = (LA.ravel()[i], LO.ravel()[i])
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > min_sep for q in pts):
            pts.append(p)
        if len(pts) >= k:
            break
    return pts


def make(seed=21, per_site=4, out='training/data'):
    rng = np.random.default_rng(seed)
    log = []
    for name, (lat, lon) in SITES.items():
        pts = high_points(lat, lon, k=per_site + 2)
        made = 0
        for (pla, plo) in pts:
            if made >= per_site:
                break
            heading = float(rng.uniform(0, 360))
            fov = float(rng.uniform(35, 75))
            pitch = float(rng.uniform(-6, 1))
            eye = float(rng.choice([1.7, 1.7, 12.0]))           # sometimes an observation tower
            try:
                v = synth.render_view(pla, plo, heading, fov, pitch, eye=eye)
            except Exception as e:                              # missing tiles etc.
                print(name, 'render failed', e)
                continue
            H = v['ground'].shape[0]
            skyfrac = 1 - v['ground'].mean()
            if skyfrac < 0.06 or skyfrac > 0.85:
                continue
            vname = f'train_sum_{name}_{made}'
            dist_km = np.where(v['ground'], v['dist'] / 1000.0, 1e6).astype(np.float32)
            meta = dict(lat=pla, lon=plo, heading=heading, fov=fov, pitch=pitch, eye=eye)
            np.savez(os.path.join(out, vname + '.npz'), ground=v['ground'], dist=dist_km,
                     skyline=v['skyline'].astype(float), meta=json.dumps(meta))
            imgs = {}
            for k in range(4):
                im, info = synth.weather(v, rng)
                im.save(os.path.join(out, f'{vname}_w{k}.jpg'), quality=92)
                imgs[k] = (np.asarray(im, np.float32) / 255.0, info)
                log.append([vname, k, info])
            # shaded clouds (w4) and valley haze on top of a weather image (w6, w7)
            k = int(rng.integers(4))
            a, info = clouds.add_clouds(imgs[k][0], v['ground'], v['skyline'], rng)
            Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(out, f'{vname}_w4.jpg'), quality=88)
            log.append([vname, 4, dict(imgs[k][1], **info, src=k)])
            for j in (6, 7):
                k = int(rng.integers(4))
                a, info = valleyhaze.add_valley_haze(imgs[k][0], v['ground'], dist_km, meta, rng)
                if imgs[k][1].get('bw'):
                    a = np.repeat(a.mean(-1, keepdims=True), 3, -1)
                Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(out, f'{vname}_w{j}.jpg'), quality=88)
                log.append([vname, j, dict(imgs[k][1], **info, src=k)])
            made += 1
        print(name, 'views', made, flush=True)
    json.dump(log, open(os.path.join(out, 'log_summits.json'), 'w'))
    return len(log)
