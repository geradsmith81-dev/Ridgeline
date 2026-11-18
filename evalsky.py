"""Score skyline detectors against the true skylines of synthetic views."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, 'ridgeline')
import skyline   # noqa: E402


def score_one(det, truth, H):
    w, h = det['work_size']
    sy = det['size'][1] / float(h)
    y = det['y'] / sy - 0.5
    has = truth < H - 1                           # columns that really have land
    err = np.abs(y - truth)
    v = det['valid']
    n = max(1, has.sum())
    return dict(
        med=float(np.median(err[has])) if has.any() else np.nan,
        within3=float(((err <= 3) & has).sum() / n),          # raw trace right (to 3 px)
        good=float(((err <= 3) & has & v).sum() / n),          # kept AND right: what the search uses
        bad_kept=float(((err > 10) & v).sum() / max(1, v.sum())),   # kept but wrong
        kept=float(v.sum() / len(v)))


def visible_skyline(z, vis_km, factor=1.3):
    g = z['ground'] & (z['dist'] <= factor * vis_km)
    return np.where(g.any(0), g.argmax(0), g.shape[0]).astype(float)


def evaluate(split='val', variants=None, weathers=None, limit=None):
    variants = variants or {'old': dict(learned=False), 'new': dict(learned=True)}
    rows = []
    wlog = {(v, k): d for f in ('log.json', 'log_clouds.json', 'log_valley.json', 'log_summits.json') if os.path.exists('training/data/' + f)
            for v, k, d in json.load(open('training/data/' + f))}
    for vp in sorted(glob.glob(f'training/data/{split}_*.npz'))[:limit]:
        z = np.load(vp)
        truth = z['skyline'].astype(float)
        H = z['ground'].shape[0]
        for jp in sorted(glob.glob(vp[:-4] + '_w*.jpg')):
            im = Image.open(jp)
            wtag = os.path.basename(jp)[-6:-4]
            wk = wlog.get((os.path.basename(vp)[:-4], int(wtag[1])), {})
            vtruth = visible_skyline(z, wk.get('vis_km', 1e9))
            for name, kw in variants.items():
                kw = dict(kw)
                model = kw.pop('model', None)
                if model:
                    import skymodel
                    skymodel._NET = skymodel.SkyNet(model)
                d = skyline.detect(im, **kw)
                r = score_one(d, truth, H)
                r.update({k + '_vis': v for k, v in score_one(d, vtruth, H).items()})
                r.update(view=os.path.basename(vp)[:-4], w=wtag, variant=name, sky=wk.get('kind'),
                         vis=wk.get('vis_km'), bw=wk.get('bw'))
                rows.append(r)
    return rows
