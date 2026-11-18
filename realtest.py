"""Real photos with GPS: fit the camera at (near) the true spot for a detected skyline, then measure how
much of the detected skyline agrees with the elevation model's horizon."""
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('RIDGELINE_TILE_CACHE', os.path.abspath('tilecache'))
sys.path.insert(0, 'ridgeline')
import search as se    # noqa: E402
import skyline         # noqa: E402


def load(path):
    return skyline.load_image(open(path, 'rb').read())


def pose_check(im, info, det_kw, radius_km=0.2, spacing_km=0.1, tol_frac=0.015):
    t0 = time.time()
    d = skyline.detect(im, **det_kw)
    hfov = skyline.hfov_from_exif(info, *im.size)
    res = se.locate(d['x'], d['y'], d['valid'], d['size'], info['lat'], info['lon'], radius_km=radius_km,
                    spacing_km=spacing_km, hfov=hfov)
    c = res['candidates'][0]
    n = len(d['x'])
    v = se.view_of(c['lat'], c['lon'], c['heading'], c['fov'], c['pitch'], c['roll'], d['size'], n,
                   tilt=c.get('tilt', 0.0))
    yd = np.array([np.nan if t is None else t for t in v['y_cols']], float)
    H = d['size'][1]
    inframe = np.isfinite(yd) & (yd > 0) & (yd < H)
    err = np.abs(d['y'] - yd)
    ok = d['valid'] & inframe & (err <= tol_frac * H)
    return dict(kept=float(d['valid'].mean()),
                agree=float(ok.sum() / max(1, (d['valid'] & inframe).sum())),     # kept columns that match
                cover=float(ok.sum() / max(1, inframe.sum())),                      # DEM skyline columns found
                med_err_pct=float(np.nanmedian(np.where(d['valid'] & inframe, err, np.nan)) / H * 100),
                off_m=1000 * se.haversine_km(info['lat'], info['lon'], c['lat'], c['lon']),
                heading=c['heading'], fov=c['fov'], hfov_exif=hfov, secs=time.time() - t0,
                det=d, view=yd)
