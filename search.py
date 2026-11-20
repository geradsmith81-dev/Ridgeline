"""
Ridgeline - searching the land around a starting point for the place whose skyline matches a photo.

1. Candidate camera positions are laid on a square grid (spacing s) inside a circle of radius R
   around the starting point.
2. For every candidate the 360-degree skyline is computed from the open elevation model (AWS Terrain
   Tiles, as in Afterglow), with Earth curvature and refraction (k = 0.13), the camera 1.7 m above the
   ground and terrain nearer than 200 m ignored.  The tiles of each distance band are loaded once for
   the whole search area; candidates on one grid row share their ray geometry in map-pixel space, so
   all skylines come from one bilinear-sampling pass.
3. The photo's skyline (angles from skyline.to_angles) is slid round every heading.  At each heading
   the camera tilt (an offset) and roll (a slope) are solved by least squares, and the fit is scored
   by a robust RMS (the worst 10 % of points are dropped), divided by the photo skyline's own relief
   so scores are comparable between photos.  When the lens is unknown, several fields of view are
   tried and the best kept.
4. Coarse-to-fine: a coarse grid first, then fine grids around the best distinct candidates.
"""
import math
import time

import numpy as np
from scipy import ndimage as ndi

import terrain as tr

R_EARTH = tr.R_EARTH
K = tr.K_REFRACTION

# (start m, end m, zoom, step m)
BANDS_COARSE = [(0, 2500, 13, 15.0), (2500, 12000, 11, 80.0), (12000, 40000, 10, 200.0), (40000, 250000, 9, 450.0)]
CLIP_FRAC = 0.0094          # residual clip = 0.94 % of the picture width (0.5 deg for a 50 deg lens)
COARSE_PITCHES = (-10.0, -5.0, 0.0, 5.0, 10.0)
COARSE_PITCHES_UNKNOWN_FOV = (-8.0, 0.0, 8.0)
BANDS_FINE = [(0, 3000, 13, 15.0), (3000, 15000, 12, 40.0), (15000, 45000, 11, 100.0), (45000, 250000, 10, 250.0)]


def grid_points(lat0, lon0, radius_km, spacing_km):
    n = int(math.floor(radius_km / spacing_km))
    k = np.arange(-n, n + 1) * spacing_km
    ex, ny = np.meshgrid(k, k)
    m = ex ** 2 + ny ** 2 <= radius_km ** 2 + 1e-9
    ex, ny = ex[m], ny[m]
    lat = lat0 + np.degrees(ny * 1000.0 / R_EARTH)
    lon = lon0 + np.degrees(ex * 1000.0 / (R_EARTH * math.cos(math.radians(lat0))))
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * R_EARTH / 1000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class AreaTerrain:
    """Tile mosaics covering all rays from every candidate in a search area."""

    def __init__(self, lat0, lon0, radius_km, max_km=120.0, bands=BANDS_COARSE, near_m=200.0, progress=None,
                 canopy_m=0.0):
        self.lat0, self.lon0 = lat0, lon0
        self.canopy_m = float(canopy_m)      # trees standing on the modelled ground (metres)
        self.max_m = max_km * 1000.0
        self.near_m = near_m
        self.bands = []
        self.tiles = self.missing = 0
        rad_m = radius_km * 1000.0 + 500.0
        active = [b for b in bands if b[0] < self.max_m]
        for i, (d0, d1, z, step) in enumerate(active):
            d1 = min(d1, self.max_m)
            start = max(d0, near_m) + step / 2.0
            if start >= d1:
                continue
            reach = rad_m + d1 + 1000.0
            dlat = math.degrees(reach / R_EARTH)
            dlon = math.degrees(reach / (R_EARTH * max(0.05, math.cos(math.radians(min(85, abs(lat0) + dlat))))))
            if progress:
                progress(f'Loading elevation tiles (zoom {z}, rays {d0 / 1000:.0f}-{d1 / 1000:.0f} km)', i / len(active))
            m = tr.Mosaic(z, lat0 - dlat, lat0 + dlat, lon0 - dlon, lon0 + dlon)
            self.tiles += m.n_tiles
            self.missing += m.n_missing
            arr = m.arr
            fill = np.nanmin(arr) if np.isfinite(arr).any() else 0.0
            arr = np.where(np.isfinite(arr), arr, min(0.0, float(fill)))
            self.bands.append(dict(z=z, d=np.arange(start, d1, step), m=m, arr=arr.astype(np.float32)))
        if not self.bands:
            raise tr.TerrainError('Search distance too short.')

    def ground(self, lat, lon):
        b = self.bands[0]
        x, y = tr.merc_px(np.asarray(lat), np.asarray(lon), b['z'])
        return self._sample(b, x, y)

    @staticmethod
    def _sample(b, x, y):
        m = b['m']
        u = np.asarray(x) - m.tx0 * 256 - 0.5
        v = np.asarray(y) - m.ty0 * 256 - 0.5
        return ndi.map_coordinates(b['arr'], [v.ravel(), u.ravel()], order=1, mode='nearest').reshape(np.shape(u))

    def horizons(self, lat, lon, az_step=0.5, height=1.7, progress=None, chunk=24, az=None):
        """Skyline elevation angle (deg) for each candidate (rows) at azimuths 0, az_step, ... (cols),
        or at the given azimuth array `az`.  Also returns the distance (km) of the skyline terrain.
        When canopy_m is set, that height is added to the ground along every ray, because in forest
        the line a photograph shows is the top of the trees and the elevation model has none."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        n = lat.size
        az = np.arange(0.0, 360.0, az_step) if az is None else np.asarray(az, float)
        gnd = self.ground(lat, lon)
        h0 = gnd + height
        floor = np.minimum(0.0, gnd)
        best_t = np.full((n, az.size), -np.inf, dtype=np.float32)
        best_d = np.zeros((n, az.size), dtype=np.float32)
        # unique grid rows share ray offsets in Mercator pixels
        lat_keys = np.round(lat, 7)
        uniq = np.unique(lat_keys)
        done = 0
        t_start = time.time()
        for b in self.bands:
            d = b['d']
            z = b['z']
            drop = (d ** 2 * (1.0 - K) / (2.0 * R_EARTH)).astype(np.float32)
            for la in uniq:
                idx = np.nonzero(lat_keys == la)[0]
                ref_lon = float(lon[idx[0]])
                pla, plo = tr.destination(float(la), ref_lon, az[:, None], d[None, :])
                px, py = tr.merc_px(pla, plo, z)
                cx, cy = tr.merc_px(np.array([la]), np.array([ref_lon]), z)
                du = (px - cx[0]).astype(np.float32)
                dv = (py - cy[0]).astype(np.float32)
                cxs, _ = tr.merc_px(np.full(idx.size, la), lon[idx], z)
                for s in range(0, idx.size, chunk):
                    ii = idx[s:s + chunk]
                    ox = (cxs[s:s + chunk] - cx[0]).astype(np.float32)
                    x = cx[0] + ox[:, None, None] + du[None]
                    y = np.broadcast_to(cy[0] + dv[None], x.shape)
                    h = self._sample(b, x, y)
                    h = np.maximum(h, floor[ii][:, None, None])
                    if self.canopy_m:
                        h = h + np.float32(self.canopy_m)      # the skyline is the treetops, not the soil
                    tt = (h - h0[ii][:, None, None] - drop[None, None, :]) / d[None, None, :].astype(np.float32)
                    j = np.argmax(tt, axis=2)
                    tm = np.take_along_axis(tt, j[..., None], axis=2)[..., 0]
                    better = tm > best_t[ii]
                    best_t[ii] = np.where(better, tm, best_t[ii])
                    best_d[ii] = np.where(better, d[j] / 1000.0, best_d[ii])
            done += 1
            if progress:
                progress(f'Computing skylines ({done}/{len(self.bands)} distance bands, {n} places)',
                         done / len(self.bands))
        return dict(az=az, angle=np.degrees(np.arctan(best_t)), dist_km=best_d, ground=gnd,
                    seconds=time.time() - t_start)


# ------------------------------------------------------------------------------------------
# matching
# ------------------------------------------------------------------------------------------
def photo_profile(th, el, valid, step, shift=0.0):
    """Average the photo skyline into bins of `step` degrees centred on multiples of step - shift."""
    th = np.asarray(th, float) + shift
    el = np.asarray(el, float)
    ok = np.asarray(valid, bool) & np.isfinite(el)
    k = np.round(th / step).astype(int)
    k0 = k.min()
    nb = k.max() - k0 + 1
    s = np.bincount(k[ok] - k0, weights=el[ok], minlength=nb)
    c = np.bincount(k[ok] - k0, minlength=nb)
    prof = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return (np.arange(nb) + k0) * step, prof


def is_pano(cam):
    return bool(cam) and cam.get('proj') in ('cyl', 'equi')


def match(H, az_step, theta, prof, circular=True, max_roll_deg=8.0, max_dpitch_deg=8.0, clip_deg=1.0,
          chunk=64, basis3=False, max_offset_deg=30.0, az_cols=None, heading_hint=None, heading_window=None):
    """Slide the photo profile (theta deg on multiples of az_step, relative to the heading; prof = world
    elevation for an assumed tilt; NaN = ignore) over candidate skylines H (n x N).

    At each heading the remaining tilt (cos theta term) and roll (sin theta term) are fitted by least
    squares (a tilted camera's level plane rises by p*cos(theta) + r*sin(theta)).  For panoramas
    (basis3=True) a constant term is fitted as well: the unknown row of the horizon in a stitched or
    cropped panorama.  The error is the RMS of the residual with each point's square clipped at
    clip_deg^2 (robust against wrong stretches).  Returns per candidate the best score, column index j,
    the fitted terms and the RMS (deg)."""
    H = np.asarray(H, np.float32)
    n, N = H.shape
    ok = np.isfinite(prof)
    th = theta[ok]
    p = prof[ok].astype(np.float32)
    m = p.size
    if m < 8:
        raise ValueError('Too little skyline to match.')
    tr_ = np.radians(th)
    rows_b = ([np.ones_like(tr_)] if basis3 else []) + [np.cos(tr_), np.sin(tr_)]
    B = np.stack(rows_b).astype(np.float32)                              # k x m
    P = np.linalg.pinv(B.T).astype(np.float32)                           # k x m
    relief = float(np.std(p - B.T @ (P @ p))) + 0.05
    offs = np.round(th / az_step).astype(int)
    if circular:
        js = np.arange(N)
        idx = (js[:, None] + offs[None, :]) % N
    else:
        js = np.arange(max(0, -offs.min()), min(N, N - offs.max()))
        if js.size == 0:
            raise ValueError('Azimuth window too narrow.')
        idx = js[:, None] + offs[None, :]
    allow = None
    if heading_hint is not None and heading_window and az_cols is not None:
        azc = np.asarray(az_cols, float)[js]                 # azimuth of each candidate heading column
        allow = np.abs((azc - float(heading_hint) + 180.0) % 360.0 - 180.0) <= float(heading_window)
        if not allow.any():
            allow = None
    c2 = np.float32(clip_deg ** 2)
    out = dict(score=np.empty(n, np.float32), j=np.empty(n, int), dpitch=np.empty(n, np.float32),
               tilt=np.zeros(n, np.float32), roll=np.empty(n, np.float32))
    if basis3:
        chunk = max(4, int(chunk * 60 / max(60, m)))
    for s0 in range(0, n, chunk):
        Hc = H[s0:s0 + chunk]
        r = Hc[:, idx] - p                                               # c x J x m  (DEM minus photo)
        coef = r @ P.T                                                   # c x J x k
        res = r - coef @ B
        e2 = np.minimum(res * res, c2)
        sc = np.sqrt(e2.mean(axis=2))
        if basis3:
            bad = ((np.abs(coef[..., 0]) > max_offset_deg) | (np.abs(coef[..., 1]) > max_dpitch_deg)
                   | (np.abs(coef[..., 2]) > max_roll_deg))
        else:
            bad = (np.abs(coef[..., 0]) > max_dpitch_deg) | (np.abs(coef[..., 1]) > max_roll_deg)
        sc = np.where(bad, np.inf, sc)
        if allow is not None:
            sc = np.where(allow[None, :], sc, np.inf)
        jj = np.argmin(sc, axis=1)
        rows = np.arange(Hc.shape[0])
        out['score'][s0:s0 + chunk] = sc[rows, jj]
        out['j'][s0:s0 + chunk] = js[jj]
        out['dpitch'][s0:s0 + chunk] = coef[rows, jj, 0]
        if basis3:
            out['tilt'][s0:s0 + chunk] = coef[rows, jj, 1]
        out['roll'][s0:s0 + chunk] = coef[rows, jj, -1]
    out['relief'] = relief
    out['rms'] = out['score'].copy()
    return out


def _ref_fov(cam):
    return float(cam.get('ref_fov') or 180.0) if is_pano(cam) else 50.0


def _kpx(W, f, cam):
    """Photo pixels per degree near the horizon for a view `f` degrees wide."""
    if is_pano(cam):
        return W / f
    return (W / 2.0) / math.tan(math.radians(f) / 2.0) / 57.29578


def photo_relief(sky_x, sky_y, valid, size, fov=None, cam=None):
    """Relief (deg) of the photo skyline after removing the fitted terms: one fixed number per photo,
    so scores stay comparable between lens angles and tilts."""
    fov = fov or _ref_fov(cam)
    valid = np.asarray(valid, bool) & np.isfinite(np.asarray(sky_y, float))
    th, el = photo_to_world(sky_x, np.where(valid, sky_y, 0.0), size, fov, 0.0, 0.0, 0.0, cam=cam)
    t, p = photo_profile(th, el, valid, 0.25)
    ok = np.isfinite(p)
    if ok.sum() < 4:
        return 0.3
    tr_ = np.radians(t[ok])
    B = np.stack(([np.ones_like(tr_)] if is_pano(cam) else []) + [np.cos(tr_), np.sin(tr_)])
    c, *_ = np.linalg.lstsq(B.T, p[ok], rcond=None)
    return float(np.std(p[ok] - B.T @ c)) + 0.05


def match_fovs(H, az, sky_x, sky_y, valid, size, fovs, skyline_mod=None, phases=1, circular=True, pitches=(0.0,),
               clip_deg=None, relief=None, cam=None, heading_hint=None, heading_window=None):
    """Try several horizontal fields of view, camera tilts and sub-step heading phases; best per
    candidate.  `az` is the (evenly spaced) azimuth array of H's columns."""
    az = np.asarray(az, float)
    step = float(az[1] - az[0])
    valid = np.asarray(valid, bool) & np.isfinite(np.asarray(sky_y, float))
    pano = is_pano(cam)
    if pano:
        pitches = (0.0,)
    # Errors are judged in photo pixels, so that lens angles can be compared fairly: the clip is a
    # fixed number of pixels and the score is the pixel RMS over the photo's pixel relief.
    W = float(size[0])
    ref = _ref_fov(cam)
    if relief is None:
        relief = photo_relief(sky_x, sky_y, valid, size, fov=ref, cam=cam)
    relief_px = relief * _kpx(W, ref, cam)
    clip_px = CLIP_FRAC * W * (50.0 / ref if pano else 1.0)
    best = None
    for f in fovs:
        clip_f = clip_px / _kpx(W, f, cam) if clip_deg is None else clip_deg
        for p0 in pitches:
            th, el = photo_to_world(sky_x, np.where(valid, sky_y, 0.0), size, f, p0, 0.0, 0.0, cam=cam)
            for ph in range(phases):
                d = ph * step / phases
                t, p = photo_profile(th, el, valid, step, shift=d)
                r = match(H, step, t, p, circular=circular, clip_deg=clip_f, basis3=pano,
                          az_cols=az + d, heading_hint=heading_hint, heading_window=heading_window)
                r['score'] = r['rms'] * _kpx(W, f, cam) / relief_px
                r['heading'] = (az[0] + r['j'] * step + d) % 360.0
                r['pitch'] = p0 + r['dpitch']
                r['fov'] = np.full(r['score'].shape, f, dtype=float)
                if best is None:
                    best = r
                    continue
                bt = r['score'] < best['score']
                for k in ('score', 'heading', 'pitch', 'tilt', 'roll', 'rms', 'fov'):
                    best[k] = np.where(bt, r[k], best[k])
    return best


def good_region(coarse, tol=0.05, max_rects=400):
    """Where else fits almost as well: the tested places whose score is within `tol` of the best.

    Returns rectangles (south, west, north, east) covering those grid cells, merged along each row of
    latitude, plus the area they cover and the share of the searched grid.  A wide region is the
    honest answer for gentle country: the photograph fixes an area, not a point."""
    lat = np.asarray(coarse['lat'], float)
    lon = np.asarray(coarse['lon'], float)
    sc = np.asarray(coarse['score'], float)
    ok = np.isfinite(sc)
    if not ok.any():
        return dict(rects=[], area_km2=0.0, n=0, share=0.0, cutoff=None)
    lat, lon, sc = lat[ok], lon[ok], sc[ok]
    best = float(sc.min())
    cut = best * (1.0 + float(tol))
    m = sc <= cut
    step = float(coarse.get('spacing_km') or 0.3)
    dlat = math.degrees(step * 1000.0 / R_EARTH) / 2.0
    rows = {}
    for la, lo in zip(lat[m], lon[m]):
        rows.setdefault(round(la, 6), []).append(lo)
    rects = []
    area = 0.0
    for la in sorted(rows):
        dlon = math.degrees(step * 1000.0 / (R_EARTH * max(0.05, math.cos(math.radians(la))))) / 2.0
        los = sorted(rows[la])
        run0 = prev = los[0]
        for lo in los[1:] + [None]:
            if lo is not None and lo - prev <= 2.2 * dlon:
                prev = lo
                continue
            rects.append([round(float(la - dlat), 6), round(float(run0 - dlon), 6),
                          round(float(la + dlat), 6), round(float(prev + dlon), 6)])
            w_km = (prev - run0) * math.cos(math.radians(la)) * 111.32 + step
            area += w_km * step
            if lo is None:
                break
            run0 = prev = lo
    if len(rects) > max_rects:                      # keep the payload small for the map
        rects = rects[:max_rects]
    return dict(rects=rects, area_km2=round(float(area), 2), n=int(m.sum()), share=round(float(m.mean()), 4),
                cutoff=round(float(cut), 5), best=round(float(best), 5), tol=float(tol))


def start_agreement(points):
    """How far apart the best places from several starting points fall (km)."""
    pts = [(float(p['lat']), float(p['lon'])) for p in points if p]
    if len(pts) < 2:
        return dict(n=len(pts), max_km=None, median_km=None)
    d = [haversine_km(a[0], a[1], b[0], b[1]) for i, a in enumerate(pts) for b in pts[i + 1:]]
    return dict(n=len(pts), max_km=round(float(max(d)), 3), median_km=round(float(np.median(d)), 3))


def distinct_top(lat, lon, score, k, min_sep_km):
    order = np.argsort(score)
    out = []
    for i in order:
        if not np.isfinite(score[i]):
            break
        if all(haversine_km(lat[i], lon[i], lat[j], lon[j]) >= min_sep_km for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------------------------------------
# the whole search
# ------------------------------------------------------------------------------------------
def fov_list(hfov=None, lo=20.0, hi=90.0, n=9):
    if hfov:
        return [float(hfov)]
    return [float(x) for x in np.round(np.geomspace(lo, hi, n), 2)]


def _offset(lat, lon, de_km, dn_km):
    return (lat + math.degrees(dn_km * 1000.0 / R_EARTH),
            lon + math.degrees(de_km * 1000.0 / (R_EARTH * math.cos(math.radians(lat)))))


def _photo_half_width(sky_x, sky_y, valid, size, fov, pitch, cam=None):
    """Largest |azimuth offset| of the photo's skyline points (deg) for a camera of this fov and tilt."""
    if is_pano(cam):
        return fov / 2.0
    v = np.asarray(valid, bool) & np.isfinite(np.asarray(sky_y, float))
    if not v.any():
        return fov / 2.0
    th, _ = photo_to_world(np.asarray(sky_x)[v], np.asarray(sky_y)[v], size, fov, pitch, 0.0, 0.0)
    return float(np.max(np.abs(th)))


def _eval_points(A, la, lo, heading, fl, pitch, sky_x, sky_y, valid, size, az_step, pad, height, cam=None,
                 heading_hint=None, heading_window=None):
    half = None
    if pad is not None:
        half = max(_photo_half_width(sky_x, sky_y, valid, size, max(fl), pitch, cam), max(fl) / 2.0) + pad + 1.0
    if pad is None or half >= 178.0:     # whole circle
        h = A.horizons(np.asarray(la, float), np.asarray(lo, float), az_step=az_step, height=height)
        m = match_fovs(h['angle'], h['az'], sky_x, sky_y, valid, size, fl, None, phases=2, circular=True,
                       pitches=(round(float(pitch), 2),), cam=cam,
                       heading_hint=heading_hint, heading_window=heading_window)
        return m, h
    az = (heading - half + np.arange(0.0, 2 * half + 1e-9, az_step)) % 360.0
    h = A.horizons(np.asarray(la, float), np.asarray(lo, float), az=az, height=height)
    m = match_fovs(h['angle'], np.unwrap(az, period=360.0), sky_x, sky_y, valid, size, fl, None,
                   phases=2, circular=False, pitches=(round(float(pitch), 2),), cam=cam)
    return m, h


def _descend(A, c, fl_fn, sky_x, sky_y, valid, size, step_km, min_step_km, az_step, pad, height, max_iter=12,
             cam=None, heading_hint=None, heading_window=None):
    """Pattern search from candidate c (dict with lat, lon, heading, pitch, fov, score): evaluate the
    3 x 3 neighbourhood, move to the best point, halve the step when the centre is best."""
    cur = dict(c)
    it = 0
    while step_km >= min_step_km - 1e-9 and it < max_iter:
        it += 1
        offs = [(0.0, 0.0)] + [(i * step_km, j * step_km) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
        pts = [_offset(cur['lat'], cur['lon'], de, dn) for de, dn in offs]
        la = np.array([p[0] for p in pts])
        lo = np.array([p[1] for p in pts])
        fl = fl_fn(cur)
        try:
            m, h = _eval_points(A, la, lo, cur['heading'], fl, cur['pitch'], sky_x, sky_y, valid, size, az_step, pad,
                                height, cam=cam, heading_hint=heading_hint, heading_window=heading_window)
        except ValueError:
            break
        j = int(np.argmin(m['score']))
        moved = j != 0 and float(m['score'][j]) < float(m['score'][0]) - 1e-6
        cur = dict(lat=float(la[j]), lon=float(lo[j]), score=float(m['score'][j]), heading=float(m['heading'][j]),
                   pitch=float(m['pitch'][j]), tilt=float(m['tilt'][j]), roll=float(m['roll'][j]),
                   fov=float(m['fov'][j]), rms_deg=float(m['rms'][j]), ground=float(h['ground'][j]),
                   coarse_score=c.get('coarse_score'))
        if not moved:
            step_km /= 2.0
    return cur


PANO_FOV_RANGE = (90.0, 360.0)


def locate(sky_x, sky_y, valid, size, lat0, lon0, radius_km=10.0, spacing_km=0.3, hfov=None,
           fov_range=None, max_km=120.0, height=1.7, n_top=30, n_fine=6, progress=None, skyline_mod=None,
           cam=None, canopy_m=0.0, canopy_try=None, heading_hint=None, heading_window=None):
    """Full coarse-to-fine search.  Returns dict(coarse=..., candidates=[...], timing=...).
    cam: None for a normal (pinhole) photo, or dict(proj='cyl'|'equi', y0=horizon row or None,
    ref_fov=...) for a panorama."""
    T = {}
    prog = progress or (lambda msg, frac: None)
    pano = is_pano(cam)
    if fov_range is None:
        fov_range = PANO_FOV_RANGE if pano else (20.0, 90.0)
    if pano:
        cam = dict(cam)
        cam.setdefault('ref_fov', float(hfov) if hfov else 180.0)
    t0 = time.time()
    lat, lon = grid_points(lat0, lon0, radius_km, spacing_km)
    az_c = 1.0
    canopy_list = [float(c) for c in (canopy_try or [])] or None
    coarse_canopy = float(np.median(canopy_list)) if canopy_list else float(canopy_m)
    A = AreaTerrain(lat0, lon0, radius_km + spacing_km, max_km, BANDS_COARSE, canopy_m=coarse_canopy,
                    progress=lambda m, f: prog(m, 0.04 * f))
    T['coarse_tiles'] = time.time() - t0
    hz = A.horizons(lat, lon, az_step=az_c, height=height, progress=lambda m, f: prog(m, 0.04 + 0.46 * f))
    T['coarse_horizons'] = hz['seconds']
    fovs = fov_list(hfov, *fov_range, n=(8 if pano else 9))
    what = 'panorama angles' if pano else 'lens angles'
    prog(f'Matching the photo against {lat.size:,} places' + ('' if hfov else f' and {len(fovs)} {what}'), 0.5)
    t1 = time.time()
    pitches = (0.0,) if pano else (COARSE_PITCHES if hfov else COARSE_PITCHES_UNKNOWN_FOV)
    mc = match_fovs(hz['angle'], hz['az'], sky_x, sky_y, valid, size, fovs, None, phases=2, pitches=pitches, cam=cam,
                    heading_hint=heading_hint, heading_window=heading_window)
    T['coarse_match'] = time.time() - t1
    coarse = dict(lat=lat, lon=lon, score=mc['score'], heading=mc['heading'], fov=mc['fov'],
                  ground=hz['ground'], az_step=az_c, spacing_km=spacing_km)
    tops = distinct_top(lat, lon, mc['score'], n_top, 1.5 * spacing_km)

    if hfov:
        fl1 = lambda c: [float(hfov)]
        fl2 = fl1
    else:
        lo_f, hi_f = fov_range

        def _fl(c, fac):
            return sorted({float(np.clip(round(c['fov'] * f, 2), lo_f, hi_f)) for f in fac})
        fl1 = lambda c: _fl(c, (0.92, 1.0, 1.08))
        fl2 = lambda c: _fl(c, (0.98, 1.0, 1.02))
    t2 = time.time()
    lvl1 = []
    for k, i in enumerate(tops):
        prog(f'Refining candidate {k + 1} of {len(tops)}', 0.55 + 0.25 * k / max(1, len(tops)))
        c = dict(lat=float(lat[i]), lon=float(lon[i]), heading=float(mc['heading'][i]), pitch=float(mc['pitch'][i]),
                 fov=float(mc['fov'][i]), score=float(mc['score'][i]), coarse_score=float(mc['score'][i]))
        lvl1.append(_descend(A, c, fl1, sky_x, sky_y, valid, size, spacing_km / 2.0, max(0.03, spacing_km / 8.0),
                             0.5, None, height, cam=cam, heading_hint=heading_hint, heading_window=heading_window))
    lvl1.sort(key=lambda c: c['score'])
    T['refine1'] = time.time() - t2
    uniq = []
    for c in lvl1:
        if all(haversine_km(c['lat'], c['lon'], o['lat'], o['lon']) > 0.15 for o in uniq):
            uniq.append(c)
    t3 = time.time()
    prog('Loading detailed elevation data for the best places', 0.8)
    best = uniq[:n_fine]
    la_ = np.array([c['lat'] for c in best])
    lo_ = np.array([c['lon'] for c in best])
    cl, cn = float(la_.mean()), float(lo_.mean())
    rad = float(max(haversine_km(cl, cn, la_, lo_).max(), 0.1)) + 0.3
    AF = AreaTerrain(cl, cn, rad, max_km, BANDS_FINE, canopy_m=coarse_canopy)
    wide = pano and max(c['fov'] for c in best) / 2.0 + 9.0 >= 178.0
    cands = []
    for k, c in enumerate(best):
        prog(f'Fine check of the best places ({k + 1} of {len(best)})', 0.82 + 0.17 * k / max(1, len(best)))
        if canopy_list:
            trials = []
            for cm in canopy_list:
                AF.canopy_m = cm
                t = _descend(AF, dict(c), fl2, sky_x, sky_y, valid, size, 0.04, 0.01, 0.2 if wide else 0.1,
                             None if wide else 8.0, height, cam=cam, heading_hint=heading_hint, heading_window=heading_window)
                t['canopy_m'] = cm
                trials.append(t)
            AF.canopy_m = coarse_canopy
            cands.append(min(trials, key=lambda t: t['score']))
        else:
            c2 = _descend(AF, c, fl2, sky_x, sky_y, valid, size, 0.04, 0.01, 0.2 if wide else 0.1,
                          None if wide else 8.0, height, cam=cam, heading_hint=heading_hint, heading_window=heading_window)
            c2['canopy_m'] = coarse_canopy
            cands.append(c2)
    T['refine2'] = time.time() - t3
    cands.sort(key=lambda c: c['score'])
    out = []
    for c in cands + uniq[n_fine:]:
        if all(haversine_km(c['lat'], c['lon'], o['lat'], o['lon']) > 0.15 for o in out):
            out.append(c)
    out.sort(key=lambda c: c['score'])
    for c in out:
        c.setdefault('tilt', 0.0)
        c.setdefault('canopy_m', coarse_canopy)
    T['total'] = time.time() - t0
    prog('Done', 1.0)
    return dict(coarse=coarse, candidates=out[:15], timing=T, tiles=A.tiles + AF.tiles,
                missing=A.missing + AF.missing, n_points=int(lat.size), cam=cam)


def _pano_geom(size, fov, cam):
    W, Hh = size
    kdeg = W / fov
    y0 = cam.get('y0')
    y0 = Hh / 2.0 if y0 is None else float(y0)
    return W, Hh, kdeg, y0


def view_of(lat, lon, heading, fov, pitch, roll, size, n_cols, max_km=120.0, height=1.7, pad=25.0, step=0.05,
            cam=None, tilt=0.0, canopy_m=0.0):
    """What the elevation model shows from a candidate: the skyline projected into the photo (pixel row
    per column, None outside), the panorama around the view, and the ridge points (lat/lon) seen."""
    W, Hh = size
    A = AreaTerrain(lat, lon, 0.05, max_km, BANDS_FINE, canopy_m=canopy_m)
    half = min(180.0, fov / 2.0 + pad)
    if half >= 180.0:
        rel = np.arange(-180.0, 180.0, step)
        az = (heading + rel) % 360.0
    else:
        az = (heading - half + np.arange(0.0, 2 * half + 1e-9, step)) % 360.0
        rel = np.arange(az.size) * step - half
    h = A.horizons(np.array([lat]), np.array([lon]), az=az, height=height)
    ang = h['angle'][0].astype(float)
    dist = h['dist_km'][0].astype(float)
    if is_pano(cam):
        _, _, kdeg, y0 = _pano_geom(size, fov, cam)
        trr = np.radians(rel)
        el_p = ang - (pitch + tilt * np.cos(trr) + roll * np.sin(trr))
        u = W / 2.0 + rel * kdeg
        if cam['proj'] == 'cyl':
            v = y0 - (W / math.radians(fov)) * np.tan(np.radians(np.clip(el_p, -80, 80)))
        else:
            v = y0 - el_p * kdeg
        u = np.where(np.abs(rel) <= fov / 2.0, u, np.nan)
    else:
        # exact pinhole projection with tilt (pitch) and roll
        a_ = np.radians(rel)
        e_ = np.radians(ang)
        x = np.cos(e_) * np.sin(a_)
        y = np.sin(e_)
        z = np.cos(e_) * np.cos(a_)
        p, r = math.radians(pitch), math.radians(roll)
        yc = y * math.cos(p) - z * math.sin(p)
        zc = y * math.sin(p) + z * math.cos(p)
        xr = x * math.cos(r) + yc * math.sin(r)
        yr = -x * math.sin(r) + yc * math.cos(r)
        f = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
        front = zc > 1e-6
        u = np.where(front, f * xr / np.where(front, zc, 1) + W / 2.0, np.nan)
        v = np.where(front, Hh / 2.0 - f * yr / np.where(front, zc, 1), np.nan)
    cols = (np.arange(n_cols) + 0.5) * W / n_cols
    okp = np.isfinite(u)
    o = np.argsort(u[okp])
    yv = np.interp(cols, u[okp][o], v[okp][o], left=np.nan, right=np.nan)
    ycol = [None if not np.isfinite(t) else round(float(t), 1) for t in yv]
    inside = np.abs(rel) <= fov / 2.0
    rlat, rlon = tr.destination(lat, lon, az[inside], dist[inside] * 1000.0)
    every = max(1, int(round(0.2 / step)))
    ridge = [[round(float(a1), 5), round(float(b1), 5)] for a1, b1 in zip(rlat, rlon)][::every]
    return dict(y_cols=ycol, rel=rel, angle=ang, dist_km=dist, az=az, ridge=ridge,
                median_km=float(np.median(dist[inside])) if inside.any() else 10.0, ground=float(h['ground'][0]))


def photo_to_world(x, y, size, fov, pitch, roll, heading, cam=None, tilt=0.0):
    """Photo skyline pixels -> world azimuth / elevation (deg) for a solved camera (inverse of view_of).
    Normal photos: exact pinhole camera with tilt (pitch) and roll.  Panoramas: azimuth is linear in x;
    elevation follows the cylindrical or equirectangular projection, plus the fitted horizon offset
    (pitch), axis tilt (cos term) and roll (sin term)."""
    if is_pano(cam):
        W, Hh, kdeg, y0 = _pano_geom(size, fov, cam)
        th = (np.asarray(x, float) - W / 2.0) / kdeg
        if fov >= 359.0:
            th = ((th + 180.0) % 360.0) - 180.0
        dy = y0 - np.asarray(y, float)
        if cam['proj'] == 'cyl':
            el = np.degrees(np.arctan(dy / (W / math.radians(fov))))
        else:
            el = dy / kdeg
        trr = np.radians(th)
        el = el + pitch + tilt * np.cos(trr) + roll * np.sin(trr)
        return heading + th, el
    W, Hh = size
    f = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
    u = np.asarray(x, float) - W / 2.0
    vv = Hh / 2.0 - np.asarray(y, float)
    n = np.sqrt(u * u + vv * vv + f * f)
    xr, yr, zc = u / n, vv / n, f / n
    p, r = math.radians(pitch), math.radians(roll)
    xc = xr * math.cos(r) - yr * math.sin(r)
    yc = xr * math.sin(r) + yr * math.cos(r)
    yw = yc * math.cos(p) + zc * math.sin(p)
    zw = -yc * math.sin(p) + zc * math.cos(p)
    az = heading + np.degrees(np.arctan2(xc, zw))
    el = np.degrees(np.arctan2(yw, np.hypot(xc, zw)))
    return az, el
