"""
Ridgeline - an oblique 3-D view of a best place: the terrain seen from a point behind and above
the place, looking the way the photo looks, with the photo's view cone tinted on the ground and
the predicted skyline ridge drawn where it lies.  Downloadable as a PNG.

How it is drawn (no graphics card needed)
-----------------------------------------
* Terrain heights come from the same open elevation tiles as the search (AWS Terrain Tiles),
  finer near the camera and coarser far away; the ground is draped with Esri satellite imagery,
  a topographic or a street map, lightly hill-shaded and hazed with distance.
* Each picture column is a ray fan from the camera along one compass direction.  Ground points
  at growing distance are projected into the column (true perspective with the camera's down
  tilt and the Earth's curvature) and painted front to back, so nearer hills hide farther ones.
  The distance of every painted pixel is kept (a depth buffer), and is used to hide the parts of
  the predicted ridge and of the other places that are behind a hill.
* Vertical scale is true (no exaggeration).
"""
import io
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import mapimage as mi
import search as se
import terrain as tr

R = tr.R_EARTH
K_REFR = tr.K_REFRACTION


class Imagery:
    """Basemap tiles of one zoom level over a lat/lon box, sampled bilinearly (RGB 0-1)."""

    def __init__(self, base, z, lat_min, lat_max, lon_min, lon_max, max_tiles=48, progress=None, lo=0.0, hi=1.0,
                 what='imagery'):
        while True:
            x0, y1 = tr.merc_px(np.array([lat_min]), np.array([lon_min]), z)
            x1, y0 = tr.merc_px(np.array([lat_max]), np.array([lon_max]), z)
            tx0, tx1 = int(x0[0] // 256), int(x1[0] // 256)
            ty0, ty1 = int(y0[0] // 256), int(y1[0] // 256)
            if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles or z <= 6:
                break
            z -= 1
        self.z, self.tx0, self.ty0 = z, tx0, ty0
        jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        tiles = mi.fetch_tiles(base, z, jobs, progress, lo, hi, what)
        arr = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), 0.45, np.float32)
        self.missing = 0
        for (tx, ty), t in zip(jobs, tiles):
            if t is None:
                self.missing += 1
                continue
            arr[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = \
                np.asarray(t.resize((256, 256)), np.float32) / 255.0
        self.arr = arr
        self.n = len(jobs)

    def sample(self, lat, lon):
        x, y = tr.merc_px(lat, lon, self.z)
        u = x - self.tx0 * 256 - 0.5
        v = y - self.ty0 * 256 - 0.5
        return np.stack([ndi.map_coordinates(self.arr[..., k], [v, u], order=1, mode='nearest') for k in range(3)], -1)


def _dem(A, lat, lon, d):
    """Terrain height (m) at lat/lon, using the elevation band that suits the distance d (m)."""
    out = np.empty(np.shape(lat), np.float32)
    ends = np.array([b['d'][-1] for b in A.bands])
    bi = np.clip(np.searchsorted(ends, d), 0, len(A.bands) - 1)
    for i, b in enumerate(A.bands):
        m = bi == i
        if m.any():
            x, y = tr.merc_px(lat[m], lon[m], b['z'])
            out[m] = A._sample(b, x, y)
    return np.maximum(out, 0.0)


def _local_xy(lat0, lon0, lat, lon):
    """East / north metres from (lat0, lon0) (small-area approximation)."""
    return (np.radians(np.asarray(lon) - lon0) * R * math.cos(math.radians(lat0)),
            np.radians(np.asarray(lat) - lat0) * R)


def render(view, base='Satellite photo', width=1600, height=1200, progress=None):
    """PNG bytes of the oblique view of the chosen place.

    view: dict(photo_name, cands, sel, ridge [[lat, lon], ...], cone_km, verdict (label, text, colour),
               place_text, pano (bool))"""
    head_h, foot_h = 150, 96
    W, Hh = width, height - head_h - foot_h
    cands = view['cands']
    sel = int(view.get('sel', 0))
    c = cands[sel]
    lat_p, lon_p, hd = float(c['lat']), float(c['lon']), float(c['heading'])
    fov = min(float(c['fov']), 360.0)
    ridge = view.get('ridge') or []
    rla = np.array([p[0] for p in ridge]) if ridge else np.array([])
    rlo = np.array([p[1] for p in ridge]) if ridge else np.array([])
    rdist = (np.hypot(*_local_xy(lat_p, lon_p, rla, rlo)) / 1000.0) if ridge else np.array([5.0])
    d_med = float(np.median(rdist)) if rdist.size else 5.0
    d_far = float(np.percentile(rdist, 95)) if rdist.size else 10.0

    # camera behind and above the place, looking along the photo's direction.  It is placed so that
    # the place sits in the lower third of the picture and the predicted skyline ridge below the
    # top edge, backing away when both do not fit.
    F = 100.0 if fov >= 300.0 else float(np.clip(fov + 25.0, 55.0, 100.0))
    f = (W / 2.0) / math.tan(math.radians(F) / 2.0)
    cy = Hh / 2.0
    vfov = 2.0 * math.atan(cy / f)
    prog = progress or (lambda f, msg: None)
    prog(0.01, 'Loading elevation data')
    A = se.AreaTerrain(lat_p, lon_p, 9.0, min(91.0, 1.3 * d_far + 12.0), se.BANDS_FINE,
                       progress=lambda msg, fr: prog(0.01 + 0.44 * fr, 'Loading elevation data'))
    prog(0.45, 'Placing the camera')
    g_p = float(A.ground(np.array([lat_p]), np.array([lon_p]))[0])
    if ridge:
        rh_all = _dem(A, rla, rlo, rdist * 1000.0)
    back = float(np.clip(0.3 * d_med + 0.5, 0.6, 5.0))
    for _ in range(10):
        cam_lat, cam_lon = (float(v_) for v_ in tr.destination(lat_p, lon_p, (hd + 180.0) % 360.0, back * 1000.0))
        g_cam = float(A.ground(np.array([cam_lat]), np.array([cam_lon]))[0])
        up = float(np.clip(220.0 * back + 150.0, 200.0, 1800.0))
        alt = max(g_cam, g_p) + up
        e_p = math.atan2(g_p + 1.7 - alt, back * 1000.0)
        if ridge:
            ex_, ny_ = _local_xy(cam_lat, cam_lon, rla, rlo)
            dc = np.hypot(ex_, ny_)
            e_r = float(np.max(np.arctan2(rh_all - alt - dc ** 2 * (1 - K_REFR) / (2 * R), dc)))
        else:
            e_r = 0.0
        e_r = max(e_r, 0.0 if fov >= 300.0 else e_r)
        if e_r - e_p <= 0.72 * vfov or back >= 8.0:
            break
        back = min(8.0, back * 1.3)
    pitch = e_p + 0.30 * vfov                      # the place 0.30 * vertical field below the centre
    if e_r - pitch > 0.42 * vfov:                  # keep the ridge inside the top of the picture
        pitch = e_r - 0.42 * vfov
    pitch = float(np.clip(pitch, math.radians(-35.0), math.radians(10.0)))
    d_max = float(np.clip(1.3 * d_far + back + 2.0, 10.0, 90.0)) * 1000.0

    prog(0.47, 'Tracing the terrain')
    # rays: one compass direction per column, geometric distance steps
    cols = np.arange(W) + 0.5
    a_rel = np.arctan((cols - W / 2.0) / f)                                  # rad, relative to heading
    az = (hd + np.degrees(a_rel)) % 360.0
    K = 1000
    d = np.geomspace(15.0, d_max, K)
    la, lo = tr.destination(cam_lat, cam_lon, az[:, None], d[None, :])     # W x K
    h = _dem(A, la, lo, np.broadcast_to(d, la.shape))
    drop = d ** 2 * (1.0 - K_REFR) / (2.0 * R)
    e = np.arctan2(h - alt - drop[None, :], d[None, :])                       # elevation angle of each ground point
    x_ = np.cos(e) * np.sin(a_rel[:, None])
    y_ = np.sin(e)
    z_ = np.cos(e) * np.cos(a_rel[:, None])
    p = pitch
    yc = y_ * math.cos(p) - z_ * math.sin(p)
    zc = y_ * math.sin(p) + z_ * math.cos(p)
    v = cy - f * yc / np.maximum(zc, 1e-6)                                    # screen row of each sample
    v = np.where(zc > 1e-6, v, Hh * 4)
    M = np.minimum.accumulate(v, axis=1)                                      # highest row reached so far
    # for every pixel (row r, column i): first sample whose running minimum reaches the row
    rows = np.arange(Hh) + 0.5
    kstar = np.empty((Hh, W), np.int32)
    for i in range(W):
        kstar[:, i] = np.searchsorted(-M[i], -rows, side='left')               # M[i] is non-increasing
    ground = kstar < K
    ks = np.clip(kstar, 1, K - 1)
    ci = np.broadcast_to(np.arange(W)[None, :], ks.shape)
    m_prev = M[ci, ks - 1]
    v_k = v[ci, ks]
    t = np.clip((m_prev - rows[:, None]) / np.maximum(m_prev - v_k, 1e-6), 0.0, 1.0)
    dpx = d[ks - 1] * (1 - t) + d[ks] * t                                     # distance of the ground at each pixel
    dpx = np.where(ground, dpx, np.inf)
    lat_px = la[ci, ks - 1] * (1 - t) + la[ci, ks] * t
    lon_px = lo[ci, ks - 1] * (1 - t) + lo[ci, ks] * t
    # hill shading from the slope along/across each ray
    dh_along = np.gradient(h, axis=1) / np.gradient(d)[None, :]
    dh_across = np.gradient(h, axis=0) / np.maximum(d[None, :] * np.gradient(a_rel)[:, None], 1.0)
    sun_az = math.radians(315.0 - hd)                                          # light from the north-west on the map
    nx, ny = -dh_across, -dh_along
    shade = (np.cos(math.radians(45)) + math.sin(math.radians(45)) * (nx * math.sin(sun_az) + ny * math.cos(sun_az))) \
        / np.sqrt(1 + nx ** 2 + ny ** 2)
    shade_px = np.clip(shade[ci, ks], 0.15, 1.2)

    # ground colour from the basemap, three detail levels by distance
    rgb = np.zeros((Hh, W, 3), np.float32)
    missing = n_tiles = 0
    prog(0.53, 'Loading imagery')
    bands_t = ((0, 2500, 16, 90, 'near imagery'), (2500, 12000, 14, 64, 'middle-distance imagery'),
               (12000, 1e9, 12, 48, 'distant imagery'))
    for bi_, (d0, d1, z, mt, what) in enumerate(bands_t):
        lo_p, hi_p = 0.53 + 0.11 * bi_, 0.53 + 0.11 * (bi_ + 1)
        msk = ground & (dpx >= d0) & (dpx < d1)
        if not msk.any():
            continue
        L, O = lat_px[msk], lon_px[msk]
        im = Imagery(base, z, float(L.min()) - 0.002, float(L.max()) + 0.002, float(O.min()) - 0.002, float(O.max()) + 0.002,
                     max_tiles=mt, progress=prog, lo=lo_p, hi=hi_p, what=what)
        missing += im.missing
        n_tiles += im.n
        rgb[msk] = im.sample(L, O)
    rgb *= (0.55 + 0.45 * shade_px)[..., None] if base != 'Satellite photo' else (0.8 + 0.2 * shade_px)[..., None]

    prog(0.86, 'Shading, haze and the view of the photo')
    # sky and haze
    tt = np.clip((rows / Hh), 0, 1)[:, None, None]
    sky = (np.array([0.16, 0.29, 0.50]) * (1 - tt) + np.array([0.80, 0.86, 0.92]) * tt).astype(np.float32)
    sky = np.broadcast_to(sky, (Hh, W, 3))
    haze_c = np.array([0.78, 0.84, 0.91], np.float32)
    hz = (1.0 - np.exp(-np.where(ground, dpx, 0.0) / 45000.0))[..., None] * 0.85
    img = np.where(ground[..., None], rgb * (1 - hz) + haze_c * hz, sky)

    # the photo's view cone, tinted on the ground
    gx, gy = _local_xy(lat_p, lon_p, lat_px, lon_px)
    gdist = np.hypot(gx, gy)
    brg = (np.degrees(np.arctan2(gx, gy)) - hd + 540.0) % 360.0 - 180.0
    cone_m = max(float(view.get('cone_km') or 1.0), 1.05 * d_far) * 1000.0
    if fov >= 300.0:
        cone = ground & (gdist <= cone_m)
    else:
        cone = ground & (np.abs(brg) <= fov / 2.0) & (gdist <= cone_m)
    warm = np.array([1.0, 0.80, 0.61], np.float32)
    img = np.where(cone[..., None], img * 0.72 + warm * 0.28, img)
    edge = cone & ~ndi.binary_erosion(cone, iterations=2)
    img = np.where(edge[..., None], np.array([1.0, 0.62, 0.42], np.float32), img)
    centre = ground & (np.abs(brg) < np.degrees(2.5 / np.maximum(gdist, 1.0)) + 0.06) & (gdist <= cone_m) & (gdist > 20)
    img = np.where(centre[..., None], np.array([1.0, 0.48, 0.24], np.float32), img)
    page_map = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert('RGBA')

    # project 3-D points (lat, lon, height) into the picture, with visibility from the depth buffer
    def project(plat, plon, ph):
        ex, ny_ = _local_xy(cam_lat, cam_lon, plat, plon)
        dist = np.hypot(ex, ny_)
        a = np.arctan2(ex, ny_) - math.radians(hd)
        a = (a + math.pi) % (2 * math.pi) - math.pi
        ee = np.arctan2(ph - alt - dist ** 2 * (1 - K_REFR) / (2 * R), dist)
        xx = np.cos(ee) * np.sin(a)
        yy = np.sin(ee)
        zz = np.cos(ee) * np.cos(a)
        ycc = yy * math.cos(p) - zz * math.sin(p)
        zcc = yy * math.sin(p) + zz * math.cos(p)
        u = W / 2.0 + f * np.tan(a)
        vv = cy - f * ycc / np.maximum(zcc, 1e-6)
        ok = (zcc > 1e-6) & (np.abs(a) < math.radians(F / 2.0 + 1)) & (u >= 0) & (u < W) & (vv >= 0) & (vv < Hh)
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(vv.astype(int), 0, Hh - 1)
        depth = np.where(ok, dpx[np.clip(vi + 2, 0, Hh - 1), ui], np.inf)
        vis = ok & (dist <= depth * 1.03 + 60.0)
        return u, vv, vis, dist

    prog(0.92, 'Drawing the skyline ridge and the places')
    S = 2
    ov = Image.new('RGBA', (W * S, Hh * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    if ridge:
        rh = rh_all + 3.0
        u, vv, vis, _ = project(rla, rlo, rh)
        seg = []
        for k in range(len(u)):
            if vis[k] and (not seg or math.hypot(u[k] - seg[-1][0] / S, vv[k] - seg[-1][1] / S) < 40):
                seg.append((u[k] * S, vv[k] * S))
            else:
                if len(seg) > 1:
                    dr.line(seg, fill=(0, 0, 0, 160), width=7 * S, joint='curve')
                    dr.line(seg, fill=mi.PINK + (255,), width=4 * S, joint='curve')
                seg = [(u[k] * S, vv[k] * S)] if vis[k] else []
        if len(seg) > 1:
            dr.line(seg, fill=(0, 0, 0, 160), width=7 * S, joint='curve')
            dr.line(seg, fill=mi.PINK + (255,), width=4 * S, joint='curve')
    # the other best places, then the chosen one, as pins
    fnum = mi._font(13 * S, True)
    order = [i for i in range(len(cands)) if i != sel] + [sel]
    for i in order:
        q = cands[i]
        qh = _dem(A, np.array([q['lat']]), np.array([q['lon']]), np.array([1000.0])) + 1.7
        u, vv, vis, dist = project(np.array([q['lat']]), np.array([q['lon']]), qh)
        if not vis[0] and i != sel:
            continue
        if not (0 <= u[0] < W and 0 <= vv[0] < Hh):
            continue
        x0, y0 = u[0] * S, vv[0] * S
        chosen = i == sel
        r = (15 if chosen else 11) * S
        stem = (46 if chosen else 30) * S
        dr.line([(x0, y0), (x0, y0 - stem)], fill=(255, 255, 255, 230), width=2 * S)
        dr.ellipse([x0 - 3 * S, y0 - 3 * S, x0 + 3 * S, y0 + 3 * S], fill=(255, 255, 255, 255))
        cx_, cy_ = x0, y0 - stem - r
        dr.ellipse([cx_ - r - 2 * S, cy_ - r - 2 * S, cx_ + r + 2 * S, cy_ + r + 2 * S], fill=(0, 0, 0, 120))
        dr.ellipse([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=(mi.ORANGE if chosen else mi.PANEL) + (255,),
                   outline=((255, 255, 255) if chosen else mi.ACCENT) + (255,), width=2 * S)
        dr.text((cx_, cy_), str(i + 1), font=fnum if chosen else mi._font(11 * S, True),
                fill=(mi.BG if chosen else mi.ACCENT) + (255,), anchor='mm')
    ov = ov.resize((W, Hh), Image.LANCZOS)
    page_map.alpha_composite(ov)

    # compass strip along the top of the view
    md = ImageDraw.Draw(page_map)
    fs = mi._font(14, True)
    for deg in range(0, 360, 10):
        a = math.radians(((deg - hd + 540) % 360) - 180)
        if abs(a) >= math.radians(F / 2.0):
            continue
        u = W / 2.0 + f * math.tan(a)
        major = deg % 30 == 0
        md.line([(u, 0), (u, 12 if major else 6)], fill=(255, 255, 255, 220), width=2 if major else 1)
        if major:
            lab = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}.get(deg, f'{deg}°')
            md.text((u, 24), lab, font=fs, fill=(255, 255, 255, 255), anchor='mm', stroke_width=2, stroke_fill=(0, 0, 0, 255))

    page = Image.new('RGB', (width, height), mi.BG)
    page.paste(page_map.convert('RGB'), (0, head_h))
    pd = ImageDraw.Draw(page)
    pd.text((24, 18), 'Ridgeline', font=mi._font(34, True), fill=mi.ACCENT)
    pd.text((210, 30), f"3-D view of place #{sel + 1}   {view.get('photo_name', '')}"[:120], font=mi._font(18), fill=mi.SUB)
    vlab, vtext, vcol = view.get('verdict') or ('', '', '#aab4c6')
    x = 24
    if vlab:
        fpill = mi._font(17, True)
        tw = pd.textlength(vlab, font=fpill)
        pd.rounded_rectangle([x, 70, x + tw + 22, 98], 12, fill=mi._hex(vcol))
        pd.text((x + 11, 84), vlab, font=fpill, fill=mi.BG, anchor='lm')
        x += tw + 36
    sub = (f"Seen from {back:.1f} km behind place #{sel + 1} and {alt - g_p:,.0f} m above it, looking "
           f"{hd:.0f}° {mi_compass(hd)}, {abs(math.degrees(pitch)):.0f}° {'down' if pitch < 0 else 'up'}. True vertical scale.")
    pd.text((x, 84), sub, font=mi._font(16), fill=mi.TEXT, anchor='lm')
    pd.text((24, 122), view.get('place_text', ''), font=mi._font(17, True), fill=mi.TEXT, anchor='lm')
    fy = head_h + Hh + 18
    fl = mi._font(15)
    x = 24
    pd.polygon([(x, fy + 16), (x + 22, fy), (x + 22, fy + 16)], fill=(200, 170, 140), outline=(255, 158, 107))
    x += 30
    pd.text((x, fy + 8), 'view of the photo (tinted ground, centre line)', font=fl, fill=mi.TEXT, anchor='lm')
    x += pd.textlength('view of the photo (tinted ground, centre line)', font=fl) + 26
    pd.line([(x, fy + 8), (x + 24, fy + 8)], fill=mi.PINK, width=4)
    x += 32
    pd.text((x, fy + 8), 'predicted skyline ridge (hidden parts left out)', font=fl, fill=mi.TEXT, anchor='lm')
    x += pd.textlength('predicted skyline ridge (hidden parts left out)', font=fl) + 26
    pd.ellipse([x, fy - 1, x + 18, fy + 17], fill=mi.ORANGE, outline=(255, 255, 255), width=2)
    x += 26
    pd.text((x, fy + 8), 'the place (numbers = rank)', font=fl, fill=mi.TEXT, anchor='lm')
    note = (f"{mi.ATTRIBUTION[base]}  |  Elevation: AWS Terrain Tiles  |  Made with Ridgeline, {date.today():%d %b %Y}")
    if missing:
        note += f'  |  {missing} of {n_tiles} map tiles could not be loaded'
    pd.text((24, fy + 46), note, font=mi._font(13), fill=mi.SUB, anchor='lm')
    prog(0.96, 'Saving the picture')
    out = io.BytesIO()
    page.save(out, format='PNG', optimize=True)
    prog(1.0, 'Done')
    return out.getvalue()


_DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def mi_compass(az):
    return _DIRS[int(round((az % 360) / 22.5)) % 16]


_CACHE = {}


def render_cached(key, view, base='Satellite photo', progress=None):
    if key not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = render(view, base, progress=progress)
    elif progress:
        progress(1.0, 'Done (already drawn)')
    return _CACHE[key]
