"""
Ridgeline - a picture (PNG) of the results map, for downloading after a search.

The map is drawn on the server, so it looks the same on a phone or a computer:
Esri basemap tiles (topographic, satellite photo or streets) with the match heat map, the search
circle, the starting point, the numbered best places, the view cone and skyline ridge of the chosen
place, the photo's own GPS position if it has one, a header with the verdict, a legend, a scale bar,
a north arrow and the basemap attribution.
"""
import io
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import terrain as tr

TILES = {
    'Topographic': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    'Satellite photo': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'Streets': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
}
ATTRIBUTION = {
    'Topographic': 'Basemap: Esri, HERE, Garmin, FAO, NOAA, USGS, EPA, NPS',
    'Satellite photo': 'Imagery: Esri, Maxar, Earthstar Geographics, GIS user community',
    'Streets': 'Basemap: Esri, HERE, Garmin, USGS, NGA, OpenStreetMap contributors',
}
MAX_Z = {'Topographic': 17, 'Satellite photo': 18, 'Streets': 18}
CACHE = os.path.join(os.path.dirname(tr.CACHE_DIR.rstrip('\\/')) or tempfile.gettempdir(), 'maptiles')
_session = requests.Session()
_session.headers['User-Agent'] = 'Ridgeline photo-locator/1.0'

BG = (11, 17, 28)
PANEL = (17, 25, 39)
TEXT = (244, 239, 232)
SUB = (201, 209, 224)
ACCENT = (255, 205, 156)
EMBER = (181, 70, 26)
PINK = (255, 79, 123)
ORANGE = (255, 122, 61)
CYAN = (126, 224, 255)
HEAT = ['#20304d', '#5b3a8c', '#c2457a', '#ff9f6b', '#ffe680']


def _font(size, bold=False):
    try:
        import matplotlib
        p = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def _hex(c):
    c = c.lstrip('#')
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _tile(base, z, x, y):
    n = 2 ** z
    x %= n
    if not 0 <= y < n:
        return None
    p = os.path.join(CACHE, base.replace(' ', '_'), str(z), str(x), f'{y}.img')
    try:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return Image.open(io.BytesIO(f.read())).convert('RGB')
    except Exception:
        pass
    try:
        r = _session.get(TILES[base].format(z=z, x=x, y=y), timeout=20)
        if r.status_code != 200 or not r.headers.get('content-type', '').startswith('image'):
            return None
        im = Image.open(io.BytesIO(r.content)).convert('RGB')
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'wb') as f:
                f.write(r.content)
        except OSError:
            pass
        return im
    except Exception:
        return None


def fetch_tiles(base, z, jobs, progress=None, lo=0.0, hi=1.0, what='map tiles'):
    """Download tiles in parallel, reporting progress as each one arrives."""
    from concurrent.futures import as_completed
    out = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_tile, base, z, j[0], j[1]): i for i, j in enumerate(jobs)}
        for n, fu in enumerate(as_completed(futs), 1):
            out[futs[fu]] = fu.result()
            if progress and (n == len(jobs) or n % 3 == 0):
                progress(lo + (hi - lo) * n / max(1, len(jobs)), f'Loading {what} ({n} of {len(jobs)})')
    return out


def _basemap(base, lat_c, lon_c, half_km, w, h, progress=None):
    """Stitched basemap of w x h pixels centred on (lat_c, lon_c) showing at least +-half_km."""
    ppk_needed = min(w, h) / (2.0 * half_km)                     # pixels per km needed
    z = 3
    for zz in range(3, MAX_Z[base] + 1):
        ppk = 256 * 2 ** zz / (40075.016686 * math.cos(math.radians(lat_c)))
        if ppk <= ppk_needed:
            z = zz
    cx, cy = tr.merc_px(np.array([lat_c]), np.array([lon_c]), z)
    cx, cy = float(cx[0]), float(cy[0])
    # draw at zoom z, then scale so that the extent fills the frame
    ppk = 256 * 2 ** z / (40075.016686 * math.cos(math.radians(lat_c)))
    scale = ppk_needed / ppk
    sw, sh = int(math.ceil(w / scale)) + 2, int(math.ceil(h / scale)) + 2
    x0, y0 = cx - sw / 2.0, cy - sh / 2.0
    tx0, ty0 = int(math.floor(x0 / 256)), int(math.floor(y0 / 256))
    tx1, ty1 = int(math.floor((x0 + sw) / 256)), int(math.floor((y0 + sh) / 256))
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    tiles = fetch_tiles(base, z, jobs, progress, 0.02, 0.75)
    canvas = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (40, 46, 58))
    missing = 0
    for (tx, ty), t in zip(jobs, tiles):
        if t is None:
            missing += 1
            continue
        canvas.paste(t, ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = x0 - tx0 * 256, y0 - ty0 * 256
    crop = canvas.crop((int(ox), int(oy), int(ox) + sw, int(oy) + sh))
    img = crop.resize((int(round(sw * scale)), int(round(sh * scale))), Image.LANCZOS).crop((0, 0, w, h))

    def to_px(lat, lon):
        px, py = tr.merc_px(np.asarray(lat, float), np.asarray(lon, float), z)
        return (px - x0) * scale, (py - y0) * scale
    return img, to_px, missing, len(jobs), ppk_needed


def _heat_rgba(coarse, start):
    lat, lon, s = np.asarray(coarse['lat']), np.asarray(coarse['lon']), np.asarray(coarse['score'], float)
    sp = coarse['spacing_km']
    dlat = math.degrees(sp * 1000 / tr.R_EARTH)
    dlon = math.degrees(sp * 1000 / (tr.R_EARTH * math.cos(math.radians(start['lat']))))
    iy = np.round((lat - start['lat']) / dlat).astype(int)
    ix = np.round((lon - start['lon']) / dlon).astype(int)
    ny, nx = iy.max() - iy.min() + 1, ix.max() - ix.min() + 1
    g = np.full((ny, nx), np.nan)
    g[iy.max() - iy, ix - ix.min()] = s
    fin = np.isfinite(g)
    best, mid = np.nanmin(g), np.nanpercentile(g, 50)
    v = np.where(fin, np.clip((mid - g) / max(1e-6, mid - best), 0, 1), 0) ** 1.5
    stops = np.array([_hex(c) for c in HEAT], float)
    pos = np.linspace(0, 1, len(HEAT))
    rgb = np.stack([np.interp(v, pos, stops[:, k]) for k in range(3)], -1)
    a = np.where(fin, (0.12 + 0.83 * v ** 0.8) * 255 * 0.7, 0)
    img = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), 'RGBA')
    north = start['lat'] + iy.max() * dlat + dlat / 2
    south = start['lat'] + iy.min() * dlat - dlat / 2
    west = start['lon'] + ix.min() * dlon - dlon / 2
    east = start['lon'] + ix.max() * dlon + dlon / 2
    return img, (north, west, south, east)


def _dashed(d, pts, fill, width, dash=14, gap=10):
    acc, on = 0.0, True
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        t = 0.0
        while t < seg:
            step = min((dash if on else gap) - acc, seg - t)
            if on:
                a = t / seg
                b = (t + step) / seg
                d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)],
                       fill=fill, width=width)
            t += step
            acc += step
            if acc >= (dash if on else gap) - 1e-9:
                acc, on = 0.0, not on


def _star(cx, cy, r):
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.45
        a = -math.pi / 2 + k * math.pi / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _nice_km(target):
    for v in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if v >= target:
            return v
    return 100


_CACHE = {}


def render_cached(key, view, base='Topographic', extent='area', progress=None):
    if key not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = render(view, base, extent, progress=progress)
    elif progress:
        progress(1.0, 'Done (already drawn)')
    return _CACHE[key]


def render(view, base='Topographic', extent='area', width=1600, height=1200, progress=None):
    """PNG bytes of the results map.

    view: dict(photo_name, start{lat,lon,label}, radius_km, coarse (lat, lon, score, spacing_km),
               cands [{lat, lon, heading, fov, rms_deg}], sel, ridge [[lat, lon], ...] of the chosen place,
               cone_km, exif {lat, lon} or None, verdict (label, text, colour hex), place_text, pano (bool))
    extent: 'area' (whole search area) or 'place' (close-up of the chosen place)."""
    head_h, foot_h = 150, 118
    mw, mh = width, height - head_h - foot_h
    st0 = view['start']
    cands = view.get('cands') or []
    sel = min(max(0, int(view.get('sel', 0))), max(0, len(cands) - 1))
    c = cands[sel] if cands else None
    if extent == 'place' and c:
        lat_c, lon_c = c['lat'], c['lon']
        half = min(max(1.2, 1.08 * float(view.get('cone_km') or 5.0)), 30.0)
    else:
        lat_c, lon_c = st0['lat'], st0['lon']
        half = 1.12 * float(view['radius_km'])
    prog = progress or (lambda f, msg: None)
    prog(0.01, 'Choosing the map area')
    img, to_px, missing, n_tiles, ppk = _basemap(base, lat_c, lon_c, half, mw, mh, progress=prog)
    prog(0.78, 'Drawing the heat map, places and view')
    img = img.convert('RGBA')

    S = 2                                                     # overlay drawn at 2x, then smoothed
    ov = Image.new('RGBA', (mw * S, mh * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    P = lambda lat, lon: tuple(float(v) * S for v in to_px(lat, lon))

    # heat map
    if view.get('coarse') is not None:
        hm, (n_, w_, s_, e_) = _heat_rgba(view['coarse'], st0)
        x1, y1 = P(n_, w_)
        x2, y2 = P(s_, e_)
        if x2 - x1 > 2 and y2 - y1 > 2:
            hm = hm.resize((int(x2 - x1), int(y2 - y1)), Image.BILINEAR)
            ov.alpha_composite(hm, (int(x1), int(y1)))
    # places that fit almost as well as the best one
    rects = (view.get('region') or {}).get('rects') or []
    if rects:
        layer = Image.new('RGBA', ov.size, (0, 0, 0, 0))
        dl = ImageDraw.Draw(layer)
        for s_lat, w_lon, n_lat, e_lon in rects:
            x1, y1 = P(n_lat, w_lon)
            x2, y2 = P(s_lat, e_lon)
            dl.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=(126, 224, 200, 80))
        ov.alpha_composite(layer)
    # search circle
    az = np.arange(0, 361, 2.0)
    cla, clo = tr.destination(st0['lat'], st0['lon'], az, view['radius_km'] * 1000.0)
    circ = [P(a, b) for a, b in zip(cla, clo)]
    _dashed(d, circ, (255, 159, 107, 255), 3 * S, dash=14 * S, gap=9 * S)
    # cone and ridge of the chosen place
    if c:
        km = float(view.get('cone_km') or 8.0) * 1000.0
        rr = view.get('ridge') or []
        if rr:      # reach the skyline ridges seen in the photo (95 % of them), at most 60 km
            dk = tr.R_EARTH / 1000.0 * np.radians(np.hypot(
                np.array([a for a, _ in rr]) - c['lat'],
                (np.array([b for _, b in rr]) - c['lon']) * math.cos(math.radians(c['lat']))))
            km = max(km, min(60000.0, 1050.0 * float(np.percentile(dk, 95))))
        fov = min(float(c['fov']), 359.9)
        if fov >= 300.0:
            # (nearly) all-round panorama: a cone would be a disk larger than the map, so draw a ring
            # of a size that fits the frame, with the direction of the picture's centre as an arrow
            km = min(km, 0.38 * min(mw, mh) / ppk * 1000.0)
        angs = np.linspace(c['heading'] - fov / 2, c['heading'] + fov / 2, 48 if fov < 300 else 181)
        pla, plo = tr.destination(c['lat'], c['lon'], angs, km)
        poly = [P(c['lat'], c['lon'])] + [P(a, b) for a, b in zip(pla, plo)]
        layer = Image.new('RGBA', ov.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(poly if fov < 300 else poly[1:], fill=(255, 205, 156, 52))
        ov.alpha_composite(layer)
        d.line((poly + [poly[0]]) if fov < 300 else poly[1:], fill=(255, 205, 156, 230), width=2 * S)
        hla, hlo = tr.destination(c['lat'], c['lon'], c['heading'], km)
        d.line([P(c['lat'], c['lon']), P(float(hla), float(hlo))], fill=ORANGE + (255,), width=3 * S)
        if fov >= 300.0:
            hx, hy = P(float(hla), float(hlo))
            ox_, oy_ = P(c['lat'], c['lon'])
            ang = math.atan2(hy - oy_, hx - ox_)
            L_ = 18 * S
            d.polygon([(hx, hy), (hx - L_ * math.cos(ang - 0.45), hy - L_ * math.sin(ang - 0.45)),
                       (hx - L_ * math.cos(ang + 0.45), hy - L_ * math.sin(ang + 0.45))], fill=ORANGE + (255,))
        ridge = view.get('ridge') or []
        if len(ridge) > 1:
            rp = [P(a, b) for a, b in ridge]
            gaps = [0] + [i for i in range(1, len(rp)) if math.hypot(rp[i][0] - rp[i - 1][0], rp[i][1] - rp[i - 1][1]) > 60 * S] + [len(rp)]
            for a0, a1 in zip(gaps[:-1], gaps[1:]):
                if a1 - a0 > 1:
                    d.line(rp[a0:a1], fill=(0, 0, 0, 150), width=6 * S, joint='curve')
                    d.line(rp[a0:a1], fill=PINK + (255,), width=3 * S, joint='curve')
    # start point
    sx, sy = P(st0['lat'], st0['lon'])
    d.ellipse([sx - 9 * S, sy - 9 * S, sx + 9 * S, sy + 9 * S], fill=EMBER + (255,), outline=(255, 255, 255, 255), width=2 * S)
    # photo GPS
    ex = view.get('exif')
    if ex:
        exx, exy = P(ex['lat'], ex['lon'])
        d.polygon(_star(exx, exy, 15 * S), fill=CYAN + (255,), outline=(0, 0, 0, 255))
    # candidates (draw the chosen one last)
    fnum = _font(13 * S, True)
    order = [i for i in range(len(cands)) if i != sel] + ([sel] if c else [])
    for i in order:
        q = cands[i]
        x, y = P(q['lat'], q['lon'])
        if not (-40 * S < x < mw * S + 40 * S and -40 * S < y < mh * S + 40 * S):
            continue
        r = 14 * S
        chosen = i == sel
        d.ellipse([x - r - 2 * S, y - r - 2 * S, x + r + 2 * S, y + r + 2 * S], fill=(0, 0, 0, 120))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(ORANGE if chosen else PANEL) + (255,),
                  outline=((255, 255, 255) if chosen else ACCENT) + (255,), width=2 * S)
        d.text((x, y), str(i + 1), font=fnum, fill=(BG if chosen else ACCENT) + (255,), anchor='mm')
    ov = ov.resize((mw, mh), Image.LANCZOS)
    img.alpha_composite(ov)

    # map furniture: north arrow and scale bar
    md = ImageDraw.Draw(img)
    fs = _font(15)
    fsb = _font(15, True)
    nx_, ny_ = mw - 46, 26
    md.rounded_rectangle([nx_ - 26, ny_ - 16, nx_ + 26, ny_ + 62], 10, fill=BG + (200,))
    md.polygon([(nx_, ny_ - 6), (nx_ - 12, ny_ + 30), (nx_, ny_ + 22), (nx_ + 12, ny_ + 30)], fill=ACCENT + (255,))
    md.text((nx_, ny_ + 46), 'N', font=fsb, fill=TEXT + (255,), anchor='mm')
    km_bar = _nice_km(0.18 * mw / ppk)
    L = km_bar * ppk
    bx, by = 22, mh - 30
    md.rounded_rectangle([bx - 10, by - 30, bx + L + 70, by + 16], 8, fill=BG + (200,))
    md.rectangle([bx, by - 4, bx + L, by + 4], fill=TEXT + (255,))
    md.rectangle([bx, by - 4, bx + L / 2, by + 4], fill=EMBER + (255,))
    lab = f'{km_bar:g} km' if km_bar >= 1 else f'{km_bar * 1000:.0f} m'
    md.text((bx + L + 8, by), lab, font=fs, fill=TEXT + (255,), anchor='lm')
    md.text((bx, by - 16), '0', font=_font(12), fill=SUB + (255,), anchor='lm')

    # page: header, map, footer
    page = Image.new('RGB', (width, height), BG)
    page.paste(img.convert('RGB'), (0, head_h))
    pd = ImageDraw.Draw(page)
    pd.text((24, 18), 'Ridgeline', font=_font(34, True), fill=ACCENT)
    pd.text((210, 30), f"where was this photo taken?   {view.get('photo_name', '')}"[:120], font=_font(18), fill=SUB)
    vlab, vtext, vcol = view.get('verdict') or ('', '', '#aab4c6')
    x = 24
    if vlab:
        fpill = _font(17, True)
        tw = pd.textlength(vlab, font=fpill)
        pd.rounded_rectangle([x, 70, x + tw + 22, 98], 12, fill=_hex(vcol))
        pd.text((x + 11, 84), vlab, font=fpill, fill=BG, anchor='lm')
        x += tw + 36
    ftx = _font(16)
    vt = vtext
    while pd.textlength(vt, font=ftx) > width - x - 24 and len(vt) > 10:
        vt = vt[:-12].rstrip() + '...'
    pd.text((x, 84), vt, font=ftx, fill=TEXT, anchor='lm')
    pd.text((24, 122), view.get('place_text', ''), font=_font(17, True), fill=TEXT, anchor='lm')
    # footer legend
    fy = head_h + mh + 18
    fl = _font(15)
    items = []
    if view.get('coarse') is not None:
        items.append(('heat', 'match: poor to good'))
    all_round = bool(c) and min(float(c['fov']), 359.9) >= 300.0
    if rects:
        items.append(('region', 'area that fits almost as well'))
    items += [('cand', 'best places (number = rank)'),
              ('ring', 'all-round view (arrow = centre of the picture)') if all_round else ('cone', 'view of the photo'),
              ('ridge', 'skyline ridge'), ('circle', 'search area'), ('start', 'starting point')]
    if ex:
        items.append(('star', 'photo GPS position'))
    x = 24
    for kind, label in items:
        if kind == 'heat':
            for k in range(60):
                col = _hex(HEAT[min(len(HEAT) - 1, int(k / 60 * len(HEAT)))])
                pd.line([(x + k, fy + 2), (x + k, fy + 14)], fill=col)
            x += 68
        elif kind == 'cand':
            pd.ellipse([x, fy - 1, x + 18, fy + 17], fill=PANEL, outline=ACCENT, width=2)
            pd.text((x + 9, fy + 8), '1', font=_font(11, True), fill=ACCENT, anchor='mm')
            x += 26
        elif kind == 'cone':
            pd.polygon([(x, fy + 16), (x + 22, fy), (x + 22, fy + 16)], fill=(120, 105, 90), outline=ACCENT)
            x += 30
        elif kind == 'region':
            pd.rectangle([x, fy, x + 22, fy + 16], fill=(96, 150, 136), outline=(126, 224, 200))
            x += 30
        elif kind == 'ring':
            pd.ellipse([x, fy - 1, x + 18, fy + 17], fill=(120, 105, 90), outline=(255, 205, 156), width=2)
            pd.line([(x + 9, fy + 8), (x + 26, fy + 8)], fill=ACCENT, width=3)
            pd.polygon([(x + 30, fy + 8), (x + 24, fy + 4), (x + 24, fy + 12)], fill=ACCENT)
            x += 38
        elif kind == 'ridge':
            pd.line([(x, fy + 8), (x + 24, fy + 8)], fill=PINK, width=4)
            x += 32
        elif kind == 'circle':
            pd.line([(x, fy + 8), (x + 9, fy + 8)], fill=(255, 159, 107), width=3)
            pd.line([(x + 14, fy + 8), (x + 23, fy + 8)], fill=(255, 159, 107), width=3)
            x += 31
        elif kind == 'start':
            pd.ellipse([x + 2, fy, x + 18, fy + 16], fill=EMBER, outline=(255, 255, 255), width=2)
            x += 26
        elif kind == 'star':
            pd.polygon(_star(x + 10, fy + 8, 10), fill=CYAN)
            x += 26
        pd.text((x, fy + 8), label, font=fl, fill=TEXT, anchor='lm')
        x += pd.textlength(label, font=fl) + 26
    note = (f"{ATTRIBUTION[base]}  |  Elevation: AWS Terrain Tiles  |  Made with Ridgeline, {date.today():%d %b %Y}")
    if missing:
        note += f'  |  {missing} of {n_tiles} map tiles could not be loaded'
    pd.text((24, fy + 46), note, font=_font(13), fill=SUB, anchor='lm')
    pd.text((24, fy + 72), 'Heat map: how well the photo skyline fits the skyline seen from each tested place '
                           '(bright = good). Trees and buildings are not in the elevation model.',
            font=_font(13), fill=SUB, anchor='lm')
    prog(0.93, 'Saving the picture')
    out = io.BytesIO()
    page.save(out, format='PNG', optimize=True)
    prog(1.0, 'Done')
    return out.getvalue()
