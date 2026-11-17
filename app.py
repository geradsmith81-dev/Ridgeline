"""
Ridgeline - find where a landscape photograph was taken by matching its skyline against the
skylines computed from open elevation data around a starting place you suggest.

Run with:  streamlit run app.py
"""
import base64
import hashlib
import html
import io
import math
import os
import time

import numpy as np
import streamlit as st

st.set_page_config(page_title='Ridgeline', page_icon='assets/ridgeline_icon.png', layout='wide',
                   initial_sidebar_state='collapsed')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import components as cm
import gdrive
import mapimage as mi
import oblique as ob
import places
import search as se
import skyline as sk
import terrain as tr

ss = st.session_state
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def compass(az):
    return DIRS[int(round((az % 360) / 22.5)) % 16]


# ------------------------------------------------------------------------------------------
# look (Afterglow theme)
# ------------------------------------------------------------------------------------------
st.markdown("""
<style>
.block-container, [data-testid="stMainBlockContainer"] {padding-top: 4.6rem; max-width: 1250px;}
.ag-title {font-size: 2.1rem; font-weight: 800; margin: 0.4rem 0 0 0; line-height: 1.25; color: #ffcd9c; letter-spacing: .5px;}
.ag-sub {color: #c9d1e0; margin: 0 0 .6rem 0;}
.ag-step {font-size: 1.25rem; font-weight: 800; color: #ffcd9c; margin: 1.1rem 0 .3rem 0;
          border-bottom: 1px solid #2a3550; padding-bottom: 4px;}
.ag-step .n {display: inline-block; width: 1.7rem; height: 1.7rem; line-height: 1.7rem; text-align: center;
             border-radius: 50%; background: #b5461a; color: #fff; font-size: 1rem; margin-right: .45rem;}
.ag-ask {background: #2a1d1a; border: 1px solid #b5461a; border-radius: 12px; padding: 10px 14px; color: #f4efe8;
         margin: 4px 0 8px;}
.ag-ask b {color: #ffcd9c;}
.ag-cards, .ag-score, .ag-tab, .ag-wrap, .ag-note {color: #f4efe8;}
.ag-cards {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 6px 0 10px;}
@media (max-width: 760px) {.ag-cards {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.ag-card {background: #111927; border: 1px solid #2a3550; border-radius: 12px; padding: 10px 12px;}
.ag-card .k {font-size: .8rem; color: #aab4c6; text-transform: uppercase; letter-spacing: .6px;}
.ag-card .v {font-size: 1.45rem; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; line-height: 1.3;}
.ag-card .s {font-size: .8rem; color: #c9d1e0; line-height: 1.35;}
.ag-c1 {border-top: 3px solid #ffd25e;} .ag-c2 {border-top: 3px solid #ff7a3d;}
.ag-c3 {border-top: 3px solid #a9dcff;} .ag-c4 {border-top: 3px solid #ff4f7b;}
.ag-note {background: #111927; border: 1px solid #2a3550; border-radius: 12px; padding: 10px 14px; margin: 6px 0;}
.ag-note b {color: #ffcd9c;}
.ag-pill {display: inline-block; text-align: center; border-radius: 10px; padding: 1px 9px; font-weight: 700; color: #0b111c;}
table.ag-tab {border-collapse: collapse; width: 100%; font-size: .9rem;}
table.ag-tab th {text-align: left; color: #aab4c6; font-weight: 600; border-bottom: 1px solid #2a3550; padding: 5px 6px;}
table.ag-tab td {border-bottom: 1px solid #1c2638; padding: 5px 6px; vertical-align: middle;}
table.ag-tab td a {color: #6fd0dc;}
.ag-wrap {overflow-x: auto;}
.ag-credit {color: #ffcd9c; font-size: .92rem; text-align: center; margin: .6rem 0 1.2rem 0;}
.ag-thumbname {font-size: .78rem; color: #dfe5ef; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
</style>
""", unsafe_allow_html=True)

# Always dark, whatever the phone or computer is set to (as in Afterglow).
_SCOPE = ':is([data-testid="stMain"], [data-testid="stDialog"], [data-testid="stPopoverBody"], [data-testid="stSidebar"])'
_TEXT = (':is(p, li, span, label, small, summary, b, strong, em, td, th, [data-testid="stMarkdownContainer"], '
         '[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *)')
# NB: no spaces between the :is(...) and :not(...) parts - a space would mean 'descendant of'
_NOT = ''.join(':not(%s)' % x for x in (
    ':is(h1, h2, h3, h4, h5, h6) *', '[class^="ag-"]', '[class*=" ag-"]', '[class^="ag-"] *', '[class*=" ag-"] *',
    '.leaflet-container *', '[style*="color"]', '[style*="color"] *', 'a', 'a *', 'code', 'code *',
    '[data-testid="stBaseButton-primary"] *', '[data-testid="stCaptionContainer"] *'))
FORCE_TEXT_SELECTOR = 'body ' + _SCOPE + ' ' + _TEXT + _NOT
_FORCE_DARK = (
    '.stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] '
    '{ background-color: #0b111c !important; }\n'
    '[data-testid="stHeader"] { background-color: rgba(11, 17, 28, 0.92) !important; }\n'
    '[data-testid="stSidebar"] > div:first-child { background-color: #111927 !important; }\n'
    + FORCE_TEXT_SELECTOR + ' { color: #f4efe8 !important; }\n'
    '[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] :is(p, span, li, b, strong, em) '
    '{ color: #dfe5ef !important; }\n'
    '[data-testid="stCaptionContainer"] a, [data-testid="stMarkdownContainer"] a { color: #6fd0dc !important; }\n'
    'h1, h2, h3, h4 { color: #ffcd9c !important; }\n')
with open(os.path.join(HERE, 'assets', 'light_mode.css'), encoding='utf-8') as _f:
    st.markdown('<style>' + _FORCE_DARK + _f.read() + '</style>', unsafe_allow_html=True)

st.markdown('<div class="ag-title">Ridgeline</div>'
            '<div class="ag-sub">Find where a landscape photo was taken: the app reads the skyline in the picture '
            'and compares it with the real hills and mountains around a place you suggest.</div>',
            unsafe_allow_html=True)


def step(n, text):
    st.markdown(f'<div class="ag-step"><span class="n">{n}</span>{html.escape(text)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------
# cached work
# ------------------------------------------------------------------------------------------
@st.cache_data(max_entries=6, show_spinner=False)
def decode_photo(data):
    im, info = sk.load_image(data)
    return im, info


@st.cache_data(max_entries=6, show_spinner=False)
def detect_skyline(h, data, smooth, edge, work_width=sk.WORK_WIDTH):
    im, info = decode_photo(data)
    d = sk.detect(im, smooth=smooth, edge_weight=edge, work_width=work_width)
    d.pop('sky', None)
    return d


@st.cache_data(max_entries=6, show_spinner=False)
def display_image(h, data, max_w=1400):
    im, _ = decode_photo(data)
    url, size = cm.image_data_url(im, max_w=max_w)
    return url, size


@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def find_places(q):
    return places.search_places(q)


@st.cache_data(ttl=600, max_entries=10, show_spinner=False)
def drive_list(folder_id):
    return gdrive.list_folder(folder_id)


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def drive_get(file_id):
    return gdrive.download(file_id)


@st.cache_data(ttl=7 * 86400, max_entries=30, show_spinner=False)
def view_for(lat, lon, heading, fov, pitch, roll, size, n_cols, max_km, height, proj=None, y0=None, tilt=0.0):
    cam = dict(proj=proj, y0=y0) if proj in ('cyl', 'equi') else None
    return se.view_of(lat, lon, heading, fov, pitch, roll, size, n_cols, max_km=max_km, height=height, cam=cam,
                      tilt=tilt, step=(0.1 if cam else 0.05))


# ------------------------------------------------------------------------------------------
# state helpers
# ------------------------------------------------------------------------------------------
def set_photo(data, name, source):
    h = hashlib.sha1(data).hexdigest()[:16]
    if ss.get('photo') and ss['photo']['hash'] == h:
        return
    ss['photo'] = dict(data=data, name=name, source=source, hash=h)
    for k in ('edit', 'res', 'sel', 'edit_nonce_n'):
        ss.pop(k, None)


def set_start(lat, lon, label, src):
    ss['start'] = dict(lat=float(lat), lon=float(lon), label=label, src=src)
    ss.pop('res', None)


def _on_pick(key='startmap'):
    v = ss.get(key)
    p = v.get('pick') if hasattr(v, 'get') else None
    if not isinstance(p, dict) or 'lat' not in p:
        return
    try:
        lat, lon = float(p['lat']), float(p['lon'])
    except (TypeError, ValueError):
        return
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        set_start(lat, lon, places.describe(lat, lon), 'gps' if p.get('src') == 'gps' else 'map')


def _on_select(key='resmap'):
    v = ss.get(key)
    p = v.get('select') if hasattr(v, 'get') else None
    if isinstance(p, dict) and isinstance(p.get('i'), int):
        ss['sel'] = p['i']
        ss['sel_box'] = p['i']


def _on_edit(key='skyed'):
    v = ss.get(key)
    p = v.get('edit') if hasattr(v, 'get') else None
    if isinstance(p, dict) and isinstance(p.get('y'), list) and isinstance(p.get('valid'), list):
        ss['edit'] = dict(nonce=p.get('nonce'), y=p['y'], valid=p['valid'])
        ss.pop('res', None)


def _do_search():
    q = (ss.get('q') or '').strip()
    ss['place_hits'] = []
    ss['search_error'] = None
    if not q:
        return
    c = places.parse_coords(q)
    if c:
        set_start(c[0], c[1], places.describe(*c), 'coords')
        return
    try:
        hits = find_places(q)
    except places.PlaceError as e:
        ss['search_error'] = str(e)
        return
    if not hits:
        ss['search_error'] = f'No place called "{q}" was found. Try a nearby town, or type coordinates such as 46.02, 7.75.'
        return
    ss['place_hits'] = hits
    set_start(hits[0]['lat'], hits[0]['lon'], hits[0]['label'], 'search')


def _choose_hit():
    i = ss.get('hit')
    hits = ss.get('place_hits') or []
    if i is not None and 0 <= i < len(hits):
        set_start(hits[i]['lat'], hits[i]['lon'], hits[i]['label'], 'search')


# ------------------------------------------------------------------------------------------
# 1  photo
# ------------------------------------------------------------------------------------------
step(1, 'Choose a photo')
t_dev, t_drive = st.tabs(['From this device', 'From Google Drive'])
with t_dev:
    up = st.file_uploader('Colour or black-and-white landscape photo', type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp',
                                                                             'heic', 'heif', 'bmp'], key='upload')
    st.caption('On a phone this opens your photo library or camera. Use the original file if you can: it keeps the '
               'lens data (EXIF) that tells the app how wide the view is. Screenshots and edited copies lose it.')
    if up is not None:
        set_photo(up.getvalue(), up.name, 'device')
with t_drive:
    st.text_input('Link to a Google Drive folder or picture', key='drive_link',
                  placeholder='https://drive.google.com/drive/folders/...')
    st.caption('In Google Drive choose Share > General access > "Anyone with the link", then copy the link here. '
               'Private folders cannot be opened.')
    if st.button('Open link', key='drive_open'):
        kind, fid = gdrive.parse_link(ss.get('drive_link'))
        ss['drive_err'] = None
        ss['drive_items'] = None
        if not fid:
            ss['drive_err'] = 'That does not look like a Google Drive link.'
        elif kind == 'file':
            try:
                with st.spinner('Downloading the picture from Google Drive...'):
                    set_photo(drive_get(fid), 'Google Drive picture', 'drive')
            except gdrive.DriveError as e:
                ss['drive_err'] = str(e)
        else:
            try:
                with st.spinner('Reading the folder...'):
                    ss['drive_items'] = drive_list(fid)
                if not ss['drive_items']:
                    ss['drive_err'] = 'No pictures were found in this folder.'
            except gdrive.DriveError as e:
                ss['drive_err'] = str(e)
    if ss.get('drive_err'):
        st.warning(ss['drive_err'])
    items = ss.get('drive_items') or []
    if items:
        per = 12
        pages = max(1, math.ceil(len(items) / per))
        pg = st.number_input(f'{len(items)} pictures - page', 1, pages, 1, key='drive_page') if pages > 1 else 1
        cols = st.columns(4)
        for k, it in enumerate(items[(pg - 1) * per: pg * per]):
            with cols[k % 4]:
                st.image(gdrive.thumbnail_url(it['id']), width='stretch')
                st.markdown(f'<div class="ag-thumbname" title="{html.escape(it["name"])}">{html.escape(it["name"])}</div>',
                            unsafe_allow_html=True)
                if st.button('Use this', key=f'drv_{it["id"]}', width='stretch'):
                    try:
                        with st.spinner('Downloading...'):
                            set_photo(drive_get(it['id']), it['name'], 'drive')
                    except gdrive.DriveError as e:
                        st.warning(str(e))

photo = ss.get('photo')
im = info = None
if photo:
    try:
        im, info = decode_photo(photo['data'])
    except Exception as e:
        st.error(f'This file could not be read as a picture ({e}).')
        ss.pop('photo', None)
        photo = None

if photo:
    exif_bits = []
    if info.get('make') or info.get('model'):
        exif_bits.append(' '.join(x for x in (info.get('make'), info.get('model')) if x))
    if info.get('focal35'):
        exif_bits.append(f"{info['focal35']:.0f} mm (35 mm equivalent)")
    elif info.get('focal_mm'):
        exif_bits.append(f"{info['focal_mm']:.1f} mm lens (no 35 mm equivalent stored)")
    if info.get('lat') is not None:
        exif_bits.append('GPS position stored in the photo')
    if info.get('pano'):
        exif_bits.append(f"panorama data: {info['pano']['hfov']:.0f}° wide")
    st.caption(f"**{html.escape(photo['name'])}** - {info['width']} x {info['height']} px, "
               f"{'colour' if info.get('is_color') else 'black and white'}"
               + (' - ' + ', '.join(exif_bits) if exif_bits else ' - no camera data (EXIF)'))

# ------------------------------------------------------------------------------------------
# 2  starting place
# ------------------------------------------------------------------------------------------
step(2, 'Where should the search start?')
st.markdown('<div class="ag-ask"><b>Where do you think this photo was taken?</b> Type a town, valley, park or peak '
            'near it, or coordinates, or tap the map. The app then searches the land within the distance you '
            'choose around that place.</div>', unsafe_allow_html=True)
c1, c2 = st.columns([3, 1], vertical_alignment='bottom')
with c1:
    st.text_input('Place or coordinates', key='q', placeholder='e.g. Zermatt, Yosemite Valley, or 46.02, 7.75',
                  on_change=_do_search)
with c2:
    st.button('Find', on_click=_do_search, width='stretch')
if ss.get('search_error'):
    st.caption(ss['search_error'])
if len(ss.get('place_hits') or []) > 1:
    st.selectbox('Other places with that name', options=list(range(len(ss['place_hits']))), index=None, key='hit',
                 format_func=lambda i: ss['place_hits'][i]['label'], placeholder='Choose the right place',
                 on_change=_choose_hit)
if photo and info.get('lat') is not None:
    if st.button(f"Start from the GPS position stored in the photo ({info['lat']:.4f}, {info['lon']:.4f})"):
        set_start(info['lat'], info['lon'], 'photo GPS: ' + places.describe(info['lat'], info['lon']), 'exif')

r1, r2 = st.columns(2)
with r1:
    radius = st.slider('Search this far around the starting point (km)', 2, 30, 10, 1, key='radius')
with r2:
    sp_opts = ['Automatic', 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    sp_choice = st.selectbox('Grid spacing (closer = slower, more thorough)', sp_opts, key='spacing',
                             format_func=lambda v: v if isinstance(v, str) else f'{v * 1000:.0f} m')
if sp_choice == 'Automatic':
    spacing = max(0.2, round(math.sqrt(math.pi * radius ** 2 / 6000.0) / 0.05) * 0.05)
else:
    spacing = float(sp_choice)
n_pts = int(math.pi * radius ** 2 / spacing ** 2)
est_s = 20 + n_pts * 0.015 + 90 + (60 if ss.get('lens', '').startswith('Not known') else 0)
if photo and info.get('kind') in ('pano', '360'):
    est_s += 60 + n_pts * 0.01
st.caption('Tip: when the skyline in the photo is close (a valley, a cliff a kilometre or two away), the view changes '
           'quickly as you move. Then a smaller search area with a 150-200 m grid works better than a big one.')
st.caption(f'About {n_pts:,} places will be tested ({spacing * 1000:.0f} m apart); this takes roughly '
           f'{max(1, round(est_s / 60))} minute{"s" if est_s >= 90 else ""} on a typical computer '
           '(the first search in a new area also downloads elevation data).')

start = ss.get('start')
map_data = dict(mode='start', start=start, radius_km=radius, heat=None, cands=[], sel=0,
                exif=dict(lat=info['lat'], lon=info['lon']) if (photo and info.get('lat') is not None) else None,
                status=(f"Starting point: {start['label']} ({start['lat']:.4f}, {start['lon']:.4f})" if start
                        else 'No starting point yet - search for a place above or tap the map.'),
                nonce=f"start:{start['lat']:.5f},{start['lon']:.5f},{radius}" if start else 'start:none')
if not ss.get('res'):
    cm.show_map(map_data, key='startmap', on_pick=_on_pick)

# ------------------------------------------------------------------------------------------
# 3  skyline
# ------------------------------------------------------------------------------------------
sky = None
hfov = None
if photo:
    step(3, 'Check the skyline')
    KINDS = ['Normal photo (one frame)', 'Panorama (stitched, e.g. a phone\'s "Pano" mode)',
             '360° photo (the whole way round, equirectangular)']
    k_default = {'photo': 0, 'pano': 1, '360': 2}[info.get('kind', 'photo')]
    kind = st.radio('What kind of picture is it?', KINDS, index=k_default, key=f"kind_{photo['hash']}",
                    horizontal=True)
    pano = kind != KINDS[0]
    if pano and k_default == 0:
        st.caption('This picture does not look like a panorama (it is not very wide and has no panorama data). '
                   'Choose "Normal photo" unless you know it was stitched.')
    with st.expander('Skyline finder settings'):
        e1, e2 = st.columns(2)
        smooth = e1.slider('Line smoothness', 0.2, 4.0, 1.0, 0.1, key='smooth',
                           help='Higher = a calmer line that ignores small bumps; lower = follows every tree top.')
        edge = e2.slider('Follow edges', 0.0, 3.0, 1.0, 0.1, key='edgew',
                         help='How strongly the line sticks to sharp light/dark edges.')
    with st.spinner('Finding the skyline...'):
        det = detect_skyline(photo['hash'], photo['data'], smooth, edge,
                             sk.PANO_WORK_WIDTH if pano else sk.WORK_WIDTH)
    url, dsize = display_image(photo['hash'], photo['data'], 2400 if pano else 1400)
    W0, H0 = det['size']
    fx = dsize[0] / W0
    nonce = f"{photo['hash']}:{smooth}:{edge}:{int(pano)}:{ss.get('edit_nonce_n', 0)}"
    ed = ss.get('edit')
    if ed and ed.get('nonce') == nonce and len(ed['y']) == len(det['y']):
        y_disp = ed['y']
        valid = [bool(v) for v in ed['valid']]
    else:
        y_disp = [round(float(v) * fx, 1) for v in det['y']]
        valid = [bool(v) for v in det['valid']]
    cm.show_editor(dict(img=url, w=dsize[0], h=dsize[1], n=len(det['y']), y=y_disp, valid=valid, nonce=nonce,
                        tools=True), key='skyed', on_edit=_on_edit)
    b1, b2 = st.columns([1, 3])
    if b1.button('Back to the automatic line'):
        ss['edit_nonce_n'] = ss.get('edit_nonce_n', 0) + 1
        ss.pop('edit', None)
        ss.pop('res', None)
        st.rerun()
    y_orig = np.array([np.nan if v is None else float(v) / fx for v in y_disp])
    valid_a = np.array(valid, bool) & np.isfinite(y_orig)
    share = valid_a.mean() if valid_a.size else 0
    b2.caption(f'{share * 100:.0f} % of the picture width has a usable skyline. Mark trees, buildings, poles or clouds '
               'that hide the real hills as "Not skyline" - a short stretch of true skyline matches better than a '
               'long stretch with mistakes.')
    sky = dict(x=np.asarray(det['x'], float), y=y_orig, valid=valid_a, size=(W0, H0))

    # lens / panorama angle
    cam = None
    if not pano:
        st.markdown('**How wide is the view?**')
        ex_fov = sk.hfov_from_exif(info, W0, H0)
        opts = []
        if ex_fov:
            opts.append(f"From the photo's camera data: {info['focal35']:.0f} mm equivalent = {ex_fov:.1f}° wide")
        opts += ['Phone main camera (26 mm equivalent)', 'Enter the 35 mm-equivalent focal length',
                 'Enter the horizontal angle of view', 'Not known - let the app try 20° to 90°']
        lens = st.radio('Lens', opts, key='lens', label_visibility='collapsed')
        if lens.startswith("From the photo"):
            hfov = ex_fov
            st.caption('If the picture was cropped, the camera data no longer fits - choose "Not known" instead.')
        elif lens.startswith('Phone'):
            hfov = sk.hfov_from_focal35(26.0, W0, H0)
            st.caption(f'{hfov:.1f}° wide for this picture shape.')
        elif lens.startswith('Enter the 35'):
            f35 = st.number_input('35 mm-equivalent focal length (mm)', 8.0, 1200.0, 50.0, 1.0, key='f35')
            hfov = sk.hfov_from_focal35(f35, W0, H0)
            st.caption(f'{hfov:.1f}° wide for this picture shape.')
        elif lens.startswith('Enter the horizontal'):
            hfov = st.number_input('Horizontal angle of view (degrees)', 2.0, 150.0, 60.0, 0.5, key='hfov_in')
        else:
            hfov = None
    else:
        gp = info.get('pano')
        if kind == KINDS[2]:
            hfov = 360.0
            proj = 'equi'
            y0 = gp['y0'] if gp and gp.get('y0') is not None else None
            st.caption('A 360° photo covers the whole horizon, so only the place and the direction of the picture\'s '
                       'centre are searched for. Use the full, unedited picture (2:1 shape).')
        else:
            st.markdown('**How much of the horizon does the panorama cover?**')
            popts = []
            if gp:
                popts.append(f"From the panorama's data: {gp['hfov']:.0f}°")
            popts += ['Enter the angle it covers (degrees)', 'Not known - let the app try 90° to 360°']
            plens = st.radio('Panorama angle', popts, key='pano_lens', label_visibility='collapsed')
            y0 = gp['y0'] if gp else None
            if plens.startswith('From the panorama'):
                hfov = gp['hfov']
            elif plens.startswith('Enter'):
                hfov = st.number_input('Angle covered (degrees, 360 = all the way round)', 60.0, 360.0, 180.0, 5.0,
                                       key='pano_deg', help='A phone panorama swept from one side to the other '
                                       'usually covers 120-240°.')
            else:
                hfov = None
            proj_names = {'cyl': 'Cylindrical (phone panoramas and most stitching programs)',
                          'equi': 'Equirectangular (Google Camera, Photo Sphere, 360° tools)'}
            pdef = gp['proj'] if gp else 'cyl'
            proj = st.selectbox('Projection', ['cyl', 'equi'], index=0 if pdef == 'cyl' else 1, key='pano_proj',
                                format_func=lambda p: proj_names[p])
            st.caption('Do not crop the sides of the panorama after stitching, or the angle no longer fits. '
                       'Small up-down shifts and a slightly tilted horizon are found by the app.')
        cam = dict(proj=proj, y0=y0, ref_fov=hfov or 180.0)
    with st.expander('More settings: camera height, distance of the skyline'):
        m1, m2 = st.columns(2)
        cam_h = m1.number_input('Camera height above the ground (m)', 0.0, 3000.0, 1.7, 0.5, key='cam_h',
                                help='Raise it for a photo from a tower, building, bridge or drone.')
        max_km = m2.select_slider('Mountains up to this far away are included (km)', [40, 60, 80, 120, 160, 200],
                                  120, key='max_km')
else:
    cam_h, max_km = 1.7, 120
    cam = None
    pano = False

# ------------------------------------------------------------------------------------------
# 4  search
# ------------------------------------------------------------------------------------------
if photo:
    step(4, 'Find where it was taken')
    ready = start is not None and sky is not None and sky['valid'].sum() >= 20
    if start is None:
        st.info('Choose a starting place in step 2 first.')
    elif sky is not None and sky['valid'].sum() < 20:
        st.warning('There is too little skyline to work with. Use "Draw skyline" or "Is skyline" on the picture.')
    params = (photo['hash'], start['lat'] if start else None, start['lon'] if start else None, radius, spacing,
              round(hfov, 3) if hfov else None, cam_h, max_km, str(cam), hashlib.sha1(np.nan_to_num(sky['y']).tobytes()
              + sky['valid'].tobytes()).hexdigest()[:12] if sky else None)
    if st.button('Search for the place', type='primary', disabled=not ready, width='stretch'):
        bar = st.progress(0.0, text='Starting...')
        t0 = time.time()

        def prog(msg, frac):
            bar.progress(min(1.0, max(0.0, frac)), text=f'{msg} - {time.time() - t0:.0f} s')
        try:
            res = se.locate(sky['x'], sky['y'], sky['valid'], sky['size'], start['lat'], start['lon'],
                            radius_km=radius, spacing_km=spacing, hfov=hfov, max_km=max_km, height=cam_h,
                            progress=prog, skyline_mod=sk, cam=cam)
            ss['res'] = dict(res=res, params=params, start=dict(start), hfov=hfov, radius=radius, cam=cam)
            ss['sel'] = 0
            ss['sel_box'] = 0
        except (tr.TerrainError, ValueError) as e:
            st.error(f'The search stopped: {e}')
        bar.empty()
    R = ss.get('res')
    if R and R['params'] != params:
        st.caption('Settings changed since the last search - press the button again to update the results.')

# ------------------------------------------------------------------------------------------
# results
# ------------------------------------------------------------------------------------------
def heat_overlay(c, st0):
    lat, lon, s = c['lat'], c['lon'], np.asarray(c['score'], float)
    sp = c['spacing_km']
    dlat = math.degrees(sp * 1000 / se.R_EARTH)
    dlon = math.degrees(sp * 1000 / (se.R_EARTH * math.cos(math.radians(st0['lat']))))
    iy = np.round((lat - st0['lat']) / dlat).astype(int)
    ix = np.round((lon - st0['lon']) / dlon).astype(int)
    ny, nx = iy.max() - iy.min() + 1, ix.max() - ix.min() + 1
    g = np.full((ny, nx), np.nan)
    g[iy.max() - iy, ix - ix.min()] = s
    fin = np.isfinite(g)
    best = np.nanmin(g)
    mid = np.nanpercentile(g, 50)
    v = np.clip((mid - g) / max(1e-6, mid - best), 0, 1)
    v = np.where(fin, v, 0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'rl', ['#20304d', '#5b3a8c', '#c2457a', '#ff9f6b', '#ffe680'])
    rgba = cmap(v ** 1.5)
    rgba[..., 3] = np.where(fin, 0.12 + 0.83 * v ** 1.2, 0)
    img = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
    b = io.BytesIO()
    img.save(b, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode('ascii')
    la_max = st0['lat'] + iy.max() * dlat + dlat / 2
    la_min = st0['lat'] + iy.min() * dlat - dlat / 2
    lo_min = st0['lon'] + ix.min() * dlon - dlon / 2
    lo_max = st0['lon'] + ix.max() * dlon + dlon / 2
    return dict(url=url, bounds=[[la_min, lo_min], [la_max, lo_max]])


def verdict(cands):
    """Strong / Likely / Unsure from how much better the best place fits than the best place more
    than 1 km away (thresholds from the synthetic and Yosemite tests, see the validation report)."""
    if not cands:
        return 'none', 'No place matched.', '#f5847c', None
    b = cands[0]
    others = [c for c in cands[1:] if se.haversine_km(b['lat'], b['lon'], c['lat'], c['lon']) > 1.0]
    ratio = (others[0]['score'] / b['score']) if others and b['score'] > 0 else float('inf')
    if ratio >= 1.5:
        return 'strong', (f'Strong match: the photo skyline fits place #1 to within {b["rms_deg"]:.2f}° and every place '
                          f'more than 1 km away fits at least {ratio:.1f} times worse.'), '#2fc46f', ratio
    if ratio >= 1.15:
        return 'likely', (f'Possible match: place #1 fits best (to within {b["rms_deg"]:.2f}°), but the next-best place more '
                          f'than 1 km away fits only {ratio:.2f} times worse. In tests about half of such results were '
                          'right: check that the blue model line follows the skyline on the photo below, and compare '
                          'the other numbered places.'), '#ffd25e', ratio
    return 'unsure', ('Not decided: several places fit about equally well (best vs next-best '
                      f'{ratio:.2f}). Clean up the skyline (mark trees and clouds as "Not skyline"), give the lens '
                      'angle if you know it, or, if you have a good idea of the area, search a smaller radius with '
                      'a finer grid (150-200 m).'), '#f0913a', ratio


R = ss.get('res')
if photo and R:
    res = R['res']
    cands = res['candidates']
    st0 = R['start']
    step(5, 'Results')
    if res.get('missing'):
        st.caption(f"{res['missing']} of {res['tiles']} elevation tiles could not be downloaded; parts of the area "
                   'may be wrong.')
    kind, text, color, _ratio = verdict(cands)
    st.markdown(f'<div class="ag-note"><span class="ag-pill" style="background:{color}">'
                f'{ {"strong": "Strong", "likely": "Possible", "unsure": "Unsure", "none": "None"}[kind] }</span> '
                f'{html.escape(text)}</div>', unsafe_allow_html=True)
    if cands:
        sel = int(ss.get('sel', 0))
        sel = min(max(0, sel), len(cands) - 1)
        c = cands[sel]
        n_cols = len(sky['y'])
        try:
            V = view_for(round(c['lat'], 6), round(c['lon'], 6), round(c['heading'], 3), round(c['fov'], 3),
                         round(c['pitch'], 3), round(c['roll'], 3), tuple(sky['size']), n_cols, max_km, cam_h,
                         (R.get('cam') or {}).get('proj'), (R.get('cam') or {}).get('y0'), round(c.get('tilt', 0.0), 3))
        except Exception as e:
            V = None
            st.caption(f'The view from this place could not be drawn ({e}).')
        d_start = float(se.haversine_km(st0['lat'], st0['lon'], c['lat'], c['lon']))
        if R.get('cam'):
            lens_sub = (f"horizon {abs(c['pitch']):.1f}° {'above' if c['pitch'] >= 0 else 'below'} the picture's "
                        f"middle row; level within {math.hypot(c.get('tilt', 0.0), c['roll']):.1f}°")
        else:
            lens_sub = (f"tilted {abs(c['pitch']):.1f}° {'up' if c['pitch'] >= 0 else 'down'}, turned "
                        f"{abs(c['roll']):.1f}° {'clockwise' if c['roll'] >= 0 else 'anticlockwise'}")
        pano_head_txt = ''
        gph = (info.get('pano') or {}).get('heading')
        if R.get('cam') and gph is not None:
            dd_ = abs(((c['heading'] - gph + 180) % 360) - 180)
            pano_head_txt = f'; the panorama\'s own data says {gph:.0f}° ({dd_:.0f}° apart)'
        exif_txt = ''
        if info.get('lat') is not None:
            d_ex = float(se.haversine_km(info['lat'], info['lon'], c['lat'], c['lon']))
            exif_txt = f'{d_ex:.2f} km from the GPS position stored in the photo'
        st.markdown(f"""<div class="ag-cards">
          <div class="ag-card ag-c1"><div class="k">Place #{sel + 1}</div><div class="v">{c['lat']:.5f}, {c['lon']:.5f}</div>
            <div class="s">{html.escape(places.describe(c['lat'], c['lon']))} · ground {c['ground']:.0f} m ·
            {d_start:.1f} km from the start{(' · ' + exif_txt) if exif_txt else ''}</div></div>
          <div class="ag-card ag-c2"><div class="k">{'Panorama centre faces' if R.get('cam') else 'Camera facing'}</div><div class="v">{c['heading']:.1f}° {compass(c['heading'])}</div>
            <div class="s">direction of the picture's centre (true north){pano_head_txt}</div></div>
          <div class="ag-card ag-c3"><div class="k">{'Panorama covers' if R.get('cam') else 'Lens'}</div><div class="v">{c['fov']:.1f}°{'' if R.get('cam') else ' wide'}</div>
            <div class="s">{lens_sub}</div></div>
          <div class="ag-card ag-c4"><div class="k">Skyline fit</div><div class="v">{c['rms_deg']:.2f}°</div>
            <div class="s">typical gap between photo and model skyline (score {c['score']:.3f}; lower is better)</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"[Open in Google Maps](https://www.google.com/maps/search/?api=1&query={c['lat']:.6f},{c['lon']:.6f}) · "
                    f"[OpenTopoMap](https://opentopomap.org/#marker=15/{c['lat']:.6f}/{c['lon']:.6f})")

        mc = [dict(lat=x['lat'], lon=x['lon'], rank=i + 1, heading=x['heading'], fov=x['fov'],
                   label=f"fit {x['rms_deg']:.2f}°, facing {x['heading']:.0f}° {compass(x['heading'])}",
                   cone_km=(V['median_km'] if (V and i == sel) else 10.0),
                   ridge=(V['ridge'] if (V and i == sel) else None)) for i, x in enumerate(cands)]
        rdata = dict(mode='results', start=st0, radius_km=R['radius'], heat=heat_overlay(res['coarse'], st0),
                     cands=mc, sel=sel, zoom=12,
                     exif=dict(lat=info['lat'], lon=info['lon']) if info.get('lat') is not None else None,
                     status=f"Tested {res['n_points']:,} places in {res['timing']['total']:.0f} s. Tap a number to see that place.",
                     nonce=f"res:{photo['hash']}:{sel}:{res['timing']['total']:.3f}")
        cm.show_map(rdata, key='resmap', on_select=_on_select)

        # pictures of the results: map image and 3-D view, each drawn with a progress bar
        pano_r = bool(R.get('cam'))
        view_img = dict(
            photo_name=photo['name'], start=st0, radius_km=R['radius'], coarse=res['coarse'], cands=cands, sel=sel,
            ridge=V['ridge'] if V else None, cone_km=V['median_km'] if V else 8.0,
            exif=dict(lat=info['lat'], lon=info['lon']) if info.get('lat') is not None else None,
            verdict=({'strong': 'Strong', 'likely': 'Possible', 'unsure': 'Unsure', 'none': 'None'}[kind], text, color),
            place_text=(f"Place #{sel + 1}: {c['lat']:.5f}, {c['lon']:.5f}  ·  "
                        f"{'centre faces' if pano_r else 'facing'} {c['heading']:.1f}° {compass(c['heading'])}  ·  "
                        f"{'covers' if pano_r else 'lens'} {c['fov']:.1f}°  ·  skyline fit {c['rms_deg']:.2f}°"))
        run_id = round(res['timing']['total'], 3)
        stem = os.path.splitext(photo['name'])[0][:40]

        def draw_with_progress(what, fn):
            """Run a drawing function, showing a progress bar with the stage, elapsed time and time left."""
            bar = st.progress(0.0, text=f'Drawing the {what}...')
            t0 = time.time()

            def prog(frac, msg):
                el = time.time() - t0
                left = ''
                if 0.08 < frac < 0.999 and el > 1.0:
                    left = f', about {max(1, round(el * (1 - frac) / frac))} s left'
                bar.progress(min(1.0, max(0.0, float(frac))),
                             text=f'Drawing the {what}: {msg} - {frac * 100:.0f} % - {el:.1f} s{left}')
            try:
                png = fn(prog)
            except Exception as e:
                bar.empty()
                st.error(f'The {what} could not be drawn ({e}).')
                return None, 0.0
            secs = time.time() - t0
            bar.progress(1.0, text=f'The {what} is ready - drawn in {secs:.1f} s')
            return png, secs

        st.markdown('**Save the results map as a picture**')
        i1, i2, i3 = st.columns([1.1, 1.5, 1.1], vertical_alignment='bottom')
        img_base = i1.selectbox('Background', list(mi.TILES), key='img_base')
        img_ext = i2.radio('Show', ['Whole search area', f'Close-up of place #{sel + 1}'], key='img_ext',
                           horizontal=True)
        img_extent = 'area' if img_ext.startswith('Whole') else 'place'
        img_key = ('map', photo['hash'], sel, img_base, img_extent, run_id)
        if i3.button('Draw map image', key='draw_map', icon=':material/map:', width='stretch'):
            png, secs = draw_with_progress('map image', lambda p: mi.render_cached(img_key, view_img, img_base,
                                                                                    img_extent, progress=p))
            if png:
                ss['pic_map'] = dict(key=img_key, png=png, secs=secs)
        pm = ss.get('pic_map')
        if pm and pm['key'] == img_key:
            d1, d2 = st.columns([1.1, 2.6], vertical_alignment='center')
            d1.download_button('Download map image (PNG)', pm['png'], mime='image/png', on_click='ignore',
                               file_name=f'ridgeline_map_{stem}_{img_extent}.png', type='primary',
                               icon=':material/download:', width='stretch')
            d2.caption(f"Drawn in {pm['secs']:.1f} s ({len(pm['png']) / 1e6:.1f} MB).")
            with st.expander('Preview of the map image'):
                st.image(pm['png'], width='stretch')
        else:
            st.caption('Press "Draw map image": a progress bar shows the drawing time, then the picture can be '
                       'downloaded. It shows the heat map, the numbered places, the view and skyline ridge of the '
                       'place selected above, and a legend.')

        st.markdown(f'**3-D view of place #{sel + 1}** - the land seen from behind and above the place, with the '
                    'photo\'s view and the predicted skyline ridge')
        o1, o2 = st.columns([1.1, 2.6], vertical_alignment='bottom')
        obl_base = o1.selectbox('Ground', list(mi.TILES), index=1, key='obl_base')
        obl_key = ('obl', photo['hash'], sel, obl_base, run_id)
        can_obl = bool(V and V.get('ridge'))
        if o2.button(f'Draw 3-D view of place #{sel + 1}', key='draw_obl', icon=':material/landscape:',
                     disabled=not can_obl):
            png, secs = draw_with_progress('3-D view', lambda p: ob.render_cached(obl_key, view_img, obl_base,
                                                                                   progress=p))
            if png:
                ss['pic_obl'] = dict(key=obl_key, png=png, secs=secs)
        po = ss.get('pic_obl')
        if po and po['key'] == obl_key:
            st.image(po['png'], width='stretch')
            e1, e2 = st.columns([1.1, 2.6], vertical_alignment='center')
            e1.download_button(f'Download 3-D view of place #{sel + 1} (PNG)', po['png'], mime='image/png',
                               on_click='ignore', file_name=f'ridgeline_3d_place{sel + 1}_{stem}.png', type='primary',
                               icon=':material/download:', width='stretch')
            e2.caption(f"Drawn in {po['secs']:.1f} s ({len(po['png']) / 1e6:.1f} MB).")
        st.caption('The first drawing in an area takes 5-15 seconds (terrain and imagery are downloaded); later ones are '
                   'quicker. Choose another place in "Show place" or tap a number on the map for its own 3-D view. '
                   'Vertical scale is true; trees and buildings come from the imagery, not from the elevation model.')
        st.selectbox('Show place', list(range(len(cands))), key='sel_box',
                     format_func=lambda i: f"#{i + 1}  {cands[i]['lat']:.4f}, {cands[i]['lon']:.4f}  - fit {cands[i]['rms_deg']:.2f}°",
                     on_change=lambda: ss.__setitem__('sel', ss['sel_box']))

        if V:
            st.markdown(f'**The photo with the skyline the elevation model predicts from place #{sel + 1}** '
                        '(blue = model, orange = photo)')
            fxd = dsize[0] / W0
            ov = [dict(y=[None if v is None else round(v * fxd, 1) for v in V['y_cols']], color='#7ee0ff', dash=[7, 4])]
            cm.show_editor(dict(img=url, w=dsize[0], h=dsize[1], n=n_cols,
                                y=[None if not np.isfinite(v) else round(float(v) * fxd, 1) for v in sky['y']],
                                valid=[bool(v) for v in sky['valid']], overlay=ov, tools=False,
                                nonce=f'view:{photo["hash"]}:{sel}:{c["lat"]:.5f}:{c["lon"]:.5f}'), key='viewed')

            # panorama
            az0 = c['heading']
            rel = V['rel']
            fig, ax = plt.subplots(figsize=(11, 3.1), dpi=110)
            fig.patch.set_facecolor('#0b111c')
            ax.set_facecolor('#111927')
            lo = min(np.nanmin(V['angle']), np.nanmin(V['angle']) - 1)
            ax.fill_between(rel, lo - 1, V['angle'], color='#1f2a40', lw=0)
            ax.plot(rel, V['angle'], color='#7ee0ff', lw=1.4, label='elevation model skyline')
            paz, pel = se.photo_to_world(sky['x'], sky['y'], sky['size'], c['fov'], c['pitch'], c['roll'], az0,
                                         cam=R.get('cam'), tilt=c.get('tilt', 0.0))
            prel = ((paz - az0 + 180) % 360) - 180
            pm = np.where(sky['valid'], pel, np.nan)
            ax.plot(prel, pm, color='#ff9f6b', lw=1.6, label='skyline in the photo')
            ax.axvspan(-c['fov'] / 2, c['fov'] / 2, color='#ffcd9c', alpha=0.07, lw=0)
            ticks = np.arange(math.ceil((rel[0] + az0) / 10) * 10, rel[-1] + az0 + 0.1, 10)
            ax.set_xticks(ticks - az0)
            ax.set_xticklabels([f'{t % 360:.0f}° {compass(t)}' for t in ticks], fontsize=8, color='#dfe5ef')
            ax.tick_params(axis='y', colors='#dfe5ef', labelsize=8)
            ax.set_ylabel('elevation (°)', color='#dfe5ef', fontsize=9)
            ax.set_xlim(rel[0], rel[-1])
            ax.set_ylim(lo - 0.5, max(np.nanmax(V['angle']), np.nanmax(pm) if np.isfinite(pm).any() else 0) + 1.5)
            for sp_ in ax.spines.values():
                sp_.set_color('#2a3550')
            leg = ax.legend(loc='upper right', fontsize=8, facecolor='#111927', edgecolor='#2a3550')
            for t in leg.get_texts():
                t.set_color('#f4efe8')
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)
            st.caption(f'Skyline-forming ridges in the view lie {np.nanmin(V["dist_km"][np.abs(rel) <= c["fov"] / 2]):.1f}-'
                       f'{np.nanmax(V["dist_km"][np.abs(rel) <= c["fov"] / 2]):.1f} km away (red line on the map). '
                       'Trees, buildings and snow cornices are not in the elevation model.')

        rows = []
        for i, x in enumerate(cands):
            rows.append({'#': i + 1, 'latitude': round(x['lat'], 6), 'longitude': round(x['lon'], 6),
                         'near': places.describe(x['lat'], x['lon']),
                         'km from start': round(float(se.haversine_km(st0['lat'], st0['lon'], x['lat'], x['lon'])), 2),
                         'facing (°)': round(x['heading'], 1), 'lens (° wide)': round(x['fov'], 1),
                         'tilt (°)': round(x['pitch'], 1), 'roll (°)': round(x['roll'], 1),
                         'skyline fit (°)': round(x['rms_deg'], 3), 'score': round(x['score'], 4),
                         'ground (m)': round(x['ground'])})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width='stretch')
        st.download_button('Download these places (CSV)', df.to_csv(index=False).encode('utf-8'),
                           file_name=f"ridgeline_{os.path.splitext(photo['name'])[0][:40]}.csv", mime='text/csv')
    if st.button('New search area'):
        ss.pop('res', None)
        st.rerun()

st.markdown('---')
st.caption('Elevation: AWS Terrain Tiles (SRTM, USGS 3DEP, GMTED2010, ETOPO1 and national models). '
           'Place search: Open-Meteo / GeoNames. Maps: OpenTopoMap, OpenStreetMap, Esri. '
           'The match is only as good as the skyline: photos with distinctive hills or mountains work best; '
           'flat land, sea horizons and views blocked by trees cannot be placed.')
st.markdown('<div class="ag-credit">Developed by Gerad Smith and Marco Sugamiele for the '
            '<i>Ancestors in Photo Negative</i> project</div>', unsafe_allow_html=True)
