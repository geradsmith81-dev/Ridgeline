"""Ridgeline - the two browser components (Streamlit components v2, runs in the page itself)."""
import base64
import io
import os

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(HERE, 'assets', name), encoding='utf-8') as f:
        return f.read()


_MAP = st.components.v2.component(
    'ridgeline_map',
    js=_read('leaflet.js') + '\n;\n' + _read('ridgemap.js'),
    css=_read('leaflet.css') + '\n' + _read('ridgemap.css'),
    isolate_styles=False,
)
_EDIT = st.components.v2.component(
    'ridgeline_skyline_editor',
    js=_read('skyedit.js'),
    css=_read('ridgemap.css') + '\n' + _read('skyedit.css'),
    isolate_styles=False,
)


def image_data_url(im, max_w=1400, quality=85):
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, max(1, round(h * max_w / w))))
    b = io.BytesIO()
    im.convert('RGB').save(b, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode('ascii'), im.size


def show_map(data, key, on_pick=None, on_select=None):
    kw = {}
    if on_pick is not None:
        kw['on_pick_change'] = on_pick
    if on_select is not None:
        kw['on_select_change'] = on_select
    return _MAP(data=data, key=key, width='stretch', height='content', **kw)


def show_editor(data, key, on_edit=None):
    kw = {}
    if on_edit is not None:
        kw['on_edit_change'] = on_edit
    return _EDIT(data=data, key=key, width='stretch', height='content', **kw)
