"""
Ridgeline - pictures from a Google Drive folder or file link.

Without any Google setup this works for folders and files shared as "Anyone with the link":
  * a folder is listed from Google's plain folder-view page
        https://drive.google.com/embeddedfolderview?id=<FOLDER_ID>
  * a file is downloaded from https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download
If a Google API key is set (environment variable GOOGLE_API_KEY or Streamlit secret
google_api_key), the official Drive API v3 is used for listing instead; it also needs the folder
to be shared by link.  Private folders would need Google sign-in (OAuth), which this app does not do.
"""
import html
import os
import re

import requests

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.heic', '.heif', '.bmp')
_session = requests.Session()


class DriveError(RuntimeError):
    pass


def parse_link(link):
    """-> ('folder' | 'file', id) or (None, None)."""
    s = (link or '').strip()
    if not s:
        return None, None
    m = re.search(r'/folders/([A-Za-z0-9_-]{10,})', s)
    if m:
        return 'folder', m.group(1)
    m = re.search(r'/file/d/([A-Za-z0-9_-]{10,})', s) or re.search(r'/d/([A-Za-z0-9_-]{10,})', s)
    if m:
        return 'file', m.group(1)
    m = re.search(r'[?&]id=([A-Za-z0-9_-]{10,})', s)
    if m:
        return ('folder' if 'folder' in s else 'file'), m.group(1)
    if re.fullmatch(r'[A-Za-z0-9_-]{20,}', s):
        return 'folder', s
    return None, None


def _api_key():
    k = os.environ.get('GOOGLE_API_KEY')
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get('google_api_key')
    except Exception:
        return None


def _is_image(name, mime=''):
    return (name or '').lower().endswith(IMAGE_EXT) or (mime or '').startswith('image/')


def parse_folder_page(text):
    """Entries from the embeddedfolderview HTML: list of dict(id, name, kind)."""
    out = []
    for m in re.finditer(r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)"(.*?)</a>', text, re.S):
        fid, body = m.group(1), m.group(2)
        t = re.search(r'<div class="flip-entry-title">(.*?)</div>', body, re.S)
        name = html.unescape(re.sub('<[^>]+>', '', t.group(1))).strip() if t else fid
        kind = 'folder' if ('/drive/folders/' in body or 'folderview' in body) else 'file'
        out.append(dict(id=fid, name=name, kind=kind))
    return out


def list_folder(folder_id, max_items=500, recurse=1):
    """Pictures in a shared folder (and, one level down, in its sub-folders)."""
    key = _api_key()
    if key:
        return _list_api(folder_id, key, max_items)
    try:
        r = _session.get('https://drive.google.com/embeddedfolderview', params={'id': folder_id}, timeout=30)
    except requests.RequestException as e:
        raise DriveError(f'Google Drive could not be reached ({e.__class__.__name__}).')
    if r.status_code in (401, 403, 404) or 'flip-entry' not in r.text and 'flip-list' not in r.text:
        raise DriveError('This folder could not be opened. In Google Drive, use Share > General access > '
                         '"Anyone with the link", then copy the link again.')
    items = []
    for e in parse_folder_page(r.text):
        if e['kind'] == 'folder':
            if recurse > 0:
                try:
                    for s in list_folder(e['id'], max_items, recurse - 1):
                        s['name'] = f"{e['name']}/{s['name']}"
                        items.append(s)
                except DriveError:
                    pass
        elif _is_image(e['name']):
            items.append(dict(id=e['id'], name=e['name']))
        if len(items) >= max_items:
            break
    return items


def _list_api(folder_id, key, max_items):
    items, token = [], None
    while True:
        p = dict(q=f"'{folder_id}' in parents and trashed = false", key=key, pageSize=200,
                 fields='nextPageToken, files(id, name, mimeType)', supportsAllDrives='true',
                 includeItemsFromAllDrives='true')
        if token:
            p['pageToken'] = token
        r = _session.get('https://www.googleapis.com/drive/v3/files', params=p, timeout=30)
        if r.status_code != 200:
            raise DriveError(f'Google Drive API answered {r.status_code}: {r.text[:200]}')
        j = r.json()
        for f in j.get('files', []):
            if _is_image(f.get('name'), f.get('mimeType')):
                items.append(dict(id=f['id'], name=f['name']))
        token = j.get('nextPageToken')
        if not token or len(items) >= max_items:
            return items[:max_items]


def thumbnail_url(file_id, width=320):
    return f'https://drive.google.com/thumbnail?id={file_id}&sz=w{int(width)}'


def download(file_id, max_mb=60):
    """Bytes of a shared file."""
    url = 'https://drive.usercontent.google.com/download'
    try:
        r = _session.get(url, params=dict(id=file_id, export='download', confirm='t'), timeout=90, stream=True)
    except requests.RequestException as e:
        raise DriveError(f'Google Drive could not be reached ({e.__class__.__name__}).')
    if r.status_code != 200:
        raise DriveError(f'Google Drive did not send the file (HTTP {r.status_code}). '
                         'Check that it is shared as "Anyone with the link".')
    data = bytearray()
    for chunk in r.iter_content(1 << 16):
        data += chunk
        if len(data) > max_mb * 1e6:
            raise DriveError(f'The file is larger than {max_mb} MB.')
    head = bytes(data[:64]).lower()
    if b'<!doctype html' in head or b'<html' in head:
        raise DriveError('Google Drive sent a web page instead of the picture - the file is probably not shared '
                         'as "Anyone with the link".')
    return bytes(data)
