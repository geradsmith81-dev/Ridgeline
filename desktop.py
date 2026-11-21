"""Desktop-only behaviour for Ridgeline.

Covers where this computer keeps Ridgeline's data, the Quit button, the automatic shutdown when
the app window is closed, the password used when the app is shared on the local network, and
clearing stored map data.

Everything here is switched off on the hosted (GitHub / Streamlit Community Cloud) copy, because
only tools/launch.py sets RIDGELINE_DESKTOP=1.
"""
import hmac
import ipaddress
import json
import logging
import os
import shutil
import threading
import time

APP = 'Ridgeline'
log = logging.getLogger('ridgeline.desktop')


# ------------------------------------------------------------------------------------------
# where things are
# ------------------------------------------------------------------------------------------
def is_desktop():
    return os.environ.get('RIDGELINE_DESKTOP') == '1'


def data_dir():
    """Per-user data folder: %LOCALAPPDATA%\\Ridgeline (map caches, logs, the running-server note)."""
    base = os.environ.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
    return os.path.join(base, APP)


def state_file():
    return os.path.join(data_dir(), 'server.json')


def share_password():
    return os.environ.get('RIDGELINE_SHARE_PASSWORD') or ''


def is_sharing():
    return bool(share_password())


def client_is_local(st):
    """True when the person using this page is sitting at this computer.

    When the app is not shared, the server only listens on this computer, so every visitor is
    local. When it is shared, only a loopback address counts as local; an unknown address does not.
    """
    if not is_sharing():
        return True
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip.split('%')[0]).is_loopback
    except ValueError:
        return ip == 'localhost'


# ------------------------------------------------------------------------------------------
# password gate (network sharing only)
# ------------------------------------------------------------------------------------------
def require_password(st):
    """Ask visitors from other devices for the sharing password; stop the page until it is given."""
    pw = share_password()
    if not pw or client_is_local(st) or st.session_state.get('_rl_auth'):
        return
    st.markdown('### Ridgeline is shared from another computer')
    st.write('Enter the password shown in the Ridgeline window on that computer.')
    with st.form('rl_pw', clear_on_submit=True):
        given = st.text_input('Password', type='password')
        ok = st.form_submit_button('Open Ridgeline')
    if ok:
        if hmac.compare_digest(given.strip().lower().encode('utf-8'), pw.lower().encode('utf-8')):
            st.session_state['_rl_auth'] = True
            st.rerun()
        time.sleep(1.5)      # slows down guessing
        st.error('That password is not right.')
    st.stop()


# ------------------------------------------------------------------------------------------
# stopping the app
# ------------------------------------------------------------------------------------------
def shutdown(reason, code=0):
    """Stop the whole Ridgeline server process."""
    log.info('Stopping Ridgeline: %s', reason)
    try:
        with open(state_file(), encoding='utf-8') as f:
            mine = json.load(f).get('pid') == os.getpid()
        if mine:
            os.remove(state_file())
    except Exception:
        pass
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    os._exit(code)


def _active_sessions():
    from streamlit.runtime import Runtime
    if not Runtime.exists():
        return None
    return Runtime.instance()._session_mgr.num_active_sessions()


def start_idle_watch(idle_s=90, first_s=600, poll_s=5):
    """Stop the server once the app window has been closed.

    Counts polls rather than clock time, so a computer waking from sleep is not mistaken for a
    long idle spell: the browser gets idle_s seconds of running time to reconnect.
    """
    need = max(1, int(round(idle_s / poll_s)))

    def run():
        t0 = time.monotonic()
        seen = False
        empty = 0
        while True:
            time.sleep(poll_s)
            try:
                n = _active_sessions()
            except Exception:
                log.exception('Automatic shutdown is off: cannot count open app windows')
                return
            if n is None:
                continue
            if n > 0:
                seen, empty = True, 0
                continue
            if not seen:
                if time.monotonic() - t0 > first_s:
                    shutdown('no app window was opened within %d seconds' % first_s)
                continue
            empty += 1
            if empty >= need:
                shutdown('the app window was closed')

    threading.Thread(target=run, name='ridgeline-idle-watch', daemon=True).start()


def _quit_now():
    shutdown('Quit button')


# ------------------------------------------------------------------------------------------
# stored map data
# ------------------------------------------------------------------------------------------
def _folder_size(path):
    n = b = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                b += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return n, b


def clear_folders(paths):
    """Delete the contents of the given cache folders. Returns (files removed, bytes removed)."""
    n_all = b_all = 0
    for p in paths:
        if not p or not os.path.isdir(p):
            continue
        n, b = _folder_size(p)
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)
        n_all += n
        b_all += b
    return n_all, b_all


# ------------------------------------------------------------------------------------------
# the controls shown at the bottom of the page on the desktop copy
# ------------------------------------------------------------------------------------------
def desktop_controls(st, cache_dirs, on_cleared=None):
    if not is_desktop() or not client_is_local(st):
        return
    with st.expander('This computer: stored map data and quitting'):
        st.caption('Elevation and map tiles are kept on this computer so that a second search in the same '
                   'area is faster. They also show which areas have been searched. Clear them before you '
                   'lend or hand on this computer, or when a project\'s data agreement asks for it.')
        sure = st.checkbox('Yes, delete the stored elevation and map tiles', key='rl_clear_sure')
        if st.button('Clear stored map data', key='rl_clear', disabled=not sure):
            n, b = clear_folders(cache_dirs)
            if on_cleared:
                on_cleared()
            st.success(f'Removed {n:,} stored tiles ({b / 1e6:,.1f} MB).')
        st.caption(f'Ridgeline\'s data folder on this computer: {data_dir()}')
        if is_sharing():
            st.caption('Ridgeline is shared on this network. Close the Ridgeline window with the password '
                       'to stop sharing.')
        if st.button('Quit Ridgeline', key='rl_quit', type='primary'):
            st.info('Ridgeline has stopped. You can close this window.')
            st.html('<script>setTimeout(function(){try{window.close()}catch(e){}}, 400)</script>',
                    unsafe_allow_javascript=True)
            threading.Timer(1.0, _quit_now).start()
