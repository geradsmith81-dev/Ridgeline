// Ridgeline map - start point / search circle, and search results (score heat map, candidates,
// view cone).  Streamlit "components v2"; Leaflet (window.L) is bundled in front of this file.
// Messages sent back: 'pick' {lat, lon, src} (new starting point) and 'select' {i} (a candidate).

const DEG = Math.PI / 180;
const R_EARTH = 6371008.8;
function dest(lat, lon, az, m) {
  const p1 = lat * DEG, l1 = lon * DEG, th = az * DEG, d = m / R_EARTH;
  const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(th));
  const l2 = l1 + Math.atan2(Math.sin(th) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2));
  return [p2 / DEG, l2 / DEG];
}
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
const DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
function compass(az) { return DIRS[Math.round((((az % 360) + 360) % 360) / 22.5) % 16]; }

function build(parent) {
  const S = {};
  const root = el('div', 'ag-root');
  root.innerHTML = `
    <div class="ag-bar">
      <button type="button" class="ag-btn ag-gps">Use my location</button>
      <span class="ag-status"></span>
    </div>
    <div class="ag-map"></div>
    <div class="ag-legend"></div>`;
  parent.appendChild(root);
  S.root = root;
  S.status = root.querySelector('.ag-status');
  S.legend = root.querySelector('.ag-legend');
  S.gps = root.querySelector('.ag-gps');
  const map = L.map(root.querySelector('.ag-map'), { zoomControl: true, worldCopyJump: true });
  S.map = map;
  const base = {
    'Topographic': L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
      maxZoom: 17, subdomains: 'abc',
      attribution: 'Map &copy; OpenStreetMap contributors, SRTM | Style &copy; OpenTopoMap (CC-BY-SA)' }),
    'Satellite photo': L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics' }),
    'Streets': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }),
  };
  base['Topographic'].addTo(map);
  S.heatLayer = L.layerGroup().addTo(map);
  S.coneLayer = L.layerGroup().addTo(map);
  S.markLayer = L.layerGroup().addTo(map);
  L.control.layers(base, { 'Match heat map': S.heatLayer, 'View cone': S.coneLayer },
                   { collapsed: true, position: 'topright' }).addTo(map);
  L.control.scale({ imperial: true, metric: true }).addTo(map);
  map.setView([20, 0], 2);

  map.on('click', (ev) => {
    const lat = ev.latlng.lat, lon = ((ev.latlng.lng + 540) % 360) - 180;
    const box = el('div', 'ag-pop');
    box.innerHTML = `<div>${lat.toFixed(5)}, ${lon.toFixed(5)}</div>`;
    const b = el('button', 'ag-btn ag-btn-primary', 'Start the search here');
    b.type = 'button';
    b.addEventListener('click', () => {
      S.status.textContent = 'Setting the starting point...';
      S.send('pick', { lat, lon, src: 'map', ts: Date.now() });
      map.closePopup();
    });
    box.appendChild(b);
    L.popup().setLatLng(ev.latlng).setContent(box).openOn(map);
  });
  S.gps.addEventListener('click', () => {
    if (!navigator.geolocation) { S.status.textContent = 'This browser cannot give a location.'; return; }
    S.status.textContent = 'Asking the browser for your location...';
    navigator.geolocation.getCurrentPosition((p) => {
      S.send('pick', { lat: p.coords.latitude, lon: p.coords.longitude, src: 'gps', ts: Date.now() });
    }, (e) => {
      S.status.innerHTML = 'Location not available (' + (e.message || 'permission refused') + ').' +
        (window.top !== window.self ? ' <a href="' + document.location.href + '" target="_top">Open full-screen</a> and try again.' : '');
    }, { enableHighAccuracy: true, timeout: 20000, maximumAge: 60000 });
  });
  S.ro = new ResizeObserver(() => map.invalidateSize());
  S.ro.observe(root);
  return S;
}

function candIcon(rank, sel) {
  return L.divIcon({ className: 'ag-cand-wrap', iconSize: [28, 28], iconAnchor: [14, 14],
    html: `<span class="ag-cand${sel ? ' ag-cand-sel' : ''}">${rank}</span>` });
}

function update(S, d) {
  const map = S.map;
  S.heatLayer.clearLayers(); S.coneLayer.clearLayers(); S.markLayer.clearLayers();
  S.gps.style.display = d.mode === 'start' ? '' : 'none';
  const st = d.start;
  if (st) {
    L.circle([st.lat, st.lon], { radius: d.radius_km * 1000, color: '#ff9f6b', weight: 2, fill: false,
                                 dashArray: '6 6', interactive: false }).addTo(S.markLayer);
    L.circleMarker([st.lat, st.lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#b5461a', fillOpacity: 1 })
      .bindTooltip('Starting point: ' + (st.label || '')).addTo(S.markLayer);
  }
  if (d.heat) {
    L.imageOverlay(d.heat.url, d.heat.bounds, { opacity: 0.62, interactive: false }).addTo(S.heatLayer);
  }
  if (d.exif) {
    L.marker([d.exif.lat, d.exif.lon], { icon: L.divIcon({ className: 'ag-cand-wrap', iconSize: [26, 26], iconAnchor: [13, 13],
      html: '<span class="ag-exif">&#9733;</span>' }) })
      .bindTooltip('Position stored in the photo (EXIF GPS)').addTo(S.markLayer);
  }
  const cands = d.cands || [];
  cands.forEach((c, i) => {
    const m = L.marker([c.lat, c.lon], { icon: candIcon(c.rank, i === d.sel), zIndexOffset: i === d.sel ? 1000 : 0 });
    m.bindTooltip(`#${c.rank} ${c.label}`);
    m.on('click', () => S.send('select', { i, ts: Date.now() }));
    m.addTo(S.markLayer);
  });
  const c = cands[d.sel];
  if (c) {
    const km = (c.cone_km || 10) * 1000;
    const pts = [[c.lat, c.lon]];
    for (let a = -c.fov / 2; a <= c.fov / 2 + 1e-6; a += c.fov / 24) pts.push(dest(c.lat, c.lon, c.heading + a, km));
    L.polygon(pts, { color: '#ffcd9c', weight: 1.5, fillColor: '#ffcd9c', fillOpacity: 0.16, interactive: false })
      .addTo(S.coneLayer);
    L.polyline([[c.lat, c.lon], dest(c.lat, c.lon, c.heading, km)], { color: '#ff7a3d', weight: 2.5, interactive: false })
      .addTo(S.coneLayer);
    if (c.ridge) {
      L.polyline(c.ridge, { color: '#ff4f7b', weight: 2.5, opacity: 0.95, interactive: false })
        .bindTooltip('Skyline ridge in the photo').addTo(S.coneLayer);
    }
  }
  // view
  if (S.nonce !== d.nonce) {
    S.nonce = d.nonce;
    if (c && d.mode === 'results') map.setView([c.lat, c.lon], d.zoom || 12);
    else if (st) map.fitBounds(L.latLng(st.lat, st.lon).toBounds(2.3 * d.radius_km * 1000));
  }
  S.status.textContent = d.status || '';
  S.legend.innerHTML = d.mode === 'results'
    ? `<span><span class="ag-sw" style="background:linear-gradient(90deg,#20304d,#5b3a8c,#c2457a,#ff9f6b,#ffe680)"></span> match: poor &rarr; good</span>
       <span><span class="ag-cand ag-cand-mini">1</span> best places (tap one)</span>
       <span style="color:#ffcd9c">&#9650; view of the photo</span>
       <span style="color:#ff4f7b">&#9644; skyline ridge</span>
       ${d.exif ? '<span style="color:#7ee0ff">&#9733; photo GPS</span>' : ''}
       <span style="color:#ff9f6b">&#9711; search area</span>`
    : `<span style="color:#ff9f6b">&#9711; search area</span> <span>Tap the map to move the starting point.</span>`;
  setTimeout(() => map.invalidateSize(), 120);
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!window.L) { parentElement.textContent = 'The map library did not load.'; return; }
  let S = parentElement.__ridgeline_map;
  if (!S || !S.root.isConnected) { S = build(parentElement); parentElement.__ridgeline_map = S; }
  S.send = setTriggerValue;
  if (data) update(S, data);
  return () => { if (S.ro) S.ro.disconnect(); };
}
