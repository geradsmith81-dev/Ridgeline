// Ridgeline skyline editor - shows the photo with the skyline found by the app and lets the user
// redraw parts of it or mark stretches that are not true skyline (trees, buildings, clouds).
// data: {img (data URL), w, h (pixels of that image), n (columns), y (n values, image px, null = none),
//        valid (n booleans), overlay (optional [{y:[...], color, label}] lines in image px), nonce, tools}
// Sends 'edit' {y, valid, nonce} after each stroke.

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function build(parent) {
  const S = { mode: 'look', hist: [] };
  const root = el('div', 'ag-root ag-ed');
  root.innerHTML = `
    <div class="ag-tools">
      <label class="ag-tool"><input type="radio" name="ag-mode" value="look" checked> Look</label>
      <label class="ag-tool"><input type="radio" name="ag-mode" value="draw"> Draw skyline</label>
      <label class="ag-tool"><input type="radio" name="ag-mode" value="off"> Not skyline</label>
      <label class="ag-tool"><input type="radio" name="ag-mode" value="on"> Is skyline</label>
      <button type="button" class="ag-btn ag-undo">Undo</button>
    </div>
    <div class="ag-canwrap"><canvas class="ag-can"></canvas></div>
    <div class="ag-hint"></div>`;
  parent.appendChild(root);
  S.root = root;
  S.can = root.querySelector('.ag-can');
  S.hint = root.querySelector('.ag-hint');
  S.img = new Image();
  S.img.onload = () => draw(S);
  root.querySelectorAll('input[name="ag-mode"]').forEach((r) => r.addEventListener('change', () => {
    S.mode = r.value;
    S.can.style.touchAction = S.mode === 'look' ? 'auto' : 'none';
    S.can.style.cursor = S.mode === 'look' ? 'default' : 'crosshair';
    hint(S);
  }));
  root.querySelector('.ag-undo').addEventListener('click', () => {
    const h = S.hist.pop();
    if (!h) return;
    S.y = h.y; S.valid = h.valid;
    draw(S); send(S);
  });
  const pos = (ev) => {
    const r = S.can.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * S.d.w, (ev.clientY - r.top) / r.height * S.d.h];
  };
  S.can.addEventListener('pointerdown', (ev) => {
    if (S.mode === 'look' || !S.d) return;
    ev.preventDefault();
    S.can.setPointerCapture(ev.pointerId);
    S.hist.push({ y: S.y.slice(), valid: S.valid.slice() });
    if (S.hist.length > 30) S.hist.shift();
    S.drag = pos(ev);
    apply(S, S.drag, S.drag);
    draw(S);
  });
  S.can.addEventListener('pointermove', (ev) => {
    if (!S.drag) return;
    ev.preventDefault();
    const p = pos(ev);
    apply(S, S.drag, p);
    S.drag = p;
    draw(S);
  });
  const end = () => { if (S.drag) { S.drag = null; send(S); } };
  S.can.addEventListener('pointerup', end);
  S.can.addEventListener('pointercancel', end);
  S.ro = new ResizeObserver(() => draw(S));
  S.ro.observe(root);
  return S;
}

function hint(S) {
  const t = {
    look: 'Choose a tool to correct the line. Orange = skyline used for matching; grey dashes = ignored.',
    draw: 'Drag along the true skyline (left to right or back). The line follows your finger or mouse.',
    off: 'Drag sideways across trees, buildings, poles or clouds that hide the real skyline.',
    on: 'Drag sideways to use a stretch of the line again.',
  };
  S.hint.textContent = t[S.mode];
}

function apply(S, a, b) {
  const n = S.d.n, colw = S.d.w / n;
  let i0 = Math.floor(a[0] / colw), i1 = Math.floor(b[0] / colw);
  const y0 = a[1], y1 = b[1];
  const lo = Math.max(0, Math.min(i0, i1)), hi = Math.min(n - 1, Math.max(i0, i1));
  for (let i = lo; i <= hi; i++) {
    if (S.mode === 'draw') {
      const f = i1 === i0 ? 0 : (i - i0) / (i1 - i0);
      S.y[i] = Math.max(0, Math.min(S.d.h, y0 + (y1 - y0) * f));
      S.valid[i] = true;
    } else if (S.mode === 'off') {
      S.valid[i] = false;
    } else if (S.mode === 'on') {
      if (S.y[i] !== null && S.y[i] !== undefined) S.valid[i] = true;
    }
  }
}

function send(S) {
  S.send('edit', { y: S.y.map((v) => (v === null || v === undefined ? null : Math.round(v * 10) / 10)),
                   valid: S.valid, nonce: S.d.nonce, ts: Date.now() });
}

function draw(S) {
  const d = S.d;
  if (!d || !S.img.complete || !S.img.naturalWidth) return;
  const cw = Math.max(200, S.root.clientWidth || 600);
  const maxH = Math.max(260, Math.round(window.innerHeight * 0.75));
  let W = cw, H = Math.round(cw * d.h / d.w);
  if (H > maxH) { H = maxH; W = Math.round(H * d.w / d.h); }
  const dpr = window.devicePixelRatio || 1;
  const c = S.can;
  c.style.width = W + 'px'; c.style.height = H + 'px';
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.drawImage(S.img, 0, 0, W, H);
  const sx = W / d.w, sy = H / d.h, n = d.n, colw = W / n;
  // masked columns
  g.fillStyle = 'rgba(255, 79, 123, 0.22)';
  for (let i = 0; i < n; i++) if (!S.valid[i]) g.fillRect(i * colw, 0, colw + 0.6, H);
  // extra lines (e.g. the elevation-model skyline of a match)
  for (const o of (d.overlay || [])) {
    g.strokeStyle = o.color || '#7ee0ff'; g.lineWidth = 2.2; g.setLineDash(o.dash || []);
    g.beginPath();
    let on = false;
    for (let i = 0; i < o.y.length; i++) {
      const v = o.y[i];
      if (v === null || v === undefined) { on = false; continue; }
      const x = (i + 0.5) * W / o.y.length, y = v * sy;
      if (!on) { g.moveTo(x, y); on = true; } else g.lineTo(x, y);
    }
    g.stroke();
  }
  g.setLineDash([]);
  // skyline: dark outline then colour, solid where used, dashed where ignored
  const seg = (want, color, width, dash) => {
    g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash);
    g.beginPath();
    let on = false;
    for (let i = 0; i < n; i++) {
      const v = S.y[i];
      if (v === null || v === undefined || S.valid[i] !== want) { on = false; continue; }
      const x = (i + 0.5) * colw, y = v * sy;
      if (!on) { g.moveTo(x, y); on = true; } else g.lineTo(x, y);
    }
    g.stroke();
  };
  if (d.show_line !== false) {
    seg(true, 'rgba(0,0,0,0.7)', 4.5, []);
    seg(true, '#ff9f6b', 2.2, []);
    seg(false, 'rgba(200,205,215,0.9)', 1.6, [5, 5]);
  }
  g.setLineDash([]);
}

function update(S, d) {
  const fresh = !S.d || S.d.nonce !== d.nonce;
  S.d = d;
  if (fresh) {
    S.y = d.y.slice();
    S.valid = d.valid.slice();
    S.hist = [];
  }
  S.root.querySelector('.ag-tools').style.display = d.tools === false ? 'none' : '';
  hint(S);
  if (S.img.src !== d.img) S.img.src = d.img; else draw(S);
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let S = parentElement.__ridgeline_ed;
  if (!S || !S.root.isConnected) { S = build(parentElement); parentElement.__ridgeline_ed = S; }
  S.send = setTriggerValue;
  if (data) update(S, data);
  return () => { if (S.ro) S.ro.disconnect(); };
}
