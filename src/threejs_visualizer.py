"""
3D Liquidity Analysis Engine — Three.js WebGL.

Generates:
  output/3d_liquidity_pro.html  — interactive multi-exchange 3D analytical surface
  output/dashboard.html         — multi-panel analytics dashboard
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_exchange_profiles(data: dict, n_bins: int = 200) -> dict:
    """
    Build per-exchange depth profiles.

    Pipeline: average → log-compress → denoise → resample → smooth →
              threshold → normalize. Keeps 6-10 meaningful peaks.
    """
    price_grids = data["price_grids"]
    exchanges = data["exchanges"]
    price_min, price_max = data["price_range"]
    price_ticks = np.linspace(price_min, price_max, n_bins)
    imbalance_series = data.get("imbalance_series", {})

    profiles = {}
    raw_profiles = {}
    global_max = 0.0

    for ex in exchanges:
        matrix = price_grids[ex]
        raw = matrix.mean(axis=0)
        raw = np.log1p(raw)
        raw = median_filter(raw, size=5)

        x_orig = np.linspace(0, 1, len(raw))
        x_new = np.linspace(0, 1, n_bins)
        resampled = np.interp(x_new, x_orig, raw)
        resampled = gaussian_filter(resampled, sigma=2.0)

        raw_profiles[ex] = resampled.copy()
        global_max = max(global_max, resampled.max())

    for ex in exchanges:
        p = raw_profiles[ex]
        if global_max > 0:
            p = p / global_max

        # Moderate threshold — keep top ~35% of signal for 6-10 peaks
        p65 = np.percentile(p, 65)
        p = np.clip((p - p65) / (1.0 - p65 + 1e-9), 0, 1)

        p = np.power(p, 0.55)
        p = gaussian_filter(p, sigma=1.5)

        pmax = p.max()
        if pmax > 0:
            p = p / pmax

        profiles[ex] = p

    # ── Detect peaks ──
    walls = {}
    all_walls = []
    for ex in exchanges:
        p = profiles[ex]
        peak_idxs, _ = find_peaks(p, height=0.3, distance=n_bins // 12, prominence=0.1)
        wall_list = []
        for idx in peak_idxs:
            wall_list.append({
                "idx": int(idx),
                "height": round(float(p[idx]), 3),
                "price": round(float(price_ticks[idx]), 2),
                "exchange": ex,
            })
        wall_list.sort(key=lambda w: w["height"], reverse=True)
        walls[ex] = wall_list[:5]
        all_walls.extend(wall_list[:5])

    # Global top 3 for labels
    all_walls.sort(key=lambda w: w["height"], reverse=True)
    top_keys = set()
    for w in all_walls[:3]:
        top_keys.add((w["exchange"], w["idx"]))

    for ex in exchanges:
        for w in walls[ex]:
            w["show_label"] = (ex, w["idx"]) in top_keys

    # ── Imbalance ──
    avg_imbalances = {}
    for ex in exchanges:
        imb = imbalance_series.get(ex, [0])
        avg_imbalances[ex] = round(float(np.mean(imb)), 4)

    # ── Price labels — 6 ticks ──
    n_labels = 6
    step = max(1, n_bins // n_labels)
    price_labels = [
        {"idx": int(i), "val": round(float(price_ticks[i]), 2)}
        for i in range(0, n_bins, step)
    ]

    profiles_out = {}
    for ex in exchanges:
        profiles_out[ex] = [round(float(v), 4) for v in profiles[ex]]

    return {
        "exchanges": exchanges,
        "profiles": profiles_out,
        "n_bins": n_bins,
        "price_labels": price_labels,
        "price_range": [round(float(price_min), 2), round(float(price_max), 2)],
        "mid_price": round(float((price_min + price_max) / 2.0), 2),
        "walls": walls,
        "avg_imbalances": avg_imbalances,
    }


# ---------------------------------------------------------------------------
# HTML generators
# ---------------------------------------------------------------------------

def generate_threejs(data: dict):
    """Generate all output files."""
    os.makedirs("output", exist_ok=True)

    payload = _build_exchange_profiles(data)
    payload_json = json.dumps(payload, separators=(",", ":"))

    exchanges = data["exchanges"]
    imbalance = data.get("imbalance_series", {})
    spark_data = {}
    for ex in exchanges:
        spark_data[ex] = [round(float(v), 4) for v in imbalance.get(ex, [0])]
    spark_json = json.dumps(spark_data, separators=(",", ":"))

    html_3d = _build_3d_html(payload_json)
    with open("output/3d_liquidity_pro.html", "w") as f:
        f.write(html_3d)
    print("  Saved output/3d_liquidity_pro.html")

    html_dash = _build_dashboard_html(payload_json, spark_json, data)
    with open("output/dashboard.html", "w") as f:
        f.write(html_dash)
    print("  Saved output/dashboard.html")


def _build_3d_html(payload_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3D Liquidity Engine — BTC/USDT</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0F172A; overflow:hidden; font-family:'Consolas','SF Mono','Menlo',monospace; }}
  canvas {{ display:block; }}

  #tooltip {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(15,23,42,0.94); border:1px solid rgba(79,209,197,0.25);
    color:#c0d8e8; padding:10px 14px; border-radius:4px;
    font-size:11px; line-height:1.6; z-index:100;
    min-width:155px; font-family:'Consolas','SF Mono',monospace;
  }}
  #tooltip .t-label {{ color:rgba(79,209,197,0.5); font-size:9px; text-transform:uppercase; letter-spacing:0.7px; }}
  #tooltip .t-val {{ color:#e2e8f0; font-size:12px; font-weight:600; }}
  #tooltip .t-row {{ margin-top:3px; }}
  #tooltip .t-wall {{ color:#F6E05E; font-size:9px; margin-top:5px; font-weight:700; }}

  #info {{
    position:fixed; top:14px; left:16px; z-index:90;
    font-size:10px; color:rgba(148,163,184,0.4);
    font-family:'Consolas','SF Mono',monospace;
  }}
  #info span {{ color:rgba(79,209,197,0.45); }}
</style>
</head>
<body>
<div id="tooltip"></div>
<div id="info">BTC/USDT <span>Order Book Depth</span></div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

const D = {payload_json};
const EX = D.exchanges;
const NB = D.n_bins;
const PROF = D.profiles;
const WALLS = D.walls;

/* ═══════ LAYOUT ═══════ */
const SW = 15;
const SD = 3.0;
const GAP = 1.0;
const HY = 2.8;
const BG = 0x0F172A;
const TD = EX.length * SD + (EX.length - 1) * GAP;
const SROWS = 22;

/* ═══════ COLOR: #1E2A38 → #2F6FA3 → #4FD1C5 → #F6E05E ═══════ */
const CS = [
  [0.00, 0.118, 0.165, 0.220],
  [0.33, 0.184, 0.435, 0.640],
  [0.66, 0.310, 0.820, 0.773],
  [1.00, 0.965, 0.878, 0.369],
];

function colorAt(t) {{
  t = Math.max(0, Math.min(1, t));
  let lo = CS[0], hi = CS[CS.length - 1];
  for (let i = 0; i < CS.length - 1; i++) {{
    if (t >= CS[i][0] && t <= CS[i + 1][0]) {{ lo = CS[i]; hi = CS[i + 1]; break; }}
  }}
  const f = hi[0] > lo[0] ? (t - lo[0]) / (hi[0] - lo[0]) : 0;
  return new THREE.Color(
    lo[1] + (hi[1] - lo[1]) * f,
    lo[2] + (hi[2] - lo[2]) * f,
    lo[3] + (hi[3] - lo[3]) * f
  );
}}

/* ═══════ RENDERER ═══════ */
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const labelR = new CSS2DRenderer();
labelR.setSize(window.innerWidth, window.innerHeight);
labelR.domElement.style.position = 'absolute';
labelR.domElement.style.top = '0';
labelR.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelR.domElement);

/* ═══════ SCENE ═══════ */
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);

/* ═══════ CAMERA — balanced, ~80% frame fill ═══════ */
const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 400);
camera.position.set(10, 7, TD + 8);

/* ═══════ CONTROLS ═══════ */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 6;
controls.maxDistance = 40;
controls.maxPolarAngle = Math.PI * 0.44;
controls.target.set(0, 0.5, TD * 0.42);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.15;

/* ═══════ HELPERS ═══════ */
function mkLabel(text, size, color, bold) {{
  const d = document.createElement('div');
  d.textContent = text;
  d.style.cssText = `color:${{color}};font-family:'Consolas','SF Mono',monospace;font-size:${{size}}px;${{bold ? 'font-weight:600;' : ''}}white-space:nowrap;pointer-events:none;`;
  return new CSS2DObject(d);
}}

function xWorld(binIdx) {{ return -SW / 2 + (binIdx / NB) * SW; }}

function addLine(pts, mat) {{
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(g, mat));
}}

/* ═══════ AXIS MATERIALS — light gray, subtle ═══════ */
const axisLine = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.25 }});
const axisTick = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.18 }});
const gridLine = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.04 }});

/* ═══════ X AXIS — Price ═══════ */
addLine([new THREE.Vector3(-SW / 2, 0, -0.15), new THREE.Vector3(SW / 2, 0, -0.15)], axisLine);
const xTitle = mkLabel('Price (USDT)', 8, 'rgba(148,163,184,0.35)', false);
xTitle.position.set(0, -0.08, -0.7);
scene.add(xTitle);

D.price_labels.forEach(pl => {{
  const x = xWorld(pl.idx);
  addLine([new THREE.Vector3(x, 0, -0.15), new THREE.Vector3(x, -0.06, -0.15)], axisTick);
  addLine([new THREE.Vector3(x, 0.003, -0.15), new THREE.Vector3(x, 0.003, TD + 0.1)], gridLine);
  const lb = mkLabel(pl.val.toLocaleString(), 7, 'rgba(148,163,184,0.28)', false);
  lb.position.set(x, -0.18, -0.15);
  scene.add(lb);
}});

/* ═══════ Y AXIS — Volume ═══════ */
const yH = HY * 1.05;
addLine([new THREE.Vector3(-SW / 2 - 0.15, 0, -0.15), new THREE.Vector3(-SW / 2 - 0.15, yH, -0.15)], axisLine);
const yTitle = mkLabel('Volume', 8, 'rgba(148,163,184,0.35)', false);
yTitle.position.set(-SW / 2 - 0.5, yH * 0.5, -0.15);
scene.add(yTitle);

for (let i = 0; i <= 4; i++) {{
  const y = (i / 4) * yH;
  addLine([new THREE.Vector3(-SW / 2 - 0.15, y, -0.15), new THREE.Vector3(-SW / 2 - 0.25, y, -0.15)], axisTick);
  addLine([new THREE.Vector3(-SW / 2, y, -0.15), new THREE.Vector3(SW / 2, y, -0.15)], gridLine);
  const lb = mkLabel(Math.round(i * 25) + '%', 7, 'rgba(148,163,184,0.22)', false);
  lb.position.set(-SW / 2 - 0.5, y, -0.15);
  scene.add(lb);
}}

/* ═══════ Z AXIS — Exchange ═══════ */
addLine([new THREE.Vector3(-SW / 2 - 0.15, 0, -0.15), new THREE.Vector3(-SW / 2 - 0.15, 0, TD + 0.15)], axisLine);

/* ═══════ PER-EXCHANGE SURFACES ═══════ */
const meshes = [];

EX.forEach((ex, ei) => {{
  const prof = PROF[ex];
  const zOff = ei * (SD + GAP);

  const geo = new THREE.PlaneGeometry(SW, SD, NB - 1, SROWS - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const cols = new Float32Array(pos.count * 3);

  for (let i = 0; i < pos.count; i++) {{
    const c = i % NB;
    const r = Math.floor(i / NB);
    const h = prof[c];
    const rT = r / (SROWS - 1);
    const fade = Math.sin(rT * Math.PI);
    const fH = h * fade;

    pos.setY(i, fH * HY);
    pos.setZ(i, pos.getZ(i) + zOff + SD / 2);

    const col = colorAt(fH);
    cols[i * 3] = col.r;
    cols[i * 3 + 1] = col.g;
    cols[i * 3 + 2] = col.b;
  }}

  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({{
    vertexColors: true,
    roughness: 0.5,
    metalness: 0.02,
    flatShading: false,
    side: THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = {{ exchange: ex, profile: prof }};
  scene.add(mesh);
  meshes.push(mesh);

  /* ── Exchange label ── */
  const exLbl = mkLabel(ex, 9, 'rgba(148,163,184,0.4)', true);
  exLbl.position.set(-SW / 2 - 0.4, 0.04, zOff + SD / 2);
  scene.add(exLbl);
  addLine([
    new THREE.Vector3(-SW / 2 - 0.15, 0, zOff + SD / 2),
    new THREE.Vector3(-SW / 2 - 0.25, 0, zOff + SD / 2),
  ], axisTick);
}});

/* ═══════ PEAK LABELS — top 3 only ═══════ */
const dropMat = new THREE.LineBasicMaterial({{ color: 0xF6E05E, transparent: true, opacity: 0.18 }});

EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP);
  const zMid = zOff + SD / 2;

  (WALLS[ex] || []).forEach(w => {{
    if (!w.show_label) return;
    const x = xWorld(w.idx);
    const y = w.height * HY;

    // Vertical drop line
    addLine([new THREE.Vector3(x, y, zMid), new THREE.Vector3(x, 0, zMid)], dropMat);

    // Small floor tick
    addLine([new THREE.Vector3(x - 0.1, 0.003, zMid), new THREE.Vector3(x + 0.1, 0.003, zMid)], dropMat);

    // Price label
    const lb = mkLabel('$' + w.price.toLocaleString(), 8, 'rgba(246,224,94,0.55)', true);
    lb.position.set(x, y + 0.18, zMid);
    scene.add(lb);
  }});
}});

/* ═══════ FLOOR ═══════ */
const floorGeo = new THREE.PlaneGeometry(SW * 1.2, TD * 1.4);
floorGeo.rotateX(-Math.PI / 2);
const floorMat = new THREE.MeshBasicMaterial({{
  color: BG, transparent: true, opacity: 0.15,
}});
const floorMesh = new THREE.Mesh(floorGeo, floorMat);
floorMesh.position.set(0, -0.01, TD * 0.42);
scene.add(floorMesh);

/* ═══════ LIGHTING — neutral, even ═══════ */
scene.add(new THREE.AmbientLight(0xffffff, 0.7));

const keyL = new THREE.DirectionalLight(0xffffff, 0.9);
keyL.position.set(6, 14, 8);
scene.add(keyL);

const fillL = new THREE.DirectionalLight(0xe2e8f0, 0.45);
fillL.position.set(-6, 8, -4);
scene.add(fillL);

const backL = new THREE.DirectionalLight(0xe2e8f0, 0.2);
backL.position.set(0, 4, -10);
scene.add(backL);

/* ═══════ TOOLTIP ═══════ */
const ray = new THREE.Raycaster();
const mPos = new THREE.Vector2();
const tip = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', (e) => {{
  mPos.x = (e.clientX / window.innerWidth) * 2 - 1;
  mPos.y = -(e.clientY / window.innerHeight) * 2 + 1;
  ray.setFromCamera(mPos, camera);
  const hits = ray.intersectObjects(meshes);

  if (hits.length > 0) {{
    const hit = hits[0];
    const ex = hit.object.userData.exchange;
    const pt = hit.point;

    const priceT = (pt.x + SW / 2) / SW;
    const price = D.price_range[0] + priceT * (D.price_range[1] - D.price_range[0]);
    const binIdx = Math.round(priceT * NB);
    const volPct = Math.max(0, (pt.y / HY) * 100);

    let wallNote = '';
    for (const w of (WALLS[ex] || [])) {{
      if (Math.abs(w.idx - binIdx) < NB * 0.03) {{
        wallNote = '<div class="t-wall">LIQUIDITY WALL — $' + w.price.toLocaleString() + '</div>';
        break;
      }}
    }}

    tip.style.display = 'block';
    tip.style.left = (e.clientX + 16) + 'px';
    tip.style.top = (e.clientY - 10) + 'px';
    tip.innerHTML =
      `<div class="t-label">Exchange</div><div class="t-val">${{ex}}</div>`
      + `<div class="t-row"><div class="t-label">Price</div><div class="t-val">${{price.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}} USDT</div></div>`
      + `<div class="t-row"><div class="t-label">Volume</div><div class="t-val">${{volPct.toFixed(1)}}%</div></div>`
      + wallNote;
  }} else {{
    tip.style.display = 'none';
  }}
}});

renderer.domElement.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});

/* ═══════ RENDER ═══════ */
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelR.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelR.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


def _build_dashboard_html(payload_json: str, spark_json: str, data: dict) -> str:
    exchanges = data["exchanges"]
    imbalance = data.get("imbalance_series", {})
    n_samples = len(data.get("timestamps", []))
    price_min, price_max = data.get("price_range", (0, 0))

    stat_rows = ""
    for ex in exchanges:
        imb = imbalance.get(ex, [0])
        last_imb = imb[-1] if imb else 0
        avg_imb = float(np.mean(imb)) if imb else 0
        imb_color = "#4FD1C5" if last_imb > 0 else "#F87171"
        stat_rows += f"""
        <tr>
          <td style="font-weight:600">{ex}</td>
          <td style="color:{imb_color}">{last_imb:+.4f}</td>
          <td>{avg_imb:+.4f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Liquidity Dashboard — BTC/USDT</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0F172A; color:#94a3b8; font-family:'Consolas','SF Mono','Menlo',monospace; }}
  .header {{
    padding:14px 20px; border-bottom:1px solid rgba(148,163,184,0.08);
    display:flex; justify-content:space-between; align-items:center;
  }}
  .header h1 {{ color:#e2e8f0; font-size:13px; font-weight:600; letter-spacing:0.4px; }}
  .header .meta {{ font-size:10px; color:rgba(148,163,184,0.35); }}
  .panels {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr auto; height:calc(100vh - 48px); }}
  .panel {{ border:1px solid rgba(148,163,184,0.06); position:relative; overflow:hidden; }}
  .panel-title {{
    position:absolute; top:8px; left:12px; font-size:9px;
    color:rgba(148,163,184,0.25); text-transform:uppercase; letter-spacing:1px; z-index:10;
  }}
  .panel-3d {{ grid-column:1/-1; min-height:60vh; }}
  .panel-3d iframe {{ width:100%; height:100%; border:none; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  th {{ text-align:left; color:rgba(148,163,184,0.3); font-weight:400; padding:8px 12px;
       text-transform:uppercase; font-size:8px; letter-spacing:0.8px;
       border-bottom:1px solid rgba(148,163,184,0.08); }}
  td {{ padding:8px 12px; border-bottom:1px solid rgba(148,163,184,0.04); }}
  .spark-panel {{ padding:16px; }}
  .spark-row {{ display:flex; align-items:center; margin-bottom:10px; gap:12px; }}
  .spark-label {{ width:72px; font-size:10px; font-weight:600; }}
  .spark-canvas {{ flex:1; height:28px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Cross-Exchange Liquidity Dashboard — BTC/USDT</h1>
  <div class="meta">{n_samples} snapshots &middot; {round(price_min,2)} — {round(price_max,2)} USDT</div>
</div>
<div class="panels">
  <div class="panel panel-3d">
    <div class="panel-title">3D Liquidity Surface</div>
    <iframe src="3d_liquidity_pro.html"></iframe>
  </div>
  <div class="panel" style="padding:16px;">
    <div class="panel-title">Exchange Metrics</div>
    <table style="margin-top:26px;">
      <tr><th>Exchange</th><th>Last Imbalance</th><th>Avg Imbalance</th></tr>
      {stat_rows}
    </table>
  </div>
  <div class="panel spark-panel">
    <div class="panel-title">Imbalance Over Time</div>
    <div id="sparklines" style="margin-top:26px;"></div>
  </div>
</div>
<script>
const sparkData = {spark_json};
const exColors = {{ Binance:'#F6E05E', Coinbase:'#4FD1C5', Kraken:'#2F6FA3' }};
const ctr = document.getElementById('sparklines');

Object.entries(sparkData).forEach(([ex, vals]) => {{
  const row = document.createElement('div');
  row.className = 'spark-row';
  const lbl = document.createElement('div');
  lbl.className = 'spark-label';
  lbl.style.color = exColors[ex] || '#94a3b8';
  lbl.textContent = ex;
  const cvs = document.createElement('canvas');
  cvs.className = 'spark-canvas';
  cvs.height = 28;
  row.appendChild(lbl);
  row.appendChild(cvs);
  ctr.appendChild(row);

  requestAnimationFrame(() => {{
    cvs.width = cvs.offsetWidth * 2;
    cvs.height = 56;
    cvs.style.height = '28px';
    const ctx = cvs.getContext('2d');
    const w = cvs.width, h = cvs.height;
    if (vals.length < 2) return;
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const rng = mx - mn || 1;
    const pad = 4;

    const zy = h - pad - ((0 - mn) / rng) * (h - pad * 2);
    ctx.strokeStyle = 'rgba(148,163,184,0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, zy);
    ctx.lineTo(w, zy);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = exColors[ex] || '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    vals.forEach((v, i) => {{
      const x = (i / (vals.length - 1)) * w;
      const y = h - pad - ((v - mn) / rng) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
}});
</script>
</body>
</html>"""
