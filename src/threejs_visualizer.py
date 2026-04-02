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

def _build_exchange_profiles(data: dict, n_bins: int = 120) -> dict:
    """
    Build per-exchange depth profiles with aggressive noise removal.

    Pipeline: average → log-compress → denoise → resample → hard threshold →
              keep only dominant structures.
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

        # Light smooth — preserve grid structure
        resampled = gaussian_filter(resampled, sigma=1.2)

        raw_profiles[ex] = resampled.copy()
        global_max = max(global_max, resampled.max())

    for ex in exchanges:
        p = raw_profiles[ex]
        if global_max > 0:
            p = p / global_max

        # Aggressive threshold — kill bottom 75%
        p75 = np.percentile(p, 75)
        p = np.clip((p - p75) / (1.0 - p75 + 1e-9), 0, 1)

        # Minimal smooth to keep grid character
        p = gaussian_filter(p, sigma=0.8)

        pmax = p.max()
        if pmax > 0:
            p = p / pmax

        profiles[ex] = p

    # ── Detect top peaks only ──
    walls = {}
    all_walls = []
    for ex in exchanges:
        p = profiles[ex]
        peak_idxs, _ = find_peaks(p, height=0.4, distance=n_bins // 10, prominence=0.15)
        wall_list = []
        for idx in peak_idxs:
            wall_list.append({
                "idx": int(idx),
                "height": round(float(p[idx]), 3),
                "price": round(float(price_ticks[idx]), 2),
            })
        wall_list.sort(key=lambda w: w["height"], reverse=True)
        walls[ex] = wall_list[:4]
        all_walls.extend(wall_list[:4])

    # Keep only global top 5 for drop lines
    all_walls.sort(key=lambda w: w["height"], reverse=True)
    top_keys = set()
    for w in all_walls[:5]:
        top_keys.add(w["price"])

    for ex in exchanges:
        for w in walls[ex]:
            w["is_top"] = w["price"] in top_keys

    # ── Imbalance ──
    avg_imbalances = {}
    for ex in exchanges:
        imb = imbalance_series.get(ex, [0])
        avg_imbalances[ex] = round(float(np.mean(imb)), 4)

    # ── Price labels — 8 ticks ──
    n_labels = 8
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
  body {{ background:#0F1115; overflow:hidden; font-family:'Consolas','SF Mono','Menlo',monospace; }}
  canvas {{ display:block; }}

  #tooltip {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(15,17,21,0.95); border:1px solid rgba(80,200,230,0.3);
    color:#c0d8e8; padding:10px 14px; border-radius:3px;
    font-size:11px; line-height:1.6; z-index:100;
    min-width:160px; font-family:'Consolas','SF Mono',monospace;
  }}
  #tooltip .t-label {{ color:rgba(80,200,230,0.5); font-size:9px; text-transform:uppercase; letter-spacing:0.8px; }}
  #tooltip .t-val {{ color:#e0f4ff; font-size:12px; font-weight:600; }}
  #tooltip .t-row {{ margin-top:3px; }}
  #tooltip .t-wall {{ color:#f0d040; font-size:9px; margin-top:5px; font-weight:700; }}

  #info {{
    position:fixed; top:14px; left:16px; z-index:90;
    font-size:10px; color:rgba(180,200,220,0.35);
    font-family:'Consolas','SF Mono',monospace;
    letter-spacing:0.3px;
  }}
  #info span {{ color:rgba(80,200,230,0.5); }}
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
const SW = 16;
const SD = 3.5;
const GAP = 0.6;
const HY = 3.2;
const BG = 0x0F1115;
const TD = EX.length * SD + (EX.length - 1) * GAP;
const SROWS = 20;

/* ═══════ HEATMAP: dark-blue → blue → cyan → yellow ═══════ */
const CS = [
  [0.00, 0.04, 0.06, 0.18],
  [0.20, 0.06, 0.12, 0.35],
  [0.40, 0.08, 0.25, 0.55],
  [0.55, 0.10, 0.45, 0.70],
  [0.70, 0.15, 0.65, 0.80],
  [0.82, 0.30, 0.82, 0.85],
  [0.92, 0.70, 0.92, 0.40],
  [1.00, 0.95, 0.88, 0.15],
];

function colorAt(t) {{
  t = Math.max(0, Math.min(1, t));
  let lo = CS[0], hi = CS[CS.length-1];
  for (let i = 0; i < CS.length-1; i++) {{
    if (t >= CS[i][0] && t <= CS[i+1][0]) {{ lo=CS[i]; hi=CS[i+1]; break; }}
  }}
  const f = hi[0]>lo[0] ? (t-lo[0])/(hi[0]-lo[0]) : 0;
  return new THREE.Color(lo[1]+(hi[1]-lo[1])*f, lo[2]+(hi[2]-lo[2])*f, lo[3]+(hi[3]-lo[3])*f);
}}

/* ═══════ RENDERER ═══════ */
const renderer = new THREE.WebGLRenderer({{ antialias:true }});
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

/* ═══════ CAMERA — analytical angle, low distortion ═══════ */
const camera = new THREE.PerspectiveCamera(35, window.innerWidth/window.innerHeight, 0.1, 500);
camera.position.set(14, 10, TD + 12);

/* ═══════ CONTROLS ═══════ */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 8;
controls.maxDistance = 50;
controls.maxPolarAngle = Math.PI * 0.42;
controls.target.set(0, 0.8, TD * 0.42);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.12;

/* ═══════ HELPERS ═══════ */
function mkLabel(text, size, color, bold) {{
  const d = document.createElement('div');
  d.textContent = text;
  d.style.cssText = `color:${{color}};font-family:'Consolas','SF Mono',monospace;font-size:${{size}}px;${{bold?'font-weight:700;':''}}white-space:nowrap;pointer-events:none;`;
  return new CSS2DObject(d);
}}

function xWorld(binIdx) {{ return -SW/2 + (binIdx/NB) * SW; }}

function addLine(pts, mat) {{
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(g, mat));
}}

/* ═══════ AXIS MATERIALS ═══════ */
const axisMain = new THREE.LineBasicMaterial({{ color:0xffffff, transparent:true, opacity:0.5 }});
const axisTick = new THREE.LineBasicMaterial({{ color:0xffffff, transparent:true, opacity:0.3 }});
const gridFloor = new THREE.LineBasicMaterial({{ color:0xffffff, transparent:true, opacity:0.04 }});

/* ═══════ X AXIS — Price ═══════ */
addLine([new THREE.Vector3(-SW/2, 0, -0.2), new THREE.Vector3(SW/2, 0, -0.2)], axisMain);
const xTitle = mkLabel('PRICE (USDT)', 9, 'rgba(255,255,255,0.4)', true);
xTitle.position.set(0, -0.1, -1.0);
scene.add(xTitle);

D.price_labels.forEach(pl => {{
  const x = xWorld(pl.idx);
  addLine([new THREE.Vector3(x, 0, -0.2), new THREE.Vector3(x, -0.1, -0.2)], axisTick);
  // Floor grid line along Z
  addLine([new THREE.Vector3(x, 0.005, -0.2), new THREE.Vector3(x, 0.005, TD + 0.2)], gridFloor);
  const lb = mkLabel(pl.val.toLocaleString(), 8, 'rgba(255,255,255,0.3)', false);
  lb.position.set(x, -0.25, -0.2);
  scene.add(lb);
}});

/* ═══════ Z AXIS — Volume (vertical) ═══════ */
const yH = HY * 1.1;
addLine([new THREE.Vector3(-SW/2 - 0.2, 0, -0.2), new THREE.Vector3(-SW/2 - 0.2, yH, -0.2)], axisMain);
const zTitle = mkLabel('VOLUME', 9, 'rgba(255,255,255,0.4)', true);
zTitle.position.set(-SW/2 - 0.6, yH * 0.5, -0.2);
scene.add(zTitle);

for (let i = 0; i <= 4; i++) {{
  const y = (i/4) * yH;
  addLine([new THREE.Vector3(-SW/2 - 0.2, y, -0.2), new THREE.Vector3(-SW/2 - 0.35, y, -0.2)], axisTick);
  // Horizontal grid across back wall
  addLine([new THREE.Vector3(-SW/2, y, -0.2), new THREE.Vector3(SW/2, y, -0.2)], gridFloor);
  const lb = mkLabel(Math.round(i * 25) + '%', 7, 'rgba(255,255,255,0.25)', false);
  lb.position.set(-SW/2 - 0.6, y, -0.2);
  scene.add(lb);
}}

/* ═══════ Y AXIS — Exchange (depth) ═══════ */
addLine([new THREE.Vector3(-SW/2 - 0.2, 0, -0.2), new THREE.Vector3(-SW/2 - 0.2, 0, TD + 0.3)], axisMain);
const yTitle = mkLabel('EXCHANGE', 9, 'rgba(255,255,255,0.4)', true);
yTitle.position.set(-SW/2 - 0.6, -0.1, TD * 0.5);
scene.add(yTitle);

/* ═══════ PER-EXCHANGE SURFACES ═══════ */
const meshes = [];

EX.forEach((ex, ei) => {{
  const prof = PROF[ex];
  const zOff = ei * (SD + GAP);

  /* ── Flat-shaded grid surface ── */
  const geo = new THREE.PlaneGeometry(SW, SD, NB - 1, SROWS - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const cols = new Float32Array(pos.count * 3);

  for (let i = 0; i < pos.count; i++) {{
    const c = i % NB;
    const r = Math.floor(i / NB);
    const h = prof[c];
    const rT = r / (SROWS - 1);
    // Flat-top taper: full height in center 60%, linear fall at edges
    const edgeDist = Math.min(rT, 1.0 - rT) * 2.0;
    const fade = Math.min(edgeDist / 0.4, 1.0);
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
    roughness: 0.55,
    metalness: 0.0,
    flatShading: true,
    side: THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = {{ exchange: ex, profile: prof }};
  scene.add(mesh);
  meshes.push(mesh);

  /* ── Wireframe grid overlay ── */
  const wfGeo = geo.clone();
  const wfMat = new THREE.MeshBasicMaterial({{
    wireframe: true,
    color: 0xffffff,
    transparent: true,
    opacity: 0.04,
  }});
  scene.add(new THREE.Mesh(wfGeo, wfMat));

  /* ── Exchange label ── */
  const exLbl = mkLabel(ex.toUpperCase(), 10, 'rgba(255,255,255,0.45)', true);
  exLbl.position.set(-SW/2 - 0.5, 0.06, zOff + SD / 2);
  scene.add(exLbl);
  addLine([
    new THREE.Vector3(-SW/2 - 0.2, 0, zOff + SD/2),
    new THREE.Vector3(-SW/2 - 0.32, 0, zOff + SD/2),
  ], axisTick);
}});

/* ═══════ VERTICAL DROP LINES from top peaks ═══════ */
const dropMat = new THREE.LineBasicMaterial({{ color: 0xf0d040, transparent: true, opacity: 0.35 }});
const dropMatSub = new THREE.LineBasicMaterial({{ color: 0xf0d040, transparent: true, opacity: 0.1 }});

EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP);
  const zMid = zOff + SD / 2;

  (WALLS[ex] || []).forEach(w => {{
    const x = xWorld(w.idx);
    const y = w.height * HY;

    // Vertical drop line
    addLine([new THREE.Vector3(x, y, zMid), new THREE.Vector3(x, 0, zMid)], w.is_top ? dropMat : dropMatSub);

    // Small floor tick
    if (w.is_top) {{
      addLine([new THREE.Vector3(x - 0.12, 0.005, zMid), new THREE.Vector3(x + 0.12, 0.005, zMid)], dropMat);
    }}
  }});
}});

/* ═══════ FLOOR GRID ═══════ */
// Cross-Z grid lines at price tick positions (already added above)
// Cross-X grid lines at exchange positions
EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP) + SD / 2;
  addLine([new THREE.Vector3(-SW/2, 0.005, zOff), new THREE.Vector3(SW/2, 0.005, zOff)], gridFloor);
}});

/* ═══════ LIGHTING — clean, even, no drama ═══════ */
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

const keyL = new THREE.DirectionalLight(0xffffff, 1.2);
keyL.position.set(8, 16, 8);
scene.add(keyL);

const fillL = new THREE.DirectionalLight(0xc0d0e0, 0.5);
fillL.position.set(-8, 10, -4);
scene.add(fillL);

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

    const priceT = (pt.x + SW/2) / SW;
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
      + `<div class="t-row"><div class="t-label">Price</div><div class="t-val">${{price.toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}} USDT</div></div>`
      + `<div class="t-row"><div class="t-label">Volume</div><div class="t-val">${{volPct.toFixed(1)}}%</div></div>`
      + wallNote;
  }} else {{
    tip.style.display = 'none';
  }}
}});

renderer.domElement.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});

/* ═══════ RENDER LOOP ═══════ */
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelR.render(scene, camera);
}}
animate();

/* ═══════ RESIZE ═══════ */
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
        imb_color = "#40b878" if last_imb > 0 else "#e05858"
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
  body {{ background:#0a0c10; color:#a0b0c0; font-family:'Consolas','SF Mono','Menlo',monospace; }}
  .header {{
    padding:14px 20px; border-bottom:1px solid rgba(255,255,255,0.06);
    display:flex; justify-content:space-between; align-items:center;
  }}
  .header h1 {{ color:#d0dce8; font-size:13px; font-weight:600; letter-spacing:0.5px; }}
  .header .meta {{ font-size:10px; color:rgba(255,255,255,0.25); }}
  .panels {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr auto; height:calc(100vh - 48px); }}
  .panel {{ border:1px solid rgba(255,255,255,0.04); position:relative; overflow:hidden; }}
  .panel-title {{
    position:absolute; top:8px; left:12px; font-size:9px;
    color:rgba(255,255,255,0.2); text-transform:uppercase; letter-spacing:1.2px; z-index:10;
  }}
  .panel-3d {{ grid-column:1/-1; min-height:60vh; }}
  .panel-3d iframe {{ width:100%; height:100%; border:none; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  th {{ text-align:left; color:rgba(255,255,255,0.2); font-weight:400; padding:8px 12px;
       text-transform:uppercase; font-size:8px; letter-spacing:0.8px;
       border-bottom:1px solid rgba(255,255,255,0.06); }}
  td {{ padding:8px 12px; border-bottom:1px solid rgba(255,255,255,0.03); }}
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
const exColors = {{ Binance:'#e0b830', Coinbase:'#3080e0', Kraken:'#7050c0' }};
const ctr = document.getElementById('sparklines');

Object.entries(sparkData).forEach(([ex, vals]) => {{
  const row = document.createElement('div');
  row.className = 'spark-row';
  const lbl = document.createElement('div');
  lbl.className = 'spark-label';
  lbl.style.color = exColors[ex] || '#a0b0c0';
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

    // Zero line
    const zy = h - pad - ((0 - mn) / rng) * (h - pad * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, zy);
    ctx.lineTo(w, zy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Sparkline
    ctx.strokeStyle = exColors[ex] || '#a0b0c0';
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
