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

def _build_exchange_profiles(data: dict, n_bins: int = 300) -> dict:
    """
    Build high-resolution depth profile per exchange.

    Pipeline: average -> log-compress -> denoise -> resample -> smooth ->
              hard threshold -> power curve -> normalize.

    Keeps only top structures for a clean, minimal output.
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

        resampled = gaussian_filter(resampled, sigma=2.5)

        raw_profiles[ex] = resampled.copy()
        global_max = max(global_max, resampled.max())

    for ex in exchanges:
        p = raw_profiles[ex]
        if global_max > 0:
            p = p / global_max

        # Hard threshold — keep only top 30% of signal
        p70 = np.percentile(p, 70)
        p = np.clip((p - p70) / (1.0 - p70 + 1e-9), 0, 1)

        # Sharpen peaks with steeper power curve
        p = np.power(p, 0.6)
        p = gaussian_filter(p, sigma=2.0)

        pmax = p.max()
        if pmax > 0:
            p = p / pmax

        profiles[ex] = p

    # ── Detect liquidity walls — keep only strongest ──
    walls = {}
    all_walls = []
    for ex in exchanges:
        p = profiles[ex]
        peak_idxs, props = find_peaks(p, height=0.45, distance=n_bins // 15, prominence=0.2)
        wall_list = []
        for idx in peak_idxs:
            wall_list.append({
                "idx": int(idx),
                "height": round(float(p[idx]), 3),
                "price": round(float(price_ticks[idx]), 2),
                "exchange": ex,
            })
        wall_list.sort(key=lambda w: w["height"], reverse=True)
        walls[ex] = wall_list[:3]
        all_walls.extend(wall_list[:3])

    # Global top 5 walls for labels
    all_walls.sort(key=lambda w: w["height"], reverse=True)
    top_wall_keys = set()
    for w in all_walls[:5]:
        top_wall_keys.add((w["exchange"], w["idx"]))

    # Mark which walls get labels
    for ex in exchanges:
        for w in walls[ex]:
            w["show_label"] = (ex, w["idx"]) in top_wall_keys

    # ── Per-exchange avg imbalance ──
    avg_imbalances = {}
    for ex in exchanges:
        imb = imbalance_series.get(ex, [0])
        avg_imbalances[ex] = round(float(np.mean(imb)), 4)

    # ── Price labels (sparse — only 5 ticks) ──
    n_labels = 5
    step = max(1, n_bins // n_labels)
    price_labels = [
        {"idx": int(i), "val": round(float(price_ticks[i]), 2)}
        for i in range(0, n_bins, step)
    ]

    # Serialize profiles
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
        "mid_idx": int(n_bins / 2),
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

    # Metrics for dashboard
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
<title>3D Liquidity Engine</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#08090c; overflow:hidden; font-family:'Inter','Helvetica Neue',sans-serif; }}
  canvas {{ display:block; }}

  #tooltip {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(10,12,18,0.92); border:1px solid rgba(80,140,180,0.2);
    color:#a0b8c8; padding:10px 14px; border-radius:6px;
    font-size:11px; line-height:1.7; z-index:100;
    backdrop-filter:blur(12px); min-width:150px;
    font-family:'Inter','Helvetica Neue',sans-serif;
  }}
  #tooltip .t-label {{ color:rgba(90,150,190,0.6); font-size:9px; text-transform:uppercase; letter-spacing:0.8px; }}
  #tooltip .t-val {{ color:#d0e4f0; font-size:12px; font-weight:500; }}
  #tooltip .t-row {{ margin-top:3px; }}
  #tooltip .t-wall {{ color:#70d4ff; font-size:9px; margin-top:5px; font-weight:600; letter-spacing:0.5px; }}

  #legend {{
    position:fixed; bottom:16px; right:16px; z-index:90;
    background:rgba(10,12,18,0.7); border:1px solid rgba(60,80,100,0.15);
    padding:12px 16px; border-radius:8px; font-size:9px;
    backdrop-filter:blur(12px); color:rgba(130,160,180,0.6);
    font-family:'Inter','Helvetica Neue',sans-serif;
  }}
  #legend .l-title {{ font-size:9px; color:rgba(130,160,180,0.4); font-weight:500; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }}
  #legend .l-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
  #legend .l-swatch {{ width:16px; height:2px; border-radius:1px; }}
</style>
</head>
<body>
<div id="tooltip"></div>
<div id="legend">
  <div class="l-title">Depth</div>
  <div class="l-row"><div class="l-swatch" style="background:#12102a"></div><span>Low</span></div>
  <div class="l-row"><div class="l-swatch" style="background:#1a3a6a"></div><span>Mid</span></div>
  <div class="l-row"><div class="l-swatch" style="background:#50c8e8"></div><span>High</span></div>
  <div class="l-row"><div class="l-swatch" style="background:#e0f4ff"></div><span>Peak</span></div>
</div>

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

/* ═══════════════════ DATA ═══════════════════ */
const D = {payload_json};
const EX = D.exchanges;
const NB = D.n_bins;
const PROF = D.profiles;
const WALLS = D.walls;

/* ═══════════════════ LAYOUT ═══════════════════ */
const SW = 14;
const SD = 3.2;
const GAP = 2.0;
const HY = 2.8;
const BG = 0x08090c;
const TD = EX.length * SD + (EX.length - 1) * GAP;
const SROWS = 24;

/* ═══════════════════ COLOR — 3 stops only ═══════════════════ */
const CS = [
  [0.00, 0.07, 0.06, 0.16],
  [0.15, 0.08, 0.08, 0.25],
  [0.35, 0.10, 0.18, 0.42],
  [0.55, 0.10, 0.30, 0.55],
  [0.75, 0.20, 0.60, 0.72],
  [0.90, 0.45, 0.82, 0.88],
  [1.00, 0.88, 0.96, 1.00],
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

/* ═══════════════════ RENDERER ═══════════════════ */
const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const labelR = new CSS2DRenderer();
labelR.setSize(window.innerWidth, window.innerHeight);
labelR.domElement.style.position = 'absolute';
labelR.domElement.style.top = '0';
labelR.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelR.domElement);

/* ═══════════════════ SCENE ═══════════════════ */
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);

/* ═══════════════════ CAMERA ═══════════════════ */
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 400);
camera.position.set(9, 6, TD + 6);

/* ═══════════════════ CONTROLS ═══════════════════ */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 5;
controls.maxDistance = 35;
controls.maxPolarAngle = Math.PI * 0.45;
controls.target.set(0, 0.3, TD * 0.4);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.15;

/* ═══════════════════ HELPERS ═══════════════════ */
function mkLabel(text, size, color, bold) {{
  const d = document.createElement('div');
  d.textContent = text;
  d.style.cssText = `color:${{color}};font-family:'Inter','Helvetica Neue',sans-serif;font-size:${{size}}px;${{bold?'font-weight:600;':''}}white-space:nowrap;pointer-events:none;`;
  return new CSS2DObject(d);
}}

function xWorld(binIdx) {{ return -SW/2 + (binIdx/NB) * SW; }}

/* ═══════════════════ MINIMAL AXES ═══════════════════ */
const axisMat = new THREE.LineBasicMaterial({{ color:0x2a3545, transparent:true, opacity:0.5 }});
const tickMat = new THREE.LineBasicMaterial({{ color:0x2a3545, transparent:true, opacity:0.35 }});

function addLine(pts, mat) {{
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(g, mat));
}}

// X axis — thin line, sparse ticks
addLine([new THREE.Vector3(-SW/2, 0, -0.3), new THREE.Vector3(SW/2, 0, -0.3)], axisMat);
const xT = mkLabel('Price (USDT)', 9, 'rgba(100,130,160,0.4)', false);
xT.position.set(0, -0.15, -0.7);
scene.add(xT);

D.price_labels.forEach(pl => {{
  const x = xWorld(pl.idx);
  addLine([new THREE.Vector3(x, 0, -0.3), new THREE.Vector3(x, -0.08, -0.3)], tickMat);
  const lb = mkLabel(pl.val.toLocaleString(), 7, 'rgba(100,130,160,0.35)', false);
  lb.position.set(x, -0.22, -0.3);
  scene.add(lb);
}});

// Y axis — minimal
const yH = HY * 1.1;
addLine([new THREE.Vector3(-SW/2-0.3, 0, -0.3), new THREE.Vector3(-SW/2-0.3, yH, -0.3)], axisMat);
const yT = mkLabel('Volume', 9, 'rgba(100,130,160,0.4)', false);
yT.position.set(-SW/2-0.6, yH*0.55, -0.3);
scene.add(yT);

// Only 3 Y ticks
for (let i = 0; i <= 2; i++) {{
  const y = (i/2)*yH;
  addLine([new THREE.Vector3(-SW/2-0.3, y, -0.3), new THREE.Vector3(-SW/2-0.45, y, -0.3)], tickMat);
  const lb = mkLabel(Math.round((i/2)*100)+'%', 7, 'rgba(100,130,160,0.3)', false);
  lb.position.set(-SW/2-0.7, y, -0.3);
  scene.add(lb);
}}

// Z axis
addLine([new THREE.Vector3(-SW/2-0.3, 0, -0.3), new THREE.Vector3(-SW/2-0.3, 0, TD+0.3)], axisMat);

/* ═══════════════════ EXCHANGE COLORS ═══════════════════ */
const EXC = {{ Binance:'rgba(200,180,120,0.5)', Coinbase:'rgba(100,140,200,0.5)', Kraken:'rgba(150,120,200,0.5)' }};

/* ═══════════════════ PER-EXCHANGE SURFACES ═══════════════════ */
const meshes = [];

EX.forEach((ex, ei) => {{
  const prof = PROF[ex];
  const zOff = ei * (SD + GAP);

  const geo = new THREE.PlaneGeometry(SW, SD, NB-1, SROWS-1);
  geo.rotateX(-Math.PI/2);
  const pos = geo.attributes.position;
  const cols = new Float32Array(pos.count*3);

  for (let i = 0; i < pos.count; i++) {{
    const c = i % NB;
    const r = Math.floor(i / NB);
    const h = prof[c];
    const rT = r / (SROWS-1);
    const fade = Math.sin(rT * Math.PI);
    const fH = h * fade;

    pos.setY(i, fH * HY);
    pos.setZ(i, pos.getZ(i) + zOff + SD/2);

    const col = colorAt(fH);
    cols[i*3] = col.r; cols[i*3+1] = col.g; cols[i*3+2] = col.b;
  }}

  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({{
    vertexColors:true, roughness:0.45, metalness:0.08,
    flatShading:false, side:THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = {{ exchange:ex, profile:prof }};
  scene.add(mesh);
  meshes.push(mesh);

  // Exchange label — subtle
  const exLbl = mkLabel(ex, 10, EXC[ex]||'rgba(140,160,180,0.5)', true);
  exLbl.position.set(-SW/2-0.5, 0.08, zOff+SD/2);
  scene.add(exLbl);
}});

/* ═══════════════════ WALL MARKERS — only labeled walls ═══════════════════ */
EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP);
  const zMid = zOff + SD/2;

  (WALLS[ex] || []).forEach(w => {{
    if (!w.show_label) return;
    const x = xWorld(w.idx);
    const y = w.height * HY;

    // Thin vertical drop line
    const dropMat = new THREE.LineBasicMaterial({{ color:0x50c8e8, transparent:true, opacity:0.12 }});
    addLine([new THREE.Vector3(x, y, zMid), new THREE.Vector3(x, 0, zMid)], dropMat);

    // Minimal label
    const lb = mkLabel('$' + w.price.toLocaleString(), 8, 'rgba(120,210,240,0.6)', true);
    lb.position.set(x, y + 0.2, zMid);
    scene.add(lb);
  }});
}});

/* ═══════════════════ FLOOR — subtle ground plane only ═══════════════════ */
const floorGeo = new THREE.PlaneGeometry(SW*1.4, TD*1.6);
floorGeo.rotateX(-Math.PI/2);
const floorMat = new THREE.MeshStandardMaterial({{
  color:0x08090c, roughness:0.9, metalness:0.0, transparent:true, opacity:0.3,
}});
const floorMesh = new THREE.Mesh(floorGeo, floorMat);
floorMesh.position.set(0, -0.02, TD*0.4);
scene.add(floorMesh);

/* ═══════════════════ LIGHTING — clean, soft ═══════════════════ */
scene.add(new THREE.AmbientLight(0x303848, 0.8));

const keyL = new THREE.DirectionalLight(0xd0e0f0, 1.0);
keyL.position.set(6, 12, 6);
scene.add(keyL);

const fillL = new THREE.DirectionalLight(0x607090, 0.4);
fillL.position.set(-6, 8, -3);
scene.add(fillL);

/* ═══════════════════ TOOLTIP ═══════════════════ */
const ray = new THREE.Raycaster();
const mPos = new THREE.Vector2();
const tip = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', (e) => {{
  mPos.x = (e.clientX/window.innerWidth)*2-1;
  mPos.y = -(e.clientY/window.innerHeight)*2+1;
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
    const exWalls = WALLS[ex] || [];
    for (const w of exWalls) {{
      if (Math.abs(w.idx - binIdx) < NB * 0.02) {{
        wallNote = '<div class="t-wall">LIQUIDITY WALL</div>';
        break;
      }}
    }}

    tip.style.display = 'block';
    tip.style.left = (e.clientX+18)+'px';
    tip.style.top = (e.clientY-12)+'px';
    tip.innerHTML = `<div class="t-label">Exchange</div><div class="t-val">${{ex}}</div>`
      + `<div class="t-row"><div class="t-label">Price</div><div class="t-val">${{price.toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}} USDT</div></div>`
      + `<div class="t-row"><div class="t-label">Relative Volume</div><div class="t-val">${{volPct.toFixed(1)}}%</div></div>`
      + wallNote;
  }} else {{
    tip.style.display = 'none';
  }}
}});

renderer.domElement.addEventListener('mouseleave', () => {{ tip.style.display='none'; }});

/* ═══════════════════ RENDER LOOP ═══════════════════ */
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelR.render(scene, camera);
}}
animate();

/* ═══════════════════ RESIZE ═══════════════════ */
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth/window.innerHeight;
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
        imb_color = "#5aaa7a" if last_imb > 0 else "#aa5a6a"
        ex_color = '#b0a070' if ex == 'Binance' else '#6080b0' if ex == 'Coinbase' else '#8070a0'
        stat_rows += f"""
        <tr>
          <td style="color:{ex_color};font-weight:500">{ex}</td>
          <td style="color:{imb_color}">{last_imb:+.4f}</td>
          <td>{avg_imb:+.4f}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Liquidity Dashboard</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#06070a; color:#7a8a98; font-family:'Inter','Helvetica Neue',sans-serif; }}
  .header {{
    padding:16px 24px; border-bottom:1px solid rgba(40,50,60,0.3);
    display:flex; justify-content:space-between; align-items:center;
  }}
  .header h1 {{ color:#90a0b0; font-size:13px; font-weight:500; letter-spacing:0.5px; }}
  .header .meta {{ font-size:10px; color:rgba(100,120,140,0.4); }}
  .panels {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr auto; height:calc(100vh - 52px); }}
  .panel {{ border:1px solid rgba(40,50,60,0.2); position:relative; overflow:hidden; }}
  .panel-title {{
    position:absolute; top:10px; left:14px; font-size:9px;
    color:rgba(100,120,140,0.35); text-transform:uppercase; letter-spacing:1.2px; z-index:10;
  }}
  .panel-3d {{ grid-column:1/-1; min-height:60vh; }}
  .panel-3d iframe {{ width:100%; height:100%; border:none; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  th {{ text-align:left; color:rgba(100,120,140,0.35); font-weight:400; padding:8px 12px;
       text-transform:uppercase; font-size:8px; letter-spacing:0.8px;
       border-bottom:1px solid rgba(40,50,60,0.2); }}
  td {{ padding:8px 12px; border-bottom:1px solid rgba(30,40,50,0.15); }}
  .spark-panel {{ padding:16px; }}
  .spark-row {{ display:flex; align-items:center; margin-bottom:10px; gap:12px; }}
  .spark-label {{ width:72px; font-size:10px; font-weight:500; }}
  .spark-canvas {{ flex:1; height:26px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Cross-Exchange Liquidity Dashboard</h1>
  <div class="meta">{n_samples} snapshots &middot; {round(price_min,2)} — {round(price_max,2)} USDT</div>
</div>
<div class="panels">
  <div class="panel panel-3d">
    <div class="panel-title">3D Liquidity Surface</div>
    <iframe src="3d_liquidity_pro.html"></iframe>
  </div>
  <div class="panel" style="padding:16px;">
    <div class="panel-title">Exchange Metrics</div>
    <table style="margin-top:28px;">
      <tr><th>Exchange</th><th>Last Imbalance</th><th>Avg Imbalance</th></tr>
      {stat_rows}
    </table>
  </div>
  <div class="panel spark-panel">
    <div class="panel-title">Imbalance Over Time</div>
    <div id="sparklines" style="margin-top:28px;"></div>
  </div>
</div>
<script>
const sparkData = {spark_json};
const exColors = {{ Binance:'#b0a070', Coinbase:'#6080b0', Kraken:'#8070a0' }};
const ctr = document.getElementById('sparklines');

Object.entries(sparkData).forEach(([ex, vals]) => {{
  const row = document.createElement('div');
  row.className = 'spark-row';
  const lbl = document.createElement('div');
  lbl.className = 'spark-label';
  lbl.style.color = exColors[ex]||'#7a8a98';
  lbl.textContent = ex;
  const cvs = document.createElement('canvas');
  cvs.className = 'spark-canvas';
  cvs.height = 26;
  row.appendChild(lbl); row.appendChild(cvs); ctr.appendChild(row);

  requestAnimationFrame(() => {{
    cvs.width = cvs.offsetWidth*2; cvs.height = 52; cvs.style.height = '26px';
    const ctx = cvs.getContext('2d');
    const w = cvs.width, h = cvs.height;
    if (vals.length < 2) return;
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const rng = mx-mn||1; const pad = 4;

    const zy = h-pad-((0-mn)/rng)*(h-pad*2);
    ctx.strokeStyle = 'rgba(50,60,70,0.2)'; ctx.lineWidth = 1;
    ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(0,zy); ctx.lineTo(w,zy); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = exColors[ex]||'#7a8a98'; ctx.lineWidth = 1.5; ctx.beginPath();
    vals.forEach((v,i) => {{
      const x = (i/(vals.length-1))*w;
      const y = h-pad-((v-mn)/rng)*(h-pad*2);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }});
}});
</script>
</body>
</html>"""
