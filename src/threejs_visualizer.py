"""
Professional 3D Liquidity Engine — Three.js WebGL.

Generates:
  output/3d_liquidity_pro.html  — interactive multi-exchange 3D surface
  output/dashboard.html         — multi-panel analytics dashboard
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_exchange_profiles(data: dict, n_bins: int = 120) -> dict:
    """
    Build a clean depth profile per exchange.

    For each exchange:
      1. Average volume across time snapshots → 1D profile (400 bins)
      2. Log-compress
      3. Median filter (kill noise spikes)
      4. Resample to n_bins
      5. Gaussian smooth for broad ridges
      6. Soft-threshold bottom 70 %
      7. Power-curve for wider structures
      8. Final polish smooth

    Also extracts per-exchange best bid/ask from the last snapshot.
    """
    price_grid = data["price_grid"]          # 1D, 400 bins
    price_grids = data["price_grids"]        # {exchange: (n_time, 400)}
    exchanges = data["exchanges"]
    price_min, price_max = data["price_range"]

    profiles = {}
    global_max = 0.0

    for ex in exchanges:
        matrix = price_grids[ex]             # (n_time, 400)
        raw = matrix.mean(axis=0)            # average across snapshots

        # Log-compress
        raw = np.log1p(raw)

        # Median filter
        raw = median_filter(raw, size=5)

        # Resample to target resolution
        x_orig = np.linspace(0, 1, len(raw))
        x_new = np.linspace(0, 1, n_bins)
        resampled = np.interp(x_new, x_orig, raw)

        # Gaussian smooth — broad ridges
        resampled = gaussian_filter(resampled, sigma=3.0)

        profiles[ex] = resampled
        global_max = max(global_max, resampled.max())

    # Normalize all to same global max, then threshold
    for ex in exchanges:
        p = profiles[ex]
        if global_max > 0:
            p = p / global_max

        # Soft-threshold: suppress bottom 70 %
        p70 = np.percentile(p, 70)
        p = np.clip((p - p70) / (1.0 - p70 + 1e-9), 0, 1)

        # Power-curve: widen dominant ridges
        p = np.power(p, 0.55)

        # Final polish
        p = gaussian_filter(p, sigma=1.5)

        # Re-normalize
        pmax = p.max()
        if pmax > 0:
            p = p / pmax

        profiles[ex] = [round(float(v), 4) for v in p]

    # Price labels for X axis (sample ~8 ticks)
    price_ticks = np.linspace(price_min, price_max, n_bins)
    n_labels = 8
    step = max(1, n_bins // n_labels)
    price_labels = [
        {"idx": int(i), "val": round(float(price_ticks[i]), 2)}
        for i in range(0, n_bins, step)
    ]

    # Best bid/ask per exchange from imbalance_series context
    # (extracted from raw data — last snapshot)
    bid_ask = {}
    for ex in exchanges:
        matrix = price_grids[ex]
        last = matrix[-1]
        mid_idx = len(last) // 2
        # bid side is left of mid, ask side is right
        bid_vol = last[:mid_idx].sum()
        ask_vol = last[mid_idx:].sum()
        # Find where volume is concentrated
        bid_peak = np.argmax(last[:mid_idx]) if bid_vol > 0 else mid_idx - 1
        ask_peak = mid_idx + np.argmax(last[mid_idx:]) if ask_vol > 0 else mid_idx
        bid_ask[ex] = {
            "bid_idx": round(float(bid_peak / len(last) * n_bins), 1),
            "ask_idx": round(float(ask_peak / len(last) * n_bins), 1),
            "bid_price": round(float(price_ticks[min(int(bid_peak / len(last) * n_bins), n_bins - 1)]), 2),
            "ask_price": round(float(price_ticks[min(int(ask_peak / len(last) * n_bins), n_bins - 1)]), 2),
        }

    # Volume scale — compute raw max for axis labels
    raw_maxes = {}
    for ex in exchanges:
        raw_maxes[ex] = round(float(np.exp(price_grids[ex].mean(axis=0).max()) - 1), 4)

    return {
        "exchanges": exchanges,
        "profiles": profiles,
        "n_bins": n_bins,
        "price_labels": price_labels,
        "price_range": [round(float(price_min), 2), round(float(price_max), 2)],
        "bid_ask": bid_ask,
        "raw_maxes": raw_maxes,
    }


def _build_metrics_summary(data: dict) -> dict:
    """Extract summary metrics for dashboard from last snapshot."""
    exchanges = data["exchanges"]
    imbalance = data.get("imbalance_series", {})

    summary = {}
    for ex in exchanges:
        imb_list = imbalance.get(ex, [0])
        summary[ex] = {
            "last_imbalance": round(float(imb_list[-1]), 4) if imb_list else 0,
            "avg_imbalance": round(float(np.mean(imb_list)), 4) if imb_list else 0,
        }
    return summary


# ---------------------------------------------------------------------------
# HTML generators
# ---------------------------------------------------------------------------

def generate_threejs(data: dict):
    """Generate all output files."""
    os.makedirs("output", exist_ok=True)

    payload = _build_exchange_profiles(data)
    metrics = _build_metrics_summary(data)
    payload_json = json.dumps(payload, separators=(",", ":"))
    metrics_json = json.dumps(metrics, separators=(",", ":"))

    # 1. Interactive 3D
    html_3d = _build_3d_html(payload_json)
    with open("output/3d_liquidity_pro.html", "w") as f:
        f.write(html_3d)
    print("  Saved output/3d_liquidity_pro.html")

    # 2. Dashboard
    html_dash = _build_dashboard_html(payload_json, metrics_json, data)
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
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #05070a; overflow: hidden; font-family: 'Consolas','Menlo',monospace; }}
  canvas {{ display: block; }}
  #tooltip {{
    position: fixed; pointer-events: none; display: none;
    background: rgba(8,12,20,0.92); border: 1px solid rgba(60,180,220,0.4);
    color: #c8dce8; padding: 8px 12px; border-radius: 4px;
    font-size: 12px; line-height: 1.5; z-index: 100;
    backdrop-filter: blur(6px);
  }}
  #tooltip .label {{ color: rgba(100,200,240,0.9); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }}
  #tooltip .val {{ color: #e0f0ff; font-size: 13px; }}
</style>
</head>
<body>
<div id="tooltip"></div>
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

/* ─────────────────── Data ─────────────────── */
const D = {payload_json};
const EXCHANGES = D.exchanges;
const N_BINS = D.n_bins;
const PROFILES = D.profiles;
const PRICE_LABELS = D.price_labels;
const BID_ASK = D.bid_ask;

/* ─────────────────── Layout constants ─────────────────── */
const SURFACE_W = 12;          // X span (price axis)
const STRIP_D = 2.5;           // Z depth per exchange strip
const GAP = 1.2;               // Z gap between strips
const HSCALE = 2.2;            // Y height multiplier
const BG = 0x05070a;
const TOTAL_D = EXCHANGES.length * STRIP_D + (EXCHANGES.length - 1) * GAP;

/* ─────────────────── Color ramp ─────────────────── */
const STOPS = [
  [0.00, 0.04, 0.02, 0.08],
  [0.15, 0.06, 0.06, 0.28],
  [0.30, 0.05, 0.15, 0.48],
  [0.48, 0.04, 0.32, 0.62],
  [0.62, 0.03, 0.50, 0.68],
  [0.76, 0.12, 0.68, 0.52],
  [0.88, 0.50, 0.82, 0.28],
  [0.96, 0.80, 0.90, 0.16],
  [1.00, 0.92, 0.94, 0.12],
];

function colorAt(t) {{
  t = Math.max(0, Math.min(1, t));
  let lo = STOPS[0], hi = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++) {{
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) {{ lo = STOPS[i]; hi = STOPS[i + 1]; break; }}
  }}
  const f = hi[0] > lo[0] ? (t - lo[0]) / (hi[0] - lo[0]) : 0;
  return new THREE.Color(lo[1]+(hi[1]-lo[1])*f, lo[2]+(hi[2]-lo[2])*f, lo[3]+(hi[3]-lo[3])*f);
}}

/* ─────────────────── WebGL Renderer ─────────────────── */
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

/* ─────────────────── CSS2D label renderer ─────────────────── */
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelRenderer.domElement);

/* ─────────────────── Scene ─────────────────── */
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.FogExp2(BG, 0.025);

/* ─────────────────── Camera ─────────────────── */
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 300);
camera.position.set(10, 7, 14);

/* ─────────────────── Controls ─────────────────── */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 4;
controls.maxDistance = 40;
controls.maxPolarAngle = Math.PI * 0.47;
controls.target.set(0, 0.8, TOTAL_D * 0.3);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.25;

/* ─────────────────── Helper: make CSS2D label ─────────────────── */
function makeLabel(text, fontSize, color, bold) {{
  const div = document.createElement('div');
  div.textContent = text;
  div.style.cssText = `color:${{color}};font-family:Consolas,Menlo,monospace;font-size:${{fontSize}}px;${{bold?'font-weight:600;':''}}white-space:nowrap;pointer-events:none;text-shadow:0 0 6px rgba(0,0,0,0.8);`;
  const obj = new CSS2DObject(div);
  return obj;
}}

/* ─────────────────── Axes ─────────────────── */
const axisColor = 0x2a3a55;
const axisMat = new THREE.LineBasicMaterial({{ color: axisColor, transparent: true, opacity: 0.7 }});
const tickMat = new THREE.LineBasicMaterial({{ color: 0x3a5070, transparent: true, opacity: 0.5 }});

// X axis (Price) — along bottom front
const xLen = SURFACE_W;
const xOrigin = new THREE.Vector3(-xLen/2, 0, -0.3);
const xAxisGeo = new THREE.BufferGeometry().setFromPoints([
  xOrigin, new THREE.Vector3(xLen/2, 0, -0.3)
]);
scene.add(new THREE.Line(xAxisGeo, axisMat));

// X axis title
const xTitle = makeLabel('Price (USDT)', 11, '#4a7a9a', true);
xTitle.position.set(0, -0.15, -1.0);
scene.add(xTitle);

// X tick marks + labels
PRICE_LABELS.forEach(pl => {{
  const xPos = -xLen/2 + (pl.idx / N_BINS) * xLen;
  const tickGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(xPos, 0, -0.3),
    new THREE.Vector3(xPos, -0.15, -0.3)
  ]);
  scene.add(new THREE.Line(tickGeo, tickMat));

  const lbl = makeLabel(pl.val.toLocaleString(), 9, '#5588aa', false);
  lbl.position.set(xPos, -0.35, -0.3);
  scene.add(lbl);
}});

// Y axis (Volume) — vertical at left
const yLen = HSCALE * 1.1;
const yAxisGeo = new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-xLen/2 - 0.3, 0, -0.3),
  new THREE.Vector3(-xLen/2 - 0.3, yLen, -0.3)
]);
scene.add(new THREE.Line(yAxisGeo, axisMat));

// Y axis title
const yTitle = makeLabel('Volume', 11, '#4a7a9a', true);
yTitle.position.set(-xLen/2 - 0.8, yLen * 0.55, -0.3);
scene.add(yTitle);

// Y tick marks (5 levels)
for (let i = 0; i <= 4; i++) {{
  const yPos = (i / 4) * yLen;
  const tickGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-xLen/2 - 0.3, yPos, -0.3),
    new THREE.Vector3(-xLen/2 - 0.5, yPos, -0.3)
  ]);
  scene.add(new THREE.Line(tickGeo, tickMat));

  const pct = Math.round((i / 4) * 100);
  const lbl = makeLabel(pct + '%', 9, '#5588aa', false);
  lbl.position.set(-xLen/2 - 0.75, yPos, -0.3);
  scene.add(lbl);
}}

// Z axis (Exchange) — along left side going back
const zAxisGeo = new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-xLen/2 - 0.3, 0, -0.3),
  new THREE.Vector3(-xLen/2 - 0.3, 0, TOTAL_D + 0.3)
]);
scene.add(new THREE.Line(zAxisGeo, axisMat));

// Z axis title
const zTitle = makeLabel('Exchange', 11, '#4a7a9a', true);
zTitle.position.set(-xLen/2 - 0.8, -0.15, TOTAL_D * 0.5);
scene.add(zTitle);

/* ─────────────────── Exchange colors for labels ─────────────────── */
const EX_COLORS = {{ 'Binance': '#f0b90b', 'Coinbase': '#0052ff', 'Kraken': '#7b3fe4' }};

/* ─────────────────── Build per-exchange surfaces ─────────────────── */
const surfaces = [];
const STRIP_ROWS = 20;   // subdivisions in Z per strip

EXCHANGES.forEach((ex, exIdx) => {{
  const profile = PROFILES[ex];
  const zStart = exIdx * (STRIP_D + GAP);
  const zEnd = zStart + STRIP_D;

  // PlaneGeometry: width segments = N_BINS-1, height segments = STRIP_ROWS-1
  const geo = new THREE.PlaneGeometry(SURFACE_W, STRIP_D, N_BINS - 1, STRIP_ROWS - 1);
  geo.rotateX(-Math.PI / 2);

  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);

  for (let i = 0; i < pos.count; i++) {{
    const col = i % N_BINS;
    const row = Math.floor(i / N_BINS);
    const h = profile[col];

    // Taper edges of strip for smooth falloff at exchange boundaries
    const rowT = row / (STRIP_ROWS - 1);
    const edgeFade = Math.sin(rowT * Math.PI);  // 0→1→0
    const finalH = h * edgeFade;

    pos.setY(i, finalH * HSCALE);
    pos.setZ(i, pos.getZ(i) + zStart + STRIP_D / 2);

    const c = colorAt(finalH);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }}

  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({{
    vertexColors: true,
    roughness: 0.45,
    metalness: 0.08,
    flatShading: false,
    side: THREE.DoubleSide,
  }});

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = {{ exchange: ex, profile: profile }};
  scene.add(mesh);
  surfaces.push(mesh);

  // Exchange label on Z axis
  const exColor = EX_COLORS[ex] || '#88aacc';
  const exLabel = makeLabel(ex, 12, exColor, true);
  exLabel.position.set(-xLen/2 - 0.6, 0.1, zStart + STRIP_D / 2);
  scene.add(exLabel);

  // Z tick mark
  const zTickGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-xLen/2 - 0.3, 0, zStart + STRIP_D / 2),
    new THREE.Vector3(-xLen/2 - 0.5, 0, zStart + STRIP_D / 2)
  ]);
  scene.add(new THREE.Line(zTickGeo, tickMat));
}});

/* ─────────────────── Bid/Ask indicator lines ─────────────────── */
const bidAskGroup = new THREE.Group();
scene.add(bidAskGroup);

EXCHANGES.forEach((ex, exIdx) => {{
  const ba = BID_ASK[ex];
  if (!ba) return;
  const zStart = exIdx * (STRIP_D + GAP);
  const zMid = zStart + STRIP_D / 2;

  // Bid line (green)
  const bidX = -SURFACE_W/2 + (ba.bid_idx / N_BINS) * SURFACE_W;
  const bidGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(bidX, 0, zMid - STRIP_D/2),
    new THREE.Vector3(bidX, HSCALE * 0.3, zMid),
    new THREE.Vector3(bidX, 0, zMid + STRIP_D/2),
  ]);
  const bidLine = new THREE.Line(bidGeo, new THREE.LineBasicMaterial({{
    color: 0x00cc66, transparent: true, opacity: 0.6
  }}));
  bidAskGroup.add(bidLine);

  // Ask line (red)
  const askX = -SURFACE_W/2 + (ba.ask_idx / N_BINS) * SURFACE_W;
  const askGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(askX, 0, zMid - STRIP_D/2),
    new THREE.Vector3(askX, HSCALE * 0.3, zMid),
    new THREE.Vector3(askX, 0, zMid + STRIP_D/2),
  ]);
  const askLine = new THREE.Line(askGeo, new THREE.LineBasicMaterial({{
    color: 0xff4444, transparent: true, opacity: 0.6
  }}));
  bidAskGroup.add(askLine);
}});

/* ─────────────────── Grid floor ─────────────────── */
const gridHelper = new THREE.GridHelper(20, 30, 0x141e30, 0x0c1420);
gridHelper.position.set(0, -0.02, TOTAL_D * 0.4);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.2;
scene.add(gridHelper);

/* ─────────────────── Subtle floor plane ─────────────────── */
const floorGeo = new THREE.PlaneGeometry(SURFACE_W * 1.6, TOTAL_D * 1.6);
floorGeo.rotateX(-Math.PI / 2);
const floorMat = new THREE.MeshStandardMaterial({{
  color: BG, roughness: 0.6, metalness: 0.3,
  transparent: true, opacity: 0.25,
}});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.position.set(0, -0.04, TOTAL_D * 0.4);
floor.receiveShadow = true;
scene.add(floor);

/* ─────────────────── Lighting ─────────────────── */
// Ambient — low, neutral
const amb = new THREE.AmbientLight(0x1a1a28, 0.6);
scene.add(amb);

// Key — top-left, white-blue, casts shadows
const key = new THREE.DirectionalLight(0xe8eeff, 1.3);
key.position.set(8, 14, 4);
key.castShadow = true;
key.shadow.mapSize.width = 1024;
key.shadow.mapSize.height = 1024;
key.shadow.camera.near = 1;
key.shadow.camera.far = 35;
key.shadow.camera.left = -12;
key.shadow.camera.right = 12;
key.shadow.camera.top = 12;
key.shadow.camera.bottom = -12;
key.shadow.bias = -0.002;
scene.add(key);

// Fill — opposite side, cool
const fill = new THREE.DirectionalLight(0x3355aa, 0.4);
fill.position.set(-7, 6, -5);
scene.add(fill);

// Rim — subtle backlight
const rim = new THREE.DirectionalLight(0x554488, 0.25);
rim.position.set(0, 4, -10);
scene.add(rim);

/* ─────────────────── Raycaster tooltip ─────────────────── */
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', (e) => {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(surfaces);

  if (hits.length > 0) {{
    const hit = hits[0];
    const ex = hit.object.userData.exchange;
    const pt = hit.point;

    // Map X back to price
    const priceT = (pt.x + SURFACE_W / 2) / SURFACE_W;
    const price = D.price_range[0] + priceT * (D.price_range[1] - D.price_range[0]);

    // Map Y back to volume %
    const volPct = Math.max(0, (pt.y / HSCALE) * 100);

    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 16) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.innerHTML = `<div class="label">Exchange</div><div class="val">${{ex}}</div>`
      + `<div class="label" style="margin-top:4px">Price</div><div class="val">${{price.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}})}} USDT</div>`
      + `<div class="label" style="margin-top:4px">Relative Volume</div><div class="val">${{volPct.toFixed(1)}}%</div>`;
  }} else {{
    tooltip.style.display = 'none';
  }}
}});

renderer.domElement.addEventListener('mouseleave', () => {{
  tooltip.style.display = 'none';
}});

/* ─────────────────── Render loop ─────────────────── */
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}}
animate();

/* ─────────────────── Resize ─────────────────── */
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


def _build_dashboard_html(payload_json: str, metrics_json: str, data: dict) -> str:
    exchanges = data["exchanges"]
    imbalance = data.get("imbalance_series", {})
    n_samples = len(data.get("timestamps", []))
    price_min, price_max = data.get("price_range", (0, 0))

    # Build per-exchange stat rows
    stat_rows = ""
    for ex in exchanges:
        imb = imbalance.get(ex, [0])
        last_imb = imb[-1] if imb else 0
        avg_imb = float(np.mean(imb)) if imb else 0
        imb_color = "#00cc66" if last_imb > 0 else "#ff4455"
        stat_rows += f"""
        <tr>
          <td style="color:{{'#f0b90b' if ex=='Binance' else '#0052ff' if ex=='Coinbase' else '#7b3fe4'}};font-weight:600">{ex}</td>
          <td style="color:{imb_color}">{last_imb:+.4f}</td>
          <td>{avg_imb:+.4f}</td>
        </tr>"""

    # Imbalance sparkline data per exchange
    spark_data = {}
    for ex in exchanges:
        spark_data[ex] = [round(float(v), 4) for v in imbalance.get(ex, [0])]
    spark_json = json.dumps(spark_data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Liquidity Dashboard — BTC/USDT</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#05070a; color:#b0c4d4; font-family:'Consolas','Menlo',monospace; }}
  .header {{
    padding: 20px 30px; border-bottom: 1px solid #141e30;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .header h1 {{ color: #d0e0f0; font-size: 16px; font-weight: 600; letter-spacing: 0.5px; }}
  .header .meta {{ font-size: 11px; color: #5a7a90; }}
  .panels {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr auto; height: calc(100vh - 60px); }}
  .panel {{
    border: 1px solid #141e30; position: relative; overflow: hidden;
  }}
  .panel-title {{
    position: absolute; top: 10px; left: 14px; font-size: 10px;
    color: #4a7090; text-transform: uppercase; letter-spacing: 1px; z-index: 10;
  }}
  .panel-3d {{ grid-column: 1 / -1; min-height: 55vh; }}
  .panel-3d iframe {{
    width: 100%; height: 100%; border: none; display: block;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; color: #4a7090; font-weight: 400; padding: 8px 12px;
       text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;
       border-bottom: 1px solid #141e30; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #0c1420; }}
  .sparkline-panel {{ padding: 16px; }}
  .spark-row {{ display: flex; align-items: center; margin-bottom: 10px; gap: 12px; }}
  .spark-label {{ width: 80px; font-size: 11px; font-weight: 600; }}
  .spark-canvas {{ flex: 1; height: 28px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Cross-Exchange Liquidity Dashboard — BTC/USDT</h1>
  <div class="meta">{n_samples} snapshots &middot; {round(price_min, 2)} — {round(price_max, 2)} USDT</div>
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
  <div class="panel sparkline-panel">
    <div class="panel-title">Imbalance Over Time</div>
    <div id="sparklines" style="margin-top:28px;"></div>
  </div>
</div>
<script>
const sparkData = {spark_json};
const exColors = {{ Binance: '#f0b90b', Coinbase: '#0052ff', Kraken: '#7b3fe4' }};
const container = document.getElementById('sparklines');

Object.entries(sparkData).forEach(([ex, vals]) => {{
  const row = document.createElement('div');
  row.className = 'spark-row';

  const label = document.createElement('div');
  label.className = 'spark-label';
  label.style.color = exColors[ex] || '#88aacc';
  label.textContent = ex;

  const canvas = document.createElement('canvas');
  canvas.className = 'spark-canvas';
  canvas.height = 28;

  row.appendChild(label);
  row.appendChild(canvas);
  container.appendChild(row);

  // Draw sparkline
  requestAnimationFrame(() => {{
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 56;
    canvas.style.height = '28px';
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    if (vals.length < 2) return;

    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;
    const pad = 4;

    // Zero line
    const zeroY = h - pad - ((0 - mn) / range) * (h - pad * 2);
    ctx.strokeStyle = 'rgba(60,80,100,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Sparkline
    ctx.strokeStyle = exColors[ex] || '#88aacc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    vals.forEach((v, i) => {{
      const x = (i / (vals.length - 1)) * w;
      const y = h - pad - ((v - mn) / range) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
}});
</script>
</body>
</html>"""
