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
    Build high-resolution depth profile per exchange with analytical metadata.

    Pipeline per exchange:
      1. Average volume across time snapshots
      2. Log-compress
      3. Median filter (denoise)
      4. Resample to n_bins
      5. Gaussian smooth (preserve structure)
      6. Soft-threshold bottom 55%
      7. Power-curve for broader ridges
      8. Final polish

    Also computes: walls, contours, bid/ask, mid-price, slope, imbalance per bin.
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

        p60 = np.percentile(p, 60)
        p = np.clip((p - p60) / (1.0 - p60 + 1e-9), 0, 1)

        p = np.power(p, 0.5)
        p = gaussian_filter(p, sigma=1.8)

        pmax = p.max()
        if pmax > 0:
            p = p / pmax

        profiles[ex] = p

    # ── Detect liquidity walls (peaks) ──
    walls = {}
    for ex in exchanges:
        p = profiles[ex]
        peak_idxs, props = find_peaks(p, height=0.35, distance=n_bins // 20, prominence=0.15)
        wall_list = []
        for idx in peak_idxs:
            wall_list.append({
                "idx": int(idx),
                "height": round(float(p[idx]), 3),
                "price": round(float(price_ticks[idx]), 2),
            })
        wall_list.sort(key=lambda w: w["height"], reverse=True)
        walls[ex] = wall_list[:6]

    # ── Floor contour paths (analytical) ──
    contour_levels = [0.15, 0.30, 0.50, 0.70, 0.90]
    contours = {}
    for ex in exchanges:
        p = profiles[ex]
        ex_contours = []
        for level in contour_levels:
            segments = []
            in_region = False
            start_idx = 0
            for i in range(n_bins):
                if p[i] >= level and not in_region:
                    in_region = True
                    start_idx = i
                elif p[i] < level and in_region:
                    in_region = False
                    segments.append([start_idx, i - 1])
            if in_region:
                segments.append([start_idx, n_bins - 1])
            ex_contours.append({"level": round(level, 2), "segments": segments})
        contours[ex] = ex_contours

    # ── Slope (absolute gradient) for color modulation ──
    slopes = {}
    for ex in exchanges:
        p = profiles[ex]
        grad = np.abs(np.gradient(p))
        grad_max = grad.max()
        if grad_max > 0:
            grad = grad / grad_max
        slopes[ex] = [round(float(v), 4) for v in grad]

    # ── Mid-price index ──
    mid_price = (price_min + price_max) / 2.0
    mid_idx = int(n_bins / 2)

    # ── Bid/Ask per exchange ──
    bid_ask = {}
    for ex in exchanges:
        matrix = price_grids[ex]
        last = matrix[-1]
        mid_raw = len(last) // 2
        bid_peak = np.argmax(last[:mid_raw]) if last[:mid_raw].sum() > 0 else mid_raw - 1
        ask_peak = mid_raw + np.argmax(last[mid_raw:]) if last[mid_raw:].sum() > 0 else mid_raw
        bid_ask[ex] = {
            "bid_idx": round(float(bid_peak / len(last) * n_bins), 1),
            "ask_idx": round(float(ask_peak / len(last) * n_bins), 1),
            "bid_price": round(float(price_ticks[min(int(bid_peak / len(last) * n_bins), n_bins - 1)]), 2),
            "ask_price": round(float(price_ticks[min(int(ask_peak / len(last) * n_bins), n_bins - 1)]), 2),
        }

    # ── Per-exchange avg imbalance ──
    avg_imbalances = {}
    for ex in exchanges:
        imb = imbalance_series.get(ex, [0])
        avg_imbalances[ex] = round(float(np.mean(imb)), 4)

    # ── Price labels (~10 ticks) ──
    n_labels = 10
    step = max(1, n_bins // n_labels)
    price_labels = [
        {"idx": int(i), "val": round(float(price_ticks[i]), 2)}
        for i in range(0, n_bins, step)
    ]

    # ── Raw max volume for axis labels ──
    raw_maxes = {}
    for ex in exchanges:
        raw_maxes[ex] = round(float(np.exp(price_grids[ex].mean(axis=0).max()) - 1), 4)

    # Serialize profiles
    profiles_out = {}
    for ex in exchanges:
        profiles_out[ex] = [round(float(v), 4) for v in profiles[ex]]

    return {
        "exchanges": exchanges,
        "profiles": profiles_out,
        "slopes": slopes,
        "n_bins": n_bins,
        "price_labels": price_labels,
        "price_range": [round(float(price_min), 2), round(float(price_max), 2)],
        "mid_price": round(float(mid_price), 2),
        "mid_idx": mid_idx,
        "bid_ask": bid_ask,
        "walls": walls,
        "contours": contours,
        "contour_levels": contour_levels,
        "avg_imbalances": avg_imbalances,
        "raw_maxes": raw_maxes,
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
    n_samples = len(data.get("timestamps", []))

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
  body {{ background:#000; overflow:hidden; font-family:'Consolas','Menlo',monospace; }}
  canvas {{ display:block; }}
  /* Vertical background gradient via overlay */
  #bg-grad {{
    position:fixed; top:0; left:0; width:100%; height:100%; z-index:-1;
    background:linear-gradient(to bottom, #02040a 0%, #000000 100%);
  }}

  #tooltip {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(6,10,18,0.94); border:1px solid rgba(50,160,210,0.35);
    color:#c0d4e4; padding:10px 14px; border-radius:4px;
    font-size:11px; line-height:1.6; z-index:100;
    backdrop-filter:blur(8px); min-width:160px;
  }}
  #tooltip .t-label {{ color:rgba(80,180,230,0.85); font-size:9px; text-transform:uppercase; letter-spacing:0.6px; }}
  #tooltip .t-val {{ color:#e0f0ff; font-size:12px; font-weight:500; }}
  #tooltip .t-row {{ margin-top:3px; }}
  #tooltip .t-wall {{ color:#f0c040; font-size:9px; margin-top:4px; }}

  #legend {{
    position:fixed; bottom:16px; right:16px; z-index:90;
    background:rgba(6,10,18,0.88); border:1px solid rgba(40,60,90,0.4);
    padding:12px 16px; border-radius:6px; font-size:10px;
    backdrop-filter:blur(8px); color:#7a9ab0; min-width:140px;
  }}
  #legend .l-title {{ font-size:11px; color:#a0c0d8; font-weight:600; margin-bottom:8px; }}
  #legend .l-row {{ display:flex; align-items:center; gap:8px; margin:3px 0; }}
  #legend .l-swatch {{ width:14px; height:3px; border-radius:1px; }}
  .l-section {{ margin-top:8px; padding-top:6px; border-top:1px solid rgba(40,60,90,0.3); }}
  .l-section-title {{ font-size:9px; color:#5a7a90; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }}
  #legend .l-axis {{ display:flex; align-items:center; gap:6px; margin:2px 0; font-size:9px; }}
</style>
</head>
<body>
<div id="bg-grad"></div>
<div id="tooltip"></div>
<div id="legend">
  <div class="l-title">Volume Scale</div>
  <div class="l-row"><div class="l-swatch" style="background:#0a0515"></div>Low</div>
  <div class="l-row"><div class="l-swatch" style="background:#0c3860"></div>Medium</div>
  <div class="l-row"><div class="l-swatch" style="background:#0a8cad"></div>High</div>
  <div class="l-row"><div class="l-swatch" style="background:#d0e830"></div>Extreme</div>
  <div class="l-section">
    <div class="l-section-title">Overlays</div>
    <div class="l-row"><div class="l-swatch" style="background:#00cc66"></div>Bid peak</div>
    <div class="l-row"><div class="l-swatch" style="background:#ff4444"></div>Ask peak</div>
    <div class="l-row"><div class="l-swatch" style="background:rgba(100,200,255,0.4)"></div>Mid-price</div>
    <div class="l-row"><div class="l-swatch" style="background:#f0c040"></div>Liquidity wall</div>
  </div>
  <div class="l-section">
    <div class="l-section-title">Axes</div>
    <div class="l-axis">X &mdash; Price (USDT)</div>
    <div class="l-axis">Y &mdash; Volume (relative)</div>
    <div class="l-axis">Z &mdash; Exchange</div>
  </div>
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
import {{ EffectComposer }} from 'three/addons/postprocessing/EffectComposer.js';
import {{ RenderPass }} from 'three/addons/postprocessing/RenderPass.js';
import {{ UnrealBloomPass }} from 'three/addons/postprocessing/UnrealBloomPass.js';

/* ═══════════════════ DATA ═══════════════════ */
const D = {payload_json};
const EX = D.exchanges;
const NB = D.n_bins;
const PROF = D.profiles;
const SLOPES = D.slopes;
const WALLS = D.walls;
const CONTOURS = D.contours;
const C_LEVELS = D.contour_levels;
const BID_ASK = D.bid_ask;

/* ═══════════════════ LAYOUT ═══════════════════ */
const SW = 14;            // surface width (X)
const SD = 3.0;           // strip depth (Z) per exchange
const GAP = 1.5;          // gap between strips
const HY = 3.4;           // height scale (Y) — exaggerated for drama
const BG = 0x020408;
const TD = EX.length * SD + (EX.length - 1) * GAP;
const SROWS = 26;         // Z subdivisions per strip

/* ═══════════════════ COLOR RAMP ═══════════════════ */
const CS = [
  [0.00, 0.03, 0.01, 0.06],
  [0.10, 0.05, 0.04, 0.22],
  [0.22, 0.04, 0.10, 0.40],
  [0.36, 0.03, 0.25, 0.56],
  [0.50, 0.03, 0.42, 0.65],
  [0.64, 0.05, 0.58, 0.62],
  [0.76, 0.18, 0.72, 0.48],
  [0.86, 0.48, 0.83, 0.28],
  [0.94, 0.75, 0.90, 0.16],
  [1.00, 0.88, 0.95, 0.10],
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
renderer.toneMappingExposure = 0.95;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

/* CSS2D for labels */
const labelR = new CSS2DRenderer();
labelR.setSize(window.innerWidth, window.innerHeight);
labelR.domElement.style.position = 'absolute';
labelR.domElement.style.top = '0';
labelR.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelR.domElement);

/* ═══════════════════ SCENE ═══════════════════ */
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.FogExp2(BG, 0.032);

/* ═══════════════════ CAMERA ═══════════════════ */
const camera = new THREE.PerspectiveCamera(34, window.innerWidth/window.innerHeight, 0.1, 400);
camera.position.set(12, 5.5, TD + 10);

/* ═══════════════════ CONTROLS ═══════════════════ */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 5;
controls.maxDistance = 50;
controls.maxPolarAngle = Math.PI * 0.46;
controls.target.set(0, 1.0, TD * 0.4);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.15;

/* ═══════════════════ HELPERS ═══════════════════ */
function mkLabel(text, size, color, bold) {{
  const d = document.createElement('div');
  d.textContent = text;
  d.style.cssText = `color:${{color}};font-family:Consolas,Menlo,monospace;font-size:${{size}}px;${{bold?'font-weight:600;':''}}white-space:nowrap;pointer-events:none;text-shadow:0 0 8px rgba(0,0,0,0.9);`;
  return new CSS2DObject(d);
}}

function xWorld(binIdx) {{ return -SW/2 + (binIdx/NB) * SW; }}

const axisMat = new THREE.LineBasicMaterial({{ color:0x1a2838, transparent:true, opacity:0.45 }});
const tickMat = new THREE.LineBasicMaterial({{ color:0x1e3048, transparent:true, opacity:0.3 }});
const guideMat = new THREE.LineBasicMaterial({{ color:0x101820, transparent:true, opacity:0.08 }});

function addLine(pts, mat) {{
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(g, mat));
}}

/* ═══════════════════ 3D AXES ═══════════════════ */
// X axis (Price)
addLine([new THREE.Vector3(-SW/2, 0, -0.4), new THREE.Vector3(SW/2, 0, -0.4)], axisMat);
const xT = mkLabel('Price (USDT)', 11, '#2a4a60', true);
xT.position.set(0, -0.2, -1.2);
scene.add(xT);

// X ticks
D.price_labels.forEach(pl => {{
  const x = xWorld(pl.idx);
  addLine([new THREE.Vector3(x, 0, -0.4), new THREE.Vector3(x, -0.12, -0.4)], tickMat);
  // Vertical guide line from tick up
  addLine([new THREE.Vector3(x, 0, -0.4), new THREE.Vector3(x, 0, TD+0.3)], guideMat);
  const lb = mkLabel(pl.val.toLocaleString(), 8, '#283848', false);
  lb.position.set(x, -0.3, -0.4);
  scene.add(lb);
}});

// Y axis (Volume)
const yH = HY * 1.15;
addLine([new THREE.Vector3(-SW/2-0.4, 0, -0.4), new THREE.Vector3(-SW/2-0.4, yH, -0.4)], axisMat);
const yT = mkLabel('Volume', 11, '#2a4a60', true);
yT.position.set(-SW/2-1.0, yH*0.55, -0.4);
scene.add(yT);

for (let i = 0; i <= 5; i++) {{
  const y = (i/5)*yH;
  addLine([new THREE.Vector3(-SW/2-0.4, y, -0.4), new THREE.Vector3(-SW/2-0.6, y, -0.4)], tickMat);
  // Horizontal guide line across floor
  addLine([new THREE.Vector3(-SW/2, y, -0.4), new THREE.Vector3(SW/2, y, -0.4)], guideMat);
  const lb = mkLabel(Math.round((i/5)*100)+'%', 8, '#283848', false);
  lb.position.set(-SW/2-0.85, y, -0.4);
  scene.add(lb);
}}

// Z axis (Exchange)
addLine([new THREE.Vector3(-SW/2-0.4, 0, -0.4), new THREE.Vector3(-SW/2-0.4, 0, TD+0.4)], axisMat);
const zT = mkLabel('Exchange', 11, '#2a4a60', true);
zT.position.set(-SW/2-1.0, -0.2, TD*0.5);
scene.add(zT);

/* ═══════════════════ BACK WALL GRID ═══════════════════ */
const backWallMat = new THREE.LineBasicMaterial({{ color:0x0c1420, transparent:true, opacity:0.06 }});
// Vertical lines on back wall
for (let i = 0; i <= 10; i++) {{
  const x = -SW/2 + (i/10)*SW;
  addLine([new THREE.Vector3(x, 0, -0.4), new THREE.Vector3(x, yH, -0.4)], backWallMat);
}}
// Horizontal lines on back wall
for (let i = 0; i <= 5; i++) {{
  const y = (i/5)*yH;
  addLine([new THREE.Vector3(-SW/2, y, -0.4), new THREE.Vector3(SW/2, y, -0.4)], backWallMat);
}}
// Side wall grid (left)
for (let i = 0; i <= 5; i++) {{
  const y = (i/5)*yH;
  addLine([new THREE.Vector3(-SW/2-0.4, y, -0.4), new THREE.Vector3(-SW/2-0.4, y, TD+0.4)], backWallMat);
}}

/* ═══════════════════ EXCHANGE COLORS ═══════════════════ */
const EXC = {{ Binance:'#f0b90b', Coinbase:'#0052ff', Kraken:'#7b3fe4' }};

/* ═══════════════════ PER-EXCHANGE SURFACES ═══════════════════ */
const meshes = [];

EX.forEach((ex, ei) => {{
  const prof = PROF[ex];
  const slope = SLOPES[ex];
  const zOff = ei * (SD + GAP);

  // ── Solid surface ──
  const geo = new THREE.PlaneGeometry(SW, SD, NB-1, SROWS-1);
  geo.rotateX(-Math.PI/2);
  const pos = geo.attributes.position;
  const cols = new Float32Array(pos.count*3);

  // Find peak threshold for this exchange (top ~20%)
  const sorted = [...prof].sort((a,b) => b-a);
  const peakThresh = sorted[Math.floor(sorted.length * 0.12)] || 0.5;

  for (let i = 0; i < pos.count; i++) {{
    const c = i % NB;
    const r = Math.floor(i / NB);
    const h = prof[c];
    const rT = r / (SROWS-1);
    const fade = Math.sin(rT * Math.PI);
    const fH = h * fade;

    pos.setY(i, fH * HY);
    pos.setZ(i, pos.getZ(i) + zOff + SD/2);

    // Visual hierarchy: dim non-peak areas, brighten peaks
    const isPeak = h >= peakThresh;
    const slopeBoost = 1.0 + (slope[c] || 0) * 0.4;
    const hierarchyDim = isPeak ? Math.min(slopeBoost, 1.4) : 0.4 + fH * 0.5;
    const col = colorAt(fH);
    col.multiplyScalar(hierarchyDim);
    cols[i*3] = col.r; cols[i*3+1] = col.g; cols[i*3+2] = col.b;
  }}

  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({{
    vertexColors:true, roughness:0.38, metalness:0.10,
    flatShading:false, side:THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = {{ exchange:ex, profile:prof }};
  scene.add(mesh);
  meshes.push(mesh);

  // ── Wireframe overlay ──
  const wfGeo = geo.clone();
  const wfMat = new THREE.MeshBasicMaterial({{
    wireframe:true, color:0x2a4060, transparent:true, opacity:0.06,
  }});
  scene.add(new THREE.Mesh(wfGeo, wfMat));

  // ── Exchange label ──
  const exLbl = mkLabel(ex, 13, EXC[ex]||'#88aacc', true);
  exLbl.position.set(-SW/2-0.7, 0.12, zOff+SD/2);
  scene.add(exLbl);
  addLine([new THREE.Vector3(-SW/2-0.4, 0, zOff+SD/2), new THREE.Vector3(-SW/2-0.55, 0, zOff+SD/2)], tickMat);
}});

/* ═══════════════════ FLOOR CONTOUR LINES ═══════════════════ */
const contourColors = [0x0c1828, 0x122848, 0x1a4068, 0x2a6888, 0x40a0b0];

EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP);
  const exContours = CONTOURS[ex];

  exContours.forEach((cObj, li) => {{
    const color = contourColors[li] || 0x2a6888;
    const cMat = new THREE.LineBasicMaterial({{ color, transparent:true, opacity:0.35 + li*0.08 }});

    cObj.segments.forEach(seg => {{
      const x0 = xWorld(seg[0]);
      const x1 = xWorld(seg[1]);
      const zMid = zOff + SD/2;
      // Compute z-extent at this contour level using sin taper inversion
      const level = cObj.level;
      // For the center of the region, find z width
      const zHalf = SD/2 * 0.85;  // scale down slightly
      const pts = [
        new THREE.Vector3(x0, 0.01, zMid - zHalf * (1-level*0.3)),
        new THREE.Vector3(x0, 0.01, zMid + zHalf * (1-level*0.3)),
        new THREE.Vector3(x1, 0.01, zMid + zHalf * (1-level*0.3)),
        new THREE.Vector3(x1, 0.01, zMid - zHalf * (1-level*0.3)),
        new THREE.Vector3(x0, 0.01, zMid - zHalf * (1-level*0.3)),
      ];
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      scene.add(new THREE.Line(g, cMat));
    }});
  }});
}});

/* ═══════════════════ LIQUIDITY WALL MARKERS + DROP LINES ═══════════════════ */
const wallMarkerMat = new THREE.MeshStandardMaterial({{
  color:0xf0c040, emissive:0xf0c040, emissiveIntensity:1.2,
  roughness:0.2, metalness:0.05, transparent:true, opacity:0.9,
}});
const dropMat = new THREE.LineBasicMaterial({{ color:0xf0c040, transparent:true, opacity:0.2 }});
const wallLabelData = [];

EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP);
  const zMid = zOff + SD/2;

  (WALLS[ex] || []).forEach(w => {{
    const x = xWorld(w.idx);
    const y = w.height * HY;

    // Marker sphere
    const sGeo = new THREE.IcosahedronGeometry(0.09, 2);
    const sMesh = new THREE.Mesh(sGeo, wallMarkerMat);
    sMesh.position.set(x, y, zMid);
    scene.add(sMesh);

    // Vertical drop line to floor
    addLine([new THREE.Vector3(x, y, zMid), new THREE.Vector3(x, 0, zMid)], dropMat);

    // Small floor cross
    addLine([new THREE.Vector3(x-0.1, 0.01, zMid), new THREE.Vector3(x+0.1, 0.01, zMid)], dropMat);
    addLine([new THREE.Vector3(x, 0.01, zMid-0.1), new THREE.Vector3(x, 0.01, zMid+0.1)], dropMat);

    // CSS2D wall label
    const lb = mkLabel('$' + w.price.toLocaleString(), 8, '#d4a830', true);
    lb.position.set(x, y + 0.18, zMid);
    scene.add(lb);
  }});
}});

/* ═══════════════════ BID/ASK LINES ═══════════════════ */
const bidMat = new THREE.LineBasicMaterial({{ color:0x00cc66, transparent:true, opacity:0.45 }});
const askMat = new THREE.LineBasicMaterial({{ color:0xff4444, transparent:true, opacity:0.45 }});

EX.forEach((ex, ei) => {{
  const ba = BID_ASK[ex];
  if (!ba) return;
  const zOff = ei * (SD + GAP);

  const bidX = xWorld(ba.bid_idx);
  addLine([
    new THREE.Vector3(bidX, 0, zOff),
    new THREE.Vector3(bidX, HY*0.25, zOff+SD/2),
    new THREE.Vector3(bidX, 0, zOff+SD),
  ], bidMat);

  const askX = xWorld(ba.ask_idx);
  addLine([
    new THREE.Vector3(askX, 0, zOff),
    new THREE.Vector3(askX, HY*0.25, zOff+SD/2),
    new THREE.Vector3(askX, 0, zOff+SD),
  ], askMat);
}});

/* ═══════════════════ MID-PRICE REFERENCE PLANE ═══════════════════ */
const mpX = xWorld(D.mid_idx);
const mpGeo = new THREE.PlaneGeometry(0.01, yH);
mpGeo.rotateY(0);
// Build a thin vertical strip spanning the full Z range
const mpVerts = new Float32Array([
  mpX, 0, -0.3,  mpX, yH, -0.3,  mpX, yH, TD+0.3,
  mpX, 0, -0.3,  mpX, yH, TD+0.3, mpX, 0, TD+0.3,
]);
const mpBufGeo = new THREE.BufferGeometry();
mpBufGeo.setAttribute('position', new THREE.BufferAttribute(mpVerts, 3));
const mpMat = new THREE.MeshBasicMaterial({{
  color:0x60c8ff, transparent:true, opacity:0.06, side:THREE.DoubleSide,
  depthWrite:false,
}});
scene.add(new THREE.Mesh(mpBufGeo, mpMat));

// Mid-price line on floor
addLine([
  new THREE.Vector3(mpX, 0.01, -0.4),
  new THREE.Vector3(mpX, 0.01, TD+0.4),
], new THREE.LineBasicMaterial({{ color:0x60c8ff, transparent:true, opacity:0.2 }}));

const mpLbl = mkLabel('Mid $' + D.mid_price.toLocaleString(), 9, 'rgba(100,200,255,0.7)', true);
mpLbl.position.set(mpX, -0.18, -0.4);
scene.add(mpLbl);

/* ═══════════════════ FLOOR ═══════════════════ */
const floorGrid = new THREE.GridHelper(24, 36, 0x0a1420, 0x060c14);
floorGrid.position.set(0, -0.01, TD*0.4);
floorGrid.material.transparent = true;
floorGrid.material.opacity = 0.12;
scene.add(floorGrid);

const floorGeo = new THREE.PlaneGeometry(SW*1.8, TD*1.8);
floorGeo.rotateX(-Math.PI/2);
const floorMat = new THREE.MeshStandardMaterial({{
  color:BG, roughness:0.6, metalness:0.25, transparent:true, opacity:0.2,
}});
const floorMesh = new THREE.Mesh(floorGeo, floorMat);
floorMesh.position.set(0, -0.03, TD*0.4);
floorMesh.receiveShadow = true;
scene.add(floorMesh);

/* ═══════════════════ LIGHTING (cinematic) ═══════════════════ */
// Very low ambient — most of scene in shadow
scene.add(new THREE.AmbientLight(0x0c0c20, 0.4));

// Spotlight on center of surface — creates dramatic falloff
const spot = new THREE.SpotLight(0xd0e0ff, 2.5, 45, Math.PI*0.28, 0.4, 1.2);
spot.position.set(4, 18, TD*0.4);
spot.target.position.set(0, 0, TD*0.4);
spot.castShadow = true;
spot.shadow.mapSize.width = 1024;
spot.shadow.mapSize.height = 1024;
spot.shadow.camera.near = 2; spot.shadow.camera.far = 40;
spot.shadow.bias = -0.002;
scene.add(spot);
scene.add(spot.target);

// Subtle fill from opposite side
const fillL = new THREE.DirectionalLight(0x1a3060, 0.2);
fillL.position.set(-8, 6, -6);
scene.add(fillL);

// Faint rim from behind for edge definition
const rimL = new THREE.DirectionalLight(0x2a2048, 0.12);
rimL.position.set(0, 4, -14);
scene.add(rimL);

// Sweeping accent light — animates across surface
const sweepL = new THREE.PointLight(0x20a0c0, 0.8, 18, 1.8);
sweepL.position.set(0, 3, TD*0.4);
scene.add(sweepL);

/* ═══════════════════ BLOOM POST-PROCESSING ═══════════════════ */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.35,   // strength — restrained, only wall markers bloom visibly
  0.4,    // radius
  0.65    // threshold — high so only emissive markers trigger bloom
);
composer.addPass(bloomPass);

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
    const prof = hit.object.userData.profile;
    const pt = hit.point;

    const priceT = (pt.x + SW/2) / SW;
    const price = D.price_range[0] + priceT * (D.price_range[1] - D.price_range[0]);
    const binIdx = Math.round(priceT * NB);
    const volPct = Math.max(0, (pt.y / HY) * 100);
    const imb = D.avg_imbalances[ex] || 0;
    const imbStr = (imb >= 0 ? '+' : '') + (imb * 100).toFixed(2) + '%';
    const imbColor = imb >= 0 ? '#00cc66' : '#ff4455';

    // Check if near a wall
    let wallNote = '';
    const exWalls = WALLS[ex] || [];
    for (const w of exWalls) {{
      if (Math.abs(w.idx - binIdx) < NB * 0.02) {{
        wallNote = '<div class="t-wall">LIQUIDITY WALL</div>';
        break;
      }}
    }}

    // Approximate bid/ask volume from profile shape
    const ba = BID_ASK[ex];
    const midBin = D.mid_idx;
    const isBid = binIdx < midBin;
    const sideLabel = isBid ? 'Bid Side' : 'Ask Side';
    const sideColor = isBid ? '#00cc66' : '#ff4455';

    tip.style.display = 'block';
    tip.style.left = (e.clientX+18)+'px';
    tip.style.top = (e.clientY-12)+'px';
    tip.innerHTML = `<div class="t-label">Exchange</div><div class="t-val">${{ex}}</div>`
      + `<div class="t-row"><div class="t-label">Price</div><div class="t-val">${{price.toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}} USDT</div></div>`
      + `<div class="t-row"><div class="t-label">Relative Volume</div><div class="t-val">${{volPct.toFixed(1)}}%</div></div>`
      + `<div class="t-row"><div class="t-label">Side</div><div class="t-val" style="color:${{sideColor}}">${{sideLabel}}</div></div>`
      + `<div class="t-row"><div class="t-label">Avg Imbalance</div><div class="t-val" style="color:${{imbColor}}">${{imbStr}}</div></div>`
      + wallNote;
  }} else {{
    tip.style.display = 'none';
  }}
}});

renderer.domElement.addEventListener('mouseleave', () => {{ tip.style.display='none'; }});

/* ═══════════════════ RENDER LOOP ═══════════════════ */
const clock = new THREE.Clock();

function animate() {{
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  // Sweep light glides across surface (signature animation)
  sweepL.position.x = Math.sin(t * 0.3) * SW * 0.45;
  sweepL.position.z = TD * 0.4 + Math.cos(t * 0.2) * TD * 0.3;

  controls.update();
  composer.render();
  labelR.render(scene, camera);
}}
animate();

/* ═══════════════════ RESIZE ═══════════════════ */
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelR.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
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
        imb_color = "#00cc66" if last_imb > 0 else "#ff4455"
        ex_color = '#f0b90b' if ex == 'Binance' else '#0052ff' if ex == 'Coinbase' else '#7b3fe4'
        stat_rows += f"""
        <tr>
          <td style="color:{ex_color};font-weight:600">{ex}</td>
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
  body {{ background:#05070a; color:#b0c4d4; font-family:'Consolas','Menlo',monospace; }}
  .header {{
    padding:16px 24px; border-bottom:1px solid #141e30;
    display:flex; justify-content:space-between; align-items:center;
  }}
  .header h1 {{ color:#d0e0f0; font-size:15px; font-weight:600; letter-spacing:0.4px; }}
  .header .meta {{ font-size:10px; color:#5a7a90; }}
  .panels {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr auto; height:calc(100vh - 52px); }}
  .panel {{ border:1px solid #141e30; position:relative; overflow:hidden; }}
  .panel-title {{
    position:absolute; top:8px; left:12px; font-size:9px;
    color:#4a7090; text-transform:uppercase; letter-spacing:1px; z-index:10;
  }}
  .panel-3d {{ grid-column:1/-1; min-height:58vh; }}
  .panel-3d iframe {{ width:100%; height:100%; border:none; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  th {{ text-align:left; color:#4a7090; font-weight:400; padding:6px 10px;
       text-transform:uppercase; font-size:9px; letter-spacing:0.5px;
       border-bottom:1px solid #141e30; }}
  td {{ padding:6px 10px; border-bottom:1px solid #0c1420; }}
  .spark-panel {{ padding:14px; }}
  .spark-row {{ display:flex; align-items:center; margin-bottom:8px; gap:10px; }}
  .spark-label {{ width:72px; font-size:10px; font-weight:600; }}
  .spark-canvas {{ flex:1; height:26px; }}
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
  <div class="panel" style="padding:14px;">
    <div class="panel-title">Exchange Metrics</div>
    <table style="margin-top:24px;">
      <tr><th>Exchange</th><th>Last Imbalance</th><th>Avg Imbalance</th></tr>
      {stat_rows}
    </table>
  </div>
  <div class="panel spark-panel">
    <div class="panel-title">Imbalance Over Time</div>
    <div id="sparklines" style="margin-top:24px;"></div>
  </div>
</div>
<script>
const sparkData = {spark_json};
const exColors = {{ Binance:'#f0b90b', Coinbase:'#0052ff', Kraken:'#7b3fe4' }};
const ctr = document.getElementById('sparklines');

Object.entries(sparkData).forEach(([ex, vals]) => {{
  const row = document.createElement('div');
  row.className = 'spark-row';
  const lbl = document.createElement('div');
  lbl.className = 'spark-label';
  lbl.style.color = exColors[ex]||'#88aacc';
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
    ctx.strokeStyle = 'rgba(50,70,90,0.3)'; ctx.lineWidth = 1;
    ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(0,zy); ctx.lineTo(w,zy); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = exColors[ex]||'#88aacc'; ctx.lineWidth = 2; ctx.beginPath();
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
