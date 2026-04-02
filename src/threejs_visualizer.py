"""
3D Liquidity Analysis Engine — Three.js WebGL.

Generates:
  output/3d_liquidity_pro.html  — interactive multi-exchange 3D analytical surface
  output/dashboard.html         — multi-panel analytics dashboard
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, label
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Data preparation — ridge-based structure extraction
# ---------------------------------------------------------------------------

def _build_exchange_profiles(data: dict, n_bins: int = 200) -> dict:
    """
    Build per-exchange depth profiles with intentional ridge structures.

    Pipeline:
      1. Average across time, log-compress, denoise
      2. Resample to n_bins
      3. Detect liquidity bands (contiguous high-volume zones)
      4. Merge nearby peaks into continuous ridges
      5. Keep only top 2 dominant ridges per exchange
      6. Sharpen ridge edges
      7. Crop to active price region (remove dead edges)
    """
    price_grids = data["price_grids"]
    exchanges = data["exchanges"]
    price_min, price_max = data["price_range"]
    imbalance_series = data.get("imbalance_series", {})

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
        resampled = gaussian_filter(resampled, sigma=1.5)

        raw_profiles[ex] = resampled.copy()
        global_max = max(global_max, resampled.max())

    # ── Normalize globally ──
    for ex in exchanges:
        if global_max > 0:
            raw_profiles[ex] = raw_profiles[ex] / global_max

    # ── Find active region across all exchanges (crop dead edges) ──
    activity = np.zeros(n_bins)
    for ex in exchanges:
        activity += raw_profiles[ex]
    activity_thresh = np.percentile(activity, 40)
    active_mask = activity > activity_thresh
    active_indices = np.where(active_mask)[0]

    if len(active_indices) > 10:
        crop_start = max(0, active_indices[0] - 8)
        crop_end = min(n_bins, active_indices[-1] + 9)
    else:
        crop_start = 0
        crop_end = n_bins

    crop_bins = crop_end - crop_start
    price_ticks = np.linspace(price_min, price_max, n_bins)
    cropped_ticks = price_ticks[crop_start:crop_end]

    # ── Ridge extraction per exchange ──
    profiles = {}
    for ex in exchanges:
        p = raw_profiles[ex][crop_start:crop_end].copy()

        # Threshold to find significant regions
        p50 = np.percentile(p, 55)
        mask = p > p50

        # Label contiguous regions (bands)
        labeled, n_regions = label(mask)

        # Score each region by total volume
        region_scores = []
        for r in range(1, n_regions + 1):
            region_mask = labeled == r
            region_vol = p[region_mask].sum()
            region_peak = p[region_mask].max()
            region_center = np.mean(np.where(region_mask)[0])
            region_width = np.sum(region_mask)
            region_scores.append({
                "id": r,
                "vol": region_vol,
                "peak": region_peak,
                "center": region_center,
                "width": region_width,
                "mask": region_mask,
            })

        # Keep top 2 ridges by volume (merge if within 15 bins)
        region_scores.sort(key=lambda r: r["vol"], reverse=True)
        kept = region_scores[:3]

        # Merge nearby ridges
        if len(kept) >= 2:
            merged = [kept[0]]
            for r in kept[1:]:
                can_merge = False
                for m in merged:
                    if abs(r["center"] - m["center"]) < crop_bins * 0.12:
                        m["mask"] = m["mask"] | r["mask"]
                        m["vol"] += r["vol"]
                        can_merge = True
                        break
                if not can_merge:
                    merged.append(r)
            kept = merged[:2]

        # Build ridge profile: zero out everything not in kept ridges
        ridge_mask = np.zeros(crop_bins, dtype=bool)
        for r in kept:
            # Expand ridge slightly for smooth transitions
            expanded = np.copy(r["mask"])
            for _ in range(4):
                shifted_l = np.roll(expanded, 1)
                shifted_r = np.roll(expanded, -1)
                expanded = expanded | shifted_l | shifted_r
            ridge_mask = ridge_mask | expanded

        p_out = np.zeros(crop_bins)
        p_out[ridge_mask] = p[ridge_mask]

        # Smooth ridge edges (not the interior)
        p_out = gaussian_filter(p_out, sigma=1.8)

        # Renormalize
        pmax = p_out.max()
        if pmax > 0:
            p_out = p_out / pmax

        # Apply contrast curve — push lows down, keep highs
        p_out = np.power(p_out, 0.7)

        # Kill residual noise below 5%
        p_out[p_out < 0.05] = 0

        profiles[ex] = p_out

    # ── Detect labeled peaks (top of each ridge) ──
    walls = {}
    all_walls = []
    for ex in exchanges:
        p = profiles[ex]
        peak_idxs, _ = find_peaks(p, height=0.25, distance=crop_bins // 10, prominence=0.1)
        wall_list = []
        for idx in peak_idxs:
            wall_list.append({
                "idx": int(idx),
                "height": round(float(p[idx]), 3),
                "price": round(float(cropped_ticks[idx]), 2),
                "exchange": ex,
            })
        wall_list.sort(key=lambda w: w["height"], reverse=True)
        walls[ex] = wall_list[:3]
        all_walls.extend(wall_list[:3])

    # Top 3 globally for labels
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

    # ── Price labels — 6 ticks across cropped range ──
    n_labels = 6
    step = max(1, crop_bins // n_labels)
    price_labels = [
        {"idx": int(i), "val": round(float(cropped_ticks[i]), 2)}
        for i in range(0, crop_bins, step)
    ]

    profiles_out = {}
    for ex in exchanges:
        profiles_out[ex] = [round(float(v), 4) for v in profiles[ex]]

    return {
        "exchanges": exchanges,
        "profiles": profiles_out,
        "n_bins": int(crop_bins),
        "price_labels": price_labels,
        "price_range": [round(float(cropped_ticks[0]), 2), round(float(cropped_ticks[-1]), 2)],
        "mid_price": round(float((cropped_ticks[0] + cropped_ticks[-1]) / 2.0), 2),
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
<div id="info">BTC/USDT <span>Liquidity Structure</span></div>

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
const SW = 14;
const SD = 3.0;
const GAP = 1.2;
const HY = 3.0;
const BG = 0x0F172A;
const TD = EX.length * SD + (EX.length - 1) * GAP;
const SROWS = 20;

/* ═══════ COLOR: dark base → blue → cyan → yellow (4 stops) ═══════ */
const CS = [
  [0.00, 0.06, 0.09, 0.16],
  [0.25, 0.12, 0.28, 0.48],
  [0.50, 0.18, 0.55, 0.68],
  [0.75, 0.31, 0.82, 0.77],
  [1.00, 0.96, 0.88, 0.37],
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

/* ═══════ CAMERA ═══════ */
const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 400);
camera.position.set(10, 7.5, TD + 8);

/* ═══════ CONTROLS ═══════ */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 6;
controls.maxDistance = 40;
controls.maxPolarAngle = Math.PI * 0.44;
controls.target.set(0, 0.6, TD * 0.42);
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

/* ═══════ AXIS MATERIALS ═══════ */
const axisLine = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.2 }});
const axisTick = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.14 }});
const gridLine = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.03 }});

/* ═══════ X AXIS — Price ═══════ */
addLine([new THREE.Vector3(-SW / 2, 0, -0.1), new THREE.Vector3(SW / 2, 0, -0.1)], axisLine);
const xTitle = mkLabel('Price (USDT)', 8, 'rgba(148,163,184,0.3)', false);
xTitle.position.set(0, -0.06, -0.6);
scene.add(xTitle);

D.price_labels.forEach(pl => {{
  const x = xWorld(pl.idx);
  addLine([new THREE.Vector3(x, 0, -0.1), new THREE.Vector3(x, -0.05, -0.1)], axisTick);
  addLine([new THREE.Vector3(x, 0.002, -0.1), new THREE.Vector3(x, 0.002, TD + 0.05)], gridLine);
  const lb = mkLabel(pl.val.toLocaleString(), 7, 'rgba(148,163,184,0.22)', false);
  lb.position.set(x, -0.15, -0.1);
  scene.add(lb);
}});

/* ═══════ Y AXIS — Volume ═══════ */
const yH = HY * 1.05;
addLine([new THREE.Vector3(-SW / 2 - 0.1, 0, -0.1), new THREE.Vector3(-SW / 2 - 0.1, yH, -0.1)], axisLine);
const yTitle = mkLabel('Volume', 8, 'rgba(148,163,184,0.3)', false);
yTitle.position.set(-SW / 2 - 0.45, yH * 0.5, -0.1);
scene.add(yTitle);

for (let i = 0; i <= 4; i++) {{
  const y = (i / 4) * yH;
  addLine([new THREE.Vector3(-SW / 2 - 0.1, y, -0.1), new THREE.Vector3(-SW / 2 - 0.2, y, -0.1)], axisTick);
  addLine([new THREE.Vector3(-SW / 2, y, -0.1), new THREE.Vector3(SW / 2, y, -0.1)], gridLine);
}}

/* ═══════ Z AXIS ═══════ */
addLine([new THREE.Vector3(-SW / 2 - 0.1, 0, -0.1), new THREE.Vector3(-SW / 2 - 0.1, 0, TD + 0.1)], axisLine);

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
    // Flat-top cross-section: full height in center 70%, steep linear fall at edges
    const edgeDist = Math.min(rT, 1.0 - rT) * 2.0;
    const fade = Math.min(edgeDist / 0.3, 1.0);
    const fH = h * fade;

    pos.setY(i, fH * HY);
    pos.setZ(i, pos.getZ(i) + zOff + SD / 2);

    // High-contrast color: push low values darker
    const colorT = fH > 0.02 ? Math.pow(fH, 0.8) : 0;
    const col = colorAt(colorT);
    cols[i * 3] = col.r;
    cols[i * 3 + 1] = col.g;
    cols[i * 3 + 2] = col.b;
  }}

  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({{
    vertexColors: true,
    roughness: 0.48,
    metalness: 0.02,
    flatShading: false,
    side: THREE.DoubleSide,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.userData = {{ exchange: ex, profile: prof }};
  scene.add(mesh);
  meshes.push(mesh);

  /* ── Exchange label ── */
  const exLbl = mkLabel(ex, 9, 'rgba(148,163,184,0.35)', true);
  exLbl.position.set(-SW / 2 - 0.35, 0.03, zOff + SD / 2);
  scene.add(exLbl);
  addLine([
    new THREE.Vector3(-SW / 2 - 0.1, 0, zOff + SD / 2),
    new THREE.Vector3(-SW / 2 - 0.2, 0, zOff + SD / 2),
  ], axisTick);
}});

/* ═══════ PEAK DROP LINES — top 3 labeled ═══════ */
const dropMat = new THREE.LineBasicMaterial({{ color: 0xF6E05E, transparent: true, opacity: 0.2 }});

EX.forEach((ex, ei) => {{
  const zOff = ei * (SD + GAP);
  const zMid = zOff + SD / 2;

  (WALLS[ex] || []).forEach(w => {{
    if (!w.show_label) return;
    const x = xWorld(w.idx);
    const y = w.height * HY;

    addLine([new THREE.Vector3(x, y, zMid), new THREE.Vector3(x, 0, zMid)], dropMat);
    addLine([new THREE.Vector3(x - 0.08, 0.002, zMid), new THREE.Vector3(x + 0.08, 0.002, zMid)], dropMat);

    const lb = mkLabel('$' + w.price.toLocaleString(), 8, 'rgba(246,224,94,0.5)', true);
    lb.position.set(x, y + 0.16, zMid);
    scene.add(lb);
  }});
}});

/* ═══════ LIGHTING — neutral, even ═══════ */
scene.add(new THREE.AmbientLight(0xffffff, 0.65));

const keyL = new THREE.DirectionalLight(0xffffff, 0.95);
keyL.position.set(6, 14, 8);
scene.add(keyL);

const fillL = new THREE.DirectionalLight(0xe2e8f0, 0.4);
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
