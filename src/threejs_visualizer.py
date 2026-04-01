"""
Three.js premium 3D liquidity surface generator.
Produces a self-contained HTML file with WebGL rendering.
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def _prepare_grid(data: dict, grid_res: int = 120) -> dict:
    """
    Aggregate volume data and resample onto a smooth grid.

    Returns dict with:
        heights: flattened list of normalised height values (row-major)
        rows: number of rows (time axis)
        cols: number of columns (price axis)
    """
    # Aggregate across exchanges
    matrices = list(data["price_grids"].values())
    agg = np.zeros_like(matrices[0])
    for m in matrices:
        agg += m

    n_time, n_price = agg.shape

    # Log-scale to compress spikes
    agg = np.log1p(agg)

    # Resample to target resolution via interpolation
    t_orig = np.linspace(0, 1, n_time)
    p_orig = np.linspace(0, 1, n_price)
    interp = RegularGridInterpolator(
        (t_orig, p_orig), agg,
        method="linear", bounds_error=False, fill_value=0,
    )

    rows = min(grid_res, max(n_time * 4, 60))
    cols = grid_res
    t_new = np.linspace(0, 1, rows)
    p_new = np.linspace(0, 1, cols)
    tg, pg = np.meshgrid(t_new, p_new, indexing="ij")
    resampled = interp((tg, pg))

    # Smooth for fluid surface
    resampled = gaussian_filter(resampled, sigma=1.2)

    # Normalise to [0, 1]
    vmax = resampled.max()
    if vmax > 0:
        resampled = resampled / vmax

    # Flatten row-major for JS consumption
    heights = resampled.flatten().tolist()
    # Round to 4 decimals to keep file size down
    heights = [round(h, 4) for h in heights]

    return {"heights": heights, "rows": rows, "cols": cols}


def generate_threejs(data: dict):
    """
    Generate a self-contained Three.js HTML file with an interactive
    3D liquidity surface.
    """
    os.makedirs("output", exist_ok=True)

    grid = _prepare_grid(data)
    grid_json = json.dumps(grid, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3D Liquidity Surface</title>
<style>
  * {{ margin: 0; padding: 0; }}
  body {{ background: #0b0f17; overflow: hidden; }}
  canvas {{ display: block; }}
</style>
</head>
<body>
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

// ── Data ──
const grid = {grid_json};
const ROWS = grid.rows;
const COLS = grid.cols;
const heights = grid.heights;

// ── Color gradient: purple → blue → cyan → yellow ──
function heightColor(t) {{
  const stops = [
    [0.00, 0.10, 0.02, 0.13],
    [0.20, 0.18, 0.11, 0.41],
    [0.40, 0.11, 0.23, 0.54],
    [0.55, 0.06, 0.44, 0.63],
    [0.70, 0.09, 0.64, 0.72],
    [0.82, 0.13, 0.79, 0.63],
    [0.92, 0.72, 0.88, 0.31],
    [1.00, 0.94, 0.91, 0.19],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {{
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {{
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }}
  }}
  const f = (hi[0] - lo[0]) > 0 ? (t - lo[0]) / (hi[0] - lo[0]) : 0;
  return new THREE.Color(
    lo[1] + (hi[1] - lo[1]) * f,
    lo[2] + (hi[2] - lo[2]) * f,
    lo[3] + (hi[3] - lo[3]) * f,
  );
}}

// ── Scene ──
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f17);

// ── Camera ──
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(5, 4, 7);

// ── Renderer ──
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
document.body.appendChild(renderer.domElement);

// ── Controls ──
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2;
controls.maxDistance = 20;
controls.target.set(0, 0.5, 0);

// ── Geometry ──
const WIDTH = 8;
const DEPTH = 6;
const HEIGHT_SCALE = 2.5;

const geometry = new THREE.BufferGeometry();
const vertices = [];
const colors = [];
const indices = [];

for (let r = 0; r < ROWS; r++) {{
  for (let c = 0; c < COLS; c++) {{
    const x = (c / (COLS - 1)) * WIDTH - WIDTH / 2;
    const z = (r / (ROWS - 1)) * DEPTH - DEPTH / 2;
    const h = heights[r * COLS + c];
    const y = h * HEIGHT_SCALE;

    vertices.push(x, y, z);

    const col = heightColor(h);
    colors.push(col.r, col.g, col.b);
  }}
}}

for (let r = 0; r < ROWS - 1; r++) {{
  for (let c = 0; c < COLS - 1; c++) {{
    const i = r * COLS + c;
    indices.push(i, i + COLS, i + 1);
    indices.push(i + 1, i + COLS, i + COLS + 1);
  }}
}}

geometry.setIndex(indices);
geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
geometry.computeVertexNormals();

// ── Material ──
const material = new THREE.MeshStandardMaterial({{
  vertexColors: true,
  roughness: 0.35,
  metalness: 0.15,
  side: THREE.DoubleSide,
}});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// ── Lighting ──
const ambient = new THREE.AmbientLight(0x404060, 0.8);
scene.add(ambient);

const dir1 = new THREE.DirectionalLight(0xffffff, 1.0);
dir1.position.set(5, 8, 4);
scene.add(dir1);

const dir2 = new THREE.DirectionalLight(0x6688cc, 0.4);
dir2.position.set(-4, 3, -5);
scene.add(dir2);

// ── Render loop ──
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// ── Resize ──
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    out = "output/3d_liquidity_premium.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"  Saved {out}")
