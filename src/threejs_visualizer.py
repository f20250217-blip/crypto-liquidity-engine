"""
Three.js WebGL 3D liquidity surface generator.
Produces a self-contained HTML file — no Plotly, no matplotlib.
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def _prepare_grid(data: dict, grid_cols: int = 140, grid_rows: int = 100) -> dict:
    """
    Aggregate volume across exchanges and resample onto a high-resolution
    grid suitable for Three.js PlaneGeometry vertex displacement.

    Returns:
        heights: row-major flattened list of normalised [0,1] values
        rows: grid rows (depth / time axis)
        cols: grid columns (price axis)
    """
    matrices = list(data["price_grids"].values())
    agg = np.zeros_like(matrices[0])
    for m in matrices:
        agg += m

    n_time, n_price = agg.shape

    # Log-scale to compress dominant spikes
    agg = np.log1p(agg)

    # Resample via bilinear interpolation onto target grid
    t_orig = np.linspace(0, 1, n_time)
    p_orig = np.linspace(0, 1, n_price)
    interp = RegularGridInterpolator(
        (t_orig, p_orig), agg,
        method="linear", bounds_error=False, fill_value=0,
    )

    t_new = np.linspace(0, 1, grid_rows)
    p_new = np.linspace(0, 1, grid_cols)
    tg, pg = np.meshgrid(t_new, p_new, indexing="ij")
    resampled = interp((tg, pg))

    # Smooth for fluid surface
    resampled = gaussian_filter(resampled, sigma=1.0)

    # Normalise to [0, 1]
    vmax = resampled.max()
    if vmax > 0:
        resampled = resampled / vmax

    heights = [round(float(h), 4) for h in resampled.flatten()]
    return {"heights": heights, "rows": grid_rows, "cols": grid_cols}


def generate_threejs(data: dict):
    """Generate output/3d_liquidity_premium.html — pure Three.js WebGL."""
    os.makedirs("output", exist_ok=True)

    grid = _prepare_grid(data)
    grid_json = json.dumps(grid, separators=(",", ":"))

    html = _build_html(grid_json)

    out = "output/3d_liquidity_premium.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"  Saved {out}")


def _build_html(grid_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3D Liquidity Surface</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
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

/* ── Data ── */
const grid = {grid_json};
const ROWS = grid.rows;
const COLS = grid.cols;
const H = grid.heights;

/* ── Color stops: purple → blue → cyan → green → yellow ── */
const STOPS = [
  [0.00, 0.10, 0.02, 0.13],
  [0.15, 0.17, 0.10, 0.41],
  [0.30, 0.11, 0.23, 0.54],
  [0.45, 0.05, 0.37, 0.63],
  [0.58, 0.09, 0.54, 0.72],
  [0.70, 0.09, 0.69, 0.65],
  [0.82, 0.50, 0.82, 0.38],
  [0.92, 0.80, 0.90, 0.24],
  [1.00, 0.94, 0.91, 0.19],
];

function colorAt(t) {{
  t = Math.max(0, Math.min(1, t));
  let lo = STOPS[0], hi = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++) {{
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) {{
      lo = STOPS[i]; hi = STOPS[i + 1]; break;
    }}
  }}
  const f = hi[0] > lo[0] ? (t - lo[0]) / (hi[0] - lo[0]) : 0;
  return new THREE.Color(
    lo[1] + (hi[1] - lo[1]) * f,
    lo[2] + (hi[2] - lo[2]) * f,
    lo[3] + (hi[3] - lo[3]) * f
  );
}}

/* ── Renderer ── */
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

/* ── Scene ── */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f17);
scene.fog = new THREE.FogExp2(0x0b0f17, 0.035);

/* ── Camera ── */
const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.1, 200
);
camera.position.set(6, 4.5, 8);

/* ── Controls ── */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 3;
controls.maxDistance = 25;
controls.maxPolarAngle = Math.PI * 0.48;
controls.target.set(0, 0.6, 0);

/* ── Geometry: PlaneGeometry displaced by volume ── */
const W = 10, D = 7, HSCALE = 3.0;
const geo = new THREE.PlaneGeometry(W, D, COLS - 1, ROWS - 1);
geo.rotateX(-Math.PI / 2);

const pos = geo.attributes.position;
const colors = new Float32Array(pos.count * 3);

for (let i = 0; i < pos.count; i++) {{
  /* PlaneGeometry lays out vertices row by row along width (x),
     then advances along depth (z). Map to our grid. */
  const col = i % COLS;
  const row = Math.floor(i / COLS);
  const h = H[row * COLS + col];

  pos.setY(i, h * HSCALE);

  const c = colorAt(h);
  colors[i * 3]     = c.r;
  colors[i * 3 + 1] = c.g;
  colors[i * 3 + 2] = c.b;
}}

geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geo.computeVertexNormals();

/* ── Material ── */
const mat = new THREE.MeshStandardMaterial({{
  vertexColors: true,
  roughness: 0.32,
  metalness: 0.12,
  flatShading: false,
  side: THREE.DoubleSide,
}});

const mesh = new THREE.Mesh(geo, mat);
scene.add(mesh);

/* ── Subtle reflection plane beneath the surface ── */
const mirrorGeo = new THREE.PlaneGeometry(W * 1.4, D * 1.4);
mirrorGeo.rotateX(-Math.PI / 2);
const mirrorMat = new THREE.MeshStandardMaterial({{
  color: 0x0b0f17,
  roughness: 0.6,
  metalness: 0.3,
  transparent: true,
  opacity: 0.35,
}});
const mirror = new THREE.Mesh(mirrorGeo, mirrorMat);
mirror.position.y = -0.05;
scene.add(mirror);

/* ── Lighting ── */
const amb = new THREE.AmbientLight(0x303050, 0.7);
scene.add(amb);

const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(6, 10, 5);
scene.add(key);

const fill = new THREE.DirectionalLight(0x5577aa, 0.5);
fill.position.set(-5, 4, -6);
scene.add(fill);

const rim = new THREE.DirectionalLight(0x8844cc, 0.3);
rim.position.set(0, 2, -8);
scene.add(rim);

/* ── Render loop ── */
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

/* ── Resize ── */
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""
