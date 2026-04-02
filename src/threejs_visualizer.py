"""
Three.js WebGL 3D liquidity surface generator.
Produces a self-contained HTML file — no Plotly, no matplotlib.
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RegularGridInterpolator


def _prepare_grid(data: dict, grid_cols: int = 120, grid_rows: int = 80) -> dict:
    """
    Aggregate volume across exchanges, extract dominant liquidity
    structures, and resample onto a clean grid for Three.js.

    Pipeline:
      1. Aggregate across exchanges
      2. Log-compress to tame spikes
      3. Median filter to kill salt-and-pepper noise
      4. Resample onto target grid
      5. Heavy Gaussian smooth for broad ridges
      6. Suppress bottom 80% (noise floor) with soft threshold
      7. Power-curve to widen dominant structures
      8. Final polish smooth
      9. Normalize to [0, 1]
    """
    matrices = list(data["price_grids"].values())
    agg = np.zeros_like(matrices[0])
    for m in matrices:
        agg += m

    n_time, n_price = agg.shape

    # Log-compress to reduce spike dominance
    agg = np.log1p(agg)

    # Median filter: removes isolated spikes while preserving edges
    agg = median_filter(agg, size=3)

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

    # Heavy Gaussian smooth — creates broad ridges instead of spikes
    resampled = gaussian_filter(resampled, sigma=3.5)

    # Normalize to [0, 1] before thresholding
    vmax = resampled.max()
    if vmax > 0:
        resampled /= vmax

    # Soft threshold: suppress bottom 80% (noise floor)
    # Values below p80 fade to zero; above p80 get rescaled
    p80 = np.percentile(resampled, 80)
    resampled = np.clip((resampled - p80) / (1.0 - p80 + 1e-9), 0, 1)

    # Power curve (sqrt) to widen ridges — makes structures broader
    resampled = np.power(resampled, 0.6)

    # Final polish smooth for silky geometry
    resampled = gaussian_filter(resampled, sigma=2.0)

    # Re-normalize
    vmax = resampled.max()
    if vmax > 0:
        resampled /= vmax

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
  body {{ background: #060a12; overflow: hidden; }}
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
import {{ EffectComposer }} from 'three/addons/postprocessing/EffectComposer.js';
import {{ RenderPass }} from 'three/addons/postprocessing/RenderPass.js';
import {{ UnrealBloomPass }} from 'three/addons/postprocessing/UnrealBloomPass.js';

/* ── Data ── */
const grid = {grid_json};
const ROWS = grid.rows;
const COLS = grid.cols;
const H = grid.heights;

/* ── Color stops: dark purple → blue → cyan → yellow (clean gradient) ── */
const STOPS = [
  [0.00, 0.05, 0.02, 0.10],
  [0.20, 0.08, 0.08, 0.35],
  [0.40, 0.06, 0.22, 0.55],
  [0.55, 0.04, 0.40, 0.68],
  [0.70, 0.05, 0.60, 0.70],
  [0.82, 0.20, 0.75, 0.50],
  [0.92, 0.60, 0.85, 0.25],
  [1.00, 0.90, 0.92, 0.15],
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
const BG = 0x060a12;
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

/* ── Scene ── */
const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.FogExp2(BG, 0.045);

/* ── Camera ── */
const camera = new THREE.PerspectiveCamera(
  42, window.innerWidth / window.innerHeight, 0.1, 200
);
camera.position.set(7, 5, 9);

/* ── Controls ── */
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.04;
controls.minDistance = 3;
controls.maxDistance = 30;
controls.maxPolarAngle = Math.PI * 0.47;
controls.target.set(0, 0.8, 0);
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;

/* ── Geometry: PlaneGeometry displaced by volume ── */
const W = 10, D = 7, HSCALE = 2.4;
const geo = new THREE.PlaneGeometry(W, D, COLS - 1, ROWS - 1);
geo.rotateX(-Math.PI / 2);

const pos = geo.attributes.position;
const colors = new Float32Array(pos.count * 3);
const emissiveMap = new Float32Array(pos.count);

for (let i = 0; i < pos.count; i++) {{
  const col = i % COLS;
  const row = Math.floor(i / COLS);
  const h = H[row * COLS + col];

  pos.setY(i, h * HSCALE);

  const c = colorAt(h);
  colors[i * 3]     = c.r;
  colors[i * 3 + 1] = c.g;
  colors[i * 3 + 2] = c.b;

  emissiveMap[i] = h;
}}

geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geo.computeVertexNormals();

/* ── Custom shader material for per-vertex emissive glow ── */
const mat = new THREE.MeshStandardMaterial({{
  vertexColors: true,
  roughness: 0.35,
  metalness: 0.18,
  flatShading: false,
  side: THREE.DoubleSide,
  envMapIntensity: 0.4,
}});

/* Inject per-vertex emissive via onBeforeCompile */
geo.setAttribute('aEmissive', new THREE.BufferAttribute(emissiveMap, 1));

mat.onBeforeCompile = (shader) => {{
  shader.vertexShader = shader.vertexShader.replace(
    'void main() {{',
    'attribute float aEmissive;\\nvarying float vEmissive;\\nvoid main() {{\\n  vEmissive = aEmissive;'
  );
  shader.fragmentShader = shader.fragmentShader.replace(
    'void main() {{',
    'varying float vEmissive;\\nvoid main() {{'
  );
  /* Emissive only on dominant peaks (top ~15%) */
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <emissivemap_fragment>',
    `#include <emissivemap_fragment>
     float ep = smoothstep(0.7, 0.95, vEmissive);
     vec3 peakGlow = mix(vec3(0.0, 0.25, 0.45), vec3(0.5, 0.75, 0.15), smoothstep(0.85, 1.0, vEmissive));
     totalEmissiveRadiance += peakGlow * ep * 0.9;`
  );
}};

const mesh = new THREE.Mesh(geo, mat);
mesh.castShadow = true;
mesh.receiveShadow = true;
scene.add(mesh);

/* ── Grid floor ── */
const gridSize = 16;
const gridDiv = 40;
const gridHelper = new THREE.GridHelper(gridSize, gridDiv, 0x1a2040, 0x0e1628);
gridHelper.position.y = -0.08;
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.25;
scene.add(gridHelper);

/* ── Subtle reflection plane beneath the surface ── */
const mirrorGeo = new THREE.PlaneGeometry(W * 1.6, D * 1.6);
mirrorGeo.rotateX(-Math.PI / 2);
const mirrorMat = new THREE.MeshStandardMaterial({{
  color: BG,
  roughness: 0.5,
  metalness: 0.4,
  transparent: true,
  opacity: 0.3,
}});
const mirror = new THREE.Mesh(mirrorGeo, mirrorMat);
mirror.position.y = -0.1;
mirror.receiveShadow = true;
scene.add(mirror);

/* ── Lighting ── */
const amb = new THREE.AmbientLight(0x1a1a30, 0.5);
scene.add(amb);

/* Key light with shadows */
const key = new THREE.DirectionalLight(0xeeeeff, 1.4);
key.position.set(6, 12, 5);
key.castShadow = true;
key.shadow.mapSize.width = 1024;
key.shadow.mapSize.height = 1024;
key.shadow.camera.near = 1;
key.shadow.camera.far = 30;
key.shadow.camera.left = -10;
key.shadow.camera.right = 10;
key.shadow.camera.top = 10;
key.shadow.camera.bottom = -10;
key.shadow.bias = -0.002;
scene.add(key);

/* Cool fill from opposite side */
const fill = new THREE.DirectionalLight(0x4466aa, 0.6);
fill.position.set(-6, 5, -7);
scene.add(fill);

/* Rim light — purple accent from behind */
const rim = new THREE.DirectionalLight(0x7733cc, 0.45);
rim.position.set(0, 3, -10);
scene.add(rim);

/* Subtle accent from below-front for depth */
const accent = new THREE.PointLight(0x00997a, 0.15, 18);
accent.position.set(2, -1, 4);
scene.add(accent);

/* ── Post-processing: Bloom (Unreal) ── */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.3,    /* strength — subtle, not overwhelming */
  0.5,    /* radius */
  0.55    /* threshold — only bright peaks bloom */
);
composer.addPass(bloomPass);

/* ── Render loop ── */
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  composer.render();
}}
animate();

/* ── Resize ── */
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""
