"""
Time-aware liquidity dashboard: heatmap (time x price), 3D surface,
imbalance trends — exported as a unified HTML page.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

# ── Theme ──
BG = "#0b0f17"
GRID = "rgba(255,255,255,0.03)"
FONT = dict(family="Arial, sans-serif", color="#CCCCCC")
COLORSCALE = [
    [0.00, "#0d0221"],
    [0.12, "#1a0533"],
    [0.25, "#2d1b69"],
    [0.38, "#1b3a8a"],
    [0.50, "#0e6fa0"],
    [0.62, "#17a2b8"],
    [0.75, "#20c9a0"],
    [0.88, "#b8e04e"],
    [1.00, "#f0e830"],
]
AXIS_STYLE = dict(gridcolor=GRID, tickfont=dict(size=10, color="#777777"))


def _aggregate_volume_matrix(data: dict) -> np.ndarray:
    """Sum volume matrices across all exchanges."""
    matrices = list(data["price_grids"].values())
    agg = np.zeros_like(matrices[0])
    for m in matrices:
        agg += m
    return agg


def _detect_walls(volume_matrix: np.ndarray, price_grid: np.ndarray,
                  threshold_pct: float = 92) -> list[dict]:
    """Detect persistent liquidity walls — price levels with sustained high volume."""
    time_avg = volume_matrix.mean(axis=0)
    nonzero = time_avg[time_avg > 0]
    if len(nonzero) == 0:
        return []
    threshold = np.percentile(nonzero, threshold_pct)

    walls = []
    for i, avg_vol in enumerate(time_avg):
        if avg_vol >= threshold:
            walls.append({"price": price_grid[i], "avg_volume": avg_vol, "bin_idx": i})

    # Merge adjacent bins
    merged = []
    for w in walls:
        if merged and abs(w["bin_idx"] - merged[-1]["bin_idx"]) <= 3:
            if w["avg_volume"] > merged[-1]["avg_volume"]:
                merged[-1] = w
        else:
            merged.append(w)
    return merged[:5]


def _build_heatmap(vol_smooth, price_grid, time_labels, walls) -> go.Figure:
    """Panel 1: Time vs Price heatmap with wall annotations."""
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=vol_smooth,
        x=price_grid,
        y=time_labels,
        colorscale=COLORSCALE,
        showscale=True,
        zmin=0,
        zmax=float(np.percentile(vol_smooth, 98)),
        colorbar=dict(
            title=dict(text="Volume", font=dict(size=11, color="#999999")),
            tickfont=dict(size=10, color="#777777"),
            thickness=12, len=0.8, outlinewidth=0,
        ),
        hovertemplate="Price: $%{x:,.2f}<br>Time: %{y}<br>Volume: %{z:.3f}<extra></extra>",
    ))

    # Wall annotations
    mid_time = time_labels[len(time_labels) // 2]
    for w in walls:
        fig.add_annotation(
            x=w["price"], y=mid_time,
            text="Wall",
            font=dict(size=9, color="#f0e830"),
            bgcolor="rgba(11,15,23,0.8)",
            borderpad=2,
            showarrow=True, arrowhead=0, arrowcolor="#f0e830",
            arrowwidth=1, ax=0, ay=-22,
        )

    fig.update_layout(
        title=dict(
            text=(
                "Liquidity Heatmap (Time vs Price)<br>"
                "<span style='font-size:12px;color:#777777'>"
                "Aggregated order book volume across exchanges</span>"
            ),
            font=dict(size=17, color="#DDDDDD"), x=0.5,
        ),
        xaxis=dict(title="Price (USDT)", **AXIS_STYLE),
        yaxis=dict(title="Time", **AXIS_STYLE),
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        height=500, margin=dict(l=60, r=40, t=70, b=40),
    )
    return fig


def _build_surface(vol_smooth, price_grid, time_indices) -> go.Figure:
    """Panel 2: 3D liquidity surface (time evolution)."""
    fig = go.Figure(data=[
        go.Surface(
            x=price_grid,
            y=time_indices,
            z=vol_smooth,
            colorscale=COLORSCALE,
            showscale=False,
            opacity=0.92,
            lighting=dict(ambient=0.45, diffuse=0.55, specular=0.08, roughness=0.8),
            contours=dict(
                z=dict(show=True, usecolormap=True,
                       highlightcolor="rgba(255,255,255,0.1)", project_z=True),
            ),
        )
    ])

    fig.update_layout(
        title=dict(
            text=(
                "3D Liquidity Surface<br>"
                "<span style='font-size:12px;color:#777777'>"
                "Volume evolution across price and time</span>"
            ),
            font=dict(size=17, color="#DDDDDD"), x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Price", gridcolor=GRID, color="#888888",
                       backgroundcolor="#0a0e16", nticks=8,
                       tickfont=dict(size=9, color="#666666")),
            yaxis=dict(title="Time (sample)", gridcolor=GRID, color="#888888",
                       backgroundcolor="#0a0e16", nticks=6,
                       tickfont=dict(size=9, color="#666666")),
            zaxis=dict(title="Volume", gridcolor=GRID, color="#888888",
                       backgroundcolor="#0a0e16", nticks=5,
                       tickfont=dict(size=9, color="#666666")),
            bgcolor=BG,
            aspectratio=dict(x=1.6, y=1.0, z=0.6),
            camera=dict(eye=dict(x=1.4, y=1.2, z=0.75)),
        ),
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        height=550, margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def _build_imbalance(data: dict, time_labels: list) -> go.Figure:
    """Panel 3: imbalance trend lines per exchange."""
    colors = {"Binance": "#f0b90b", "Coinbase": "#0052ff", "Kraken": "#5741d9"}

    fig = go.Figure()
    for ex in data["exchanges"]:
        imb = data["imbalance_series"][ex]
        color = colors.get(ex, "#17a2b8")
        fig.add_trace(go.Scatter(
            x=time_labels, y=imb,
            mode="lines", name=ex,
            line=dict(color=color, width=1.8),
            hovertemplate="%{y:+.4f}<extra>" + ex + "</extra>",
        ))

    # Zero reference line as a trace
    fig.add_trace(go.Scatter(
        x=[time_labels[0], time_labels[-1]], y=[0, 0],
        mode="lines", showlegend=False,
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    ))

    fig.update_layout(
        title=dict(
            text="Order Flow Imbalance Over Time",
            font=dict(size=17, color="#DDDDDD"), x=0.5,
        ),
        xaxis=dict(title="Time", **AXIS_STYLE),
        yaxis=dict(title="Imbalance", **AXIS_STYLE),
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
        height=350, margin=dict(l=60, r=40, t=50, b=40),
        legend=dict(
            font=dict(size=11, color="#AAAAAA"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
    )
    return fig


def generate_time_dashboard(data: dict):
    """
    Build all panels and combine into a single HTML dashboard.
    """
    os.makedirs("output", exist_ok=True)

    timestamps = data["timestamps"]
    price_grid = data["price_grid"]
    n_times = len(timestamps)

    # Aggregate and smooth
    vol_matrix = _aggregate_volume_matrix(data)
    vmax = vol_matrix.max()
    if vmax > 0:
        vol_matrix = vol_matrix / vmax
    vol_smooth = gaussian_filter(vol_matrix, sigma=(1.0, 2.0))

    time_labels = [t.strftime("%H:%M:%S") for t in timestamps]
    time_indices = np.arange(n_times)
    walls = _detect_walls(vol_smooth, price_grid)

    # Build individual figures
    heatmap = _build_heatmap(vol_smooth, price_grid, time_labels, walls)
    surface = _build_surface(vol_smooth, price_grid, time_indices)
    imbalance = _build_imbalance(data, time_labels)

    # Combine into single HTML
    heatmap_html = heatmap.to_html(full_html=False, include_plotlyjs="cdn")
    surface_html = surface.to_html(full_html=False, include_plotlyjs=False)
    imbalance_html = imbalance.to_html(full_html=False, include_plotlyjs=False)

    combined = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Liquidity Intelligence Dashboard</title>
<style>
  body {{
    background: {BG};
    margin: 0;
    padding: 20px 40px;
    font-family: Arial, sans-serif;
    color: #CCCCCC;
  }}
  h1 {{
    text-align: center;
    font-size: 22px;
    color: #DDDDDD;
    margin-bottom: 2px;
  }}
  .subtitle {{
    text-align: center;
    font-size: 13px;
    color: #777777;
    margin-bottom: 30px;
  }}
  .panel {{
    margin-bottom: 20px;
  }}
</style>
</head>
<body>
<h1>Liquidity Intelligence Dashboard</h1>
<div class="subtitle">Time-Aware Order Book Depth Analysis — BTC/USDT — {n_times} samples</div>
<div class="panel">{heatmap_html}</div>
<div class="panel">{surface_html}</div>
<div class="panel">{imbalance_html}</div>
</body>
</html>"""

    out = "output/dashboard.html"
    with open(out, "w") as f:
        f.write(combined)
    print(f"  Saved {out}")

    # Static preview of heatmap
    try:
        heatmap.write_image("output/dashboard_preview.png", width=1800, height=600, scale=2)
        print("  Saved output/dashboard_preview.png")
    except Exception:
        print("  Note: install 'kaleido' for static PNG export")
