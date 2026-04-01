"""
Advanced visualizer: interactive Plotly dashboard with order book heatmaps,
3D liquidity surface, and imbalance charts.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator


# --- Theme constants ---
BG = "#0b0f17"
GRID = "rgba(255,255,255,0.05)"
FONT = dict(family="Arial, sans-serif", color="#CCCCCC")
AXIS_COMMON = dict(
    gridcolor=GRID,
    zerolinecolor=GRID,
    tickfont=dict(size=11, color="#888888"),
)


def _smooth_volume(vol: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Apply light Gaussian smoothing and clip outlier spikes."""
    smoothed = gaussian_filter1d(vol, sigma=sigma)
    cap = np.percentile(smoothed, 97)
    return np.clip(smoothed, 0, cap)


# ──────────────────────────────────────────────
#  1. Order Book Depth Heatmap
# ──────────────────────────────────────────────
def build_heatmap(
    processed: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> go.Figure:
    """
    Create a depth heatmap per exchange showing bid/ask volume intensity
    across price levels.
    """
    exchanges = list(processed.keys())
    fig = make_subplots(
        rows=len(exchanges), cols=1,
        subplot_titles=[f"{ex} — Order Book Depth" for ex in exchanges],
        vertical_spacing=0.12,
    )

    for i, ex in enumerate(exchanges, 1):
        bids, asks = processed[ex]

        # Bids: descending price, flip so lowest price on left
        bid_prices = bids["price"].values[::-1]
        bid_vols = _smooth_volume(bids["volume"].values[::-1])

        ask_prices = asks["price"].values
        ask_vols = _smooth_volume(asks["volume"].values)

        prices = np.concatenate([bid_prices, ask_prices])
        vols = np.concatenate([bid_vols, ask_vols])
        sides = np.concatenate([np.ones(len(bid_vols)), -np.ones(len(ask_vols))])

        # Heatmap: 2 rows (bid/ask) x N price columns
        z = np.array([bid_vols, ask_vols])
        # Normalise per exchange
        zmax = z.max()
        if zmax > 0:
            z = z / zmax

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=np.concatenate([bid_prices, ask_prices]),
                y=["Bids", "Asks"],
                colorscale="Viridis",
                showscale=(i == 1),
                colorbar=dict(
                    title=dict(text="Volume", font=dict(size=12, color="#AAAAAA")),
                    tickfont=dict(size=10, color="#888888"),
                    len=0.3,
                ) if i == 1 else None,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        title=dict(text="Order Book Depth Heatmap", font=dict(size=18, color="#DDDDDD"), x=0.5),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=FONT,
        height=300 * len(exchanges),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    for ax_key in [k for k in fig.layout.to_plotly_json() if k.startswith("xaxis") or k.startswith("yaxis")]:
        fig.layout[ax_key].update(gridcolor=GRID)

    return fig


# ──────────────────────────────────────────────
#  2. 3D Liquidity Surface
# ──────────────────────────────────────────────
def build_3d_surface(
    processed: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> go.Figure:
    """
    Create a 3D surface: X=price, Y=exchange index, Z=volume.
    Interpolated for smooth rendering.
    """
    exchanges = list(processed.keys())
    n_exchanges = len(exchanges)

    # Build a common price grid across all exchanges
    all_prices = []
    for bids, asks in processed.values():
        all_prices.extend(bids["price"].tolist())
        all_prices.extend(asks["price"].tolist())
    price_min, price_max = min(all_prices), max(all_prices)
    n_grid = 120
    price_grid = np.linspace(price_min, price_max, n_grid)

    # Build volume matrix: rows=exchanges, cols=price grid points
    vol_matrix = np.zeros((n_exchanges, n_grid))
    for i, (ex, (bids, asks)) in enumerate(processed.items()):
        # Combine bids + asks into price->volume mapping
        prices = np.concatenate([bids["price"].values, asks["price"].values])
        vols = np.concatenate([bids["volume"].values, asks["volume"].values])
        # Sort by price
        order = np.argsort(prices)
        prices, vols = prices[order], vols[order]
        # Interpolate onto common grid
        vol_matrix[i] = np.interp(price_grid, prices, vols, left=0, right=0)

    # Smooth and normalise
    vol_matrix = gaussian_filter1d(vol_matrix, sigma=1.0, axis=1)
    vmax = vol_matrix.max()
    if vmax > 0:
        vol_matrix = vol_matrix / vmax

    # Upsample exchange axis for smooth surface (3 exchanges -> more rows)
    if n_exchanges >= 2:
        ex_orig = np.arange(n_exchanges)
        ex_fine = np.linspace(0, n_exchanges - 1, max(n_exchanges * 8, 24))
        interp = RegularGridInterpolator(
            (ex_orig, np.arange(n_grid)), vol_matrix,
            method="linear", bounds_error=False, fill_value=0,
        )
        ex_grid, p_grid = np.meshgrid(ex_fine, np.arange(n_grid), indexing="ij")
        vol_smooth = interp((ex_grid, p_grid))
    else:
        ex_fine = np.array([0.0])
        vol_smooth = vol_matrix

    fig = go.Figure(
        data=[
            go.Surface(
                x=price_grid,
                y=ex_fine,
                z=vol_smooth,
                colorscale="Viridis",
                opacity=0.92,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Volume", side="right", font=dict(size=12, color="#AAAAAA")),
                    tickfont=dict(size=10, color="#888888"),
                    thickness=14,
                    len=0.5,
                ),
            )
        ]
    )

    # Tick labels for exchange axis
    tickvals = list(range(n_exchanges))
    ticktext = exchanges

    fig.update_layout(
        title=dict(
            text="3D Cross-Exchange Liquidity Surface",
            font=dict(size=18, color="#DDDDDD"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Price (USDT)", gridcolor=GRID, color="#999999", backgroundcolor="#0e1218"),
            yaxis=dict(
                title="Exchange", gridcolor=GRID, color="#999999", backgroundcolor="#0e1218",
                tickvals=tickvals, ticktext=ticktext,
            ),
            zaxis=dict(title="Volume (norm.)", gridcolor=GRID, color="#999999", backgroundcolor="#0e1218"),
            bgcolor=BG,
            aspectratio=dict(x=1.5, y=1.0, z=0.7),
            camera=dict(eye=dict(x=1.4, y=1.3, z=0.8)),
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=FONT,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ──────────────────────────────────────────────
#  3. Imbalance + Spread Dashboard
# ──────────────────────────────────────────────
def build_dashboard(metrics: pd.DataFrame) -> go.Figure:
    """
    Interactive bar charts for imbalance and spread comparison.
    """
    exchanges = metrics.index.tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Order Flow Imbalance", "Spread (basis points)"],
        horizontal_spacing=0.12,
    )

    # Imbalance bars
    imb_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in metrics["imbalance"]]
    fig.add_trace(
        go.Bar(
            x=exchanges, y=metrics["imbalance"],
            marker=dict(color=imb_colors, line=dict(width=0)),
            name="Imbalance",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Spread bars
    fig.add_trace(
        go.Bar(
            x=exchanges, y=metrics["spread_bps"],
            marker=dict(color="#3498db", line=dict(width=0)),
            name="Spread",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text="Cross-Exchange Liquidity Comparison — BTC/USDT",
            font=dict(size=18, color="#DDDDDD"),
            x=0.5,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=FONT,
        height=450,
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_xaxes(gridcolor=GRID)

    return fig


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────
def generate_all(
    processed: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    metrics: pd.DataFrame,
):
    """
    Generate and save all interactive HTML dashboards.
    """
    os.makedirs("output", exist_ok=True)

    dashboard = build_dashboard(metrics)
    dashboard.write_html("output/dashboard.html", auto_open=False)
    print("  Saved output/dashboard.html")

    heatmap = build_heatmap(processed)
    heatmap.write_html("output/heatmap.html", auto_open=False)
    print("  Saved output/heatmap.html")

    surface = build_3d_surface(processed)
    surface.write_html("output/3d_liquidity.html", auto_open=True)
    print("  Saved output/3d_liquidity.html")
