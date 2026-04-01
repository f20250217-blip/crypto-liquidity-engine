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


# --- Theme constants ---
BG = "#0b0f17"
GRID = "rgba(255,255,255,0.04)"

# Premium colorscale: deep purple -> blue -> cyan -> yellow
LIQUIDITY_COLORSCALE = [
    [0.00, "#1a0533"],
    [0.15, "#2d1b69"],
    [0.30, "#1b3a8a"],
    [0.45, "#0e6fa0"],
    [0.60, "#17a2b8"],
    [0.75, "#20c9a0"],
    [0.90, "#b8e04e"],
    [1.00, "#f0e830"],
]
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
                colorscale=LIQUIDITY_COLORSCALE,
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
#  2. Liquidity Map (replaces 3D surface)
# ──────────────────────────────────────────────
def _detect_walls(prices: np.ndarray, vols: np.ndarray, threshold_pct: float = 85):
    """
    Detect liquidity walls — price levels where volume exceeds the
    given percentile threshold.

    Returns list of dicts with price, volume, and side label.
    """
    threshold = np.percentile(vols[vols > 0], threshold_pct) if np.any(vols > 0) else 0
    walls = []
    for i, (p, v) in enumerate(zip(prices, vols)):
        if v >= threshold and v > 0:
            walls.append({"price": p, "volume": v, "idx": i})
    return walls


def build_liquidity_map(
    processed: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> go.Figure:
    """
    Create a professional liquidity map with:
      - High-resolution heatmap showing volume intensity across price and exchange
      - Bid/ask depth curves overlaid
      - Liquidity wall annotations at volume spike zones
    """
    exchanges = list(processed.keys())
    n_exchanges = len(exchanges)

    # Build common price grid (300 bins)
    all_prices = []
    for bids, asks in processed.values():
        all_prices.extend(bids["price"].tolist())
        all_prices.extend(asks["price"].tolist())
    price_min, price_max = min(all_prices), max(all_prices)
    n_bins = 300
    price_grid = np.linspace(price_min, price_max, n_bins)

    # Build heatmap matrix: one row per exchange, volume interpolated onto grid
    heat_matrix = np.zeros((n_exchanges, n_bins))
    mid_prices = {}
    exchange_data = {}  # store per-exchange bid/ask curves for overlay

    for i, (ex, (bids, asks)) in enumerate(processed.items()):
        bid_prices = bids["price"].values
        bid_vols = bids["volume"].values
        ask_prices = asks["price"].values
        ask_vols = asks["volume"].values

        mid_prices[ex] = (bid_prices[0] + ask_prices[0]) / 2

        # Combine into single price-volume series
        prices = np.concatenate([bid_prices, ask_prices])
        vols = np.concatenate([bid_vols, ask_vols])
        order = np.argsort(prices)
        prices, vols = prices[order], vols[order]

        # Interpolate onto common grid
        interp_vols = np.interp(price_grid, prices, vols, left=0, right=0)
        heat_matrix[i] = interp_vols

        # Cumulative depth curves
        bid_cum = np.cumsum(bid_vols)
        ask_cum = np.cumsum(ask_vols)
        exchange_data[ex] = {
            "bid_prices": bid_prices,
            "bid_cum": bid_cum,
            "ask_prices": ask_prices,
            "ask_cum": ask_cum,
            "prices_combined": prices,
            "vols_combined": vols,
        }

    # Normalise heatmap rows independently for consistent color intensity
    for i in range(n_exchanges):
        row_max = heat_matrix[i].max()
        if row_max > 0:
            heat_matrix[i] = heat_matrix[i] / row_max

    # ── Layout: heatmap on top, depth curves per exchange below ──
    n_rows = 1 + n_exchanges
    row_heights = [0.45] + [0.55 / n_exchanges] * n_exchanges
    subtitles = ["Liquidity Heatmap"] + [f"{ex} — Depth Curve" for ex in exchanges]

    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=row_heights,
        subplot_titles=subtitles,
        vertical_spacing=0.06,
    )

    # ── Heatmap ──
    fig.add_trace(
        go.Heatmap(
            z=heat_matrix,
            x=price_grid,
            y=exchanges,
            colorscale=LIQUIDITY_COLORSCALE,
            showscale=True,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="Volume Intensity", font=dict(size=11, color="#AAAAAA")),
                tickfont=dict(size=10, color="#888888"),
                thickness=12,
                len=0.35,
                y=0.82,
                outlinewidth=0,
            ),
            hovertemplate="Price: $%{x:,.2f}<br>Exchange: %{y}<br>Intensity: %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Detect and annotate liquidity walls on heatmap
    for i, ex in enumerate(exchanges):
        row_vols = heat_matrix[i]
        walls = _detect_walls(price_grid, row_vols, threshold_pct=90)
        for w in walls[:2]:  # max 2 wall markers per exchange
            fig.add_annotation(
                x=w["price"], y=ex,
                text="Wall",
                font=dict(size=9, color="#f0e830"),
                bgcolor="rgba(11,15,23,0.8)",
                borderpad=2,
                showarrow=True,
                arrowhead=0,
                arrowcolor="#f0e830",
                arrowwidth=1,
                ax=0, ay=-25,
                row=1, col=1,
            )

    # ── Depth curves per exchange ──
    for i, ex in enumerate(exchanges):
        row_idx = 2 + i
        d = exchange_data[ex]

        # Bid depth (green, right-to-left cumulative)
        fig.add_trace(
            go.Scatter(
                x=d["bid_prices"], y=d["bid_cum"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.15)",
                line=dict(color="#2ecc71", width=1.5),
                name=f"{ex} Bids",
                showlegend=False,
                hovertemplate="$%{x:,.2f}<br>Cum. Vol: %{y:.4f} BTC<extra>Bids</extra>",
            ),
            row=row_idx, col=1,
        )

        # Ask depth (red, left-to-right cumulative)
        fig.add_trace(
            go.Scatter(
                x=d["ask_prices"], y=d["ask_cum"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="#e74c3c", width=1.5),
                name=f"{ex} Asks",
                showlegend=False,
                hovertemplate="$%{x:,.2f}<br>Cum. Vol: %{y:.4f} BTC<extra>Asks</extra>",
            ),
            row=row_idx, col=1,
        )

        # Mid-price vertical line
        fig.add_vline(
            x=mid_prices[ex],
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            row=row_idx, col=1,
        )

    # ── Global layout ──
    total_height = 400 + 220 * n_exchanges
    fig.update_layout(
        title=dict(
            text=(
                "Cross-Exchange Liquidity Map<br>"
                "<span style='font-size:13px;color:#888888'>"
                "Real-Time Order Book Depth Analysis</span>"
            ),
            font=dict(size=19, color="#DDDDDD"),
            x=0.5,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=FONT,
        height=total_height,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    # Style all axes
    for key in fig.layout.to_plotly_json():
        if key.startswith("xaxis"):
            fig.layout[key].update(
                gridcolor=GRID, tickfont=dict(size=10, color="#777777"),
            )
        if key.startswith("yaxis"):
            fig.layout[key].update(
                gridcolor=GRID, tickfont=dict(size=10, color="#777777"),
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

    liquidity_map = build_liquidity_map(processed)
    liquidity_map.write_html("output/liquidity_heatmap.html", auto_open=True)
    print("  Saved output/liquidity_heatmap.html")
