# Cross-Exchange Crypto Liquidity & Order Flow Engine

Real-time order book depth analysis across Binance, Coinbase, and Kraken with interactive 3D visualization and automated liquidity wall detection.

## 3D Liquidity Surface

![3D Liquidity Engine](output/3d_preview.png)

Interactive version: open `output/3d_liquidity_pro.html` locally after running the pipeline.

## What It Shows

The 3D surface maps **order book liquidity** across three dimensions:

| Axis | Dimension | Description |
|------|-----------|-------------|
| **X** | Price (USDT) | BTC/USDT price range with numeric tick labels |
| **Z** | Exchange | Binance, Coinbase, Kraken — each as a separate surface strip |
| **Y** | Volume | Aggregated order book depth at each price level (relative %) |

## How to Read the Visualization

**Liquidity walls** — Sharp vertical peaks at price levels with concentrated resting orders. Yellow-tipped peaks mark the strongest walls. Vertical drop lines connect peaks to the floor grid for precise price identification.

**Cross-exchange comparison** — Separated flat-shaded strips let you compare depth structure across exchanges. Matching peaks signal consensus support/resistance.

**Heatmap** — Dark blue-gray (low) through blue and cyan (high) to soft yellow (extreme concentration).

## Features

- Time-series order book collection (configurable sample count and interval)
- 200-bin smooth depth profiles with noise removal (top 35% signal)
- Smooth-shaded surfaces with natural curves
- Automated liquidity wall detection — top 3 peaks labeled with drop lines
- Professional heatmap: #1E2A38 → #2F6FA3 → #4FD1C5 → #F6E05E
- Hover tooltips: exchange, price, volume, wall detection
- Multi-panel dashboard with imbalance sparklines

## Metrics

| Metric | Description |
|--------|-------------|
| Best Bid / Ask | Top-of-book prices |
| Spread | Ask minus bid (absolute and basis points) |
| Bid / Ask Volume | Total volume across top N levels |
| Imbalance | (bid_vol - ask_vol) / total_vol |

## Installation

```bash
git clone https://github.com/f20250217-blip/crypto-liquidity-engine.git
cd crypto-liquidity-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Configure in `main.py`: `N_SAMPLES` (default 30), `INTERVAL_SEC` (default 10), `LIMIT` (default 50).

## Output

| File | Description |
|------|-------------|
| `output/3d_liquidity_pro.html` | Interactive 3D liquidity surface with grid surfaces, axes, walls, tooltips |
| `output/dashboard.html` | Multi-panel dashboard: 3D view + metrics table + imbalance sparklines |

## Project Structure

```
crypto-liquidity-engine/
├── src/
│   ├── data_fetcher.py          # Multi-exchange order book retrieval (ccxt)
│   ├── orderbook_processor.py   # Bid/ask DataFrame extraction
│   ├── metrics.py               # Spread, volume, imbalance computation
│   ├── time_collector.py        # Time-series snapshot collector
│   └── threejs_visualizer.py    # 3D engine + dashboard generator
├── output/                      # Generated visualizations
├── main.py                      # Pipeline entry point
├── requirements.txt
└── README.md
```

## License

MIT
