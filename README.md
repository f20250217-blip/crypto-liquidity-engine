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

**Liquidity walls** — Tall peaks marked with yellow price labels. These are price levels where large volumes of resting orders concentrate. Walls often act as support/resistance because filling them requires significant capital.

**Cross-exchange comparison** — Separated strips along the Z axis let you compare where each exchange concentrates depth. Matching peaks across exchanges signal consensus-driven support/resistance. Mismatches may indicate arbitrage opportunity or exchange-specific positioning.

**Color gradient** — Maps volume intensity: deep purple (low) through blue and cyan to white (extreme concentration).

## Features

- Time-series order book collection (configurable sample count and interval)
- 300-bin high-resolution depth profiles per exchange
- Automated liquidity wall detection via peak finding (prominence + height thresholds)
- Minimal label system — only strongest walls labeled (max 5 across all exchanges)
- Clean 3-color gradient: deep purple to blue to cyan/white
- Hover tooltips: exchange, price, relative volume, wall detection
- Compact legend panel
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
| `output/3d_liquidity_pro.html` | Interactive 3D liquidity surface with axes, walls, contours, tooltips |
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
