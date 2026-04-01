# Cross-Exchange Crypto Liquidity & Order Flow Engine

> Comparing order book depth, spread, and flow imbalance across major crypto exchanges

## Overview

Collects time-series order book data from Binance, Coinbase, and Kraken, then builds an interactive Three.js 3D surface showing how liquidity evolves over time — with vertex-displaced geometry, custom color gradients, and cinematic lighting.

## Features

- Time-series order book collection (30 samples at configurable intervals)
- Cross-exchange volume aggregation across 400 price bins
- WebGL 3D liquidity surface with bilinear-interpolated, Gaussian-smoothed height data
- Per-exchange order flow imbalance tracking
- Console metrics: spread, depth, imbalance per exchange

## Metrics

| Metric | Description |
|--------|-------------|
| Best Bid / Ask | Top-of-book prices |
| Spread | Ask minus bid (absolute and bps) |
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

Configure collection in `main.py`: `N_SAMPLES` (default 30), `INTERVAL_SEC` (default 10), `LIMIT` (default 50).

## Output

### Console

Formatted table with per-exchange metrics (best bid/ask, spread, volumes, imbalance).

### 3D Liquidity Surface

`output/3d_liquidity_premium.html` — self-contained WebGL visualization:

- **PlaneGeometry** with 140x100 vertex grid displaced by aggregated volume
- **9-stop color gradient** (purple → blue → cyan → green → yellow) mapped to height
- **3-point lighting** (key, fill, rim) with ACES filmic tone mapping
- **OrbitControls** for interactive camera rotation and zoom
- **Fog and reflection plane** for depth and atmosphere

## Project Structure

```
crypto-liquidity-engine/
├── src/
│   ├── data_fetcher.py          # Multi-exchange order book retrieval
│   ├── orderbook_processor.py   # Bid/ask extraction
│   ├── metrics.py               # Liquidity and imbalance computation
│   ├── time_collector.py        # Time-series snapshot collector
│   └── threejs_visualizer.py    # Three.js WebGL 3D surface generator
├── output/                      # Generated outputs
├── main.py                      # Pipeline entry point
├── requirements.txt
└── README.md
```

## License

MIT
