# Cross-Exchange Crypto Liquidity & Order Flow Engine

Real-time order book depth analysis across Binance, Coinbase, and Kraken with interactive 3D visualization.

## 3D Liquidity Surface

![3D Liquidity Engine](output/3d_preview.png)

Interactive version: open `output/3d_liquidity_pro.html` locally after running the pipeline.

## What It Shows

The 3D surface maps **order book liquidity** across three dimensions:

| Axis | Dimension | Description |
|------|-----------|-------------|
| **X** | Price (USDT) | BTC/USDT price range across all observed levels |
| **Z** | Exchange | Binance, Coinbase, Kraken — each as a separate surface strip |
| **Y** | Volume | Aggregated order book depth at each price level |

Peaks represent **liquidity walls** — price levels where large volumes of orders are concentrated. The color gradient maps volume intensity from dark purple (low) through blue and cyan to yellow (extreme).

## Features

- Time-series order book collection (configurable sample count and interval)
- Per-exchange depth profiles with noise suppression and ridge extraction
- Multi-exchange 3D surface with labeled axes, tick marks, and price labels
- Bid/ask indicator lines per exchange
- Hover tooltips showing exchange, price, and relative volume
- Multi-panel dashboard with imbalance sparklines
- Interactive WebGL camera (rotate, zoom, pan)

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
| `output/3d_liquidity_pro.html` | Interactive 3D liquidity surface (WebGL) |
| `output/dashboard.html` | Multi-panel dashboard with 3D view + metrics + sparklines |

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
