# Cross-Exchange Crypto Liquidity & Order Flow Engine

> Comparing order book depth, spread, and flow imbalance across major crypto exchanges

## Overview

Fetches live BTC/USDT order books from Binance, Coinbase, and Kraken, computes liquidity metrics per exchange, and generates cross-exchange comparison charts.

## Features

- Real-time order book fetching from 3 exchanges via ccxt
- Bid/ask volume aggregation across top 50 levels
- Spread calculation in absolute and basis-point terms
- Order flow imbalance detection
- Cross-exchange comparison visualization

## Metrics

| Metric | Description |
|--------|-------------|
| Best Bid / Ask | Top-of-book prices |
| Spread | Ask minus bid (absolute and bps) |
| Bid / Ask Volume | Total volume across top N levels |
| Imbalance | (bid_vol - ask_vol) / total_vol |

## Installation

```bash
git clone https://github.com/<your-username>/crypto-liquidity-engine.git
cd crypto-liquidity-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Output

Console table with per-exchange metrics and `output/liquidity_comparison.png` with imbalance and spread comparison charts.

## Project Structure

```
crypto-liquidity-engine/
├── src/
│   ├── data_fetcher.py        # Multi-exchange order book retrieval
│   ├── orderbook_processor.py # Bid/ask extraction
│   ├── metrics.py             # Liquidity and imbalance computation
│   └── visualizer.py          # Comparison charts
├── output/                    # Generated plots
├── main.py                    # Pipeline entry point
├── requirements.txt
└── README.md
```

## License

MIT
