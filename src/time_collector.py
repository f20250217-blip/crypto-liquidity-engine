"""
Time-series order book collector: samples order books at regular intervals
and builds a historical dataset for time-frequency analysis.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_fetcher import fetch_all_order_books, EXCHANGE_CONFIG
from src.orderbook_processor import process_all
from src.metrics import compute_metrics


def collect_snapshots(
    n_samples: int = 30,
    interval_sec: float = 10.0,
    limit: int = 50,
) -> dict:
    """
    Collect order book snapshots over time.

    Args:
        n_samples: Number of snapshots to collect.
        interval_sec: Seconds between samples.
        limit: Order book depth per side.

    Returns:
        Dict with:
          timestamps: list of datetime objects
          price_grids: dict[exchange] -> 2D array (n_samples x n_bins)
          imbalance_series: dict[exchange] -> list of floats
          exchanges: list of exchange names that responded
          price_range: (min, max) across all snapshots
    """
    timestamps = []
    # Raw storage: list of dicts per snapshot
    raw_snapshots = []
    active_exchanges = set()

    print(f"\nCollecting {n_samples} snapshots ({interval_sec}s apart)...")
    print(f"  Estimated time: {n_samples * interval_sec:.0f}s\n")

    for i in range(n_samples):
        t0 = time.time()
        ts = datetime.now()

        books = fetch_all_order_books(limit=limit)
        if not books:
            print(f"  [{i+1}/{n_samples}] No data — skipping")
            elapsed = time.time() - t0
            if elapsed < interval_sec and i < n_samples - 1:
                time.sleep(interval_sec - elapsed)
            continue

        processed = process_all(books)
        snapshot = {}
        for ex, (bids, asks) in processed.items():
            active_exchanges.add(ex)
            m = compute_metrics(ex, bids, asks)
            snapshot[ex] = {
                "bids": bids,
                "asks": asks,
                "imbalance": m["imbalance"],
                "mid_price": (m["best_bid"] + m["best_ask"]) / 2,
            }

        timestamps.append(ts)
        raw_snapshots.append(snapshot)
        print(f"  [{i+1}/{n_samples}] {ts.strftime('%H:%M:%S')} — "
              f"{len(snapshot)} exchanges")

        elapsed = time.time() - t0
        if elapsed < interval_sec and i < n_samples - 1:
            time.sleep(interval_sec - elapsed)

    exchanges = sorted(active_exchanges)
    if not timestamps:
        return {"timestamps": [], "exchanges": []}

    # Build common price grid across all snapshots
    all_prices = []
    for snap in raw_snapshots:
        for ex_data in snap.values():
            all_prices.extend(ex_data["bids"]["price"].tolist())
            all_prices.extend(ex_data["asks"]["price"].tolist())

    price_min, price_max = min(all_prices), max(all_prices)
    n_bins = 400
    price_grid = np.linspace(price_min, price_max, n_bins)

    # Build time-series matrices per exchange
    # Shape: (n_valid_samples, n_bins) — volume at each price bin at each time
    price_grids = {}
    imbalance_series = {}

    for ex in exchanges:
        matrix = np.zeros((len(timestamps), n_bins))
        imb_list = []

        for t_idx, snap in enumerate(raw_snapshots):
            if ex not in snap:
                imb_list.append(0.0)
                continue

            d = snap[ex]
            bids, asks = d["bids"], d["asks"]
            prices = np.concatenate([bids["price"].values, asks["price"].values])
            vols = np.concatenate([bids["volume"].values, asks["volume"].values])
            order = np.argsort(prices)
            prices, vols = prices[order], vols[order]

            matrix[t_idx] = np.interp(price_grid, prices, vols, left=0, right=0)
            imb_list.append(d["imbalance"])

        price_grids[ex] = matrix
        imbalance_series[ex] = imb_list

    print(f"\nCollection complete: {len(timestamps)} snapshots, "
          f"{len(exchanges)} exchanges")

    return {
        "timestamps": timestamps,
        "price_grid": price_grid,
        "price_grids": price_grids,
        "imbalance_series": imbalance_series,
        "exchanges": exchanges,
        "price_range": (price_min, price_max),
    }
