"""
Metrics module: computes liquidity and order flow metrics from processed order books.
"""

import pandas as pd


def compute_metrics(exchange: str, bids: pd.DataFrame, asks: pd.DataFrame) -> dict:
    """
    Compute liquidity metrics for a single exchange.

    Returns:
        Dict with: exchange, best_bid, best_ask, spread, spread_bps,
                   bid_volume, ask_volume, imbalance.
    """
    best_bid = bids["price"].iloc[0]
    best_ask = asks["price"].iloc[0]
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2
    spread_bps = (spread / mid) * 10_000

    bid_volume = bids["volume"].sum()
    ask_volume = asks["volume"].sum()
    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total > 0 else 0.0

    return {
        "exchange": exchange,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_bps": spread_bps,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "imbalance": imbalance,
    }


def compute_all_metrics(
    processed: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Compute metrics for all exchanges and return as a DataFrame.
    """
    rows = [compute_metrics(name, bids, asks) for name, (bids, asks) in processed.items()]
    return pd.DataFrame(rows).set_index("exchange")
