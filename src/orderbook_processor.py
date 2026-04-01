"""
Order book processor: extracts structured bid/ask data from raw order books.
"""

import pandas as pd


def extract_sides(order_book: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw bid/ask lists into DataFrames.

    Args:
        order_book: Dict with 'bids' and 'asks' as [[price, volume], ...].

    Returns:
        (bids_df, asks_df) each with columns: price, volume.
    """
    # Some exchanges return extra columns (e.g. Kraken adds timestamp) — take first two
    bids_raw = [row[:2] for row in order_book["bids"]]
    asks_raw = [row[:2] for row in order_book["asks"]]
    bids = pd.DataFrame(bids_raw, columns=["price", "volume"])
    asks = pd.DataFrame(asks_raw, columns=["price", "volume"])
    return bids, asks


def process_all(order_books: list[dict]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Process order books for all exchanges.

    Returns:
        Dict mapping exchange name to (bids_df, asks_df).
    """
    return {ob["exchange"]: extract_sides(ob) for ob in order_books}
