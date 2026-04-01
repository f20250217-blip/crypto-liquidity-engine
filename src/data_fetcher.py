"""
Data fetcher module: retrieves order book data from multiple exchanges via ccxt.
"""

import ccxt


# Exchange configs — symbol may differ per exchange
EXCHANGE_CONFIG = {
    "Binance": {"class": ccxt.binance, "symbol": "BTC/USDT"},
    "Coinbase": {"class": ccxt.coinbase, "symbol": "BTC/USDT"},
    "Kraken": {"class": ccxt.kraken, "symbol": "BTC/USDT"},
}


def fetch_order_book(exchange_name: str, limit: int = 50) -> dict:
    """
    Fetch order book for a single exchange.

    Args:
        exchange_name: Key from EXCHANGE_CONFIG.
        limit: Number of levels to fetch per side.

    Returns:
        Dict with keys: exchange, bids, asks.
        Each side is a list of [price, volume] pairs.

    Raises:
        RuntimeError on API failure.
    """
    config = EXCHANGE_CONFIG[exchange_name]
    try:
        exchange = config["class"]({"enableRateLimit": True})
        book = exchange.fetch_order_book(config["symbol"], limit=limit)
    except ccxt.BaseError as e:
        raise RuntimeError(f"Failed to fetch order book from {exchange_name}: {e}") from e

    return {
        "exchange": exchange_name,
        "bids": book["bids"][:limit],
        "asks": book["asks"][:limit],
    }


def fetch_all_order_books(limit: int = 50) -> list[dict]:
    """
    Fetch order books from all configured exchanges.
    Skips exchanges that fail and prints a warning.

    Returns:
        List of order book dicts (one per exchange).
    """
    results = []
    for name in EXCHANGE_CONFIG:
        try:
            book = fetch_order_book(name, limit=limit)
            results.append(book)
            print(f"  {name}: {len(book['bids'])} bid levels, {len(book['asks'])} ask levels")
        except RuntimeError as e:
            print(f"  {name}: SKIPPED — {e}")
    return results
