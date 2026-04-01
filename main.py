"""
Cross-Exchange Crypto Liquidity & Order Flow Engine
=====================================================
Pipeline: Fetch Order Books -> Process -> Compute Metrics -> Visualize
"""

from src.data_fetcher import fetch_all_order_books
from src.orderbook_processor import process_all
from src.metrics import compute_all_metrics
from src.visualizer import plot_metrics

LIMIT = 50  # order book depth per side


def main():
    # 1. Fetch order books
    print(f"Fetching top {LIMIT} order book levels from exchanges...")
    order_books = fetch_all_order_books(limit=LIMIT)

    if not order_books:
        print("No exchange data available. Exiting.")
        return

    # 2. Process
    processed = process_all(order_books)

    # 3. Compute metrics
    metrics = compute_all_metrics(processed)

    # 4. Print results
    print("\n" + "=" * 72)
    print("CROSS-EXCHANGE LIQUIDITY COMPARISON — BTC/USDT")
    print("=" * 72)

    for exchange in metrics.index:
        row = metrics.loc[exchange]
        print(f"\n  {exchange}")
        print(f"    Best Bid:    ${row['best_bid']:>12,.2f}")
        print(f"    Best Ask:    ${row['best_ask']:>12,.2f}")
        print(f"    Spread:      ${row['spread']:>12.4f}  ({row['spread_bps']:.2f} bps)")
        print(f"    Bid Volume:   {row['bid_volume']:>12.4f} BTC")
        print(f"    Ask Volume:   {row['ask_volume']:>12.4f} BTC")
        print(f"    Imbalance:    {row['imbalance']:>+12.4f}")

    print("\n" + "=" * 72)

    # 5. Visualize
    print("\nGenerating comparison plots...")
    plot_metrics(metrics)
    print("Done.")


if __name__ == "__main__":
    main()
