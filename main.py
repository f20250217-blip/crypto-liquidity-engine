"""
Cross-Exchange Crypto Liquidity & Order Flow Engine
=====================================================
Pipeline: Collect Time-Series -> Compute Metrics -> Build 3D Visualization
"""

from src.data_fetcher import fetch_all_order_books
from src.orderbook_processor import process_all
from src.metrics import compute_all_metrics
from src.time_collector import collect_snapshots
from src.threejs_visualizer import generate_threejs

# --- Configuration ---
LIMIT = 50                # order book depth per side
N_SAMPLES = 30            # snapshots to collect
INTERVAL_SEC = 10         # seconds between samples


def run():
    """Collect time-series order book data and build 3D visualization."""
    data = collect_snapshots(
        n_samples=N_SAMPLES,
        interval_sec=INTERVAL_SEC,
        limit=LIMIT,
    )

    if not data["timestamps"]:
        print("No data collected. Exiting.")
        return

    # Print latest snapshot metrics
    print("\nFetching final snapshot for summary...")
    order_books = fetch_all_order_books(limit=LIMIT)
    if order_books:
        processed = process_all(order_books)
        metrics = compute_all_metrics(processed)
        _print_metrics(metrics)

    print("\nBuilding 3D liquidity engine...")
    generate_threejs(data)
    print("\nOutputs:")
    print("  Interactive 3D  → output/3d_liquidity_pro.html")
    print("  Dashboard       → output/dashboard.html")


def _print_metrics(metrics):
    """Print formatted metrics table."""
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


def main():
    run()
    print("Done.")


if __name__ == "__main__":
    main()
