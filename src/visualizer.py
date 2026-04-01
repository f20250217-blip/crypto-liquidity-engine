"""
Visualizer module: generates comparison charts for cross-exchange liquidity metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(metrics: pd.DataFrame):
    """
    Create two side-by-side charts:
      1. Order flow imbalance per exchange (bar chart)
      2. Spread in basis points per exchange (bar chart)

    Saves to output/liquidity_comparison.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Cross-Exchange Liquidity Comparison — BTC/USDT", fontsize=15, fontweight="bold")

    exchanges = metrics.index.tolist()

    # --- Imbalance ---
    ax = axes[0]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in metrics["imbalance"]]
    ax.bar(exchanges, metrics["imbalance"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Order Flow Imbalance")
    ax.set_ylabel("Imbalance  (bid − ask) / total")
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.3)

    # --- Spread ---
    ax = axes[1]
    ax.bar(exchanges, metrics["spread_bps"], color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_title("Spread (basis points)")
    ax.set_ylabel("Spread (bps)")
    ax.grid(axis="y", alpha=0.3)

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/liquidity_comparison.png", dpi=150)
    print("Plot saved to output/liquidity_comparison.png")
    plt.show()
