"""
Day 11 section 2: vary CVaR confidence level beta and we will evaluate
how the weights and the risks shift.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_prices, compute_log_returns
from src.optimizers.mean_cvar import generate_scenarios, solve_mean_cvar


BETAS = [0.90, 0.925, 0.95, 0.975, 0.99]


def main():
    prices = download_prices()
    returns = compute_log_returns(prices)

    # Use the most recent 36-month window
    window_end = returns.index[-1]
    window_start = window_end - pd.DateOffset(months=36)
    window = returns.loc[window_start:window_end]
    print(f"window: {window.index[0].date()} to {window.index[-1].date()} ({len(window)} days)")

    # Generate scenarios once - same set used for all beta values
    scenarios = generate_scenarios(window, n_scenarios=5000, seed=42)

    # Solve at each beta
    results = []
    for beta in BETAS:
        w = solve_mean_cvar(scenarios, beta=beta)
        w.index = window.columns
        port_returns = scenarios @ w.values
        losses = -port_returns
        var = np.quantile(losses, beta)
        cvar = losses[losses >= var].mean()
        results.append({
            "beta": beta,
            "weights": w,
            "expected_return": port_returns.mean(),
            "var": var,
            "cvar": cvar,
        })

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "weights"} for r in results])
    print()
    print("scenario-based metrics by beta:")
    print(summary.to_string(index=False))
    summary.to_csv("data/processed/day11_beta_sensitivity.csv", index=False)

    # Build weights DataFrame: rows = beta, cols = tickers
    weights_df = pd.DataFrame({r["beta"]: r["weights"] for r in results}).T
    weights_df.index.name = "beta"
    weights_df.to_csv("data/processed/day11_beta_weights.csv")

    # Figure: 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: top 10 weights as grouped bars across betas
    avg_weights = weights_df.mean().sort_values(ascending=False).head(10)
    top_tickers = avg_weights.index.tolist()
    width = 0.15
    x = np.arange(len(top_tickers))
    for i, beta in enumerate(BETAS):
        ax1.bar(x + i * width, weights_df.loc[beta, top_tickers].values,
                width=width, label=f"β = {beta}")
    ax1.set_xticks(x + 2 * width)
    ax1.set_xticklabels(top_tickers, rotation=45)
    ax1.set_ylabel("Weight")
    ax1.set_title("Min-CVaR weights by confidence level")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: VaR / CVaR / expected return as functions of beta
    ax2.plot(summary["beta"], summary["var"] * 100, "o-", linewidth=2, label="VaR (daily loss)")
    ax2.plot(summary["beta"], summary["cvar"] * 100, "s-", linewidth=2, label="CVaR (daily loss)")
    ax2.set_xlabel("Confidence level β")
    ax2.set_ylabel("Daily loss (%)")
    ax2.set_title("Risk metrics vs β")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=2))

    ax2_right = ax2.twinx()
    ax2_right.plot(summary["beta"], summary["expected_return"] * 100, "^--", color="green",
                   linewidth=2, label="Expected return (daily)")
    ax2_right.set_ylabel("Daily expected return (%)", color="green")
    ax2_right.tick_params(axis="y", labelcolor="green")
    ax2_right.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("figures/day11_beta_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("done.")


if __name__ == "__main__":
    main()