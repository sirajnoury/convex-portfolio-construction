"""
Day 11: visualize allocation evolution through 2008 and 2020 crises.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_prices, compute_log_returns, UNIVERSE
from src.estimators import compute_sample_cov
from src.optimizers.min_variance import solve_min_variance, validate_weights
from src.optimizers.mean_cvar import generate_scenarios, solve_mean_cvar
from src.backtest.walk_forward import walk_forward_backtest


# Helpers to plug each strategy into backtest
def min_var(history):
    return validate_weights(solve_min_variance(compute_sample_cov(history)))

def min_cvar(history):
    scenarios = generate_scenarios(history, n_scenarios=2000, seed=42)
    w = solve_mean_cvar(scenarios, beta=0.95)
    w.index = history.columns
    return validate_weights(w)


def extract_weights_df(backtest_result, tickers):
    weights_per_date = {date: ws for date, ws in backtest_result["weights"].items()}
    return pd.DataFrame(weights_per_date).T[tickers]


def ticker_to_sector():
    """
    Build a {ticker: sector}
    """
    out = {}
    for sector, tickers in UNIVERSE.items():
        for t in tickers:
            out[t] = sector
    return out


def plot_crisis(weights_mv, weights_cvar, start, end, crisis_name, save_path):
    sectors = ticker_to_sector()
    sector_colors = plt.get_cmap("tab20").colors

    # Slice to crisis window
    mv = weights_mv.loc[start:end]
    cv = weights_cvar.loc[start:end]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    # Sort tickers by sector for a coherent stacked-area appearance
    sector_order = list(UNIVERSE.keys())
    ordered_tickers = []
    for s in sector_order:
        ordered_tickers.extend([t for t in UNIVERSE[s] if t in mv.columns])

    color_map = {sector: sector_colors[i] for i, sector in enumerate(sector_order)}
    ticker_colors = [color_map[sectors[t]] for t in ordered_tickers]

    # Top-left: Min-var stacked area
    axes[0, 0].stackplot(mv.index, mv[ordered_tickers].T.values, colors=ticker_colors, labels=ordered_tickers)
    axes[0, 0].set_title("Min-variance: full allocation")
    axes[0, 0].set_ylabel("Weight")
    axes[0, 0].set_ylim(0, 1)

    # Top-right: Min-var top 5
    avg_weights_mv = mv.mean().sort_values(ascending=False).head(5)
    for ticker in avg_weights_mv.index:
        axes[0, 1].plot(mv.index, mv[ticker], linewidth=2, label=ticker)
    axes[0, 1].set_title("Min-variance: top 5 holdings")
    axes[0, 1].set_ylabel("Weight")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 0.105)

    # Bottom-left: CVaR stacked area
    axes[1, 0].stackplot(cv.index, cv[ordered_tickers].T.values, colors=ticker_colors, labels=ordered_tickers)
    axes[1, 0].set_title("Min-CVaR: full allocation")
    axes[1, 0].set_ylabel("Weight")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Date")

    # Bottom-right: CVaR top 5
    avg_weights_cv = cv.mean().sort_values(ascending=False).head(5)
    for ticker in avg_weights_cv.index:
        axes[1, 1].plot(cv.index, cv[ticker], linewidth=2, label=ticker)
    axes[1, 1].set_title("Min-CVaR: top 5 holdings")
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 0.105)

    fig.suptitle(f"Allocation evolution during {crisis_name}", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def main():
    prices = download_prices()
    returns = compute_log_returns(prices)
    tickers = sorted(returns.columns)

    print("running min-variance backtest...")
    mv_result = walk_forward_backtest(returns, min_var)
    print("running min-CVaR backtest...")
    cvar_result = walk_forward_backtest(returns, min_cvar)

    weights_mv = extract_weights_df(mv_result, tickers)
    weights_cvar = extract_weights_df(cvar_result, tickers)

    # Save weights for reproducibility
    weights_mv.to_csv("data/processed/day11_weights_min_var.csv")
    weights_cvar.to_csv("data/processed/day11_weights_min_cvar.csv")

    # Plot the two crisis windows
    plot_crisis(weights_mv, weights_cvar,
                start="2008-08-01", end="2009-04-30",
                crisis_name="2008 financial crisis",
                save_path="figures/day11_crisis_2008.png")
    plot_crisis(weights_mv, weights_cvar,
                start="2020-01-31", end="2020-05-31",
                crisis_name="2020 COVID crash",
                save_path="figures/day11_crisis_2020.png")

    print("done.")


if __name__ == "__main__":
    main()