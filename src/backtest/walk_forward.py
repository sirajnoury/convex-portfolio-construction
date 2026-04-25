"""
Walk-forward backtesting harness.
"""

import numpy as np
import pandas as pd

def get_rebalance_dates(returns, estimation_months=36):
    """
    Return month-end dates from the data, skipping the first estimation_months.
    """
    month_ends = returns.resample("ME").last().index
    return month_ends[estimation_months:]

def walk_forward_backtest(returns, optimizer_fn, estimation_months=36):
    """
    For each rebalanced date, an estimate, optimize, hold for one month.
    """
    rebalance_dates = get_rebalance_dates(returns, estimation_months)

    rows = []
    for i in range(len(rebalance_dates) - 1):
        decision_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]

        window_start = decision_date - pd.DateOffset(months=estimation_months)
        history = returns.loc[window_start:decision_date]
        weights = optimizer_fn(history)

        holding_period = returns.loc[decision_date:next_date].iloc[1:]
        period_simple_returns = np.exp(holding_period) - 1
        portfolio_return = (period_simple_returns @ weights).sum()

        rows.append({
            "decision_date": decision_date,
            "next_date": next_date,
            "return": portfolio_return,
            "weights": weights,
        })

    return pd.DataFrame(rows).set_index("decision_date")

def plot_backtest_results(results, save_path=None):
    """
    Plot cumulative returns and drawdowns for 1+ strategies.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, returns_series in results.items():
        cum = (1 + returns_series).cumprod()
        dd = cum / cum.cummax() - 1
        ax1.plot(cum.index, cum.values, linewidth=2, label=label)
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, label=label)

    ax1.set_ylabel("Cumulative return ($1 → ?)")
    ax1.set_title("Walk-forward backtest")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig