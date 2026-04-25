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

        history = returns.loc[:decision_date]
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