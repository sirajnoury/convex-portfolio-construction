"""
Metric calculations for backtest analysis.
"""

import numpy as np
import pandas as pd

def compute_metrics(returns_series, periods_per_year=12):
    r = returns_series.dropna()
    n = len(r)

    ann_return = (1 + r).prod() ** (periods_per_year / n) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    losses = -r
    var_95 = np.quantile(losses, 0.95)
    cvar_95 = losses[losses >= var_95].mean()

    return {
        "n_periods": n,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def compute_turnover(weights_df):
    """Average per-period turnover"""
    return weights_df.diff().abs().sum(axis=1).mean()


def metrics_table(results_dict, weights_dict=None):
    """Build a comparison DataFrame"""
    rows = []
    for name, r in results_dict.items():
        m = compute_metrics(r)
        m["strategy"] = name
        if weights_dict is not None and name in weights_dict:
            m["turnover"] = compute_turnover(weights_dict[name])
        rows.append(m)

    df = pd.DataFrame(rows).set_index("strategy")
    cols = ["n_periods", "ann_return", "ann_vol", "sharpe", "max_dd", "cvar_95"]
    if weights_dict is not None:
        cols.append("turnover")
    return df[cols]


def to_markdown_table(df, formatters=None):
    """Convert a DataFrame to a table string."""
    if formatters is None:
        formatters = {}
    lines = []
    lines.append("| Strategy | " + " | ".join(df.columns) + " |")
    lines.append("|" + "|".join(["---"] * (len(df.columns) + 1)) + "|")
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            if col in formatters:
                cells.append(formatters[col](v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)