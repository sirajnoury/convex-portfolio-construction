### CLAUDE CODE:

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import download_prices, compute_log_returns
from src.estimators import compute_sample_cov, compute_ledoit_wolf_cov
from src.optimizers.min_variance import (
    solve_min_variance, validate_weights, equal_weight_portfolio,
)
from src.optimizers.mean_cvar import generate_scenarios, solve_mean_cvar
from src.backtest.walk_forward import walk_forward_backtest
from src.metrics import compute_metrics, compute_turnover, metrics_table, to_markdown_table


OUT = Path("data/processed")


def min_var_sample(history):
    return validate_weights(solve_min_variance(compute_sample_cov(history)))

def min_var_lw(history):
    return validate_weights(solve_min_variance(compute_ledoit_wolf_cov(history)))

def eq_weight(history):
    return equal_weight_portfolio(compute_sample_cov(history))

def min_cvar(history):
    scenarios = generate_scenarios(history, n_scenarios=2000, seed=42)
    w = solve_mean_cvar(scenarios, beta=0.95)
    w.index = history.columns
    return validate_weights(w)


def extract_weights_df(backtest_result, tickers):
    weights_per_date = {date: ws for date, ws in backtest_result["weights"].items()}
    return pd.DataFrame(weights_per_date).T[tickers]


def percent_fmt(v):
    return f"{v*100:.2f}%"

def two_decimal_fmt(v):
    return f"{v:.2f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading data...")
    prices = download_prices()
    returns = compute_log_returns(prices)
    tickers = sorted(returns.columns)

    print("running 4 backtests...")
    print("  1/4 equal-weight")
    ew = walk_forward_backtest(returns, eq_weight)
    print("  2/4 min-var sample")
    mv = walk_forward_backtest(returns, min_var_sample)
    print("  3/4 min-var Ledoit-Wolf")
    mvlw = walk_forward_backtest(returns, min_var_lw)
    print("  4/4 min-CVaR")
    mc = walk_forward_backtest(returns, min_cvar)

    results = {
        "1/N equal-weight": ew["return"],
        "Min-var (sample)": mv["return"],
        "Min-var (LW)":     mvlw["return"],
        "Min-CVaR (95%)":   mc["return"],
    }

    weights_dict = {
        "Min-var (sample)": extract_weights_df(mv, tickers),
        "Min-var (LW)":     extract_weights_df(mvlw, tickers),
        "Min-CVaR (95%)":   extract_weights_df(mc, tickers),
    }

    # ---------- Main comparison table ----------
    print("building main metrics table...")
    table_main = metrics_table(results, weights_dict=weights_dict)
    table_main.to_csv(OUT / "metrics_main.csv")
    print(table_main.round(4))

    # ---------- Crisis period performance ----------
    print("building crisis-period metrics...")
    crisis_windows = {
        "2008 crisis (Sep 2008 - Mar 2009)": ("2008-09-30", "2009-03-31"),
        "2020 COVID (Feb - Apr 2020)":       ("2020-02-29", "2020-04-30"),
    }
    crisis_rows = []
    for label, (start, end) in crisis_windows.items():
        for strat, r in results.items():
            sub = r.loc[start:end]
            cum_return = (1 + sub).prod() - 1
            worst_month = sub.min()
            crisis_rows.append({
                "period": label,
                "strategy": strat,
                "cumulative_return": cum_return,
                "worst_month": worst_month,
                "n_months": len(sub),
            })
    crisis_df = pd.DataFrame(crisis_rows)
    crisis_df.to_csv(OUT / "metrics_crisis_periods.csv", index=False)
    print(crisis_df.round(4))

    # ---------- Markdown tables for README ----------
    print("writing markdown summary...")
    main_fmt = {
        "n_periods": str,
        "ann_return": percent_fmt,
        "ann_vol":    percent_fmt,
        "sharpe":     two_decimal_fmt,
        "max_dd":     percent_fmt,
        "cvar_95":    percent_fmt,
        "turnover":   percent_fmt,
    }
    md_main = to_markdown_table(table_main, formatters=main_fmt)

    crisis_pivot = crisis_df.pivot_table(
        index="period",
        columns="strategy",
        values="cumulative_return",
    )
    crisis_md_fmt = {col: percent_fmt for col in crisis_pivot.columns}
    md_crisis = to_markdown_table(crisis_pivot, formatters=crisis_md_fmt)

    md_text = "# Project metrics summary\n\n"
    md_text += "## Walk-forward backtest, 2008–2026\n\n"
    md_text += md_main + "\n\n"
    md_text += "## Crisis-period cumulative returns\n\n"
    md_text += md_crisis + "\n\n"
    md_text += ("## β-sensitivity (single 36-month window, most recent)\n\n"
                "See `data/processed/day11_beta_sensitivity.csv`.\n")

    (OUT / "metrics_summary.md").write_text(md_text)
    print(f"\nmarkdown summary written to {OUT / 'metrics_summary.md'}")
    print("done.")


if __name__ == "__main__":
    main()