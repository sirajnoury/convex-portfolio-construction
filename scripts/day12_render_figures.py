"""
From Claude Code: re-render all 6 README figures 
in a consistent style.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import download_prices, compute_log_returns, UNIVERSE
from src.estimators import compute_sample_cov, compute_mu, cov_to_corr
from src.optimizers.min_variance import (
    solve_min_variance, validate_weights, equal_weight_portfolio, efficient_frontier,
)
from src.optimizers.mean_cvar import generate_scenarios, solve_mean_cvar
from src.backtest.walk_forward import walk_forward_backtest
from src.plotting import setup_style, COLORS, pct_formatter, save_fig


FIG_DIR = Path("figures")


def figure_1_correlation_heatmap(returns):
    """Sector-ordered correlation heatmap."""
    sector_order = list(UNIVERSE.keys())
    ordered_tickers = [t for s in sector_order for t in UNIVERSE[s] if t in returns.columns]
    cov = compute_sample_cov(returns)
    corr = cov_to_corr(cov).loc[ordered_tickers, ordered_tickers]

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.7}, linewidths=0)
    ax.set_title("Asset correlation matrix (2005–2026), sector-ordered")
    save_fig(fig, FIG_DIR / "fig01_correlation_heatmap.png")
    plt.close(fig)


def figure_2_efficient_frontier(mu, cov):
    """Efficient frontier with annotated min-variance point."""
    frontier = efficient_frontier(mu, cov, n_points=30)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier["volatility"], frontier["achieved_return"], "o-",
            color=COLORS["min_var"], linewidth=2, markersize=5)

    # Annotate the min-variance corner
    min_idx = frontier["volatility"].idxmin()
    mv_x = frontier["volatility"].iloc[min_idx]
    mv_y = frontier["achieved_return"].iloc[min_idx]
    ax.annotate("Global min-variance",
                xy=(mv_x, mv_y), xytext=(mv_x + 0.015, mv_y + 0.005),
                arrowprops=dict(arrowstyle="->", color=COLORS["neutral"]),
                fontsize=10)

    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Efficient frontier (sample covariance, full sample)")
    ax.xaxis.set_major_formatter(pct_formatter())
    ax.yaxis.set_major_formatter(pct_formatter())
    save_fig(fig, FIG_DIR / "fig02_efficient_frontier.png")
    plt.close(fig)


def figure_3_walk_forward_comparison(results):
    """2-panel: cumulative returns + drawdowns, with crisis annotations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    color_map = {
        "1/N equal-weight": COLORS["equal_weight"],
        "Min-var (sample)": COLORS["min_var"],
        "Min-var (LW)":     COLORS["min_var_lw"],
        "Min-CVaR (95%)":   COLORS["min_cvar"],
    }

    for label, returns_series in results.items():
        cum = (1 + returns_series).cumprod()
        dd = cum / cum.cummax() - 1
        c = color_map[label]
        ax1.plot(cum.index, cum.values, linewidth=2, label=label, color=c)
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color=c, label=label)

    # Crisis markers
    for date, label in [("2008-09-15", "Lehman"), ("2020-03-01", "COVID")]:
        for ax in (ax1, ax2):
            ax.axvline(pd.Timestamp(date), color=COLORS["neutral"],
                       linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.text(pd.Timestamp(date), ax2.get_ylim()[0] * 0.95, label,
                 fontsize=9, ha="left", va="bottom", color=COLORS["neutral"])

    ax1.set_ylabel("Cumulative return ($1 → ?)")
    ax1.set_title("Walk-forward backtest, 2008–2026")
    ax1.legend(loc="upper left")
    ax1.axhline(1.0, color=COLORS["neutral"], linestyle=":", linewidth=0.5)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(pct_formatter())

    save_fig(fig, FIG_DIR / "fig03_walk_forward_comparison.png")
    plt.close(fig)


def figure_4_crisis_2008(weights_mv, weights_cvar):
    """2x2: stacked area + top-5 lines for both strategies, 2008 window."""
    start, end = "2008-08-01", "2009-04-30"
    mv = weights_mv.loc[start:end]
    cv = weights_cvar.loc[start:end]

    sectors = {t: s for s, ts in UNIVERSE.items() for t in ts}
    sector_order = list(UNIVERSE.keys())
    sector_colors = plt.get_cmap("tab20").colors
    color_map = {s: sector_colors[i] for i, s in enumerate(sector_order)}

    ordered = [t for s in sector_order for t in UNIVERSE[s] if t in mv.columns]
    ticker_colors = [color_map[sectors[t]] for t in ordered]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)

    axes[0, 0].stackplot(mv.index, mv[ordered].T.values, colors=ticker_colors)
    axes[0, 0].set_title("Min-variance: full allocation")
    axes[0, 0].set_ylabel("Weight")
    axes[0, 0].set_ylim(0, 1)

    top_mv = mv.mean().sort_values(ascending=False).head(5).index
    for t in top_mv:
        axes[0, 1].plot(mv.index, mv[t], linewidth=2, label=t)
    axes[0, 1].set_title("Min-variance: top 5 holdings")
    axes[0, 1].set_ylabel("Weight")
    axes[0, 1].set_ylim(0, 0.105)
    axes[0, 1].legend(loc="lower right", ncol=2)

    axes[1, 0].stackplot(cv.index, cv[ordered].T.values, colors=ticker_colors)
    axes[1, 0].set_title("Min-CVaR: full allocation")
    axes[1, 0].set_ylabel("Weight")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylim(0, 1)

    top_cv = cv.mean().sort_values(ascending=False).head(5).index
    for t in top_cv:
        axes[1, 1].plot(cv.index, cv[t], linewidth=2, label=t)
    axes[1, 1].set_title("Min-CVaR: top 5 holdings")
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylim(0, 0.105)
    axes[1, 1].legend(loc="lower right", ncol=2)

    fig.suptitle("Allocation evolution during 2008 crisis", fontsize=14, y=1.00)
    plt.tight_layout()
    save_fig(fig, FIG_DIR / "fig04_crisis_2008_allocations.png")
    plt.close(fig)


def figure_5_beta_sensitivity(beta_summary, beta_weights):
    """2-panel: weights bar chart + risk metrics by beta."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    top_tickers = beta_weights.mean().sort_values(ascending=False).head(10).index.tolist()
    betas = beta_summary["beta"].values
    width = 0.15
    x = np.arange(len(top_tickers))
    palette = [COLORS["equal_weight"], COLORS["min_var"], COLORS["min_var_lw"],
               COLORS["min_cvar"], COLORS["highlight"]]

    for i, beta in enumerate(betas):
        ax1.bar(x + i * width, beta_weights.loc[beta, top_tickers].values,
                width=width, label=f"β = {beta}", color=palette[i])
    ax1.set_xticks(x + 2 * width)
    ax1.set_xticklabels(top_tickers, rotation=45)
    ax1.set_ylabel("Weight")
    ax1.set_title("Min-CVaR weights by confidence level β")
    ax1.legend(loc="upper right", ncol=1, fontsize=9)
    ax1.set_ylim(0, 0.115)

    # Right panel: only VaR and CVaR (drop expected return — it confuses the chart)
    ax2.plot(beta_summary["beta"], beta_summary["var"] * 100, "o-",
             linewidth=2, label="VaR", color=COLORS["min_var"], markersize=7)
    ax2.plot(beta_summary["beta"], beta_summary["cvar"] * 100, "s-",
             linewidth=2, label="CVaR", color=COLORS["min_cvar"], markersize=7)
    ax2.set_xlabel("Confidence level β")
    ax2.set_ylabel("Daily loss (%)")
    ax2.set_title("Tail-risk metrics vs β")
    ax2.legend(loc="upper left")

    # Annotate the two key betas with arrows for visual anchoring
    cvar_at_95 = beta_summary[beta_summary["beta"] == 0.95]["cvar"].iloc[0] * 100
    cvar_at_99 = beta_summary[beta_summary["beta"] == 0.99]["cvar"].iloc[0] * 100
    ax2.annotate(f"{cvar_at_95:.2f}%", xy=(0.95, cvar_at_95),
                 xytext=(0.96, cvar_at_95 - 0.15), fontsize=9, color=COLORS["min_cvar"])
    ax2.annotate(f"{cvar_at_99:.2f}%", xy=(0.99, cvar_at_99),
                 xytext=(0.98, cvar_at_99 - 0.2), fontsize=9, color=COLORS["min_cvar"])

    plt.tight_layout()
    save_fig(fig, FIG_DIR / "fig05_beta_sensitivity.png")
    plt.close(fig)


def figure_6_turnover(weights_mv, weights_cvar, returns):
    """Bar chart: avg monthly turnover by strategy."""
    n_assets = weights_mv.shape[1]
    eq_weights_const = pd.DataFrame(np.ones_like(weights_mv) / n_assets,
                                    index=weights_mv.index, columns=weights_mv.columns)

    turnover_mv = weights_mv.diff().abs().sum(axis=1).mean()
    turnover_cvar = weights_cvar.diff().abs().sum(axis=1).mean()
    turnover_eq = eq_weights_const.diff().abs().sum(axis=1).mean()

    strategies = ["1/N equal-weight", "Min-variance", "Min-CVaR (95%)"]
    values = [turnover_eq * 100, turnover_mv * 100, turnover_cvar * 100]
    colors = [COLORS["equal_weight"], COLORS["min_var"], COLORS["min_cvar"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, values, color=colors)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=11)

    ax.set_ylabel("Average monthly turnover")
    ax.set_title("Portfolio turnover by strategy (2008–2026)")
    ax.yaxis.set_major_formatter(pct_formatter())

    save_fig(fig, FIG_DIR / "fig06_turnover_comparison.png")
    plt.close(fig)


def main():
    setup_style()
    FIG_DIR.mkdir(exist_ok=True)

    print("loading data...")
    prices = download_prices()
    returns = compute_log_returns(prices)
    mu = compute_mu(returns)
    cov = compute_sample_cov(returns)

    print("loading cached backtest data...")
    weights_mv = pd.read_csv("data/processed/day11_weights_min_var.csv",
                             index_col=0, parse_dates=True)
    weights_cvar = pd.read_csv("data/processed/day11_weights_min_cvar.csv",
                               index_col=0, parse_dates=True)
    beta_summary = pd.read_csv("data/processed/day11_beta_sensitivity.csv")
    beta_weights = pd.read_csv("data/processed/day11_beta_weights.csv", index_col=0)

    # We need backtest returns for figure 3 — re-run
    def min_var(history):
        return validate_weights(solve_min_variance(compute_sample_cov(history)))

    def min_var_lw_optim(history):
        from src.estimators import compute_ledoit_wolf_cov
        return validate_weights(solve_min_variance(compute_ledoit_wolf_cov(history)))

    def eq_weight(history):
        return equal_weight_portfolio(compute_sample_cov(history))

    def min_cvar(history):
        scenarios = generate_scenarios(history, n_scenarios=2000, seed=42)
        w = solve_mean_cvar(scenarios, beta=0.95)
        w.index = history.columns
        return validate_weights(w)

    print("running 4 backtests for cumulative-return figure...")
    print("  1/4 equal-weight")
    ew = walk_forward_backtest(returns, eq_weight)
    print("  2/4 min-var")
    mv = walk_forward_backtest(returns, min_var)
    print("  3/4 min-var LW")
    mvlw = walk_forward_backtest(returns, min_var_lw_optim)
    print("  4/4 min-CVaR")
    mc = walk_forward_backtest(returns, min_cvar)

    results = {
        "1/N equal-weight": ew["return"],
        "Min-var (sample)": mv["return"],
        "Min-var (LW)":     mvlw["return"],
        "Min-CVaR (95%)":   mc["return"],
    }

    print("rendering figures...")
    figure_1_correlation_heatmap(returns)
    print("  1/6 done")
    figure_2_efficient_frontier(mu, cov)
    print("  2/6 done")
    figure_3_walk_forward_comparison(results)
    print("  3/6 done")
    figure_4_crisis_2008(weights_mv, weights_cvar)
    print("  4/6 done")
    figure_5_beta_sensitivity(beta_summary, beta_weights)
    print("  5/6 done")
    figure_6_turnover(weights_mv, weights_cvar, returns)
    print("  6/6 done")

    print("all figures saved to figures/fig01_*.png through fig06_*.png")


if __name__ == "__main__":
    main()