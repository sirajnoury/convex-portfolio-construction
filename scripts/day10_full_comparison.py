"""
Day 10: full walk-forward comparison of all 4 strategies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data import download_prices, compute_log_returns
from src.estimators import compute_sample_cov, compute_ledoit_wolf_cov
from src.optimizers.min_variance import solve_min_variance, validate_weights, equal_weight_portfolio
from src.optimizers.mean_cvar import generate_scenarios, solve_mean_cvar
from src.backtest.walk_forward import walk_forward_backtest, plot_backtest_results


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


def main():
    prices = download_prices()
    returns = compute_log_returns(prices)

    print('running 1/4: equal-weight...')
    ew = walk_forward_backtest(returns, eq_weight)
    print('running 2/4: min-var sample...')
    mv = walk_forward_backtest(returns, min_var_sample)
    print('running 3/4: min-var Ledoit-Wolf...')
    mv_lw = walk_forward_backtest(returns, min_var_lw)
    print('running 4/4: min-CVaR (this takes longest)...')
    mc = walk_forward_backtest(returns, min_cvar)

    results = {
        '1/N equal-weight': ew['return'],
        'Min-var (sample)': mv['return'],
        'Min-var (LW)': mv_lw['return'],
        'Min-CVaR (95%)': mc['return'],
    }

    plot_backtest_results(results, save_path='figures/day10_full_comparison.png')

    print()
    print(f'{"strategy":25s}  {"return":>8s}  {"vol":>7s}  {"Sharpe":>7s}  {"max DD":>8s}  {"CVaR95":>8s}')
    for name, r in results.items():
        ann_r = (1 + r).prod() ** (12 / len(r)) - 1
        ann_v = r.std() * np.sqrt(12)
        cum = (1 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        losses = -r
        var = np.quantile(losses, 0.95)
        cvar = losses[losses >= var].mean()
        print(f'{name:25s}  {ann_r*100:7.2f}%  {ann_v*100:6.2f}%  {ann_r/ann_v:7.2f}  {max_dd*100:7.2f}%  {cvar*100:7.2f}%')


if __name__ == "__main__":
    main()