"""
Minimum variance portfolio optimization.
"""

import cvxpy as cp
import numpy as np
import pandas as pd

def solve_min_variance(cov, mu=None, target_return=None, max_weight=0.10):
    """
    Solve the long-only minimum variance. When mu and target_return are 
    given: mu @ w >= target_return.
    """
    n = cov.shape[0]
    Sigma = cov.values

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]
    if mu is not None and target_return is not None:
        constraints.append(mu.values @ w >= target_return)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != "optimal":
        raise RuntimeError(f"Solver failed: status = {problem.status}")

    return pd.Series(w.value, index=cov.index)

def validate_weights(weights, max_weight = 0.1, tol=1e-6):
    """
    Check that weights sum to 1, are in [0, max_weight].
    """
    weights = weights.clip(lower=0, upper=max_weight)
    weights = weights / weights.sum()

    if not np.isclose(weights.sum(), 1.0, atol=tol):
        raise ValueError(f"weights sum to {weights.sum()}, not 1")
    if (weights < -tol).any():
        raise ValueError(f"negative weights found: {weights[weights < -tol]}")
    if (weights > max_weight + tol).any():
        raise ValueError(f"weights above cap: {weights[weights > max_weight + tol]}")
    return weights

def equal_weight_portfolio(cov):
    """
    Returns the 1/N equally weighted portfolio.
    """
    n = cov.shape[0]
    return pd.Series(np.ones(n) / n, index=cov.index)

def efficient_frontier(mu, cov, n_points=30, max_weight=0.10):
    """
    Takes mu and cov and determines the feasible target return range
    returning a DataFrame of (target, achieved returns, vol, weights).
    """
    r_min = mu.min()
    r_max = max_weight * mu.nlargest(int(1 / max_weight)).sum()
    targets = np.linspace(r_min, r_max * 0.999, n_points)

    rows = []
    for target in targets:
        try:
            w = solve_min_variance(cov, mu=mu, target_return=target, max_weight=max_weight)
            w = validate_weights(w, max_weight=max_weight)
            rows.append({
                "target_return": target,
                "achieved_return": float(mu @ w),
                "volatility": float(np.sqrt(w.values @ cov.values @ w.values)),
                "weights": w,
            })
        except RuntimeError:
            continue
    return pd.DataFrame(rows)

def plot_efficient_frontier(frontier, save_path=None, title="Efficient Frontier"):
    """
    Plots the portoflio's efficient frontier.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier["volatility"], frontier["achieved_return"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig