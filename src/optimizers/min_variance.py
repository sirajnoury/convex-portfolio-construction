"""
Minimum variance portfolio optimization.
"""

import cvxpy as cp
import numpy as np
import pandas as pd

def solve_min_variance(cov, max_weight = 0.1):
    """
    Solve the long only minimum variance, returns the weight.
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
