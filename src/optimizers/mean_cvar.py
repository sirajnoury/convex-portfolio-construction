### Rockafellar Uryasev Mean-CVar portfolio optimization

import cvxpy as cp
import numpy as np
import pandas as pd

def generate_scenarios(returns, n_scenarios=5000, seed=None):
    """
    Build n_scenarios samples by randomly using past days from the 
    historical window (with replacement). The CVaR optimizer needs 
    many scenarios to evaluate worst-case losses.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(returns), size=n_scenarios, replace=True)
    return returns.values[indices]

def solve_mean_cvar(scenarios, beta=0.95, target_return=None, max_weight=0.10):
    """
    Solves the long-only Rockafellar-Uryasev LP for mean-CVaR optimization.
    Set target_return = None for pure min-CVaR, else constrains the scenario-mean
    portfolio return >= target_return.
    """
    S, n = scenarios.shape
    mu = scenarios.mean(axis=0)

    w = cp.Variable(n)
    alpha = cp.Variable()
    u = cp.Variable(S, nonneg=True)

    objective = cp.Minimize(alpha + cp.sum(u) / ((1 - beta) * S))
    constraints = [
        u >= -scenarios @ w - alpha,
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]
    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != "optimal":
        raise RuntimeError(f"Solver failed: status = {problem.status}")

    return pd.Series(w.value, index=range(n))