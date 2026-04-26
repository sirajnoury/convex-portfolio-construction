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