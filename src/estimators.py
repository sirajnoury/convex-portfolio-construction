"""
Estimate expected returns (mu) and covariance matrices (Sigma)
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS_PER_YEAR = 252

def compute_mu(returns, annualize = True):
    """
    Returns the per assest expected returns in vector form
    """
    mu = returns.mean()
    if annualize:
        mu = mu * TRADING_DAYS_PER_YEAR
    return mu

def compute_sample_cov(returns, annualize=True):
    """
    Returns the annualized sample covariance matrix.
    """
    cov = returns.cov()
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov

def compute_ledoit_wolf_cov(returns, annualize=True):
    """
    Returns the annualized Ledoit-Wolf shrinkage covariance.
    """
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov

def cov_to_corr(cov):
    """
    Convert a covariance matrix to a correlation matrix to normalize the
    individual assets with high volatility.
    """
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def plot_correlation_heatmaps(returns, save_path=None):
    """
    Plot sample vs Ledoit-Wolf correlation heatmaps side-by-side.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sample_corr = cov_to_corr(compute_sample_cov(returns, annualize=False))
    lw_corr = cov_to_corr(compute_ledoit_wolf_cov(returns, annualize=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(sample_corr, ax=axes[0], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Sample correlation")

    sns.heatmap(lw_corr, ax=axes[1], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Ledoit-Wolf correlation")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig