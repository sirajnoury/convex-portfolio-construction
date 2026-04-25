"""Download, cache, and clean price data for the portfolio universe."""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

### Paths resolved
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

### 30-stock universe, ~3 per sector
UNIVERSE = {
    "Energy":                 ["XOM", "CVX", "COP"],
    "Materials":              ["APD", "SHW", "ECL"],
    "Industrials":            ["CAT", "HON", "UNP"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD"],
    "Consumer Staples":       ["PG", "KO", "WMT"],
    "Health Care":            ["JNJ", "UNH", "PFE"],
    "Financials":             ["JPM", "BAC", "GS"],
    "Information Technology": ["AAPL", "MSFT", "NVDA"],
    "Communication Services": ["GOOGL", "VZ", "DIS"],
    "Utilities":              ["NEE", "DUK"],
    "Real Estate":            ["AMT"],
}

TICKERS = sorted(t for tickers in UNIVERSE.values() for t in tickers)
START = "2005-01-01"

def download_prices(tickers=TICKERS, start=START, use_cache=True):
    """
    Return a (dates × tickers) DataFrame of split/dividend-adjusted closes.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / "adj_close.csv"

    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    prices = raw["Close"].copy()
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    prices.to_csv(cache_path)
    return prices

def compute_log_returns(prices):
    """
    Drops the first row (undefined) and any fully-empty rows.
    A single missing price propagates one NaN return and then recovers.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")

def checker_function(prices, returns, extreme_threshold=0.5):
    """
    Produce a readable dictionnary of the data-quality.
    """
    report = {}

    # 1. Date range and row count
    report["date_range"] = (prices.index.min(), prices.index.max())
    report["n_trading_days"] = len(prices)

    # 2. Per-ticker coverage
    report["coverage"] = prices.count().rename("n_days_observed")

    # 3. Tickers missing more than 1% of its rows?
    missing_frac = prices.isna().mean()
    report["high_missing"] = missing_frac[missing_frac > 0.01]

    # 4. Extreme daily moves (|log return| > threshold)
    extreme_counts = (returns.abs() > extreme_threshold).sum()
    report["extreme_moves"] = extreme_counts[extreme_counts > 0]

    # 5. Any non-positive prices (impossible — flags data errors)
    nonpos = (prices <= 0).sum()
    report["nonpositive_prices"] = nonpos[nonpos > 0]

    return report