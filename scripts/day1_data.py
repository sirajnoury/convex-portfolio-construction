"""
Day 1: pull data, compute log returns, run the data-quality checker.
"""

import sys
from pathlib import Path

# Make `src` importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import download_prices, compute_log_returns, checker_function


def main():
    prices = download_prices()
    returns = compute_log_returns(prices)
    report = checker_function(prices, returns)

    print(f"Prices:  {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"Returns: {returns.shape[0]} days × {returns.shape[1]} tickers")
    print(f"Date range: {report['date_range'][0].date()} → {report['date_range'][1].date()}")
    print()
    print("Coverage:")
    print(report["coverage"].describe())
    print()
    print(f"High-missing tickers: {list(report['high_missing'].index) or 'none'}")
    print(f"Extreme-move tickers: {list(report['extreme_moves'].index) or 'none'}")
    print(f"Non-positive prices:  {list(report['nonpositive_prices'].index) or 'none'}")


if __name__ == "__main__":
    main()