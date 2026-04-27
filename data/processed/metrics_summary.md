# Project metrics summary

## Walk-forward backtest, 2008–2026

| Strategy | n_periods | ann_return | ann_vol | sharpe | max_dd | cvar_95 | turnover |
|---|---|---|---|---|---|---|---|
| 1/N equal-weight | 219.0 | 16.43% | 14.81% | 1.11 | -26.98% | 8.35% | nan% |
| Min-var (sample) | 219.0 | 11.49% | 11.89% | 0.97 | -19.24% | 7.37% | 5.83% |
| Min-var (LW) | 219.0 | 11.49% | 11.90% | 0.97 | -19.21% | 7.38% | 5.76% |
| Min-CVaR (95%) | 219.0 | 11.39% | 11.90% | 0.96 | -21.75% | 7.35% | 26.40% |

## Crisis-period cumulative returns

| Strategy | 1/N equal-weight | Min-CVaR (95%) | Min-var (LW) | Min-var (sample) |
|---|---|---|---|---|
| 2008 crisis (Sep 2008 - Mar 2009) | 5.19% | -6.82% | -4.58% | -4.28% |
| 2020 COVID (Feb - Apr 2020) | 5.60% | 2.75% | -0.23% | -0.30% |

## β-sensitivity (single 36-month window, most recent)

See `data/processed/day11_beta_sensitivity.csv`.
