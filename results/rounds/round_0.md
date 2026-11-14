# Round 0 leaderboard (dev data only, net of costs)

score = 0.5*SR(2004-2020) + 0.5*min(SR 2004-2020, SR 2x cost, SR lag+1)

| strategy | version | score | SR_04_20 | SR_dev | SR_2xcost | SR_lag+1 | ann_ret | vol | maxDD | calmar | turnover | cost | beta | sub | trials | lookahead | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s02_factor_momentum | v1 | 0.36 | 0.36 | 1.06 | 0.76 | 1.03 | 8.6% | 8.0% | -28.6% | 0.30 | 23.43 | 2.3% | 0.00 | [1.252, 1.641, 0.327] | 1.00 | True | 1975-01-02 |
| s10_turn_of_month | v1 | 0.19 | 0.23 | 0.23 | 0.15 | 0.45 | 1.1% | 4.8% | -10.8% | 0.09 | 18.34 | 0.4% | 0.09 | [0.089, 0.307, 0.275] | 1.00 | True | 2004-01-02 |
| s01_gp_trend | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s03_pca_statarb | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s04_volume_attention | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s05_principal_portfolios | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s06_curve_pca_rv | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s07_industry_seasonality | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s08_ml_stock_ranker | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| s09_vrp_timing | - | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |

## Weekly return correlations 2004-2020

|                     |   s02_factor_momentum |   s10_turn_of_month |
|:--------------------|----------------------:|--------------------:|
| s02_factor_momentum |                     1 |                  -0 |
| s10_turn_of_month   |                    -0 |                   1 |

## Equal-risk combination of all active strategies (2005-2020, weights from trailing data)
Sharpe 0.34, ann.ret 3.6%, vol 10.6%, maxDD -24.1%, calmar 0.13, subperiods [0.489, 0.131, 0.386], positive years 10/16
