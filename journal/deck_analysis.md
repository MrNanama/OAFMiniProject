# Pitch-deck analysis (4 of 8 decks provided: BQ, Macro Momentum, RL Pairs, Volume Filters)

The user asked for NEW strategies that take inspiration from the *type* of ideas in the decks (and the repo brief says
not to reuse the deck ideas). Below: what each deck did, what it actually shows, and the lesson we carry forward.

| Deck | Idea type | What they did | Reported result | Weaknesses we see | Lesson / inspiration |
|---|---|---|---|---|---|
| Bayesian Quadrature | Bayesian / probabilistic robust parameter selection for a trend signal | MACD on single commodity futures; WSABI-L Gaussian-process surrogate over (fast, slow) half-lives; ensemble weights = quadrature weight x exp(3 x backtest Sharpe); tuned 2012-16, tested 2016-23 | OOS Sharpe 0.11 +/- 0.33 across 10 random futures; uniform weights equally good (0.11); MACD(12,26) 0.01 | one asset at a time (no diversification); weighting by in-sample Sharpe adds noise; tiny edge | parameter *ensembles* help robustness; diversify across assets; prefer priors to in-sample Sharpe weights; GP uncertainty is useful |
| Principles of Macro Momentum | macro / latent-factor (PCA + ridge) cross-asset allocation | macro indicators (FRED/Yahoo/Bloomberg: growth, inflation, 3M bill, curve, FX, VIX), PCA (90% var) + ridge + rolling z-scores + momentum, vol-scaled, walk-forward; trades global equity indices + US 10Y, monthly vs weekly | Sharpe 1.28 monthly (0.82 weekly), max DD -10% (1990-2016) | FRED data are revised (latest vintage = look-ahead) and released with lags; many features vs few assets; fails in sudden crises (2008) | latent factors + ridge are good anti-overfitting tools; monthly beats weekly; use *real-time market-implied* information instead of revised macro data |
| RL in Pairs Trading | ML / RL relative value | DDQN on 4 hand-picked cointegrated pairs (JPM/BAC, WMT/TGT, CAT/DE, CVX/XOM), train 2010-20, test 2020-22, 2 bps fee; baselines = z-score bands and MACD on the ratio | noisy cumulative returns of +/-10-25%; authors flag overfitting and parameter sensitivity | 4 pairs = no breadth; pairs chosen in-sample; RL overfits; ~1 yr test | breadth beats cleverness: relative value should be run over the whole cross-section with a simple statistical model (factor residuals / OU) |
| Volume Filters | volume-conditioned price signals | after >2 sigma daily moves in 50 S&P stocks: momentum if normalised volume < 3, mean reversion if > 3; 1-day hold | Sharpe 2.07 (momentum+filter), 1.62 (reversion+filter), 2020-24 | no transaction costs with millions of 1-day trades; P&L visibly dominated by one stock; survivor universe; ~4-yr sample. Our predecessor (allsam, iters 1 & 3) found related daily signals die after costs | volume tells you *who* is trading, but the payoff must be harvested at slower horizons and across a broad, cost-aware cross-section |
| (CORPS, not provided) | cross-sectional momentum (per repo README) | - | - | - | momentum type: apply it to new objects (factors, seasonal lags, residuals) |

## Strategy map (initial ten, all NEW ideas; inspiration = deck idea *type*)

| id | strategy | universe | inspired by (type) |
|---|---|---|---|
| s01 | Gaussian-process Bayesian trend (posterior slope / uncertainty) across asset classes | multi-asset ETFs | Bayesian Quadrature (GP machinery) + momentum |
| s02 | Factor momentum: time-series momentum in equity factor returns (US + intl.) | Ken French factors | cross-sectional momentum deck (momentum type) |
| s03 | Eigenportfolio (PCA) residual stat-arb with OU s-scores | S&P 500 stocks | RL pairs trading (relative value), done with breadth |
| s04 | Abnormal-volume attention premium at monthly horizon | S&P 500 stocks | Volume Filters (volume as information) |
| s05 | Principal portfolios: cross-asset lead-lag from the SVD of the prediction matrix | Ken French portfolios | Macro Momentum (PCA/latent factors) |
| s06 | Treasury-curve PCA relative value (level-hedged slope/curvature reversion) | Treasury ETFs + yields | Macro Momentum's Salomon PCA source + pairs RV |
| s07 | Industry return seasonality (same-calendar-month) | Ken French 49 industries | momentum type (momentum at annual lags) |
| s08 | Walk-forward ML stock ranker (gradient boosting on price/volume features) | S&P 500 stocks | ML / Macro deck methodology (regularised learners, walk-forward) |
| s09 | Volatility-risk-premium harvest timed by VIX term structure | VIX ETP + SPY | wildcard (diversifier, crisis-convex counterpart to trend) |
| s10 | Turn-of-month / calendar flow effects | SPY, TLT | wildcard (flows) |
