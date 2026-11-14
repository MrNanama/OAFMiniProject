# s03_pca_statarb v1: eigenportfolio residual stat-arb (Avellaneda & Lee 2010) on S&P 500 stocks

## Verdict
* The gross edge is real but small and fading. Net of 5 bps/side costs and the 1-day execution lag, the recommended
  v1 earns **net SR 0.20 over 2000-2020 (t = 0.92, bootstrap CI [-0.25, 0.65]) and ~0.0 over 2004-2020**. 2015-2020 is
  negative. It is not significant (DSR 0.53 with 3 distinct trials).
* The faithful Avellaneda-Lee design (daily decisions, bands 1.25 / -0.50 / +0.75) has **gross SR 0.81 but net SR
  -0.34**. It turns the book over 130x/yr, which costs 6.5%/yr against 4.8%/yr of gross alpha.
* v1 holds positions until the residual returns to equilibrium, decides weekly and puts the residuals in volume
  ("trading") time. That cuts turnover to 49x/yr (cost 2.5%/yr) at gross SR 0.72.
* It could serve as a diversifier at best: beta to SPY 0.02, correlation with the Ken French short-term-reversal
  factor 0.07. It should not be allocated to on this evidence.

## Thesis
Idiosyncratic (factor-residual) price moves of large US stocks partly reflect temporary liquidity demand and
overreaction, so they mean-revert over days to weeks (Lehmann 1990; Lo & MacKinlay 1990; Avellaneda & Lee 2010;
Blitz, Huij, Lansdorp & Verbeek 2013 on residual reversal). Inspiration was the RL Pairs Trading deck (relative value).
That deck's weakness was no breadth: 4 hand-picked pairs, chosen in-sample, traded by an RL agent that overfits. Here
every eligible stock is paired with its own factor-mimicking portfolio built from PCA eigenportfolios, and a simple
Ornstein-Uhlenbeck (OU) s-score with hysteresis replaces the RL agent. Breadth (~150 positions) replaces cleverness.

## Implementation (v1 as committed, `oafbt/strategies/s03_pca_statarb.py`)
* **Universe:** `ds.stock_universe()`, i.e. point-in-time S&P 500 member, priced, at least 252d of history, not stale,
  liquid. That gives 175 names in 1996 rising to 437 in 2020. Raw daily returns come from `ds.ret` (cleaned).
* **PCA (weekly, first trading day of each week):** standardised returns over the trailing 252 days of the stocks
  eligible that day (at least 95% of days observed). I take the top k = 15 eigenvectors of the correlation matrix,
  computed via the 252x252 Gram matrix (same subspace as an SVD, about 3x faster). Eigenportfolio weights are
  Q_ij = v_ij / sigma_i. The 15 PCs explain 58% of variance on average (43-80%).
  * The schedule uses the first trading day of each period. That is known in real time and does not depend on where
    the data stop, unlike period ends.
* **Residuals (each decision day d):** eigenportfolio returns F come from the latest PCA weights at or before d,
  applied to the trailing 60 daily returns. Each stock's 60 returns are regressed on [1, F] in one QR projection for
  all stocks, at most 3 missing returns allowed (filled with 0).
* **Trading time (A&L sec. 6, residual-level adaptation):** eps_t is multiplied by clip(10-day average volume / volume_t,
  0.25, 4) and re-centred. Idiosyncratic moves on abnormally heavy volume (news-like) get less weight than moves on thin
  volume (liquidity-like).
  * I scale the residuals rather than the raw returns so that market-wide volume surges do not leak factor moves into
    the residual.
* **OU fit:** X = cumsum(eps). Fit the AR(1) X_{k+1} = a + b X_k + zeta and derive kappa = -ln(b)*252,
  m = a/(1-b) and sigma_eq = sqrt(var(zeta)/(1-b^2)).
* **Centred s-score:** s = (X_60 - (m - <m>)) / sigma_eq, where <m> is the cross-sectional mean over valid fits.
  X_60 = 0 because the OLS residuals with an intercept sum to zero.
* **Mean-reversion filter (A&L):** kappa > 252/30, i.e. b < exp(-1/30) (half-life < 21 days, within the "< ~30d"
  brief). Fits with b <= 0 or b >= 1 are invalid.
* **Rules, evaluated weekly:**
  * Open long if s < -1.25 and open short if s > +1.25 (A&L entry band, only if the filter is passed).
  * Close at equilibrium: long when s > 0, short when s < 0.
  * Also close if the stock leaves the universe or its fit becomes invalid.
* **Book:** equal dollars per open position, with each leg scaled to gross 1, so the book is dollar-neutral at
  constant gross 2. There are ~74 longs and ~74 shorts on average (53 rising to 99). A weight row is emitted only on
  decision days when the set of positions changes. The runner applies the 1-day lag, 5 bps/side costs, 30 bps/yr
  borrow, and 10% vol targeting with the 2x stock leverage cap.
* **Hedge choice: dollar-neutral, no SPY hedge.**
  * The book's market beta is small: mean +0.004, mean |beta| 0.06, sd 0.08 (rolling 252d betas, 2000-2020). Both legs
    are diversified baskets picked on factor residuals.
  * The engine rescales the whole stock book whenever the gross of the relative weights changes, because the 2x cap
    binds on gross. An SPY leg would therefore cost about 0.5%/yr with daily rows and 0.15%/yr with weekly rows, in SPY
    trades plus rescaling churn.
  * The hedge would remove only ~1.5% of vol (in quadrature) from a 5.4%-vol book, so it does not pay. `hedge="spy"`
    is kept as an option.
* **Look-ahead and runtime:** my own truncation tests matched with 0 mismatches (cuts 2001-06-29 and 2003-03-31), and
  the built-in check passed. A full `target_weights` call takes ~10-20 s; the whole battery took 49 s.

## Results
Committed v1 (`scripts/run_strategy.py`, run 4 = a verification re-run of run 3's parameters):
```
s03_pca_statarb v1 [2000-01-03..2020-12-31, 20.97y] SR=0.20 (04-20: 0.01, 2xcost: -0.25, lag+1: 0.15) ret=1.1% vol=5.4% maxDD=-23.3% calmar=0.04 sortino=0.30 skew=0.31 turn=49.49x cost=2.5% lev=1.95 beta=0.02 t=0.92 dsr=0.53 sub=[0.48, 0.256, -0.232] CI=[-0.25,0.65] posyrs=11/21 lookahead=True (49.1s)
```

**Gross vs net, by period** (gross = before costs and borrow fees; returns are % per year at the realised ~1.95x gross
leverage):

| period | gross SR | net SR | gross ret | cost | fees | net ret | turnover |
|---|---|---|---|---|---|---|---|
| 2000-2020 | **0.72** | **0.20** | 3.87% | 2.47% | 0.29% | 1.10% | 49.5x |
| 2000-2003 | 1.16 | 0.78 | 8.38% | 2.44% | | 5.66% | 49x |
| 2004-2008 | 0.53 | 0.01 | 2.84% | 2.49% | | 0.05% | 50x |
| 2009-2014 | 0.90 | 0.30 | 4.13% | 2.44% | | 1.40% | 49x |
| 2015-2020 | 0.32 | -0.30 | 1.46% | 2.52% | | -1.36% | 50x |
| 2004-2020 | 0.58 | 0.01 | 2.81% | 2.48% | | 0.03% | 50x |

**Net return by year (%):**

| year | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| net % | 9.2 | -4.0 | 18.1 | -0.8 | 1.9 | -1.5 | -3.0 | 0.7 | 2.3 | 12.6 | 1.8 | -3.8 | -1.4 | -2.1 | 1.3 | 0.7 | 0.6 | -5.1 | -2.0 | -3.6 | 1.1 |

**All trials (4 runs, 3 distinct parameterisations; ledger `results/ledger/s03_pca_statarb.csv`).** Artefacts of each
run are in `results/runs/s03_pca_statarb/variants/`.

| run | design | net SR 00-20 | net SR 04-20 | gross SR 00-20 | turnover | cost/yr | 2x cost | lag+1 | maxDD |
|---|---|---|---|---|---|---|---|---|---|
| 1 | faithful A&L: daily, open 1.25, close long -0.50 / short +0.75 | -0.34 | -0.54 | 0.81 | 130x | 6.5% | -1.43 | -0.45 | -52.8% |
| 2 | weekly decisions, exit at equilibrium (s = 0) | -0.06 | -0.33 | 0.47 | 49x | 2.4% | -0.53 | -0.02 | -35.5% |
| 3 | run 2 + trading-time (volume) residuals | **0.20** | 0.01 | 0.72 | 49x | 2.5% | -0.25 | 0.15 | -23.3% |
| 4 | verification re-run of committed v1 (= run 3) | 0.20 | 0.01 | 0.72 | 49x | 2.5% | -0.25 | 0.15 | -23.3% |

The daily series of runs 3 and 4 are identical. The pairwise correlation of the variants' net daily returns is
0.55-0.73.

## Diagnostics I looked at
1. **s-score and OU filter.**
   * s is close to N(0, 1): mean -0.004, sd 0.97, and 21% of names sit beyond ±1.25 each week.
   * The kappa filter passes **99% of fits**. Median b is 0.87, a half-life of ~5 days.
   * The reason is structural. With an intercept, X starts near 0 and is pinned at X_60 = 0 (a Brownian bridge), and
     AR(1) estimates on 59 points are biased down (Dickey-Fuller bias). So even a pure random walk looks "fast mean
     reverting".
   * In effect, the s-score is a linearly weighted 60-day residual-reversal statistic, and the filter is not
     selective.
2. **Signal decay of the entry signal (run-1 configuration).** I measured the equal-weighted long-bucket minus
   short-bucket spread for |s| > 1.25, in bps per day after a decision at close t.
   * Day t+1 is lost to the execution lag and carries 5.8 bps.
   * Days t+2 onward accrue a cumulative 11.5 / 17.2 / 24.3 / 37.0 bps by 5 / 10 / 20 / 30 days.
   * A round trip costs ~20 bps per unit of spread (5 bps x 2 sides x 2 legs), plus re-weighting churn.
   * A&L's exits (average hold 9 days) give the costs back before the edge is earned. That motivated run 2.
   * **Regime split:** 2000-08 accrued 22.6 / 36.8 / 56.0 bps at 10 / 20 / 30 days; 2009-20 accrued only 13.0 / 14.9 /
     22.5 bps. **After 2008 the edge per round trip is at or below the cost under any band or schedule choice.** That is
     the core reason the strategy is marginal.
3. **Turnover map of alternative rules on the same s-scores (weights only, no P&L).**

   | rule | turnover | average hold | positions |
   |---|---|---|---|
   | daily, A&L bands | 135x | 9 days | 114 |
   | daily, exit at 0 | 73x | 17 days | 166 |
   | weekly, A&L bands | 79x | 14 days | 103 |
   | weekly, exit at 0 (v1) | 49x | 23 days | 148 |
   | daily, exit at the opposite band | 27x | 48 days | 320 |

   I rejected the opposite-band exit: positions older than 30 days earned ~0 bps/day in run 1's position-age profile.
4. **Turnover sources in run 1.** About 80% comes from entries and exits: ~2,950 opens/yr, each worth ~1/52 of a leg
   in and out, which is about 113x of the 137x weight turnover. About 20% is equal-weight re-normalisation when
   position counts change; the engine cap normalises gross, so that churn is unavoidable with equal-dollar legs. Daily
   drift correction adds about 6x/yr.
5. **Book beta and hedge cost:** see the hedge choice above. The book beta moved by 0.032 per day in run 1's daily
   design, so an SPY hedge would trade ~8x/yr. In v1 it moves 0.038 per weekly row, ~2x/yr.
6. **Factor attribution of gross returns** (daily regression on the Ken French short-term-reversal factor ST_Rev,
   Mkt, SMB and HML, 2000-2020):
   * Run 1: correlation with ST_Rev 0.29.
   * v1: correlation with ST_Rev 0.07, R² 0.02, gross alpha 3.45%/yr, residual SR 0.65.
   * The volume-time residual signal is largely orthogonal to the plain reversal factor.
7. **Drawdown:** the maximum drawdown of -23% runs from a peak on 2010-08-02 to a trough on 2020-03-18, so the
   strategy was **underwater for almost the whole 2010s**. The best years, 2000, 2002 (+18%) and 2009 (+12.6%), were
   high-dispersion recovery regimes. The worst months were about -3%.
8. **Leverage and vol:** the 2x gross cap binds (average gross leverage 1.95), so realised vol is 5.4% rather than 10%.
   Sharpe is unaffected, but return levels are not comparable with uncapped strategies.

## Weaknesses
* **Statistically insignificant and decaying.** Net SR 0.20 over 2000-2020 (t 0.92), ~0 over 2004-2020, and negative
  in 2015-2020; 11 of 21 years are positive. Run 2 and the hold-to-equilibrium idea were chosen from an **in-sample**
  decay diagnostic of run 1, a mild selection bias on top of the 3 logged trials.
* **Cost-fragile:** at 2x costs the SR is -0.25. The faithful A&L version loses outright.
* **Survivorship bias:** the Yahoo stock universe lacks most delisted S&P 500 members (272 of 500 members covered in
  1996, 453 in 2020; the data audit estimates +1.7%/yr for long-only EW). Buying recent losers benefits from bankruptcies
  being absent, so the true net Sharpe is probably lower. The data audit says to treat stock long-short SR differences
  below ~0.2 as noise.
* **The OU model is not doing what it claims:** the kappa filter is non-binding (see Diagnostics 1). The signal is a
  weighted residual-reversal statistic with a model veneer.
* **Weekly decisions** forgo the fastest part of the edge (days 1-5 after a signal). The execution lag alone costs
  ~15% of the 30-day edge.
* **The volume proxy for news is crude:** there are no earnings or news dates. Market-wide volume surges deflate all
  residuals equally.
* **Low absolute return** (1.1%/yr at 5.4% vol) because of the 2x leverage cap.
* **Borrow cost** is a flat 30 bps/yr. Hard-to-borrow names in the short leg (recent winners) could cost more.

## Improvement ideas for the next round
1. **News-free reversal filter** (Da, Liu & Schaumburg 2014: reversal is concentrated in moves without news). Skip
   entries for stocks with an abnormal-volume spike (e.g. above 3x the 20-day average) or a gap move in the last few
   days, as an earnings or news proxy. Better still, use scheduled earnings dates if they can be sourced point-in-time.
   This should raise gross alpha per trade without adding turnover.
2. **Asymmetric schedule:** enter on daily signals, which captures days 1-5 of the edge, but evaluate exits only weekly
   and at equilibrium. Expected turnover is ~55-60x against 49x, with more of the early edge captured.
   Alternatively, require the signal to hold for 2 days before entering, to filter noise crossings.
3. **Risk- and signal-weighted sizing:** size positions by |s| / sigma_eq, or equal-risk (1/residual vol), instead of
   equal dollars, to concentrate capital on the strongest, least noisy signals. Optionally use beta-neutral leg sizing
   in place of an SPY leg (no extra instrument and no rescaling churn).
4. **Dispersion-timed exposure:** the edge is concentrated in high cross-sectional dispersion regimes (2000-02, 2009,
   2020). Scale gross with lagged realised residual dispersion, or run the book only when dispersion is above its
   trailing median, to cut costs in the low-edge years (2011-2019).
5. **Engine-side partial rebalancing:** a no-trade band or "keep" semantics would cut the ~20-25% of turnover that comes
   from re-weighting incumbents and drift corrections (see the shared-code notes). Combined with Gârleanu-Pedersen-style
   partial trading toward the target, this is the most direct lever on the cost problem.
