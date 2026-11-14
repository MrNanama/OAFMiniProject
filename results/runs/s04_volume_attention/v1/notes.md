# s04_volume_attention v1 - abnormal-volume (attention) premium on S&P 500 stocks

Module: `oafbt/strategies/s04_volume_attention.py` (class `VolumeAttention`, `STRATEGY = VolumeAttention`, version `v1`,
universe `stocks`).

Inspired by the Volume Filters deck, which used normalised volume on >2-sigma days to choose between 1-day momentum
and 1-day reversal in 50 stocks, with no costs. This strategy uses volume as a *visibility* signal instead. It harvests
the high-volume return premium over a month, across the whole point-in-time S&P 500, net of costs.

**Verdict: negative result. Do not allocate.** Two literature-based specifications both earn about zero gross over
2000-2020, and costs make them negative. v1 is kept as the record of the a priori specification. The diagnostics below
show why the symmetric high-minus-low design fails in this universe, and point to two better-founded redesigns
(ideas 1-2).

## Final result (dev, 2000-2020, net of costs and fees, 10% vol target under the 2x stock gross cap)

```
s04_volume_attention v1 [2000-01-03..2020-12-31, 20.97y] SR=-0.31 (04-20: -0.27, 2xcost: -0.54, lag+1: -0.33) ret=-2.0% vol=6.5% maxDD=-48.0% calmar=-0.05 sortino=-0.44 skew=0.29 turn=29.36x cost=1.5% lev=1.95 beta=0.00 t=-1.52 dsr=0.03 sub=[-0.566, -0.161, -0.187] CI=[-0.75,0.08] posyrs=7/21 lookahead=True (26.9s)
```

| metric | value | metric | value |
|---|---|---|---|
| Sharpe, net (2000-20 / 2004-20) | -0.31 / -0.27 | Sharpe, gross before costs and fees (2000-20 / 2004-20) | -0.04 / +0.03 |
| ann. return / vol | -2.0% / 6.5% | CAGR | -2.2% |
| P&L split, % NAV/yr | gross -0.25, trading cost -1.47, borrow -0.30, net -2.02 | turnover | 29.4x/yr |
| max DD | -48.0%: a 20-year bleed, not a crash (peak 2000-10-17, trough 2020-11-06) | worst / best month | -7.2% / +8.4% |
| beta / corr to SPY | 0.00 / 0.01 | HAC t (alpha t) | -1.52 (-1.54) |
| 2x costs / lag+1 | -0.54 / -0.33 | PSR(0) / DSR (2 distinct trials) | 0.08 / 0.03 |
| bootstrap 95% CI | [-0.75, 0.08] | positive years / monthly hit rate | 7/21 / 42% |
| avg gross leverage | 1.95 (the 2x cap binds; book vol is about 6.5%, below the 10% target) | avg positions | 199 (about 100 long and 99 short: the union of 4 overlapping tranches of 47-88 names each) |

The vol target cannot be reached because a dollar-neutral quintile book at gross 2 has only about 6% ex-ante vol. The
Sharpe is unaffected by this. Look-ahead: the built-in truncation check passes (0 mismatches at the cuts 2008-06-30,
2014-03-31 and 2019-09-30). Extra cuts at 2010-05-07 (Fri), 2016-11-09 (Wed) and 2020-03-20 (Fri) also give 0 mismatches.

## Thesis

* **The high-volume return premium (HVRP).** Gervais, Kaniel & Mingelgrin (2001) found that stocks with unusually high
  (low) trading volume over a day or a week appreciate (depreciate) over the following month, with no later reversal.
  Kaniel, Ozoguz & Starks (2012) report the same in most developed markets.
* **The mechanism is visibility.** A volume shock attracts attention, the investor base widens (Merton 1987 investor
  recognition), and under short-sale constraints the new demand lifts the price (Miller 1977).
* **What is new here, relative to the deck.**
  * A cross-sectional, monthly-horizon premium instead of 1-day trades.
  * The whole point-in-time S&P 500 instead of 50 survivor stocks.
  * Net of 5 bps costs plus borrow.
  * Volume is used as a stand-alone signal, neutralised so it is not momentum or reversal.

## Implementation (a weight row dated d uses only data dated <= d)

1. **Abnormal volume.**
   * `AV = log(mean V[t-20..t] / mean V[t-272..t-21])`: the last 21 days against the 252 days before them.
   * V is Yahoo share volume. It is split-adjusted: there is no step at AAPL's 7:1 split on 2014-06-09 or its 4:1 split
     on 2020-08-31. A stock's own volume ratio is therefore invariant to past and future splits.
   * Zero prints are treated as missing (0.7% of observations). The recent window needs at least 15/21 days and the
     baseline at least 201/252.
2. **Neutralisation.** On each formation date, a cross-sectional OLS over eligible stocks with complete inputs
   (`ds.stock_universe()`): `AV ~ 1 + z(r21) + |z(r21)| + z(log DV_base) + z(beta252) + z(vol21)`. The signal is the
   residual. Controls are z-scored and clipped at +-3; AV is winsorised at +-3 sd.
   * **r21 and |r21|:** the same-window log return, and its absolute value for the V-shape of volume against returns.
     GKM sort within return deciles for the same reason.
   * **DV_base:** mean dollar volume (close x volume) over the baseline window, a size and liquidity proxy. It is
     invariant to split adjustment.
   * **beta252:** beta vs SPY over 252 days, computed with vectorised cumulative sums; it matches `utils.rolling_beta`
     to within 6e-13.
   * **vol21:** 1-month realised volatility. It controls for the idiosyncratic-volatility anomaly (Ang et al. 2006),
     which would otherwise mask the effect, since AV correlates +0.24 with vol21.
   * The controls were fixed BEFORE run 1 from signal-only diagnostics (see "Diagnostics" 1). Mean R² is 0.20, and the
     residual keeps a 0.88 rank correlation with raw AV.
3. **Portfolio.**
   * On the last NYSE session of each week (`utils.period_ends(cal, "W")`, published calendar), a tranche goes long
     the top quintile and short the bottom quintile of the residual. Weights are equal, each leg sums to 1, and the
     book is dollar neutral.
   * The book is the average of the last 4 tranches: each is held 4 weeks, which diversifies timing luck.
   * At every rebalance, names no longer eligible are dropped and that tranche's leg is re-normalised.
   * Rows are emitted only on formation dates; positions drift in between.
4. **Engine.** Lag 1 (decided at Friday's close, executed at the next session's close), 5 bps one-way, 30 bps/yr
   borrow, 10% vol target, 2x gross cap.
5. **Runtime.** About 6 s per `target_weights()` call. Results are cached per (data dir, end date, params), so the full
   battery including the look-ahead check takes about 27 s.

## Development path (3 logged runs, 2 distinct parameterisations; ledger `results/ledger/s04_volume_attention.csv`)

| run | configuration | SR 00-20 | SR 04-20 | 2x cost | lag+1 | turnover | cost/yr | gross/yr | sub-period SRs |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **a priori spec**: AV 21d vs prior 252d, residual on 5 controls, 4 weekly tranches, quintile EW L/S | -0.31 | -0.27 | -0.54 | -0.33 | 29x | 1.5% | -0.25% | -0.57 / -0.16 / -0.19 |
| 2 | GKM-faithful shock: AV 5d vs prior 50d (GKM: weekly formation vs ~49-day reference), otherwise identical | -0.52 | -0.62 | -0.94 | -0.56 | 51x | 2.6% | -0.26% | -0.46 / -0.55 / -0.57 |
| 3 | = run 1 (module defaults, same params hash), final v1 re-run to restore the v1 artefacts and count trials honestly | -0.31 | -0.27 | -0.54 | -0.33 | 29x | 1.5% | -0.25% | -0.57 / -0.16 / -0.19 |

* **Why run 2.** Run 1's deciles showed the lowest-AV decile earning +0.35%/month in 2000-06. Against a 252-day
  baseline, a 21-day ratio partly measures the slow volume TREND, which predicts negatively (Haugen & Baker 1996;
  Lee & Swaminathan 2000). GKM's own design isolates a short shock against a ~2-month reference, so run 2 moves to that
  definition (1 week vs 50 days). Both choices come from the paper, not from a search.
* **Why v1 = run 1.** It is the specification fixed before any P&L was seen. It is also the better of the two on the
  round score (2004-20 SR -0.27 vs -0.62, at half the turnover). Artefacts of both variants are in
  `results/runs/s04_volume_attention/variants/`.
* **Deliberately NOT run.** The redesigns suggested by the diagnostics: long only the top-shock decile, or long both
  tails. They were found in-sample, so they are written up as pre-specified ideas below instead of being tried on the
  same data.

## Diagnostics

**1. Signal-only construction diagnostics, before run 1 (no forward returns used).**

Cross-sectional Spearman correlations of AV with each control (mean over weekly dates, sd over time in brackets):

| AV | same-window return | \|return\| | log DV_base | beta252 | vol21 |
|---|---|---|---|---|---|
| AV21 | -0.06 (0.12) | +0.18 | -0.05 | -0.00 (0.15) | +0.24 |
| AV5 | +0.01 | +0.22 | -0.03 | -0.00 (0.14) | +0.17 |

* The beta correlation averages zero but swings with sd 0.15 over time, so an uncontrolled book would carry a
  time-varying market bet. That is why beta was added.
* **Persistence.** The cross-sectional rank autocorrelation of AV21 is 0.43 / 0.27 / 0.27 / 0.12 / 0.10 / -0.02 at
  lags of 21 / 42 / 63 / 84 / 126 / 252 days. Lag 63 is no lower than lag 42, a quarterly earnings-cycle bump. For AV5
  the values are 0.50 / 0.26 / 0.20 at lags 5 / 21 / 63.
* **Index additions.** Members of less than 273 days (about 14 per date) have mean AV21 of -0.005, against -0.015 for
  the rest. Inclusion spikes do not contaminate the signal, so no tenure filter was added.
* **Construction.** Target turnover before drift is 27.7x/yr. One tranche holds 47-88 names per leg as the eligible
  universe grows from 233 to 442; the 4-tranche book holds about 72 (2000) to 137 (2020) names per leg.

**2. Decile monotonicity.**

Next-month returns, measured over days +2..+21 after the decision (that is, after the 1-day execution lag). They are
equal-weighted and market-adjusted (decile minus the mean of the eligible universe), in % per month. There are 1,091
weekly formations over 2000-2020, with Newey-West t-stats in brackets.

| signal | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D10-D1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1 residual (AV21)** | +0.16 (1.4) | -0.02 | -0.00 | -0.06 | -0.05 | -0.04 | -0.06 | -0.05 | +0.05 | +0.07 (0.8) | **-0.09 (t -0.6)** |
| raw AV21 | +0.10 | -0.08 | -0.05 | -0.12 | -0.10 | -0.08 | +0.09 | -0.04 | +0.14 | +0.16 | +0.06 (t 0.3) |
| run-2 residual (AV5/50) | +0.05 | +0.10 (2.1) | -0.02 | -0.05 | -0.02 | -0.10 (-2.2) | -0.04 | -0.04 | -0.07 | **+0.18 (3.2)** | +0.14 (t 1.2) |
| raw AV5/50 | +0.02 | +0.05 | -0.07 | -0.05 | -0.03 | -0.12 | -0.07 | -0.00 | +0.10 | +0.16 (2.0) | +0.14 (t 1.2) |

* **No monotonicity.** For the v1 signal, the Spearman correlation of decile rank with mean return is -0.07.
* **v1 sub-periods (D10-D1):** 2000-06 -0.28 (t -1.0); 2007-13 -0.15 (t -0.6); 2014-20 +0.17 (t 0.9).
* **The shape is a U.** Both tails beat the middle. Only the most extreme high-volume shocks (run 2, D10) show the GKM
  pop.

**3. Event time: does the premium appear later?**

The Q5-Q1 spread of the v1 residual, in % over each window:

| window (days after decision) | +2..+6 | +2..+21 | +22..+41 | +42..+63 | +64..+126 |
|---|---|---|---|---|---|
| Q5-Q1, % (t) | 0.00 | -0.01 (-0.1) | -0.14 (-1.3) | +0.03 | -0.45 (-1.6) |

* There is no premium at any horizon. The mild negative drift after month 3 resembles attention-driven price pressure
  that reverses (Da, Engelberg & Gao 2011).
* For the run-2 signal, the high-volume side measured against the EW universe gives the following:

| run-2 leg vs EW universe | month 1 (+2..+21) | month 2 | month 3 | months 4-6 |
|---|---|---|---|---|
| top decile | +0.18% (t 3.2) | -0.05% | +0.08% | +0.19% (t 2.0) |
| top quintile | +0.06% (t 1.3) | -0.01% | +0.01% | +0.05% |
| bottom quintile | +0.08% (t 1.5) | +0.11% (t 2.1) | +0.14% (t 2.3) | +0.30% (t 2.4) |

* **Extreme shocks get a short pop.** That is the GKM transition, as more investors notice the stock.
* **Quiet stocks earn a slow, persistent premium.** This is consistent with Merton's (1987) premium for low investor
  recognition, the neglected-firm effect.
* **So a high-minus-low book shorts the wrong thing.** The low leg should not be shorted.

**4. Long vs short leg contribution.**

Taken from the engine's held positions after vol scaling, for v1 over 2000-2020, in % of NAV per year:

| | P&L | relative to the EW eligible universe at the same notional |
|---|---|---|
| long leg (high AV) | +11.14 | **+0.20** |
| short leg (low AV) | -11.39 | **-0.43**: the shorted low-AV stocks beat the universe |
| total | gross -0.25, cost -1.47, borrow -0.30, net -2.02 | average notional 0.98 long / 0.98 short |

Run 2 is similar: long +0.32, short -0.61, gross -0.26, cost -2.57, net -3.12. In both runs the damage comes from the
short leg.

Calendar years for v1, in % (leg columns are relative to the EW universe; positive = good for the strategy):

| year | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long rel | 6.3 | -3.7 | 3.5 | -4.1 | 1.1 | -3.5 | 0.7 | -6.0 | 3.2 | 2.5 | -1.9 | 1.0 | 2.3 | -0.7 | -2.1 | 3.0 | 4.4 | -0.5 | -0.2 | 1.6 | -2.7 |
| short rel | 10.3 | -10.3 | -4.0 | -7.5 | -1.2 | -5.5 | 1.9 | 1.9 | 7.6 | -6.5 | 4.8 | 0.0 | 1.5 | -4.6 | -0.3 | 0.1 | 3.2 | 1.2 | 0.9 | -0.7 | -1.9 |
| gross | 17.0 | -13.9 | -0.2 | -11.6 | -0.1 | -9.0 | 2.5 | -4.0 | 10.4 | -3.9 | 3.0 | 0.7 | 3.8 | -5.4 | -2.4 | 3.0 | 7.7 | 0.8 | 0.4 | 0.9 | -5.0 |
| net | 15.3 | -15.7 | -2.0 | -13.4 | -1.9 | -10.8 | 0.7 | -5.7 | 8.8 | -5.3 | 1.2 | -1.2 | 1.9 | -7.1 | -4.1 | 1.2 | 5.9 | -1.0 | -1.4 | -1.0 | -6.7 |

Rolling 3-year net Sharpe: -0.9 to -1.8 in 2003-07, around 0 in 2008-12, -0.4 to -0.7 in 2013-15, +0.2 to +0.4 in
2016-18, and -0.3 / -0.5 in 2019-20.

**5. Factor exposure.**

Daily gross returns regressed on FF5 + Mom + ST_Rev, 2000-20, with HAC errors:
* **v1:** R² 0.085. Alpha -0.37%/yr (t -0.3). Loadings HML +0.11 (t 6.6), SMB -0.07 (t -4.6), RMW +0.05 (t 3.1),
  Mom -0.04 (t -3.3); market, CMA and ST_Rev are about 0. Alpha by sub-period: 2000-03 -4.3%/yr (t -1.1), 2004-20
  +0.7%/yr (t 0.6).
* **Run 2:** alpha -1.1%/yr (t -0.9), with RMW +0.11 and ST_Rev +0.05.
* **Reading.** Neutralisation did keep the book off momentum, reversal, size and market. The large 2000-03 swings are
  not factor bets; most likely they are industry clusters of volume shocks (tech), which cannot be neutralised without
  industry data.

## Weaknesses

* **The premise fails in this universe.** Gross Sharpe is about 0 in both specifications (-0.04 and +0.03 for v1), and
  no decile spread is significant. A visibility shock should matter least for S&P 500 names, which are already widely
  followed. The data have no small caps, where a visibility effect should be strongest.
* **Symmetric design against an asymmetric effect.** Both tails beat the middle deciles (diagnostics 2-3), so long-high /
  short-low cancels.
* **Costs exceed any plausible large-cap premium.** 29x turnover costs 1.5%/yr plus 0.3%/yr borrow, against a gross
  P&L of about 0. With 2x costs the Sharpe is -0.54. The GKM-style shock (run 2) is worse, at 51x turnover.
* **Earnings-cycle contamination (likely).** AV21 partly measures "the stock announced earnings in the last month" (the
  lag-63 bump). Low-AV stocks are then more likely to announce next month and collect the earnings-announcement premium
  (Frazzini & Lamont 2007). That fits the short leg beating the universe by 0.43%/yr, but it cannot be verified without
  announcement dates.
* **Unexplained year swings.** 2000 +17%, 2001 -14%, 2003 -12% gross, at a factor R² of only 0.09. The likely cause,
  industry concentration of volume shocks, cannot be removed without industry data.
* **Survivorship.** Only 759 of 1,209 historical members have Yahoo data. Missing names are mostly acquisition targets,
  which have large volume shocks, and failures. Per the data audit, long-short Sharpe differences below ~0.2 are noise.
* **Researcher degrees of freedom.** The run-2 definition and all redesign ideas below were informed by in-sample
  diagnostics. The holdout, or a different universe, is the real test.

## Improvement ideas for the next round (pre-specify each before testing)

1. **Expected-attention (earnings-announcement) premium, using volume seasonality.** Predict next month's abnormal
   volume from the same phase of the earnings cycle: average AV over the windows lagged 63/126/189/252 days and shifted
   to the coming month. Go long stocks whose volume is predicted to surge and short those predicted to be quiet. This
   is Frazzini & Lamont (2007), "The Earnings Announcement Premium and Trading Volume". It is published, it needs no
   earnings dates, the lag-63 bump confirms the seasonality in our data, and it would also explain why v1's short leg
   won. Holding is one month. The legs rotate with the quarterly cycle, so almost the whole book is replaced every
   month (about 40-50x turnover, or 2-2.5%/yr at 5 bps). The test must therefore include costs, and could hold only
   stocks predicted to announce in the coming 2-3 weeks.
2. **Asymmetric attention book.** Go long the most extreme high-volume shocks (top decile of the 5d/50d residual) for 4
   weeks, and long the quietest quintile for 3-6 months, the slow neglect premium (+0.1%/month, t 2.1-2.4 in months
   2-6). Short the EW middle of the universe (deciles 3-8) or RSP. This makes the observed U-shape the design. Expected
   net is modest: the top decile alone is about +2.3%/yr gross against about 1.3%/yr cost. It must be pre-registered
   because it was found in-sample.
3. **Industry or peer neutralisation of the volume shock.** Measure abnormal volume relative to the stock's 20
   most-correlated peers (252-day return correlation), or residualise log-volume changes on their first k principal
   components (volume co-movement). This isolates stock-specific attention and should shrink the 2000-03 swings.
4. **Cost-aware construction**, once any gross premium exists. Use a buy/hold buffer (enter in the top 20%, exit below
   the top 40%; Novy-Marx & Velikov 2016) or half-gap partial adjustment to cut turnover from 29x to about 15x. Use
   longer tranches for the slow (neglect) leg.
5. **Use the shock as a feature, not a stand-alone strategy.** Add AV5/50 and AV21/252 as inputs to the s08
   walk-forward ML ranker. Volume shocks may matter mainly through interactions, such as volume x return (Llorente,
   Michaely, Saar & Wang 2002) or volume x size, which a tree model can find at no extra trial cost to s04.

## Shared-code / data issues (reported, not modified)

* **The `volume` panel embeds future split factors, including holdout-period splits.** For example, GOOGL's March 2014
  share volume in the dev panel (~1.2e8/day) is adjusted for the 2022 20:1 split, and its close (28.98) is too.
  * **Safe:** returns, a stock's own volume ratios (as used here), and dollar volume (close x volume, where the factors
    cancel).
  * **Look-ahead:** share-volume LEVELS compared across stocks or over time, for example a raw "volume" feature in an ML
    ranker. This is the same issue `stock_universe()` documents for price floors, but it is not documented for volume.
    Suggest a note in `DataStore.panel` and `results/data_audit.md`.
* **`utils.rolling_beta` is slow on the 759-stock panel** (pandas rolling cov). s04's private `_rolling_beta`
  (cumulative sums, pairwise-complete) gives identical results (max diff 6e-13) in under a second. Suggest promoting it
  to utils.
* **`DataStore.stock_universe()` is not cached** and recomputes a 63-day rolling median over 759 columns on every call
  (about 3 s). The standard battery calls `target_weights()` 7 times (3 in `run_strategy.py` plus 4 in the look-ahead
  check), so every stock strategy has to memoise itself (s03 and s04 do). Suggest an `lru_cache` on `stock_universe`
  keyed by its arguments, or a shared memo of `target_weights` in the runner.
