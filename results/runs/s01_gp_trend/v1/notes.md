# s01_gp_trend v1 - Gaussian-process Bayesian trend across asset classes

Module: `oafbt/strategies/s01_gp_trend.py` (class `GPTrend`, `STRATEGY = GPTrend`, version `v1`, universe `multiasset`).
Inspired by: Bayesian Quadrature deck (GP machinery, robust ensembling over a trend parameter) + the momentum idea type.

## Final result (dev, 2004-2020, net of costs/fees, scaled to 10% ex-ante vol)

```
s01_gp_trend v1 [2004-01-02..2020-12-31, 16.98y] SR=0.31 (04-20: 0.31, 2xcost: 0.24, lag+1: 0.39) ret=3.3% vol=10.7% maxDD=-20.6% calmar=0.14 sortino=0.43 skew=-0.26 turn=27.06x cost=0.8% lev=2.15 beta=-0.06 t=1.34 dsr=0.67 sub=[0.226, 0.586, 0.123] CI=[-0.12,0.78] posyrs=11/17 lookahead=True (12.8s)
```

| metric | value | metric | value |
|---|---|---|---|
| Sharpe (net) | 0.31 | gross Sharpe (before costs/fees) | 0.44 |
| ann. return / vol | 3.3% / 10.7% | CAGR | 2.8% |
| max DD | -20.6% (peak 2006-05-10, trough 2007-08-16, recovered 2008-10-09; 814 days) | Calmar | 0.14 |
| turnover | 27x/yr | trading cost / fees | 0.77% / 0.58% per yr (borrow 0.30%, leverage spread 0.28%) |
| beta / corr to SPY | -0.06 / -0.11 | HAC t-stat (alpha t) | 1.34 (1.59) |
| 2x costs / lag+1 | 0.24 / 0.39 | DSR (3 distinct trials) | 0.67 |
| bootstrap 95% CI | [-0.12, 0.78] | positive years | 11/17 |
| avg gross / net leverage | 2.15 / 0.66 | avg # positions | 32 of 35 |

Crisis behaviour: 2008 +19.0% (SPY excess -38%); Covid crash 2020-02-19..03-23 +4.9% (SPY -33.6%); Q4 2018 +2.4%
(SPY -14.1%). In the worst decile of SPY months the strategy averages +1.9% (SPY -7.7%) versus +0.1% otherwise
(monthly corr with SPY -0.08). Rolling 3-year Sharpe: 0.57-0.91 in 2008-2011, 0.26-0.60 in 2013-2016, 0.09-0.24 in 2017-2020
(the industry-wide post-2015 decline of trend following is visible here too).

## Thesis

Trends in asset-class returns persist (time-series momentum: Moskowitz-Ooi-Pedersen 2012; Hurst-Ooi-Pedersen 2017).
The BQ deck ensembled MACD settings on single futures, weighting them by in-sample backtest Sharpe (OOS Sharpe ~0.1).
This strategy is NEW in three ways: (1) the GP is fitted to each asset's price path, not to the parameter space, and
the trend is the GP posterior drift scored by its own uncertainty; (2) trend horizons and drift amplitudes are
weighted by Bayesian evidence (marginal likelihood of each GP on the asset's own trailing year), not by backtest Sharpe;
(3) it is diversified across 35 ETFs in four risk groups.

## Implementation

**Data.** 35 ETFs (long-history `_X` twins where available, never an ETF together with its own twin):
equity SPY_X IWM_X QQQ EFA_X VGK_X EWJ EWG EWU EWC EWA EEM_X EWZ EWY EWT FXI VNQ_X; rates TLT_X IEF_X TIP_X SHY;
credit LQD HYG; commodity GLD_X DBC_X SLV USO DBA DBB; FX UUP FXE FXY FXB FXA FXC FXF. Signals use the EXCESS log
return `log(tri_t/tri_{t-1}) - log(1+rf_t)` (ds.tri + ds.rf). Excess rather than total return matters: on total returns
SHY or the FX ETFs would "trend" on their cash yield, which a funded position does not earn.

**GP model (per asset, per weekly decision date t, data <= t).**
* `e_s = clip(r_s / sigma_{s-1}, +-5)`, where sigma is the EWMA vol (halflife 21d) known at s-1, for s = t-251..t (window 252).
* `e_s = mu(s) + eps_s`, with `eps ~ N(0,1)` and `mu ~ GP(0, (sr^2/252) * exp(-(s-s')^2 / (2 l^2)))`.
* This is a GP on the vol-normalised log-price path `y = cumsum(e)` whose kernel is an integrated-RBF trend plus a
  Brownian noise term. The "posterior slope" of the smooth part is the drift mu.
* 20 models: lengthscale l in {21, 63, 126, 252} (1/3/6/12-month trends) x amplitude sr in {0.25, 0.5, 1, 2, 4}
  (prior sd of the drift in annualised Sharpe units, a log-uniform grid). The amplitude is integrated out rather than hand-set.
* On the fixed grid with fixed hyper-parameters, three things are fixed:
  * the posterior mean of the average drift over the holding week (sessions t+2..t+6, i.e. lag-1 execution plus a one-week hold) is a fixed linear filter `m_k = h_k'e`;
  * its posterior variance `v_k` is a constant;
  * the log evidence is a fixed quadratic form `-0.5 e'(K_k+I)^-1 e - 0.5 log|K_k+I|`.
* Everything is precomputed from one eigendecomposition of the 252x252 unit RBF matrix per lengthscale; amplitudes only
  rescale the eigenvalues. A full `target_weights()` call takes ~2-3 s, and the whole battery with the look-ahead check ~13 s.
* BMA weights are `w_k ∝ p(e | l_k, sr_k)` with a uniform prior on the grid. The posterior is the mixture `sum_k w_k N(m_k, v_k)`,
  so `z = E[mu|e] / sd[mu|e]`, using the mixture mean and variance (law of total variance). Model disagreement therefore
  lowers |z|.

**Positions.**
* The target for asset i is `clip(z_i/2, -1, 1) / max(vol_i, 3%)` (EWMA vol, halflife 42d), divided by the number of live
  assets in its risk group (equity / fixed income = rates+credit / commodity / FX). This gives each group equal risk units.
* The book is long and short. Decisions are taken on the last NYSE session of each week: a session is a week end if the
  next session falls in a later week, using the next data date or, for the last data date, the exchange's published
  schedule via `utils.next_session`.
* Each week the book closes half of the gap to the target (partial adjustment, Garleanu & Pedersen 2013).
* The engine applies a 1-day execution lag, costs and fees, and scales the book to 10% ex-ante vol (3x gross cap).

**Look-ahead safeguards.**
* Everything is computed from trailing windows and causal EWMAs. The only thing taken from the future is the published
  holiday calendar, and only to decide whether the last data date ends a week.
* The partial-adjustment recursion runs over past decisions only.
* The built-in truncation check passes (Monday cuts 2008-06-30, 2014-03-31 and 2019-09-30).
* I also ran extra cuts on 2005-03-24 (the Thursday before Good Friday), 2010-05-07 (Fri), 2016-11-09 (Wed),
  2020-03-20 (Fri) and 2020-12-30 (Wed). All give 0 mismatches.

## Development path (4 logged runs, 3 distinct parameterisations; ledger `results/ledger/s01_gp_trend.csv`)

| run | config | SR 04-20 | 2x cost | lag+1 | turnover | notes |
|---|---|---|---|---|---|---|
| 1 | spec: l = 10/21/63/126, amplitude grid BMA, plain inverse-vol risk parity, full weekly rebalance | 0.03 | -0.14 | 0.21 | 63x | gross SR 0.25; costs 1.76% + fees 0.55%/yr ate the edge |
| 2 | l = 21/63/126/252, BMA, 4-group risk budget, 3% vol floor, partial adjustment 0.5/wk | **0.31** | 0.24 | 0.39 | 27x | all five classes net positive |
| 3 | = run 2 with uniform weights over the 20 GP models (temper = 0): **ablation of the evidence weighting** | 0.36 | 0.28 | 0.42 | 30x | difference to run 2 within noise |
| 4 | = run 2 as module defaults (same params hash), final v1 | 0.31 | 0.24 | 0.39 | 27x | restores v1 artefacts |

Why the run-1 to run-2 changes (all diagnostics-driven, none from a parameter search):
* Run 1's attribution showed gross SR 0.25 but 63x turnover. The turnover came from signal changes, not from vol-target
  rescaling (62.9x with the scale held fixed). SHY alone generated 25x through its notional, and the lag+1 Sharpe of 0.21
  versus 0.03 shows that the fastest component chases moves that reverse within days.
* Run 2 therefore uses the classic 1/3/6/12-month horizons (MOP 2012 / HOP 2017) instead of the two-week model.
  Construction-only diagnostics put target turnover at 63x with the run-1 horizons and 45x with 1-12 months.
* A Garleanu-Pedersen partial adjustment brings turnover down further, from 46x to 27x.
* Forecasting the drift over a month instead of a week was tried as a construction diagnostic only (no P&L) and
  dropped: it changed turnover by only 2-3%, because z is a mean/sd ratio.
* The 4-group risk budget and the 3% vol floor were planned BEFORE run 1 from construction diagnostics.
  * With plain risk parity, the 16 correlated equity ETFs carried 55% of ex-ante risk. With the budget the split is
    equity 27%, commodity 26%, fixed income 27%, FX 20%.
  * SHY (1-2% vol) took 24% of gross notional and the 3x cap bound on 24% of rebalances. In that pre-run-1 diagnostic,
    the floor halved SHY's notional and cut cap binding under the group budget from 39% to 21%, with almost no change in
    risk contributions.
  * Final config: equity 27% / commodity 26% / fixed income 27% / FX 20% of ex-ante risk. SHY is 15% of gross notional
    (25% without budget and floor), and the cap binds on 24% of rebalances.
* I decided before run 3 to keep BMA unless the uniform ensemble beat it by more than 0.15 Sharpe, roughly 2 SE of the
  difference for variants with z-correlation ~0.96. It beat BMA by 0.05, so v1 keeps BMA, the a priori design.

## Per-asset-class contribution (final v1, 2004-2020, % of NAV per year)

| class | gross P&L | trading cost | borrow | net (before leverage spread) | turnover | avg gross notional | stand-alone gross SR |
|---|---|---|---|---|---|---|---|
| equity (16) | 1.09 | 0.11 | 0.02 | **0.95** | 3.6x | 0.31 | 0.23 |
| commodity (6) | 1.01 | 0.14 | 0.06 | **0.81** | 4.1x | 0.29 | 0.22 |
| credit (2) | 0.76 | 0.07 | 0.02 | **0.67** | 3.7x | 0.34 | 0.39 |
| rates (4) | 0.92 | 0.19 | 0.07 | **0.66** | 9.3x | 0.74 | 0.27 |
| FX (7) | 0.92 | 0.26 | 0.12 | **0.54** | 6.4x | 0.48 | 0.24 |
| **total** | 4.70 | 0.77 | 0.30 | 3.62 (- 0.28 leverage spread = 3.34 net) | 27x | 2.15 | 0.44 |

Every class contributes, and none dominates. Diversification lifts the gross Sharpe from 0.22-0.39 per class to 0.44.
* Best assets (gross % per year): LQD 0.70, FXE 0.43, GLD_X 0.34, TIP_X 0.31, FXY 0.29, IEF_X 0.25.
* Worst: IWM_X -0.10, FXB -0.06, VNQ_X -0.05.

Calendar-year gross P&L by class (%), with net total:

| year | commodity | credit | equity | FX | rates | net total |
|---|---|---|---|---|---|---|
| 2004 | -1.6 | 1.1 | -0.6 | 0.0 | 1.9 | -0.1 |
| 2005 | -1.7 | -1.4 | 1.3 | 0.0 | -1.9 | -4.9 |
| 2006 | -0.8 | 0.0 | 5.2 | 0.5 | -1.0 | 2.8 |
| 2007 | 1.7 | -1.2 | 1.1 | 0.3 | -1.0 | -0.6 |
| 2008 | 7.3 | 0.2 | 6.1 | 3.6 | 1.7 | 18.3 |
| 2009 | -0.5 | 1.1 | 3.9 | 1.0 | -1.8 | 3.2 |
| 2010 | 3.2 | 2.8 | -1.0 | 3.1 | 4.4 | 11.4 |
| 2011 | 1.8 | -0.4 | -3.8 | 1.0 | 6.3 | 4.1 |
| 2012 | -2.7 | 3.2 | -0.1 | -2.5 | -0.6 | -4.5 |
| 2013 | 2.9 | -1.2 | 4.4 | 3.6 | 0.2 | 8.1 |
| 2014 | 4.7 | 1.9 | -2.2 | 9.0 | 1.1 | 12.6 |
| 2015 | 4.1 | -0.1 | -1.4 | -0.7 | -2.0 | -1.4 |
| 2016 | -1.7 | 2.8 | -2.2 | 1.4 | 2.8 | 1.5 |
| 2017 | -5.1 | 1.4 | 14.3 | -2.7 | -2.7 | 3.5 |
| 2018 | 2.7 | -0.8 | 0.9 | 0.7 | 0.3 | 2.1 |
| 2019 | -2.2 | 6.0 | -4.2 | -2.3 | 3.6 | -0.8 |
| 2020 | 4.9 | -2.6 | -3.2 | -0.4 | 4.3 | 1.6 |

For comparison, run 1 (equity 55% of risk, 63x turnover) was net -0.39% per year in rates, -0.35% in FX and only +0.17% in equity.

## Diagnostics looked at

1. **GP on levels vs. increments (synthetic random walks, no market data).** RBF + white noise fitted to random-walk LEVELS,
   with ML-II per window, picks l = 10 in 100% of windows. The median log-evidence gaps are -41/-102/-150 nats for
   l = 21/63/126, and the gap reaches 280 nats. That BMA degenerates to the shortest horizon. On the increments,
   evidence gaps stay within about 2 nats. Hence the drift formulation, with no ad-hoc tempering.
2. **Is the evidence informative? (signals only.)**
   * With one fixed amplitude (prior SR sd 1), BMA weights are about uniform: the largest weight has p90 0.32, and z
     correlates 0.995-0.997 with the uniform ensemble.
   * With the amplitude grid, lengthscale weights still average 0.256/0.248/0.248/0.249. The 6-12-month models dominate
     (combined weight above 0.6) in only 2.6% of decisions.
   * The evidence acts on the amplitude instead: small amplitudes (sr <= 0.5) dominate 58% of decisions, which shrinks
     weak trends, and sr >= 2 dominates 8.6%, which keeps strong ones.
   * Net effect: a convex response to trend strength, with z correlation 0.96 to the uniform ensemble. BMA |z| is
     0.90x the uniform ensemble's outside the strongest-decile trends and 1.06x inside them. On synthetic
     constant-drift windows (final lengthscales), median z is 0.22 / 0.49 / 1.38 at window SR 1 / 2 / 4
     (uniform ensemble: 0.23 / 0.43 / 0.69).
   * Conclusion: one year of daily data cannot distinguish trend horizons. The SNR is about 0.004 per day, which
     reproduces from first principles the deck's "uniform weights equally good".
3. **Signal persistence and turnover (construction only).** Week-on-week z autocorrelation was 0.82 for l = 10-126 and is
   0.91 for 21-252. Target turnover went 63x, then 45x, then 27x once partial adjustment was added.
4. **Leverage and risk split (construction only).** Ex-ante risk contribution by group and gross-cap binding frequency;
   see the development path.
5. **Attribution** of runs 1 and 2/4 by class, asset and year (tables above), and gross vs net decomposition.
6. **Timing.** lag+1 is better than lag 1 in every run (+0.18, +0.08, +0.06). Part of each weekly signal change reverses
   within a day or two, probably short-term reversal or ETF premium/discount effects in international and bond ETFs.
   This is not tuned: picking the lag would be curve-fitting.
7. **Pre-inception `_X` segments.** VGK_X, DBC_X, VNQ_X and GLD_X are mutual-fund or futures proxies until 2005-03,
   2006-02, 2004-09 and 2004-11. They could not have been shorted, yet the engine charges ETF borrow. s01's average short
   notional in those segments is 0.1-2.2% of NAV, so the effect is negligible here.

## Weaknesses

* **Weak statistical evidence.** Sharpe 0.31, t = 1.34, bootstrap CI [-0.12, 0.78], DSR 0.67 with 3 trials. On its own
  this is not significant. Its value is as a crisis-convex diversifier (negative SPY beta, positive in SPY's worst months).
* **The core novelty adds nothing measurable.** Evidence weighting scored 0.31 against 0.36 for the uniform ensemble.
  The useful parts are the GP drift filters, vol-normalisation, risk grouping and cost control. The "Bayesian weights"
  are mostly inert across horizons.
* **Decay.** Rolling 3-year Sharpe is 0.1-0.2 over 2017-2020, and the P&L is concentrated in 2008, 2010, 2013 and 2014.
  There is a long, deep drawdown in 2006-07 (-20.6%, 814 days to recover).
* **Implementation drag.** ETFs are funded, so leverage spread plus borrow cost 0.58%/yr on 2.15x gross leverage, on top
  of 0.77% trading costs. That is 1.35% of a 4.7% gross edge. The same signals on futures would keep more of it.
* **Diagnostics-driven redesign.** The run-2 construction was chosen after seeing run 1, which is mild selection bias.
  DSR accounts for 3 trials, but the holdout is the real test.
* **Correlations ignored inside groups.** Equity ETFs overlap (EFA_X, VGK_X, EWG, EWU), so within-group risk is concentrated.
* **Timing leakage.** The lag+1 > lag pattern means the weekly decision captures some noise that reverts.

## Improvement ideas for the next round

1. **Pool the evidence across assets (hierarchical BMA).** Sum the per-asset log-evidences within each risk group to get
   group-level horizon and amplitude weights: 6-16x more data per decision, so the horizon choice can become informative.
   It is cheap, because per-asset evidences are already computed. If even pooled evidence stays flat, simplify to the
   uniform ensemble.
2. **A carry-informed prior mean for the drift.** Replace the zero-mean GP with a mean taken from real-time carry:
   * bond term spread from ^TNX - ^IRX;
   * FX-ETF interest differential, i.e. ETF excess return minus spot change, trailing;
   * commodity roll yield from =F vs ETF, used as a signal only.
   This gives "trend + carry" in one Bayesian model (Koijen et al. 2018). Carry is a premium that is weakly correlated with trend.
3. **Execution and cost.** Investigate the lag+1 > lag pattern by class: Friday decision vs mid-week, international ETFs'
   asynchronous closes. Replace the fixed 0.5 partial adjustment with a no-trade band sized by each asset's cost and
   signal persistence (Garleanu-Pedersen aim portfolio).
4. **Correlation-aware sizing.** Size on a shrunk covariance (Ledoit-Wolf, or GP-posterior-based), e.g. equal risk
   contributions across groups using actual correlations rather than counting assets.
5. **Longer evidence window for the slow model.** With a 252-day window, the l = 252 model is close to a flat 12-month
   average and the long-horizon evidence is truncated. A 504-day evidence window (assets enter at 2y) or a Matern kernel
   (rougher drift) may separate horizons better.

## Shared-code issues (reported, not modified)

* `utils.trading_calendar`, as first written, raised `exchange_calendars` DateOutOfBounds whenever `cal[-1] + 1.6*ahead+10`
  days fell on a non-session; for example, `ahead=10` with data ending 2008-06-30 failed. The parent already fixed it in
  commit 05df522 (`nyse_sessions` / `next_session`), and s01 uses `next_session`.
* `utils.period_ends(index, "W")` (and "M"/"Q") marks the LAST AVAILABLE date as a period end. The schedule then depends
  on where the data stop: on a truncated store with a mid-week end it creates a rebalance that the full run does not
  have. A weekly strategy using it fails the look-ahead check at the Monday cut dates, and in live use it would trade on
  a spurious "week end". Suggest a shared helper like s01's private `week_ends()`, which uses the next session from the
  published calendar.
* `checks.lookahead_check` default cut dates are all Monday month-ends. It never tests a cut on a typical rebalance day
  (Friday) or mid-week. Suggest adding, for example, 2010-05-07 (Fri) and 2016-11-09 (Wed).
* The engine charges ETF cost and borrow on `_X` twins before their ETF existed, while the series is a mutual-fund or
  futures proxy that could not be shorted (VGK_X before 2005-03, DBC_X 2006-02, VNQ_X 2004-09, GLD_X 2004-11, all within
  the 2004+ evaluation window). Suggest forbidding shorts, or charging `mutual_fund` borrow, before the inception date.
  The impact on s01 is negligible.
