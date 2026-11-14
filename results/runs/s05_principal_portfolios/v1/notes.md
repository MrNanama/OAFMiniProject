# s05_principal_portfolios v1: principal portfolios of the prediction matrix (Kelly, Malamud & Pedersen 2023)

## Verdict
* **v1 earns net SR 0.63 over 1975-2020 and 0.56 over 2004-2020.** It is robust to 2x costs (0.52) and to one more
  day of lag (0.58). Subperiod SRs are even: 0.53 / 0.84 / 0.50. HAC t = 3.6, bootstrap CI [0.26, 1.00].
  * The book is dollar-neutral, with market beta 0.00.
  * Max drawdown is -30.9% (1981). The worst post-2004 drawdown is -26% (2015-17).
  * v1 is two separate prediction-matrix books, combined at equal risk: 25 size x B/M VW portfolios and 12 VW
    industries, both on 12-1 momentum ranks.
* **The faithful KMP baseline is a fading edge and should not be traded.** That baseline is FF25 with a 1-month
  signal (run 1).
  * It had gross SR ~2 before 2005 and **gross SR 0.00 in 2005-2020**. Net SR is 0.81 over the full period but
    -0.38 over 2004-2020, with a -61% drawdown over 2011-18.
  * This matches the disappearance of short-horizon cross-autocorrelation in small/value portfolios after
    decimalisation (the daily AR(1) of size/value spreads fell from 0.16-0.35 to about 0).
* **Most of v1 is style and industry momentum.** It loads 0.34 on UMD (FF5+Mom+ST_Rev regression on monthly net
  returns). Correlation with s02 factor momentum is 0.48 monthly.
  * Before 2004 the monthly alpha is 0.2%/yr (t = 0.1).
  * In 2004-2020 the alpha is **4.4%/yr (t = 2.1)**, net of costs against gross paper factors.
  * The principal-portfolio machinery mostly chooses *which* momentum directions to trade. The lead-lag (PAP)
    sleeves are weak.
* **Daily-return Sharpe overstates the pre-2005 numbers.** On monthly returns, v1's SR is 0.51 for 1975-2020 and
  0.55 for 2004-2020. The 2004-2020 figure is unaffected (see shared-code issues).

## Thesis
* **Predictability.** Style portfolios (size x B/M) and industries are predictable over 1-12 months by their own
  past returns *and by each other's*.
  * Style momentum: Barberis & Shleifer 2003; Chen & De Bondt 2004.
  * Industry momentum: Moskowitz & Grinblatt 1999.
  * Lewellen (2002) shows size/BM portfolio momentum comes from cross-serial correlation.
  * Industry lead-lag through customer/supplier links: Menzly & Ozbas 2010; Cohen & Frazzini 2008.
* **KMP's result.** Kelly, Malamud & Pedersen (2023, JF) show that the optimal linear strategy w_t = L S_t
  maximises <L, Pi>, where Pi = E[R_{t+1} S_t'] is the prediction matrix.
  * The symmetric part of Pi gives principal exposure portfolios (PEP): each eigen-portfolio is timed by its own
    signal.
  * The antisymmetric part gives principal alpha portfolios (PAP), which are pure lead-lag trades.
* **Inspiration.** The Macro Momentum deck used latent factors, PCA plus ridge regularisation, and walk-forward
  estimation. The new object here is the prediction matrix of a cross-section, regularised by rank truncation and
  shrinkage.

## Implementation (`oafbt/strategies/s05_principal_portfolios.py`, v1 = `default_params`)
* **Assets.** KF VW portfolios in two blocks: `25_Portfolios_5x5_Daily|VW` (FF25) and
  `12_Industry_Portfolios_daily|VW` (IND12).
  * Daily excess returns come from `utils.kf_portfolio_returns`.
  * Monthly returns are compounded from daily KF total returns. Only complete calendar months are used, via
    `utils.period_ends`.
* **Signal.** S_t is the cross-sectional rank (`xs_rank`) of the 12-1 month return (months t-11..t-1), scaled to
  sum|S| = 1. In run 1 it was the rank of the month-t return.
* **Prediction matrix, per block, walk-forward.**
  * Pi_T is the mean of the pairs (S_t, u_{t+1}) over 1963-07 <= t and t+1 <= T. The window is expanding and needs
    at least 120 pairs, so the first decision is 1973-07.
  * u is the cross-sectionally demeaned return divided by an **ex-ante** dispersion d_t (RMS of demeaned returns
    over the 12 months <= t).
  * Demeaning makes Pi 1 = 0 = 1'Pi, so every principal portfolio is dollar-neutral and there is no market timing.
  * The d_t scaling stops 1999-2002 and 2008 from dominating the estimate.
* **Shrinkage (Ledoit-Wolf style, no tuning).**
  * Pi is replaced by (1-delta) Pi + delta c J, where c = tr(Pi)/(N-1) and J = I - 11'/N. The target is "equal
    own-signal predictability, no cross-predictability", i.e. the simple factor strategy.
  * delta = (sum of sampling variances of the entries) / ||Pi - target||_F^2.
  * Mean delta: FF25 0.34 (0.57 at the start, 0.32 at the end); IND12 0.70 (0.93 to 0.57). The IND12 book is
    therefore mostly industry momentum with a cross-predictability tilt.
* **Principal portfolios.**
  * PEP: the top 3 eigenvalues of Pi_s by |lambda|, with L = sum sign(lambda_k) v_k v_k'.
  * PAP: the top 3 pairs of Pi_a, from the Hermitian eigenproblem of i*Pi_a, with L = sum (y_k x_k' - x_k y_k') =
    2 Im(Z Z^H).
  * Identities verified numerically: <L_PEP, Pi> = sum|lambda|, <L_PAP, Pi> = 2 sum sigma, w_PAP'S = 0, both books
    dollar-neutral.
* **Book.** w_PEP = L_PEP S_T and w_PAP = L_PAP S_T are each scaled to unit ex-ante vol (252-day daily covariance,
  data <= T) and summed. Each block's book is then scaled to unit vol and the blocks are summed (equal risk).
* **Rebalancing.** Rows are emitted on the last trading day of each month only. The runner applies the 1-day lag,
  10% vol target, 3x cap, 5 bps costs, 30 bps/yr borrow and 10 bps/yr drag.
* **Runtime and look-ahead.** One `target_weights` call takes 0.7 s. My own truncation tests (1990-06-29,
  2008-06-30, 2014-03-31) showed 0 difference, and the built-in check passed (0 mismatches at all 3 cuts).

## Results
Committed v1 (run 4, `scripts/run_strategy.py`):
```
s05_principal_portfolios v1 [1975-01-02..2020-12-31, 46.04y] SR=0.63 (04-20: 0.56, 2xcost: 0.52, lag+1: 0.58) ret=4.6% vol=7.3% maxDD=-30.9% calmar=0.14 sortino=0.89 skew=-0.23 turn=15.18x cost=0.8% lev=2.89 beta=0.00 t=3.60 dsr=1.00 sub=[0.533, 0.84, 0.503] CI=[0.26,1.00] posyrs=33/46 lookahead=True (11.5s)
```

**All trials.** Everything was run on dev data, with 4 runs and 4 distinct parameter sets. The project ranking score
is 0.5*SR04-20 + 0.5*min(SR04-20, SR2x, SRlag+1).

| run | change | SR 75-20 | SR 04-20 | 2x cost | lag+1 | maxDD | turn/yr | cost/yr | subperiods | score |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | KMP baseline: FF25, 1-month rank signal, PEP3+PAP3 | 0.81 | -0.38 | 0.52 | 0.83 | -61.4% | 42.9 | 2.1% | 1.67 / 1.44 / -0.44 | -0.38 |
| 2 | run 1 with the 12-1 momentum signal | 0.64 | 0.36 | 0.55 | 0.61 | -26.5% | 13.9 | 0.7% | 0.61 / 1.03 / 0.33 | 0.36 |
| 3 | run 2 + Pi projected on top return-PCs (Kaiser rule, about 4 PCs) | 0.47 | 0.23 | 0.39 | 0.48 | -39.1% | 13.5 | 0.7% | 0.59 / 0.73 / 0.17 | 0.23 |
| **4 = v1** | run 2 + separate IND12 block, equal risk | **0.63** | **0.56** | 0.52 | 0.58 | -30.9% | 15.2 | 0.8% | 0.53 / 0.84 / 0.50 | **0.54** |

**v1 by period.** SR "d" uses daily returns (√252) and SR "m" uses monthly returns (√12). Returns are %/yr at the
realised leverage.

| period | net SR d | gross SR d | net SR m | gross SR m | gross ret | cost | fees | net ret | vol d | vol m | gross lev | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 0.56 | 0.84 | 0.39 | 0.57 | 5.1 | 0.7 | 1.0 | 3.4 | 6.1 | 9.4 | 2.98 | 14.3 |
| 1990-2004 | 0.76 | 0.98 | 0.57 | 0.73 | 7.5 | 0.8 | 0.9 | 5.8 | 7.6 | 10.5 | 2.81 | 15.7 |
| 2005-2020 | 0.57 | 0.78 | 0.56 | 0.77 | 6.3 | 0.8 | 0.9 | 4.6 | 8.0 | 8.2 | 2.87 | 15.5 |
| 1975-2020 | 0.63 | 0.86 | 0.51 | 0.69 | 6.3 | 0.8 | 0.9 | 4.6 | 7.3 | 9.4 | 2.89 | 15.2 |

**Block and sleeve decomposition of v1.** Each component was run through the same engine at 10% vol. This is a
diagnostic and was not logged. Figures are gross SR on daily returns.

| period | FF25 book | FF25 PEP | FF25 PAP | IND12 book | IND12 PEP | IND12 PAP | FF25 book net | IND12 book net |
|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 0.88 | 1.24 | -0.18 | 0.45 | 0.36 | 0.34 | 0.62 | 0.26 |
| 1990-2004 | 1.20 | 0.71 | 0.91 | 0.37 | 0.47 | -0.02 | 0.99 | 0.23 |
| 2005-2020 | 0.56 | 0.47 | 0.22 | 0.71 | 0.61 | 0.35 | 0.37 | 0.57 |
| 1975-2020 | 0.85 | 0.79 | 0.34 | 0.51 | 0.49 | 0.22 | 0.64 | 0.36 |

Monthly correlations (1975-2020):
* FF25 book vs IND12 book: 0.21.
* PEP vs PAP within a block: 0.21 (FF25) and -0.04 (IND12).
* v1 vs run 1 (the 1-month principal portfolios): 0.11.
* v1 vs s02 factor momentum: 0.48.

## Diagnostics I looked at
* **Run 1, by period.** Gross SR was 2.19 / 1.78 / -0.00 on daily returns (1.53 / 1.29 / 0.00 on monthly returns).
  * Both sleeves worked before 2005: PEP gross 1.72 / 1.06, PAP 1.06 / 1.32, correlation about -0.1.
  * Both died after 2005: PEP -0.14, PAP 0.19.
  * The calendar years 2012-2017 were all negative (-23%, -17%, -7%, -12%, -1%, -13%).
  * The book correlated -0.27 with ST_Rev before 2005.
* **Structure before any run (risk only).**
  * Unit-vol rows have gross ~43, so 10% vol needs about 4.3x gross. The 3x cap therefore binds: in v1 it binds on
    64% of days and realised vol is 7.3%.
  * Monthly turnover/gross is 1.35 for the 1-month signal and 0.42 for 12-1.
  * The demeaned FF25 covariance has 4 clear PCs (44 / 19 / 7 / 5% of variance) and then a flat bulk.
* **Daily vs monthly vol of FF25 spreads at gross 2.**
  * SMB-like: ratio 1.16 in 1975-89 and 1.60 in 1990-2004.
  * HML-like: 1.60 and 1.42, with daily AR(1) up to 0.35.
  * About 0.85-0.95 after 2005.
* **PCA projection (run 3).** It cut gross by only ~10% and weakened both sleeves in 2005-20 (PEP 0.47 to 0.25, PAP
  0.22 to 0.09). Predictive structure in the FF25 is not confined to the top-variance PCs, so I rejected it.
* **Factor regression of monthly net returns on FF5 + Mom + ST_Rev (HAC t).**

  | period | alpha | t | Mom | RMW | SMB | CMA | R² |
  |---|---|---|---|---|---|---|---|
  | 1975-2020 | 3.1%/yr | 2.5 | 0.34 | -0.14 | 0.11 | | 0.31 |
  | 1975-2004 | 0.2%/yr | 0.1 | 0.40 | | | | 0.38 |
  | 2004-2020 | 4.4%/yr | 2.1 | 0.24 | | | -0.15 | 0.19 |

* **Components at 2020-12** (the shrunk Pi):
  * Pi_s holds 81% of ||Pi||² (FF25) and 72% (IND12). The diagonal holds 17% and 39%.
  * FF25 PEP1 (lambda 0.046) is small-value minus big-growth momentum, i.e. the SMB+HML diagonal. PEP2 is the
    small-growth corner.
  * FF25 PAP1: past small-growth-minus-big-value performance predicts future value-minus-growth performance, and
    past value-minus-growth performance predicts future small-growth-minus-big-value *under*performance.
  * IND12 PEP1 is energy/manufacturing/chemicals against durables/telecoms/retail/finance. PEP2 is health/business
    equipment.
  * IND12 PAP1: durables and manufacturing lead energy, a plausible demand-to-input-prices channel.
* **Drawdowns deeper than 15%.**
  * 1980-11 to 1981-11: -30.9%, recovered 1983-06.
  * 1983-06 to 1984-08: -18.0%, recovered only in 1990-08. The longest underwater spell is 1,809 days.
  * 1993-08 to 1996-08: -18.4%.
  * 2000-03 to 2001-05: -23.7%, the dot-com reversal.
  * 2015-08 to 2017-12: -26.2%, recovered 2020-07.
  * Worst month: 1978-10, -9.2%.

## Weaknesses
* **It is mostly a momentum strategy.** The UMD loading is 0.34 and the correlation with s02 is 0.48. Before 2004
  there is no alpha beyond Mom, and the drawdowns are momentum reversals (1981, 2000-01, 2016). Diversification
  within the project portfolio is limited.
* **The selected variant is chosen in-sample.** The IND12 block was added after seeing runs 1-3. Its contribution is
  concentrated after 2005 (gross SR 0.71 vs 0.37-0.45 before). There are 4 trials. Note that the summary's DSR used
  n_trials = 3.
* **The KMP-specific lead-lag (PAP) alpha is weak at the 12-1 horizon.** PAP gross SR is 0.34 for FF25 and 0.22 for
  IND12 over the full period, and run 3 suggests it sits partly in low-variance directions.
* **The 1-month cross-predictability KMP emphasise is gone in this data after 2005.**
* **Leverage and fees.** The 3x cap binds on 64% of days, so the book runs at about 7.3% vol. Fees (~0.9%/yr of
  borrow, holding drag and leverage spread at 2.9x gross) are a fixed drag of about 0.12 SR.
* **Paper portfolios.** The FF25 book takes large positions in small-cap portfolios (ME1/ME2). The 5 bps plus
  10 bps/yr cost model is probably optimistic for replicating them. The 2x-cost run still gives SR 0.52.
* **The expanding window adapts slowly.** Run 1 kept betting on the pre-2005 structure for 15 years. v1 would react
  just as slowly to a regime change.

## Improvement ideas for the next round
1. **Momentum-crash control.** Scale the book by the inverse of its own trailing realised vol (Barroso &
   Santa-Clara 2015), or neutralise its ex-ante UMD beta. The goal is to cut the 1981 / 2000-01 / 2016 drawdowns
   and lower the overlap with s02. Measure whether the residual (PAP-heavy) part keeps the post-2004 alpha.
2. **Cost-aware principal portfolios.** Blend towards last month's drifted holdings, w_t = (1-tau) w_{t-1} + tau L S_t,
   with tau set from the signal's autocorrelation. Alternatively, choose the PEP/PAP sets to maximise
   <L, Pi> - kappa * E|Delta w|. Either should cut turnover from 15x and fees at the cap.
3. **Multi-horizon prediction matrix.** Stack the 1-, 3-, 6- and 12-1-month signals as signal blocks and select
   principal portfolios across horizons, with shrinkage per horizon, following KMP's multi-signal extension. The
   1-month book is nearly uncorrelated with v1 (0.11), so a turnover-gated version could add diversification if
   any short-horizon edge survives. Its post-2005 gross SR of 0.00 argues against trading it unchanged.
4. **Adaptive estimation and rank.**
   * Use an exponentially weighted Pi (half-life of about 15 years) instead of the expanding window.
   * Choose K per block, keeping components whose shrunk eigenvalue clears a noise bound (e.g. the
     Marchenko-Pastur edge of the sampling noise) instead of a fixed K = 3. This targets the slow adaptation seen
     in run 1.
5. **More blocks for breadth.** Add the 49 VW industries (with stronger shrinkage) and ME x Prior-12-2 portfolios as
   further blocks. Allocate across blocks by trailing out-of-sample book Sharpe with shrinkage to equal risk,
   rather than fixed equal risk.

## Shared-code issues (not modified; reported)
* **Daily-return Sharpe and vol targeting are biased for autocorrelated KF portfolios before 2005.**
  * Daily-measured vol understates monthly vol by 1.2-1.7x for size/value spreads. So `perf_stats` overstates the
    Sharpe (v1: 0.63 on daily returns vs 0.51 on monthly) and `exante_vol` under-sizes the risk.
  * Suggestion: report a monthly-return Sharpe (or the Lo 2002 autocorrelation-adjusted SR) in `perf_stats`, and use
    overlapping 5-day returns or a Newey-West variance in `exante_vol`.
* **`runner.run` DSR is off by one.** It computes `n_trials` from the trials logged *before* the current run
  (`n_prev`), so the current trial is not counted. The 4th distinct variant reports n_trials = 3, which slightly
  overstates the DSR.
* **`LEV_CAP['kf'] = 3` binds for dollar-neutral KF-portfolio books** (64% of days here). Realised vol is about 7.3%,
  not 10%, so `ann_ret` and `max_dd` are not comparable across strategies. Sharpe is.
* **`utils.period_ends` edge case before 1985.** It checks the last period against `nyse_sessions()`, which starts in
  1985. If data were truncated mid-month before 1985, the incomplete month would be kept as a phantom month-end
  (verified: `bdate_range(1980-05-01, 1980-06-13)` keeps 1980-06-13). The current look-ahead cut dates are all after
  2008, so the check would not catch it.
