# Iteration 1: Stress-conditioned reaction to large single-stock moves

## Thesis
Whether a big idiosyncratic move reverses or continues depends on who is on the other side.
Market-wide stress + market shock => forced/liquidity selling => rebound (Nagel 2012).
Calm market + stock-specific move => news => drift. The volume-filter deck tries to separate these with volume;
we use market stress (trailing percentile of realised S&P vol) and the market's move that day instead.

## Data / setup (sandbox constraint)
Yahoo/FRED/Stooq are blocked, so only data bundled in pip packages was usable: 20 US large caps, daily, 1990-2022 (skfolio).
**All 20 are survivors** (EW basket +18.5%/yr vs SPX +8.8%), so everything is market-hedged and tested for drift contamination.
Walk-forward ridge, refit each January on past events only (5d purge), test 2004-2022. Costs per side, SPX beta hedge.

## Results (OOS 2004-2022, net of 5 bps/side unless stated)
| Strategy | Sharpe |
|---|---|
| 1-day x-sec reversal (deck style) | -1.9 (costs ~27%/yr) |
| 12-1 momentum, monthly | +0.40 |
| Extreme-move reversal / continuation, no regime | -0.81 / -0.08 |
| Regime model, dense (all stock-days) | -0.05 |
| **Regime model, events |z|>=2** | **+0.34** (95% CI -0.03..0.68; 0 bps: 0.55, 10 bps: 0.14, breakeven ~13 bps) |

Checks: survivorship-neutral Sharpe 0.347 (same); label-shuffle placebo mean -0.29, 0/60 reach 0.34; both halves positive (0.29 / 0.39).
Caveats: DSR(N=20 trials)=0.33 (not significant); ex-2020 Sharpe 0.23, ex-2020&2011 0.14; 11/19 positive years;
edge only after DOWN moves (0.45), UP moves lose (-0.15); realised vol 31%, maxDD -63% (scaled to 10% vol: ~3%/yr, DD -25%).
Regime features were chosen after looking at the full-sample event study, so the 2004+ test is not fully clean.

## Verdict
Not yet alpha. Real-looking but small, lumpy, and cost-sensitive. Weak momentum baseline (B2) is comparable once costs are included.

## Files
oaf/data.py, oaf/backtest.py, 01_event_study.py, 02_walkforward.py (--placebo), 03_diagnostics.py, results/

---

# Iteration 2: Diversification failure as a state variable (correlation-timed market exposure) - NEGATIVE RESULT

## Data audit (new, matters for every future iteration)
Network allows only PyPI/GitHub-API. Bundled data found: `arch.data` (S&P500 + Nasdaq Composite OHLCV 1999-2018, Fama-French monthly 1926-2018,
BAA/AAA spreads, WTI/Brent, CPI, VIX 2014-18) + skfolio (20 stocks 1990-2022, 5 factor ETFs 2014-22). FTSE100/Nasdaq-1455 sets are blocked (GitHub download).
**Trap found:** the S&P500 `Open` is stale (= prior close on ~95% of days through 2005, unreliable until ~2008), so any overnight/intraday
analysis on SPX is an artifact (overnight vol 2.6%/yr). Nasdaq Open looks genuine. I abandoned an overnight-drift idea for this reason.
Not usable for survivorship-free stock selection; used the 20-stock panel only to build a *market-state* feature (avg pairwise correlation).

## Thesis
When correlation among large caps spikes, idiosyncratic risk turns systematic, signalling forced deleveraging; the market then pays a premium
(correlation risk premium) beyond what S&P volatility alone captures. Expected: timing S&P exposure with correlation beats plain vol-targeting.

## Evidence
- Correlation adds ~nothing to vol forecasting (OOS R2 0.488 -> 0.490).
- "Correlation spike normalising -> all clear" idea: sign flips between 1990-2005 and 2006-2022. Rejected.
- Predictive regression, 21d fwd excess return on 63d avg-corr (z): +0.43..0.55%/month per sd, HAC t~2.1-2.2, same sign in both halves, but quintile profile is lumpy
  (crisis-rebound driven). ~12 specifications examined before this, so not significant after deflation.

## Locked walk-forward test (expanding fit, 21d purge, monthly rebalance, 1bp cost, avg exposure normalised to 1x, 2000-2022)
| Strategy | Sharpe | MaxDD |
|---|---|---|
| Buy & hold SPX (price index, excess of rf) | 0.25 | -66% |
| Vol-target | 0.33 | -57% |
| Markowitz, constant mu (no corr) | 0.34 | -52% |
| **Corr-timed Markowitz** | **0.30** | -54% |

Corr-timed minus constant-mu: +0.1%/yr, t=0.05. **Signal adds nothing out-of-sample**; sub-periods 2000-10 / 2011-18 / 2019-22 are all within noise of each other.
Vol-targeting itself gives only ~+0.08 Sharpe here. (Index is price-only, rf=0 after 2018-11: approximations affect all strategies similarly.)

## Verdict
No alpha. The in-sample t~2 was a mirage once a vol-managed baseline was included: correlation level is largely a proxy for crisis regime, which vol already captures.
Reported as a negative result rather than tuned until it works (that would be overfitting).

## Where to go in iteration 3 (my recommendation)
1. Binding constraint is DATA, not ideas. If you can drop a broader survivorship-free price/volume CSV into the repo (or allow another data host), cross-sectional strategies become testable.
2. Best surviving lead is still iteration 1 (stress-conditioned reaction to big single-stock moves, SR 0.34 market-hedged). Combine with a vol-managed overlay.
3. Untested but data-valid: (a) Nasdaq overnight-vs-intraday premium with cost-aware gating (only trade nights with strongest expected premium; pure daily version loses to ~1bp/side costs);
   (b) 90-year Fama-French monthly + credit-spread regime study with a true 1926-1965 / 1966-2018 out-of-sample split.

Files: oaf/corr.py, 04_corr_timing.py, results/corr_*.{pkl,csv}

---

# Iteration 3 (allsam): "Who moved the price?" Overnight-gap drift on a broad universe - NEGATIVE RESULT

## Data (new: network opened to Yahoo/Ken French/Stooq; FRED still fails)
`python -m oaf.fetch` pulls adjusted OHLCV for 510 US large caps (412 with full history), 2005-01 to 2026-11 (~5,400 days). Not committed (data/ is gitignored).
Hygiene: open always inside [low, high]; 0.6% zero gaps; 33 extreme obs. **Survivorship:** ticker list is from memory and skews to survivors (49 of 559 tickers returned nothing).
Biggest bias risk is for reversal-of-losers; irrelevant for the conclusion below because the result is negative anyway.

## Thesis
Overnight return = information (news); intraday return = order flow. Flow-driven moves should reverse, news-driven moves should continue. Classic 1-day reversal fails
on costs (iteration 1: -1.9 Sharpe) because it lumps both; separating them should isolate a tradeable piece.

## Evidence
- Next-day return by intraday-z decile: weakly monotone reversal, D1-D10 spread only ~4.4 bps/day vs ~10 bps round-trip cost. Dead on costs.
- Overnight-z: not linear (U-shaped deciles); only extreme drops continue (-10.7 bps next day, t=-4.0), i.e. news drift.
- Gap-and-go vs gap-and-fade does NOT discriminate: both continue (fade even more). The news/flow split as framed is not supported.
- Event t-stats (2-5) are inflated: events cluster on crash days and are not independent.

## Locked backtest (market-neutral vs EW basket, 5-day hold, 5 bps/side, 2006-2026)
| Variant | Net Sharpe | Gross Sharpe | Gross 2006-15 | Gross 2016-26 |
|---|---|---|---|---|
| all gaps |z|>2 | -0.15 | 0.16 | 0.51 | 0.03 |
| gap & fade | -0.05 | 0.30 | 0.41 | 0.23 |
| gap & go | -0.18 | 0.00 | 0.29 | -0.11 |
| down gaps only | -0.04 | 0.25 | 0.44 | 0.12 |

All-gaps: 95% CI [-0.46, 0.20], DSR(N=10 trials) 0.01. Breakeven cost ~2.5 bps/side (Sharpe 0.00); hold 2/10/20d: -0.14/-0.09/+0.02. 2024-26 is negative even gross.

## Verdict
No alpha. The gap-drift effect decayed (gross Sharpe 0.5 -> ~0 across the two halves), is smaller than costs, and the thesis mechanism is not supported.
Short-horizon single-signal price/volume effects on liquid US stocks look arbitraged away; iterations 1-3 all point the same way.

## Suggested iteration 4
Stop mining short-horizon price signals. Options: (a) slower, diversified, cost-insensitive premia built on the new broad data (residual momentum, quality/low-vol, vol-managed factor combos) with
a survivorship haircut; (b) use Ken French factors (daily, incl. momentum) for factor timing with long out-of-sample history; (c) alternative-data-like cross-asset signals via Yahoo
(futures, rates, FX ETFs) where survivorship is not an issue.

Files: oaf/universe.py, oaf/fetch.py, oaf/gap.py, 05_gap_drift.py, results/gap_*.

---

# Iteration 4 (allsam): Learned cross-industry lead-lag network (Ken French 49 industries, 1926-2026) - NEGATIVE RESULT (decayed)

Literature scan first (WebSearch): post-publication anomaly returns are ~58% lower (McLean-Pontiff); widely-known anomalies decay most; trend following has 137y evidence (overlaps the macro-momentum deck);
industry/production-network spillovers look under-explored. Hence this idea, tested on survivorship-free portfolio data (`oaf/kf.py`, data from the Ken French library, not committed).

Design (locked): weekly market-relative industry returns; ridge of next-week returns of all 49 industries on all 49 lagged returns (lambda by last-5y validation inside the training window);
refit each January on past data only; 20y burn-in so OOS = 1947-2026; long top-10/short bottom-10, weekly, 5 bps/side. Own-lag-only model as control; time-shifted-predictor placebo.
(An earlier "scrambled network" placebo was invalid: full ridge is permutation-invariant. Replaced.)

| Period | Gross Sharpe | Net Sharpe (5bp) |
|---|---|---|
| 1947-2026 | 1.10 | 0.26 (95% CI 0.05..1.12 block-bootstrap) |
| 1970-89 | 2.62 | 1.75 |
| 1990-2009 | 0.55 | -0.15 |
| 2010-26 | 0.16 | -0.67 |

Placebo (time-shifted predictors) net Sharpe mean -1.01 (max -0.77): the network signal is real in aggregate. Turnover ~150x/yr; breakeven cost ~2 bps/side pre-1990 only, ~0 post-1990.
Own-lag-only model: net -0.26. **Verdict: a real effect that was arbitraged away (strong pre-1990, ~0 gross after 2010); not tradeable.** Files: oaf/kf.py, oaf/leadlag.py, 06_industry_leadlag.py, results/ll_*.

---

# Iteration 5 (allsam): Slow, fixed-rule premia sleeves (factor momentum + industry momentum + turn-of-month) - REAL ON PAPER, WEAK WHEN IMPLEMENTABLE

Motivation: iterations 1-4 show fast single signals decay/die on costs. Literature (McLean-Pontiff; Ehsani-Linnainmaa; Hurst-Ooi-Pedersen) says slow structural premia survive better.
No fitted parameters: all rules are textbook (12m trailing sign for factor momentum, 12-1 industry momentum top/bottom 10, turn-of-month window = last day + first 3 days).
Data: Ken French daily factors (3/5-factor, UMD, ST/LT reversal) and 49 industries, 1926-2026 (survivorship-free but paper portfolios).

## Sleeves (net of flip costs: 10bp factor_mom, 5bp ind_mom, 1bp tom), Sharpe by era
| Sleeve | all | 1927-62 | 1963-89 | 1990-2009 | 2010-26 |
|---|---|---|---|---|---|
| factor momentum (SMB,HML,RMW,CMA,UMD) | 0.76 | 0.41 | 1.45 | 0.84 | 0.17 |
| industry momentum 12-1 | 0.51 | 0.41 | 0.99 | 0.43 | 0.41 |
| turn-of-month (Mkt) | 0.79 | 1.06 | 0.94 | 0.48 | 0.41 |
| vol-managed market | 0.48 | 0.43 | 0.44 | 0.46 | 0.72 |
| market buy&hold | 0.46 | 0.45 | 0.36 | 0.33 | 0.76 |
Sleeve correlations are low (factor/industry mom 0.5; tom 0.4 with market; others ~0).

## Equal-risk combination "slow3" (fm + im + tom, each scaled to 10% vol with lagged trailing vol)
Sharpe: all 1.28; 1928-62 1.27; 1963-89 1.86; 1990-2009 1.16; **2010-26 0.53; 2020-26 0.66** (market B&H 0.76 in 2010-26). 9.0%/yr at 7% vol, maxDD -18%, beta 0.09.
Alpha vs market: +8.3%/yr (t=10.6) all; +7.8% (t=4.5) 1990-2009; **+2.6% (t=1.6) 2010-26, not significant**. Block-bootstrap Sharpe CI 2010+: [0.11, 0.97].

## Implementability check (Yahoo ETFs, 2000-2026)  -> this is the sobering part
| Instrument | Sharpe 2000-26 | 2020-26 |
|---|---|---|
| 9 SPDR sector ETFs, 12-1 mom, top3/bottom3, 0bp | 0.08 | 0.28 |
| same, 5bp/side | 0.04 | 0.25 |
| SPY turn-of-month, 1bp / 5bp | 0.37 / 0.25 | 0.54 / 0.42 |
| SPY buy&hold | 0.50 | 0.79 |
Industry momentum disappears with 9 coarse sector ETFs (0.4 with 49 granular industries); TOM survives but is a low-vol diversifier, not a Sharpe beater.

## Verdict
The long-run premia are statistically strong (t>10) but (1) alpha has shrunk to ~2.6%/yr and is insignificant in the last 17 years, (2) factor/industry portfolios are paper (underlying rebalancing costs unmodelled),
(3) tradeable proxies are much weaker. This is a defensible low-beta diversified core, not a standalone alpha claim. Files: oaf/premia.py, 07_premia_sleeves.py, 08_premia_combo.py, 09_implementable_check.py, results/premia_*, combo_*, impl_pnl.pkl.

---

# Iteration 6 (allsam): Variance risk premium - implied-vol sizing/tilt and direct short-vol harvest - NEGATIVE

Data (Yahoo): ^SP500TR (1988+), ^VIX (1990+), ^VIX3M (2006+), SVXY (2011+), ^IRX. Code: oaf/vrp.py, 10_vrp_timing.py, 11_short_vol.py.
- VIX is a better variance forecast than realised vol (in-sample R2 0.62 vs 0.56 pre-2008; 0.54 vs 0.45 after). But VRP (VIX vs realised) as a RETURN predictor is unstable (2015-26 t=-2.7; ratio|rv21 t~2 in both eras in-sample).
- Locked walk-forward equity timing (1998-2026, 21d purge, 1bp): B&H 0.43; vol-target(RV) 0.49; vol-target(VIX) 0.43; Markowitz const-mu 0.40; VRP-tilt 0.43. VIX sizing - RV sizing: -1.0%/yr (t=-1.5); tilt - vol-target: -0.5%/yr (t=-0.3). Better vol forecast did NOT translate into better Sharpe (same as iteration 2).
- Direct short vol (SVXY, 2011-2026): Sharpe 0.52 always-short, 0.56 with VIX/VIX3M<0.95 filter; 1.0-1.1 until 2017 but ~0 since 2018; worst day -83% (Feb 2018); corr 0.51 with SPX, whose own Sharpe is 0.85 over the same window. Equity-crash insurance selling, not independent alpha.

---

# Iteration 7 (allsam): Crypto cross-sectional momentum (market-neutral) - FIRST CANDIDATE, NOT PROVEN

Why crypto: less efficient, less crowded than US large caps (iterations 1-6 found nothing there). Data: Yahoo, 59 coins from 2017-11 to 2026-11 (`data/crypto.pkl`, not committed).
Universe = top-30 by lagged 30d dollar volume at each date (point-in-time liquidity filter), long top-5 / short bottom-5 by trailing k-day return, rebalanced weekly, 10 bps/side. Code: oaf/crypto.py, 12-14_*.py.

## Results (net unless stated, Sharpe on daily data x sqrt(365))
- Signal scan (8 signals tried): mom7 -0.10, **mom14 0.53 (gross 0.71)**, mom30 0.43, mom60 0.12, mom90 0.02; reversal 1/3/7d: -0.13/-0.58/-0.41. Reversal clearly negative; momentum positive in a hump at 14-30d.
- Sensitivity grid (81 combos: universe 20/30/40, k 14/21/30, q 3/5/8, rebalance 3/7/14d): ALL positive net Sharpe (median 0.44, range 0.09-0.82); 96% positive in both halves. A plateau, not a spike.
- Placebo (300 random long/short books, same structure): gross Sharpe mean 0.00, sd 0.37, p99 0.88; mom14 gross 0.71 -> p=0.027 (before deflating for ~8 signals tried).
- Market-neutral: beta to BTC -0.04. Top-5 contributing coins = 66% of PnL; without them net Sharpe 0.35 (base 0.53).
- Walk-forward lookback choice (each Jan, best trailing-2y Sharpe among 7/14/30/60): net Sharpe 0.46 (2020-26) vs fixed mom14 0.60, mom30 0.49; BTC buy&hold 0.84.
- Blend 14d+30d: Sharpe 0.53 (2018-21 0.80, 2022-26 0.24). Bootstrap 95% CI [-0.13, 1.18]. Short funding 0/5/10/20%/yr on full short notional (vol-targeted version): Sharpe 0.77/0.68/0.59/0.42.
- Vol-targeted to 20% (lagged 60d vol, cap 3x): 16.9% ann (13.0% after 10% funding), maxDD -26%, avg leverage 0.39.
- Year by year (vol-targeted, 10% funding): 2018 -0.23, 2019 1.33, 2020 0.59, 2021 1.55, 2022 -0.24, 2023 0.22, 2024 1.32, 2025 1.41, **2026 YTD -1.31 (-23.5%)**.

## Honest verdict
Best lead so far: a structurally different, market-neutral, parameter-robust signal with a sensible hump shape and a placebo p-value of ~0.03 (before multiple-testing correction), consistent with the published crypto-momentum literature (1-4 week horizon).
NOT proven: only ~8 years; CI includes zero; **coin list is hindsight-selected (survivorship, direction ambiguous for momentum)**; performance is concentrated in a few coins; the most recent year is a large loss
(possible regime change/crowding); shorting requires perps (funding/liquidity) and capacity is small. The 2026 drawdown is the first thing to understand before trusting it.

## Next
(1) Point-in-time universe (historical market-cap snapshots incl. dead coins) - requires data we cannot reach yet; (2) diagnose 2026 (is it momentum crash on rebound? beta? one coin?);
(3) add orthogonal crypto signals (size, volume-shock, BTC-relative residual momentum); (4) combine with the slow premia core from iteration 5 as a diversifier.
