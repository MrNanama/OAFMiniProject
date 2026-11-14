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
