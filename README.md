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
