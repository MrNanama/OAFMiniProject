# Iteration log

Protocol: each round, every active strategy is diagnosed and improved on DEV data only (<= 2020-12-31) with a small
trial budget (every run is logged in `results/ledger/`). A change is kept only if it has an economic/diagnostic reason and
does not rely on a single sub-period. Strategies that remain clearly inferior with no credible path are replaced.
Ranking score = 0.5*SR(2004-2020) + 0.5*min(SR 2004-2020, SR at 2x cost, SR with one extra day of lag).

## Round 0 (initial implementations)
* s02 factor momentum v1 -> v2 (done early as the reference strategy): 2004-20 SR 0.36 -> 0.75 by removing paper-only
  reversal factors (800 bps/yr drag), redundant regions, and 1-month sign flips (turnover 23x -> 15x).
* s10 turn-of-month v1 (SPY days -1..+3, SR 0.23) -> v2 month-turn flows (Treasuries days -2..-1 for index
  extension + SPY days +1..+3): SR 0.82. The day windows came from a dev-period event study (3 exploratory looks,
  logged as trials); the Treasury effect holds in raw prices for 6 bond ETFs independently (t = 2.4 to 6.1).
* Research note (s10, logged as exploratory trial): SPY close-to-close return on scheduled FOMC announcement days
  = +21 bps (t=2.1) in 1994-2007 and +35 bps (t=2.8) in 2008-2020 (other days 1-5 bps); day -1/+1 not stable across
  halves. Candidate sleeve for s10 v3: long SPY over announcement days (schedule published a year ahead, scraped from
  federalreserve.gov -> `ds.fomc_dates()`).

## Engine/data review (between round 0 and round 1)
An adversarial review (3 independent reviewers: look-ahead, accounting, data) produced 80 findings; the confirmed
high/medium ones were fixed before any further research (`git log` for details):
* the data layer itself was untracked (`.gitignore` 'data/' matched `oafbt/data/`) - fixed;
* dev panels contained 84 stocks that only joined the S&P 500 after 2020 (future index additions) - dev now holds
  only stocks that were members by 2020; renamed members (FB->META etc.) re-linked; reused tickers excluded;
* Yahoo spot FX is stamped one session late since 2010-08-04 -> FX delayed consistently; FX/futures/indices are now
  signal-only (non-tradable), lag >= 1 enforced;
* cleaning deleted genuine crash days (AIG 2008-09-17, FNMA -90%) and dropped returns across data holes -> spikes are
  masked only if they revert AND have no volume surge; returns across gaps are booked on resumption; corporate-action
  adjustment errors (double-counted/misdated) repaired;
* risk-free from ^IRX (KF daily RF is rounded to 1bp/day); drawdown counts initial capital; DSR counts the current
  trial; perf trims only the engine warm-up; ex-ante vol uses only days with prices; row normalisation removes the
  dependence on a strategy's weight scale; pending orders execute on the next print;
* the look-ahead harness was too weak (aligned cut dates, NaN==0, cached data). New harness: 5 edge-date cuts +
  random cuts + cuts at the strategy's own rebalance dates, process-wide truncation, fresh instances, hold-vs-flat and
  column-union comparisons; regression tests prove it catches the reviewers' leaky strategies.

## Round 0 results (corrected engine)
See `results/rounds/round_0.md`. Strong: s09 VRP (0.71, 2011-20), s02 factor momentum (0.68; 1.06 over 1975-2020),
s10 month-turn flows (0.82), s05 principal portfolios (0.56). Weak: s01 GP trend (0.31), s07 seasonality (0.11).
Failing: s03 PCA stat-arb (-0.02: gross edge eaten by 49x turnover), s04 volume attention (-0.26: no gross premium
in large caps), s08 ML ranker (-0.43: worked 2000-06, negative after; -68% DD in 2009), s06 curve butterfly (-0.49:
structural - a 2%-vol ETF butterfly cannot pay ETF costs and short-borrow at a 3x leverage cap).
Equal-risk combination of all 10: Sharpe 0.48 (2005-2020).

### Round 1 decisions
* RETIRE s06_curve_pca_rv (structural cost problem, no credible path) -> NEW s11_cross_asset_carry.
* s04: pivot within the volume type to expected attention (earnings-cycle volume seasonality).
* s03, s08: one substantive attempt each; replaced in round 2 if still not positive.
* Others: targeted improvements (costs, drawdown control, additional sleeves with ex-ante rationale).

## Round 1 results
Leaderboard: `results/rounds/round_1.md`. **Equal-risk combination of the 10 active strategies: Sharpe 1.28
(2005-2020), 12.1%/yr at 9.4% vol, max DD -12.2%, 15/16 positive years** (round 0: 0.48).
| change | before -> after (SR 2004-20) | why |
|---|---|---|
| s10 v3: + scheduled-FOMC-day SPY sleeve | 0.82 -> 0.95 | pre-announcement drift, positive in both halves of the dev sample |
| s06 retired -> s11 cross-asset carry | -0.49 -> 0.43 (v2) | carry inferred from prices; v1 all classes -0.03; per-class diagnostics: commodity roll yield +0.36, FX -0.44, equity div-yield -0.12; v2 = commodities only, 1y window (in-sample class choice, logged) |
| s04 -> v2 expected attention, then retired | -0.26 -> -0.08 | second volume-type idea also has no net premium in large caps |
| s03 -> v2 stricter entry, then retired | -0.02 -> 0.02 | cost wall is structural |
| new s12 crypto trend (replaces s04) | 1.76 (2015-20) | long-only multi-horizon TSMOM on BTC/ETH; short sample, CI 0.77-2.71 |
| new s14 BAB (replaces s03) | 0.28 | beat the other candidate, s13 residual momentum (-0.14, rejected) |
| s08 v2: residual target, rolling 10y fit, beta-neutral | -0.43 -> 0.33 | DD -68% -> -25% |
| s01 v2: drop 1-month GP model, 2y window | 0.31 -> 0.32 | lower turnover (27x -> 20x), little change |
| s07 v2: rank buffer | 0.11 -> 0.06 | turnover barely moves (seasonality reshuffles every month) -> replacement candidate |
| s02, s05, s09 | unchanged | drawdowns are pre-1983 (few factors); no change warranted |
Lesson: in S&P 500 large caps, simple price/volume signals (reversal, volume, residual momentum, raw ML) do not
survive costs after 2007; what works there is low-turnover, beta-neutral construction (s08 v2, s14).
