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
