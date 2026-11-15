# Independent audit: `opus-multistrat-backtester` (audited 2026-11-15)

Scope: read the engine + data pipeline + 10 frozen strategies, re-run the frozen code on a fresh Yahoo download, rebuild
the testable strategies from scratch with my own (independent) engine, run train/test, cost, placebo and permutation tests.

## 1. Can the evaluation code see the future?  -> No material look-ahead found
| check | result |
|---|---|
| Engine timing (decision at close d -> trade at close d+lag -> earns d+1..) | correct; lag>=1 enforced |
| Author's look-ahead harness re-run with 9 cut dates INSIDE 2021-26 (author only used dev cuts) | s15, s12, s09, s11, s10: all pass |
| Data-cleaning look-ahead (de-spike / corp-action repair use future prices; author's harness cannot see this) | touches 20 ETF cells (0.003%), 6 futures, 331 stock cells (0.007%); both directions; immaterial |
| Truncated-raw rebuild vs full rebuild at 2015-12-31 (ret/adjclose/xret/printed/rf/sp500/tri) | 0 differing cells |
| Frozen code on a FRESH download (my data, not the author's) | s15 1.03, s12 0.52, s09 0.21, s11 0.30, s01 0.08 holdout SR: identical to the author's to 2 dp |
| Author's headline combination recomputed from their per-strategy P&L | 1.60 dev / 0.89 holdout: reproduces exactly |
| My own from-scratch s15 (raw Yahoo, own engine) | dev 0.59 (author 0.52), holdout 1.00 (author 1.03) |

## 2. Why the headline still overstates what to expect
1. **The holdout is a single 5.9-year draw.** SE of a Sharpe over 5.9y is ~0.41. The only genuinely unseen data is 2021+; the
   lock is a code convention (the same process held the 2026 data), and the model that designed the strategies may know the
   2021-25 market from its own training data. Nothing newer than 2026-11-13 exists to re-test on.
2. **The combined Sharpe is front-loaded and s15-dependent.** Combination SR: 1.21 (2021-22) -> 0.76 (2023-24) -> 0.63 (2025-26).
   s15 = 24% of weight and 34% of P&L; drop it and the holdout SR is 0.64. s15 itself: SR 1.77 in 2021-24, **-0.54 in 2025-26**.
3. **Dev rank does not predict holdout rank** (Spearman 0.27 across the 10 strategies); mean single-strategy SR 0.67 -> 0.31.
4. **"10% vol" is only while invested.** s15 is invested ~11% of days: realised vol 3.2-3.8%, return 2-3%/yr; s10 33% of days.
5. **Cost sensitivity.** s15 (author's engine): holdout SR 1.03 -> 0.87 (2x) -> 0.70 (3x) -> 0.36 (5x); dev 0.51 -> 0.36 -> 0.20 -> -0.10.
   In my engine at 5 bps one-way (vs 2 assumed) the 2014-26 SR is 0.31; at 10 bps it is ~0. s09 / s10 / s01 vanish at 2-3x.
6. **Survivorship / hindsight lists:** S&P stocks (Yahoo lacks delisted names, ~+1.7%/yr upward bias), ETF universe = ETFs alive
   today, crypto = BTC+ETH chosen ex post, commodity borrow assumed 50 bps even for UNG/USO/UGA.

## 3. Per-strategy verdict
| strategy | my evidence | verdict |
|---|---|---|
| s15 rebalancing flows | Real in the data: permutation p<0.01; placebo windows (other days of month) show no edge. But train-refit (243 configs, 2004-13) has corr(train,test SR) = -0.29, top-5 train picks test -0.17..0.42; test 2014-26 SR 0.49 [bootstrap 95% CI -0.11, 1.13]; 2007-11 SR 1.32, 2012-19 0.42, 2020 -0.82; 2025-26 negative; very cost-sensitive. | **Promising, not proven.** Plan on SR 0.4-0.6 not 1.0. |
| s11 commodity carry | Own rebuild: dev 0.45, holdout 0.51 (author 0.28/0.30). ~Half of gross P&L is short UNG. Borrow 1500 bps on UNG/USO/UGA: SR train -0.06, test 0.25. t~1.5. Realised DD -21..-28% at 10% vol. | **Plausible but small, borrow-dependent.** |
| s12 crypto trend | Correlation with BTC buy&hold 0.93-0.97; Sharpe ~ BTC buy&hold (1.17 vs 1.11 full; 0.63 vs 0.54 in 2021-26). 2025-26 SR -0.66. | **Not an edge: it is BTC beta.** Fine as a deliberate crypto allocation, not as alpha/diversifier. |
| s10 month-turn flows | Bond-extension effect decayed: TLT days -2,-1 +25 bps (t 4.3) in 2004-13 -> +6 bps (t 1.5) in 2014-26. Equity leg never significant (t~1). Placebo-shifted windows reach SR 0.71 in test. Dies at 3x cost. | **Reject / insufficient evidence.** |
| s09 VRP timing | My rebuild holdout SR 0.06; "always short VIXY" 0.51 > timed 0.06; shifting execution one day moves holdout SR 0.21 -> 0.72 (pure noise); +30% VIXY day on 2026-11-05 cost ~5.7% at 10% sizing. | **Reject.** |
| s01 GP trend | Plain trend-following (author: evidence weighting adds nothing). Holdout 0.08, DD -33%, negative at 2x cost. | **Reject.** |
| s08 ML ranker, s16 auctions | Holdout -0.21 / -0.17 (author). s08 survivor universe. Not re-run. | **Reject.** |
| s02 factor momentum, s05 principal portfolios | **Could not be re-tested**: Ken French (and TreasuryDirect/FOMC) hosts are blocked from this sandbox. Code read: no look-ahead found; but both trade paper portfolios with an *assumed* 100-300 bps/yr implementation drag; holdout SR ~0.53 / DD -12% / -22%. | **Unverified.** Re-run on a machine with network access before any decision. |

## 4. Limits of this audit
* s02, s05, s16 and the FOMC leg of s10 were not reproduced (blocked data sources). s10 numbers above exclude FOMC.
* My "train/test" windows (2004-13 / 2014-26) overlap the author's own development period, so they test stability, not
  unseen-data performance. 2021-26 is the only clean window and the author has already used it once.
* The train-refit exercise was done in full only for s15; the others were tested with fixed parameters across sub-periods.
* Nothing here is investment advice; Sharpe estimates over <=6 years carry +-0.8 (95%) uncertainty.

Scripts: `audit/t_s15.py, t_s12_s09.py, t_s11_s10.py, leak_test.py, cleaned_cells.py, repro.py, stress.py, lookahead_holdout.py`
(these import the audited worktree at `/tmp/claude-0/audit` and the raw Yahoo download under its `data/raw`).
