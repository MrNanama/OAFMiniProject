# opus-multistrat-backtester

Author: **opus** (an AI agent; branch naming per the repo conventions). Status: **in progress** - see `journal/ITERATION_LOG.md`.

## Goal
Build a realistic back-tester on real market data, derive **ten new strategies** inspired by the *types* of ideas in the
OAF pitch decks (without reusing the deck ideas themselves), and iterate on them over ~6 rounds to maximise risk-adjusted
performance, replacing ideas that show no promise. Every number reported during iteration is from the **dev period
(<= 2020-12-31)**; **2021-01-01 to 2026-11-13 is a locked holdout** evaluated once at the very end.

## Data (all real, downloaded by `python -m oafbt.data.fetch`, not committed)
| source | content | use |
|---|---|---|
| Yahoo Finance | 101 ETFs, 20 index mutual funds (long-history proxies), 41 continuous futures, 28 FX spot, 33 indices (VIX family, yields, world indices), BTC/ETH | ETFs (and `_X` long-history twins) are the tradeable multi-asset instruments; futures/indices are **signals only** (Yahoo continuous commodity futures omit roll yield: CL=F +9%/yr vs USO -11%/yr, 2007-20) |
| Yahoo Finance + fja05680/sp500 | 759 of 1,209 historical S&P 500 members (1996-2026) with point-in-time membership | stock strategies trade only names that were members *at the time*; remaining survivorship bias measured at ~+1.7%/yr for a long-only EW book |
| Kenneth French data library | 29 daily tables: US/intl. factors, 49/12 industries, 25 size-B/M, momentum portfolios (1926-2026) | survivorship-free research universe (paper portfolios -> implementation drag charged) |

Data audit: `results/data_audit.md` (bad prints, spikes, stale prices, splice checks, futures-vs-ETF roll gap, survivorship and placebo tests).

## Back-tester (`oafbt/`)
* **Holdout lock**: research code reads `data/processed/dev` (truncated at 2020-12-31); full data sits in `data/vault` and is readable only once a `FROZEN` file exists.
* **Timing**: strategies emit target weights by *decision date* using information up to that close; execution at the next close (lag 1); positions drift between rebalances.
* **Costs** (one-way): ETFs 2-4 bps, VIX ETPs 10, stocks 5, FX 1, crypto 15, KF factors 10 (+100-300 bps/yr drag), KF portfolios 5 (+10 bps/yr); short borrow 30-300 bps/yr; 50 bps/yr spread on leverage.
* **Sizing**: every strategy scaled to 10% ex-ante volatility (historical simulation of today's weights), leverage caps 3x (multi-asset/KF), 2x (stocks).
* **Integrity checks per run**: look-ahead truncation test (weights up to date c must not change when data after c is removed), 2x-cost and lag+1 sensitivity, block-bootstrap Sharpe CI, PSR/DSR with the number of logged trials (`results/ledger/`).
* Tests: `pytest tests/`.

## How to run
```bash
pip install -r requirements.txt && pip install -e .
python -m oafbt.data.fetch            # download (~10 min)
python -m oafbt.data.store build      # clean/align, write dev panels + vault
python scripts/run_strategy.py s02_factor_momentum
python scripts/round_summary.py 1
```

## Strategies
See `journal/deck_analysis.md` for the deck analysis and the idea map. Results table: filled in at the end of each round
(`results/rounds/round_<n>.md`).
