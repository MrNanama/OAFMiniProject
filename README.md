# opus-multistrat-backtester

Author: **opus** (an AI agent; branch naming per the repo conventions). Status: **complete** - research frozen at
commit `9f93ca5` (`FROZEN`), one-shot holdout evaluated. Full history: `journal/ITERATION_LOG.md`.

## Goal
Build a realistic back-tester on real market data, derive **ten new strategies** inspired by the *types* of ideas in
the OAF pitch decks (without reusing the deck ideas themselves), iterate on them for six rounds to maximise
risk-adjusted performance (replacing ideas with no promise), and then test the final versions **once** on data never
seen during research.

* Research (dev) data: up to **2020-12-31**.
* Holdout: **2021-01-01 to 2026-11-13**, locked in code (`data/vault` is only readable after the `FROZEN` commit).

## Headline results (net of costs, every strategy scaled to 10% ex-ante vol)

| | Sharpe dev (2005-2020) | **Sharpe holdout (2021-2026)** | holdout return / vol | holdout max DD | holdout positive years |
|---|---|---|---|---|---|
| **Equal-risk combination of the 10 strategies** | 1.60 | **0.89** (t = 2.2) | 6.4% / 7.2% | **-8.2%** | **6/6** |
| SPY (excess of cash), same holdout | - | 0.68 | 11.4% / ~17% | -25.3% | - |

| # | strategy (final version) | idea (deck type that inspired it) | SR dev 2004-20 | **SR holdout** | holdout max DD | dev trials |
|---|---|---|---|---|---|---|
| s15 | rebalancing flows (v3) | balanced-fund rebalancing pushes the stock-bond spread against month-to-date performance at month end, then reverts (flows; macro cross-asset) | 0.52 | **1.03** | -5.8% | 4 |
| s02 | factor momentum (v2) | time-series momentum in US + international equity factors (cross-sectional-momentum deck -> momentum of factors) | 0.68 (1975-2020: 1.06) | **0.54** | -12.1% | 2 |
| s05 | principal portfolios (v1) | cross-asset lead-lag from the eigen-decomposition of the prediction matrix (Macro Momentum deck's PCA/latent factors) | 0.56 | **0.53** | -21.6% | 4 |
| s12 | crypto trend (v1) | multi-horizon time-series momentum on BTC/ETH, long/flat (momentum type, new asset class) | 1.76 (2015-20) | **0.52** | -16.9% | 1 |
| s11 | cross-asset carry (v3) | commodity carry measured as realised roll yield = ETF return minus Yahoo's non-roll-adjusted future (new premium; macro type) | 0.28 | **0.30** | -26.4% | 6 |
| s10 | month-turn flows (v4) | Treasury index-extension demand on days -2/-1, equity turn-of-month +1..+3, scheduled FOMC days (flows) | 0.93 | **0.29** | -9.4% | 8 |
| s09 | VRP timing (v1) | short VIX futures ETP only in contango with implied > realised vol (wildcard diversifier) | 0.71 (2011-20) | **0.21** | -12.3% | 2 |
| s01 | Gaussian-process trend (v4) | Bayesian GP trend across asset classes, lengthscales combined by GP evidence (Bayesian Quadrature deck) | 0.41 | **0.08** | -33.4% | 6 |
| s16 | Treasury auction cycle (v2) | long bonds cheapen into 10/20/30y auctions and recover after (supply flows) | 0.54 | **-0.17** | -19.7% | 3 |
| s08 | ML stock ranker (v2) | walk-forward gradient boosting + ridge on 15 price/volume features, residual target, beta-neutral (ML / regularised learning) | 0.33 | **-0.21** | -30.2% | 2 |

Holdout file: `results/holdout/holdout.md` (incl. holdout correlations: max |rho| 0.34). Ken French-based strategies (s02,
s05) end 2026-09 (data availability).

### Honest verdict
* **The combination is the product.** Single-strategy Sharpe ratios decayed from 0.73 (dev mean) to 0.31 (holdout
  mean), roughly the 30-60% haircut the literature and our trial counts predict. Diversifying ten low-correlated
  streams kept the combined Sharpe at **0.89** with a -8% drawdown, positive in every holdout year, versus SPY's 0.68
  with -25%.
* **What held up**: rebalancing flows (improved out of sample, consistent with the growth of rebalancing AUM), factor
  momentum, principal portfolios, crypto trend, commodity carry.
* **What failed out of sample**: the ML stock ranker (survivor-only large-cap universe; the reviewer had flagged that
  its dev headline leaned on 2000-04) and the Treasury auction cycle. Calendar strategies (s10) and the VRP harvest
  kept a positive but much smaller edge.
* Multiple testing: 51 distinct dev variants across 16 strategy ideas (+36 sensitivity runs, reported but not used for
  selection); every run is in `results/ledger/`. Six ideas were retired on dev evidence (below).

## What was tried and retired (dev evidence; nothing was retired using the holdout)
| idea (deck type) | dev SR 2004-20 | why retired |
|---|---|---|
| s03 PCA eigenportfolio stat-arb (RL pairs deck -> breadth) | -0.02 / +0.02 | gross edge ~0.7 SR, but the per-round-trip edge (13-15 bps) is below costs after 2008 |
| s04 high-volume premium; expected-attention / earnings-cycle volume seasonality (Volume Filters deck) | -0.26 / -0.08 | no net premium in S&P 500 large caps |
| s06 Treasury curvature butterfly (Macro deck's Salomon PCA) | -0.49 | a 2%-vol ETF butterfly cannot pay ETF costs + short borrow at a 3x leverage cap |
| s07 industry return seasonality (momentum at annual lags) | 0.06 | real long-run effect (1975-2020 SR 0.37) but monthly reshuffling forces ~45x turnover |
| s13 residual momentum (candidate) | -0.14 | rejected |
| s14 betting against beta / correlation | 0.28 / 0.33 | weakest and decaying; replaced by s16 |

**Lesson:** in S&P 500 large caps, simple price/volume signals did not survive costs after 2007 (consistent with the
earlier agent's findings on this repo). What survived are low-turnover or structurally motivated premia: flows,
supply, carry, the variance risk premium and factor momentum.

## Data (all real, downloaded by `python -m oafbt.data.fetch`, not committed)
| source | content | use |
|---|---|---|
| Yahoo Finance | 101 ETFs, 20 index mutual funds, 41 continuous futures, 28 FX spot, 33 indices (VIX family, yields, world indices), BTC/ETH | ETFs, `_X` long-history twins (ETF spliced with its index fund) and crypto are tradeable; futures, FX spot and indices are **signal-only** (Yahoo continuous commodity futures omit roll yield: CL=F +9%/yr vs USO -11%/yr, 2007-20; spot FX lacks carry and is stamped one session late since 2010-08-04) |
| Yahoo + fja05680/sp500 | 532 historical S&P 500 members (dev) with point-in-time membership | members traded only while in the index; renamed members re-linked; reused tickers excluded; **survivorship remains** (Yahoo lacks most delisted names; ~+1.7%/yr bias for a long-only EW book) |
| Kenneth French data library | 29 daily tables (US/intl factors, 49/12 industries, 25 size-B/M, momentum portfolios, 1926-2026) | survivorship-free research universe; paper portfolios -> implementation drag charged (100-300 bps/yr on factors) |
| federalreserve.gov | scheduled FOMC announcement dates 1994-2027 | s10 (schedule published a year ahead) |
| TreasuryDirect API | 1,700 note/bond auctions 1998-2026 with **announcement** dates | s16 (an auction is usable only after it is announced) |

Data audit: `results/data_audit.md` (bad prints, spikes, gaps, splice checks, futures-vs-ETF roll gap, survivorship and
placebo tests: a random monthly long-short stock book loses exactly its costs).

## Back-tester (`oafbt/`)
* **Holdout lock**: research code reads `data/processed/dev`; `data/vault` (full history) opens only after `FROZEN`.
* **Timing**: strategies emit target weights by decision date (information up to that close); execution at the next
  close (lag >= 1 enforced); positions drift between rebalances; orders for assets without a print stay pending.
* **Costs** (one-way): ETFs 2-4 bps, VIX ETPs 10, stocks 5, crypto 15, KF factors 10 (+ drag), KF portfolios 5; short
  borrow 30-300 bps/yr; 50 bps/yr on leverage; holding drag for paper portfolios.
* **Sizing**: each rebalance row scaled to 10% ex-ante vol (historical simulation), leverage caps 3x / 2x (stocks).
* **Integrity**: look-ahead harness (5 edge-date cuts + random cuts + cuts at the strategy's own decision dates,
  process-wide truncation, fresh instances, hold-vs-flat and column-union comparisons; regression tests prove it
  catches leaky strategies), 2x-cost and lag+1 sensitivity, block-bootstrap CIs, PSR/DSR with the logged trial counts.
* **Reviews**: an adversarial engine/data review (80 findings, the confirmed ones fixed before research continued) and
  a final independent strategy review before the freeze (`journal/ITERATION_LOG.md`, Round 6).
* Tests: `pytest tests/` (17 tests).

## How to run
```bash
pip install -r requirements.txt && pip install -e .
python -m oafbt.data.fetch            # download (~15 min)
python -m oafbt.data.store build      # clean/align, write dev panels + vault
python scripts/run_strategy.py s02_factor_momentum      # dev backtest + checks
python scripts/round_summary.py 6                       # leaderboard, correlations, combination
python scripts/sensitivity.py                           # parameter plateaus
python scripts/run_holdout.py                           # only works when FROZEN exists
```

## Files
`oafbt/` package (data, engine, strategies) | `scripts/` | `tests/` | `results/rounds/` leaderboards per round |
`results/runs/<strategy>/<version>/` metrics, daily P&L, tear sheets | `results/ledger/` every trial |
`results/holdout/` | `results/sensitivity.md` | `journal/deck_analysis.md` | `journal/ITERATION_LOG.md`

## Caveats
Yahoo is not a survivorship-free or institutional-grade source; stock results are upper bounds. Ken French portfolios
are paper portfolios (implementation drag is an estimate). Crypto has only five dev years. Calendar strategies assume
market-on-close execution on scheduled days. Short-borrow availability is assumed at the stated fees.
