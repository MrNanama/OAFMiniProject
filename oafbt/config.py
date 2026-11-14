"""Global configuration: paths, the dev/holdout split, risk defaults and the cost model.

THE HOLDOUT RULE: everything after DEV_END is invisible to research code. `oafbt.data.store` truncates
every panel at DEV_END unless the repository contains a FROZEN file (created exactly once, after the final
round, together with a git tag). Never create FROZEN while iterating.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                 # gitignored, rebuilt by scripts/fetch_data.py
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
JOURNAL = ROOT / "journal"
FROZEN_FILE = ROOT / "FROZEN"

DEV_END = "2020-12-31"
HOLDOUT_START = "2021-01-01"

# default evaluation windows (signals may use earlier data as warm-up)
EVAL_START = {"multiasset": "2004-01-01", "stocks": "2000-01-01", "kf": "1975-01-01", "crypto": "2015-01-01"}

TARGET_VOL = 0.10                    # every strategy is scaled to 10% annualised ex-ante volatility
LEV_CAP = {"multiasset": 3.0, "stocks": 2.0, "kf": 3.0, "crypto": 1.5}
TRADING_DAYS = 252

# ---- cost model (one-way, in basis points of traded notional) ----
COST_BPS = {
    "etf_core": 2.0,      # SPY, QQQ, TLT, IEF, GLD, EFA, EEM, HYG, LQD ...
    "etf": 4.0,           # sector / country / commodity / currency ETFs
    "etf_vol": 10.0,      # VIX ETPs
    "fut_fin": 1.5,       # equity-index, rates and FX futures
    "fut_cmdty": 3.0,     # commodity futures
    "fx": 1.0,            # FX spot
    "crypto": 15.0,
    "stock": 5.0,         # S&P 500 members (sensitivity run at 10)
    "kf_factor": 10.0,    # trading a Ken French long-short factor (two legs of many stocks)
    "kf_port": 5.0,       # a Ken French portfolio / industry proxy (sector-ETF like)
    "mutual_fund": 2.0,   # long-history proxies, only used where ETFs did not yet exist
    "index": 2.0,         # non-tradable index used as an asset proxy
}
# annual short-borrow fee (bps of short notional) and leverage spread (bps over cash on borrowed notional)
BORROW_BPS_YR = {"stock": 30.0, "etf_core": 30.0, "etf": 50.0, "etf_vol": 300.0, "crypto": 500.0,
                 "kf_factor": 0.0, "kf_port": 30.0, "mutual_fund": 1e9, "index": 30.0,
                 "fut_fin": 0.0, "fut_cmdty": 0.0, "fx": 0.0}
LEVER_SPREAD_BPS_YR = 50.0
# Ken French factors are paper portfolios: charge an annual implementation drag for their internal rebalancing
KF_FACTOR_DRAG_BPS_YR = {"Mkt-RF": 5, "SMB": 100, "HML": 100, "RMW": 100, "CMA": 100, "Mom": 300,
                         "ST_Rev": 800, "LT_Rev": 100}
