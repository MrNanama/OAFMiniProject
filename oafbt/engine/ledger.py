"""Append-only trials ledger: every backtest variant that is evaluated gets a row. Used to report the number
of trials behind each headline Sharpe and to deflate it (DSR). One CSV per strategy to avoid write conflicts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd

from ..config import RESULTS

LEDGER = RESULTS / "ledger"


def params_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]


def log_trial(strategy: str, version: str, params: dict, stats: dict, note: str = "", tag: str = "dev"):
    LEDGER.mkdir(parents=True, exist_ok=True)
    row = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "strategy": strategy,
           "version": version, "tag": tag, "phash": params_hash(params), "params": json.dumps(params, default=str),
           "sharpe": stats.get("sharpe"), "ann_ret": stats.get("ann_ret"), "max_dd": stats.get("max_dd"),
           "turnover_yr": stats.get("turnover_yr"), "note": note}
    f = LEDGER / f"{strategy}.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)


def n_trials(strategy: str | None = None, tag: str = "dev") -> int:
    """number of DISTINCT parameterisations evaluated (per strategy, or across all if None)"""
    if not LEDGER.exists():
        return 0
    files = [LEDGER / f"{strategy}.csv"] if strategy else sorted(LEDGER.glob("*.csv"))
    n = 0
    for f in files:
        if f.exists():
            d = pd.read_csv(f)
            d = d[d["tag"] == tag]
            n += d.drop_duplicates(subset=["strategy", "version", "phash"]).shape[0]
    return n


def trials_sr_std(tag: str = "dev") -> float | None:
    """cross-sectional std of Sharpe across all logged trials (for DSR); None if too few"""
    if not LEDGER.exists():
        return None
    d = pd.concat([pd.read_csv(f) for f in LEDGER.glob("*.csv")], ignore_index=True)
    d = d[d["tag"] == tag].drop_duplicates(subset=["strategy", "version", "phash"])
    return float(d["sharpe"].std()) if len(d) > 5 else None
