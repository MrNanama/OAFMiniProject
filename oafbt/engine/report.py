"""Cross-strategy summaries: metrics table, correlation matrix, equal-risk combined portfolio."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from .metrics import perf_stats, fmt_table

RUNS = C.RESULTS / "runs"


def load_run(name: str, version: str) -> tuple[dict, pd.DataFrame]:
    d = RUNS / name / version
    return json.loads((d / "metrics.json").read_text()), pd.read_parquet(d / "daily.parquet")


def combined(returns: dict, start=None, end=None, target_vol=0.10, lookback=252, max_gross=3.0,
             cost_bps=5.0) -> pd.Series:
    """inverse-vol (equal-risk) combination of strategy return streams. Each stream is NaN before its first
    non-zero day (warm-up is not 'zero risk'); vols are floored at 0.25x the cross-sectional median; weights are
    decided at month end from trailing data and applied from the next day; the overlay is scaled to target vol with
    lagged realised vol, total gross capped, and weight changes charged `cost_bps`."""
    R = pd.DataFrame(returns).sort_index()
    for c in R.columns:
        nz = R[c].fillna(0).ne(0)
        R.loc[: nz.idxmax() if nz.any() else R.index[-1], c] = np.nan
    vol = R.rolling(lookback, min_periods=63).std()
    floor = vol.median(axis=1) * 0.25
    vol = vol.clip(lower=floor, axis=0)
    w = (1.0 / vol)
    w = w.div(w.sum(axis=1), axis=0)
    month_end = R.index.to_series().groupby(R.index.to_period("M")).transform("max") == R.index.to_series()
    w = w.where(month_end).ffill().shift(1).fillna(0.0)
    port = (w * R.fillna(0.0)).sum(axis=1)
    live = w.sum(axis=1) > 0
    rv = port.where(live).rolling(lookback, min_periods=63).std().shift(1) * np.sqrt(252)
    s = (target_vol / rv).clip(upper=max_gross).fillna(0.0)
    out = port * s - (w.mul(s, axis=0).diff().abs().sum(axis=1).fillna(0.0) * cost_bps / 1e4)
    out = out[s > 0]
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    return out


def corr_matrix(returns: dict, start=None, end=None) -> pd.DataFrame:
    R = pd.DataFrame(returns)
    if start is not None:
        R = R[R.index >= pd.Timestamp(start)]
    if end is not None:
        R = R[R.index <= pd.Timestamp(end)]
    R = R.loc[:, R.std() > 0]
    return R.resample("W").sum().corr()
