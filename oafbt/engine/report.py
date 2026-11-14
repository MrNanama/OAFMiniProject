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


def combined(returns: dict, start=None, end=None, target_vol=0.10, lookback=252) -> pd.Series:
    """inverse-vol (equal-risk) combination of strategy return streams, weights from trailing data only,
    rebalanced monthly, then scaled to target vol with trailing realised vol (lagged)."""
    R = pd.DataFrame(returns).sort_index()
    if start is not None:
        R = R[R.index >= pd.Timestamp(start)]
    if end is not None:
        R = R[R.index <= pd.Timestamp(end)]
    vol = R.rolling(lookback, min_periods=63).std().shift(1)
    w = (1.0 / vol)
    w = w.div(w.sum(axis=1), axis=0)
    month_end = R.index.to_series().groupby(R.index.to_period("M")).transform("max") == R.index.to_series()
    w = w.where(month_end).ffill().shift(1)            # decided at month end, applied from next day
    port = (w * R.fillna(0.0)).sum(axis=1)
    rv = port.rolling(lookback, min_periods=63).std().shift(1) * np.sqrt(252)
    s = (target_vol / rv).clip(upper=5.0)
    return (port * s).dropna()


def corr_matrix(returns: dict, start=None, end=None) -> pd.DataFrame:
    R = pd.DataFrame(returns)
    if start is not None:
        R = R[R.index >= pd.Timestamp(start)]
    if end is not None:
        R = R[R.index <= pd.Timestamp(end)]
    R = R.loc[:, R.std() > 0]
    return R.resample("W").sum().corr()
