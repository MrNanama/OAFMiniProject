"""Integrity checks every strategy must pass."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.store import DataStore


def lookahead_check(strategy, cut_dates=("2008-06-30", "2014-03-31", "2019-09-30"), atol=1e-9) -> dict:
    """Weights decided on or before each cut date must be IDENTICAL whether or not data after the cut exists.
    Any difference means the strategy used future information (centred windows, full-sample scaling, bfill...)."""
    full = strategy.target_weights(DataStore()).astype(float)
    out = {}
    for c in cut_dates:
        c = pd.Timestamp(c)
        part = strategy.target_weights(DataStore(end=c)).astype(float)
        a = full.loc[:c]
        b = part.reindex(index=a.index, columns=a.columns)
        # compare only where the full run trades; treat NaN==NaN as equal
        both_nan = a.isna() & b.isna()
        diff = (a.fillna(0.0) - b.fillna(0.0)).abs()
        bad = (diff > atol) & ~both_nan
        n_bad = int(bad.to_numpy().sum())
        first = None
        if n_bad:
            rows = bad.any(axis=1)
            first = str(rows[rows].index[0].date())
        # extra: a strategy must not output rows after the cut
        extra = int((part.index > c).sum())
        out[str(c.date())] = {"n_mismatch": n_bad, "first_mismatch": first, "rows_after_cut": extra,
                              "rows_compared": int(len(a))}
    out["passed"] = all(v["n_mismatch"] == 0 and v["rows_after_cut"] == 0 for k, v in out.items() if k != "passed")
    return out
