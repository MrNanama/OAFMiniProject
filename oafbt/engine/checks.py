"""Integrity checks every strategy must pass.

lookahead_check: weights decided on or before a cut date c must be IDENTICAL whether or not data after c exists.
* many cuts: random sessions (seeded) + fixed edge cases (Friday, Wednesday, month start, day before a holiday,
  mid-month), so no rebalance schedule can hide behind aligned cut dates;
* the cut is enforced PROCESS-WIDE (store.global_end), so DataStore() objects a strategy creates internally are
  truncated too, and every run uses a FRESH strategy instance (no caching across runs);
* hold rows (all-NaN) and flat rows (all-zero) are different decisions and are compared separately;
* the UNION of columns is compared (weight in a column that exists in only one run is a mismatch);
* asset_returns() overrides are compared as well;
* a run on truncated data must not emit rows after the cut.
Data-level leaks (cleaning that used future prices, survivorship) are outside what this test can see - see the
data audit for those.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .. import config as C
from ..data.store import DataStore, global_end

FIXED_CUTS = ["2005-03-24", "2010-05-07", "2012-03-01", "2016-11-09", "2018-06-15"]


def _cuts(n_random: int, seed: int) -> list:
    from ..strategies.utils import nyse_sessions
    ses = nyse_sessions()
    ses = ses[(ses >= pd.Timestamp("2003-01-01")) & (ses <= pd.Timestamp(C.DEV_END) - pd.Timedelta(days=30))]
    rng = np.random.default_rng(seed)
    rnd = list(pd.DatetimeIndex(rng.choice(ses, size=n_random, replace=False)))
    return sorted(set(rnd) | {pd.Timestamp(x) for x in FIXED_CUTS})


def _fresh(strategy):
    return type(strategy)(**copy.deepcopy(strategy.params))


def lookahead_check(strategy, n_random: int = 2, n_reb: int = 4, seed: int = 7, atol: float = 1e-9,
                    cuts=None) -> dict:
    """cuts = 5 fixed edge dates + n_random random sessions + n_reb of the strategy's OWN rebalance dates (sampled
    from the full run after its first year). Cutting exactly at a decision row is what exposes leaks that only
    change the last decision before the cut (e.g. 'go flat if the market falls in 2 days')."""
    full_ds = DataStore()
    full = _fresh(strategy).target_weights(full_ds).astype(float)
    if cuts is None:
        cuts = _cuts(n_random, seed)
        rows = full.index[full.notna().any(axis=1)]
        rows = rows[(rows >= rows[0] + pd.Timedelta(days=365)) & (rows <= pd.Timestamp(C.DEV_END) - pd.Timedelta(days=10))] \
            if len(rows) else rows
        if len(rows):
            rng = np.random.default_rng(seed + 1)
            pick = rng.choice(len(rows), size=min(n_reb, len(rows)), replace=False)
            cuts = sorted(set(cuts) | set(rows[pick]))
    overrides_returns = type(strategy).asset_returns is not getattr(type(strategy).__mro__[-2], "asset_returns", None)
    out = {}
    for c in cuts:
        c = pd.Timestamp(c)
        with global_end(c):
            ds_c = DataStore(end=c)
            st = _fresh(strategy)
            part = st.target_weights(ds_c).astype(float)
            ret_part = None
            if overrides_returns and len(full.columns):
                cols = [x for x in full.columns][:200]
                ret_part = st.asset_returns(ds_c, cols)
        a = full.loc[:c]
        U = a.columns.union(part.columns)
        a2 = a.reindex(columns=U)
        b2 = part.reindex(index=a.index, columns=U)
        hold_a, hold_b = a2.isna().all(axis=1), b2.isna().all(axis=1)
        hold_mis = int((hold_a != hold_b).sum())
        both = ~hold_a & ~hold_b
        diff = (a2[both].fillna(0.0) - b2[both].fillna(0.0)).abs()
        val_mis = int((diff > atol).to_numpy().sum())
        first = None
        if hold_mis or val_mis:
            rows = (hold_a != hold_b) | pd.Series((diff > atol).any(axis=1), index=a2.index).reindex(a2.index).fillna(False)
            first = str(rows[rows].index[0].date()) if rows.any() else None
        ret_mis = 0
        if ret_part is not None:
            rf_full = _fresh(strategy).asset_returns(full_ds, list(ret_part.columns)).loc[:c]
            rr = (rf_full - ret_part.reindex(index=rf_full.index)).abs()
            ret_mis = int((rr > 1e-12).to_numpy().sum())
        extra = int((part.index > c).sum())
        out[str(c.date())] = {"hold_mismatch": hold_mis, "value_mismatch": val_mis, "returns_mismatch": ret_mis,
                              "first_mismatch": first, "rows_after_cut": extra, "rows_compared": int(len(a))}
    out["passed"] = all(v["hold_mismatch"] == 0 and v["value_mismatch"] == 0 and v["returns_mismatch"] == 0
                        and v["rows_after_cut"] == 0 for k, v in out.items() if k != "passed")
    return out
