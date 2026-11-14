"""s07 - Same-calendar-month return seasonality across industries (momentum TYPE at annual lags; NEW object).

Thesis (Heston & Sadka 2008; Keloharju, Linnainmaa & Nyberg 2016 'Return Seasonalities'): an industry that did well
in a given calendar month in past years tends to do well in that month again (recurring cash-flow news, seasonal
demand, tax/dividend/fiscal calendars, recurring investor flows). Using 'same-month minus other-month' average
returns removes the industry's general level, isolating the seasonal component.

v1: Ken French 49 industries (value-weighted, excess returns). Signal for month m+1, formed at the last trading day
of month m: mean return in that calendar month over the previous N=20 years (>= 10 obs) minus the mean over all
other months in the same 20-year window. Long top 10, short bottom 10, equal weight, monthly.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import kf_portfolio_returns, period_ends, keep_rows

TABLE = "49_Industry_Portfolios_daily|VW"


class IndustrySeasonality(Strategy):
    name = "s07_industry_seasonality"
    version = "v1"
    universe = "kf"
    thesis = "Industries repeat their calendar-month relative performance (return seasonality)."
    inspired_by = "cross-sectional momentum deck (momentum type, at annual lags)"

    def default_params(self):
        return {"years": 20, "min_obs": 10, "n_side": 10}

    def asset_returns(self, ds, columns):
        return kf_portfolio_returns(ds, TABLE, prefix="KF:P:49VW").reindex(columns=columns)

    def target_weights(self, ds):
        p = self.params
        D = kf_portfolio_returns(ds, TABLE, prefix="KF:P:49VW")
        ends = period_ends(D.index, "M")
        lr = np.log1p(D)
        grp = np.searchsorted(ends, D.index)                     # month bucket of each day (<= its month end)
        ok = grp < len(ends)
        M = np.expm1(lr[ok].groupby(grp[ok]).sum(min_count=1))
        M.index = ends[: len(M)]
        cnt = D[ok].notna().groupby(grp[ok]).sum()
        cnt.index = M.index
        M = M.where(cnt >= 10)                                   # need most of the month's days
        N = p["years"]
        same_sum = 0
        same_n = 0
        for k in range(1, N + 1):
            x = M.shift(12 * k - 1)                               # same calendar month as month m+1, k years ago
            same_sum = same_sum + x.fillna(0.0)
            same_n = same_n + x.notna()
        tot_sum = M.rolling(12 * N, min_periods=12 * p["min_obs"]).sum()
        tot_n = M.notna().astype(float).rolling(12 * N, min_periods=12 * p["min_obs"]).sum()
        same_mean = same_sum / same_n.where(same_n >= p["min_obs"])
        other_mean = (tot_sum - same_sum) / (tot_n - same_n)
        sig = (same_mean - other_mean).where(M.notna())
        rk = sig.rank(axis=1, ascending=False)
        nvalid = sig.notna().sum(axis=1)
        n = p["n_side"]
        W = (rk <= n).astype(float) / n - (rk > (nvalid - n).values[:, None]).astype(float) / n
        W = W.where(sig.notna(), 0.0)
        W = W[nvalid >= 2 * n]
        W = W.reindex(D.index)
        return keep_rows(W, W.dropna(how="all").index)


STRATEGY = IndustrySeasonality
