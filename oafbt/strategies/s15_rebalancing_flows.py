"""s15 - Balanced-fund / pension rebalancing flows around month end (replaces s07; flows TYPE, NEW signal).

Thesis (Harvey, Mazzoleni & Melone 2025 'The Unintended Consequences of Rebalancing'): funds that hold fixed
stock/bond weights must SELL the asset class that outperformed during the month and BUY the laggard near month end.
This temporary price pressure moves the stock-bond spread against month-to-date (MTD) performance in the last days
of the month, and the pressure reverses once the flows are done (first days of the next month).
Dev event study (logged): corr(MTD SPY-TLT, spread on days -2..-1) = -0.25 (2007-2020), quarter-end months -0.26
(1993-2006) / -0.20 (2007-2020); corr(MTD, spread on days +1..+2) = +0.33 (2007-2020).

v1: z = MTD log return of (SPY - TLT) at the close of day -4, divided by the trailing 36-month std of that MTD measure,
clipped to +-2. Days -2..-1: hold -z x (SPY long / TLT short, each leg inverse-vol); days +1..+2: hold +z x the same
spread (decided at the close of day -1, i.e. using only the MTD measure known at day -4). Flat otherwise. The
published NYSE calendar locates day -k/+k in advance; signals use data up to the decision close.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import ewm_vol, trading_calendar
from .s10_turn_of_month import month_positions


class RebalancingFlows(Strategy):
    name = "s15_rebalancing_flows"
    version = "v2"
    universe = "multiasset"
    calendar_timed = True     # trades scheduled in advance (MOC orders): lag+1 test not meaningful
    thesis = "Rebalancing to fixed stock/bond weights pushes the stock-bond spread against MTD performance at month end."
    inspired_by = "flows type (calendar), Macro deck's cross-asset (stock vs bond) framing"

    def default_params(self):
        # v2: skip months with a weak rebalancing signal (|z| < 0.5) - they pay full costs for little expected edge
        return {"stock": "SPY", "bond": "TLT", "z_months": 36, "z_cap": 2.0, "late": [-2, -1], "early": [1, 2],
                "z_min": 0.5}

    def target_weights(self, ds):
        p = self.params
        a, b = p["stock"], p["bond"]
        X = ds.xret([a, b])
        cal = ds.calendar
        full = trading_calendar(ds)
        fe, fs = month_positions(full)
        lr = np.log1p(X[a]) - np.log1p(X[b])
        per = cal.to_period("M")
        mtd = lr.groupby(per).cumsum()                           # MTD spread up to each close
        sig_day = fe.reindex(cal) == -4
        rp = mtd.where(sig_day)
        rp_m = rp.dropna()
        sd = rp_m.rolling(p["z_months"], min_periods=24).std().shift(1)
        z = (rp_m / sd).clip(-p["z_cap"], p["z_cap"])
        z = z.where(z.abs() >= p.get("z_min", 0.0), 0.0)
        zd = z.reindex(cal).ffill(limit=30)                        # latest month's z (known from day -4 on)
        vol = ewm_vol(X, halflife=63)
        late_on = fe.isin(p["late"]).shift(-2, fill_value=False).reindex(cal)     # decide 2 sessions ahead (lag 1)
        early_on = fs.isin(p["early"]).shift(-2, fill_value=False).reindex(cal)
        sgn = pd.Series(0.0, index=cal)
        sgn[late_on.fillna(False).to_numpy()] = -1.0
        sgn[early_on.fillna(False).to_numpy()] = 1.0
        # the early leg uses the z of the month that just ENDED: at its decision dates (days -1 and +1) zd still
        # holds that month's value until the next day -4
        expo = sgn * zd
        W = pd.DataFrame({a: expo / vol[a], b: -expo / vol[b]}, index=cal).fillna(0.0)
        W = W.loc[X.dropna().index[0]:]
        chg = W.diff().abs().sum(axis=1).fillna(1.0) > 0
        return W.where(chg, np.nan)


STRATEGY = RebalancingFlows
