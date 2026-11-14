"""s09 - Volatility-risk-premium harvesting timed by the VIX term structure (wildcard diversifier).

Thesis: implied volatility exceeds subsequently realised volatility on average (variance risk premium) and VIX
futures roll down towards spot in contango, so a short position in short-term VIX futures earns a premium. The
premium is paid for crash risk: when the VIX curve inverts (VIX > VIX3M) or implied falls below realised, risk is
being repriced and the position is closed (Simon & Campasano 2014; term-structure timing).

v1: short VIXY (1x short-term VIX futures ETP; borrow 300 bps/yr charged) when VIX/VIX3M < 0.95 AND VIX exceeds
SPY's 21-day realised vol; flat otherwise. Signals use closes of day d; execution at the close of d+1. Engine sizes
the short to 10% ex-ante vol when active.
"""
import numpy as np
import pandas as pd

from .base import Strategy


class VRPTiming(Strategy):
    name = "s09_vrp_timing"
    version = "v1"
    universe = "multiasset"
    thesis = "Short VIX futures earn the variance risk premium; exit when the VIX curve inverts or VRP turns negative."
    inspired_by = "wildcard (crisis-convex diversifier)"

    def default_params(self):
        return {"ratio_max": 0.95, "rv_window": 21}

    def target_weights(self, ds):
        p = self.params
        lv = ds.panel("close", ["^VIX", "^VIX3M"])
        spy = ds.ret(["SPY"])["SPY"]
        rv = spy.rolling(p["rv_window"]).std() * np.sqrt(252) * 100
        x = ds.xret(["VIXY"])["VIXY"]
        on = (lv["^VIX"] / lv["^VIX3M"] < p["ratio_max"]) & (lv["^VIX"] > rv)
        on = on & x.notna() & lv.notna().all(axis=1)
        W = pd.DataFrame({"VIXY": np.where(on, -1.0, 0.0)}, index=lv.index)
        W = W.loc[x.first_valid_index():]
        chg = W["VIXY"].diff().fillna(1.0) != 0
        return W.where(chg, np.nan)


STRATEGY = VRPTiming
