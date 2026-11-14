"""s06 - Treasury-curve relative value: curvature mean reversion with a DV01-neutral ETF butterfly.

Inspired by the Macro Momentum deck's source 'Principles of Principal Components' (Salomon Brothers 2000: the third
principal component of the curve - curvature - is a stationary, mean-reverting factor that can be isolated with
level- and slope-neutral butterflies) and by the relative-value idea of the pairs deck. NEW here: trade the curve's
curvature with Treasury ETFs, sized by empirically estimated durations.

v1: fly = 2*y10 - y5 - y30 (Yahoo ^TNX, ^FVX, ^TYX; signal-only). z = (fly - 1y mean) / 1y std.
    Position (weekly): long belly IEF / short wings IEI (5y) and TLT (30y) in DV01 terms 2 : -1 : -1, scaled by
    clip(z, -2, 2)/2 -> profits when the fly reverts. Durations are rolling 1y regressions of each ETF's return on
    its matching yield change (information <= t). Trading starts once IEI exists (2007).
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends, keep_rows


class CurveFlyRV(Strategy):
    name = "s06_curve_pca_rv"
    version = "v1"
    universe = "multiasset"
    thesis = "Treasury curvature (2*10y - 5y - 30y) mean-reverts; a DV01-neutral butterfly harvests it."
    inspired_by = "Macro Momentum deck (Salomon PCA source) + pairs/relative value"

    def default_params(self):
        return {"z_window": 252, "dur_window": 252, "z_cap": 2.0, "freq": "W"}

    def target_weights(self, ds):
        p = self.params
        y = ds.panel("close", ["^FVX", "^TNX", "^TYX"]) / 100.0
        fly = 2 * y["^TNX"] - y["^FVX"] - y["^TYX"]
        z = (fly - fly.rolling(p["z_window"], min_periods=126).mean()) / fly.rolling(p["z_window"], min_periods=126).std()
        legs = {"IEI": "^FVX", "IEF": "^TNX", "TLT": "^TYX"}
        R = ds.ret(list(legs))
        dy = y.diff()
        dur = {}
        for etf, yl in legs.items():
            cov = R[etf].rolling(p["dur_window"], min_periods=126).cov(dy[yl])
            var = dy[yl].rolling(p["dur_window"], min_periods=126).var()
            dur[etf] = (-cov / var).clip(lower=0.5)
        D = pd.DataFrame(dur)
        s = z.clip(-p["z_cap"], p["z_cap"]) / p["z_cap"]
        W = pd.DataFrame({"IEF": 2.0 / D["IEF"], "IEI": -1.0 / D["IEI"], "TLT": -1.0 / D["TLT"]}).mul(s, axis=0)
        W = W.where(R.notna().all(axis=1) & D.notna().all(axis=1) & z.notna(), np.nan).dropna(how="any")
        return keep_rows(W, period_ends(W.index, p["freq"]))


STRATEGY = CurveFlyRV
