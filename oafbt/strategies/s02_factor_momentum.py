"""s02 - Factor momentum (inspired by the cross-sectional-momentum deck type; NEW object: equity factors).

Thesis: returns of equity *factors* (size, value, profitability, investment, momentum; US and international)
are positively autocorrelated - factor momentum (Ehsani & Linnainmaa 2022, Gupta & Kelly 2019, Arnott et al. 2023).
Long factors whose trailing returns are positive, short those with negative trailing returns, risk-parity sized.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import kf_factor_returns, period_ends, keep_rows


REGIONS = {"US": ("F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily", "Mom"),
           "Dev": ("Developed_5_Factors_Daily", "Developed_Mom_Factor_Daily", "WML"),
           "Eur": ("Europe_5_Factors_Daily", "Europe_Mom_Factor_Daily", "WML"),
           "Jpn": ("Japan_5_Factors_Daily", "Japan_Mom_Factor_Daily", "WML"),
           "APxJ": ("Asia_Pacific_ex_Japan_5_Factors_Daily", "Asia_Pacific_ex_Japan_Mom_Factor_Daily", "WML"),
           "NA": ("North_America_5_Factors_Daily", "North_America_Mom_Factor_Daily", "WML")}


def factor_specs(regions, include_reversal=True):
    specs = {}
    for reg in regions:
        f5, fm, mcol = REGIONS[reg]
        for c in ["SMB", "HML", "RMW", "CMA"]:
            specs[f"KF:F:{reg}:{c}"] = (f5, c)
        specs[f"KF:F:{reg}:{mcol}"] = (fm, mcol)
    if include_reversal and "US" in regions:
        specs["KF:F:US:ST_Rev"] = ("F-F_ST_Reversal_Factor_daily", "ST_Rev")
        specs["KF:F:US:LT_Rev"] = ("F-F_LT_Reversal_Factor_daily", "LT_Rev")
    return specs


class FactorMomentum(Strategy):
    name = "s02_factor_momentum"
    version = "v1"
    universe = "kf"
    thesis = "Factor returns are autocorrelated; time-series momentum across factors earns a premium."
    inspired_by = "cross-sectional momentum deck (momentum type), applied to factors"

    def default_params(self):
        return {"regions": ["US", "Dev", "Eur", "Jpn", "APxJ", "NA"], "lookbacks_m": [1, 3, 6, 12],
                "vol_halflife": 63, "include_reversal": True}

    def _rets(self, ds):
        return kf_factor_returns(ds, factor_specs(self.params["regions"], self.params["include_reversal"]))

    def asset_returns(self, ds, columns):
        return self._rets(ds).reindex(columns=columns)

    def target_weights(self, ds):
        p = self.params
        R = self._rets(ds)
        lr = np.log1p(R.fillna(0.0)).where(R.notna())
        sig = 0
        for m in p["lookbacks_m"]:
            n = 21 * m
            tr = lr.rolling(n, min_periods=int(0.8 * n)).sum()
            sig = sig + np.sign(tr)
        sig = sig / len(p["lookbacks_m"])
        vol = np.sqrt((R ** 2).ewm(halflife=p["vol_halflife"], min_periods=60).mean()) * np.sqrt(252)
        W = sig / vol
        W = W.where(R.notna() & vol.notna())
        reb = period_ends(R.index, "M")
        return keep_rows(W, reb)


STRATEGY = FactorMomentum
