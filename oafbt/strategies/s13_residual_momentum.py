"""s13 - Residual (idiosyncratic) momentum with Fama-French factor neutralisation (candidate replacement for s03).

Inspiration TYPE: cross-sectional momentum deck + Macro Momentum deck's latent-factor idea. NEW object: momentum in
the part of each stock's return NOT explained by common factors (Blitz, Huij & Martens 2011; Blitz, Hanauer &
Vidojevic 2020 'The idiosyncratic momentum anomaly'). Because factor exposures are removed, the strategy does not
load on the momentum crash risk of raw momentum (2009), and residual returns underreact to firm-specific news.

v1: monthly. For each eligible stock, regress its monthly excess returns over the trailing 36 months on the Ken
French Mkt-RF, SMB, HML monthly factors (built from daily); residuals e. Signal at month-end t = sum of e over
months t-11..t-1 (skip the latest month) divided by std(e) over the 36-month window. Long top decile, short bottom
decile, equal weight, dollar-neutral.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends, keep_rows, long_short


def monthly(df: pd.DataFrame, me: pd.DatetimeIndex, min_days: int = 15) -> pd.DataFrame:
    grp = np.searchsorted(me, df.index)
    ok = grp < len(me)
    m = np.expm1(np.log1p(df[ok]).groupby(grp[ok]).sum(min_count=min_days))
    m.index = me[: len(m)]
    return m


class ResidualMomentum(Strategy):
    name = "s13_residual_momentum"
    version = "v1"
    universe = "stocks"
    thesis = "Firm-specific (factor-residual) returns underreact and trend; factor-neutral, so no momentum crashes."
    inspired_by = "cross-sectional momentum deck + latent factors (Macro Momentum deck)"

    def default_params(self):
        return {"window": 36, "min_months": 24, "formation": 11, "skip": 1, "q": 0.1}

    def target_weights(self, ds):
        p = self.params
        U = ds.stock_universe()
        cols = list(U.columns)
        me = period_ends(U.index, "M")
        R = monthly(ds.xret(cols), me)
        f = ds.kf("F-F_Research_Data_Factors_daily")[["Mkt-RF", "SMB", "HML"]]
        F = monthly(f.reindex(U.index), me, min_days=10)
        w, L = p["window"], p["formation"]
        sig = pd.DataFrame(np.nan, index=R.index, columns=cols)
        for i in range(w - 1, len(R)):
            Y = R.iloc[i - w + 1:i + 1]
            Xf = F.iloc[i - w + 1:i + 1]
            ok_rows = Xf.notna().all(axis=1).to_numpy()
            Xm = np.column_stack([np.ones(ok_rows.sum()), Xf.to_numpy()[ok_rows]])
            Ym = Y.to_numpy()[ok_rows]
            have = (~np.isnan(Ym)).sum(axis=0) >= p["min_months"]
            if not have.any():
                continue
            Yf = np.where(np.isnan(Ym), 0.0, Ym)
            mask = ~np.isnan(Ym)
            res = np.full(Ym.shape, np.nan)
            for j in np.flatnonzero(have):
                mj = mask[:, j]
                b, *_ = np.linalg.lstsq(Xm[mj], Yf[mj, j], rcond=None)
                res[mj, j] = Yf[mj, j] - Xm[mj] @ b
            form = res[-(L + p["skip"]):-p["skip"]] if p["skip"] else res[-L:]
            sd = np.nanstd(res, axis=0, ddof=1)
            s = np.nansum(form, axis=0) / sd
            s[~have | (np.isnan(form).sum(axis=0) > 2)] = np.nan
            sig.iloc[i] = s
        elig = U.reindex(sig.index).fillna(False)
        W = long_short(sig, q=p["q"], eligible=elig)
        W = W.reindex(U.index)
        return keep_rows(W, W.dropna(how="all").index)


STRATEGY = ResidualMomentum
