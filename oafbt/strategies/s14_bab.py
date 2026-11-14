"""s14 - Betting against beta within the S&P 500 (candidate replacement for s03; risk/portfolio-construction type).

Thesis (Frazzini & Pedersen 2014): leverage-constrained investors overweight high-beta stocks, flattening the
security market line; a beta-neutral book long (levered) low-beta and short (de-levered) high-beta stocks earns
a premium. v1: monthly; betas = 0.6 * rolling 252d beta vs SPY + 0.4 (Vasicek-style shrinkage to 1); stocks below
/ above the median beta, rank-weighted within each side; long side scaled by 1/beta_L, short side by 1/beta_H
(beta-neutral, net long dollars, financed at the leverage spread).
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends, keep_rows
from .s08_ml_stock_ranker import _fast_beta


class BettingAgainstBeta(Strategy):
    name = "s14_bab"
    version = "v2"
    universe = "stocks"
    thesis = "Leverage constraints make low-beta stocks earn more per unit of beta than high-beta stocks."
    inspired_by = "portfolio-construction / risk type"

    def default_params(self):
        # v2 'betting against correlation' (Asness, Frazzini, Gormsen & Pedersen 2020): within volatility quintiles,
        # long low-correlation / short high-correlation stocks, beta-neutral. Isolates the correlation part of the
        # low-risk effect, which they find more robust than BAB (v1 BAB: SR 0.28, 2004-20).
        return {"beta_window": 252, "shrink": 0.6, "mode": "bac", "vol_buckets": 5}

    def target_weights(self, ds):
        p = self.params
        U = ds.stock_universe()
        cols = list(U.columns)
        R = ds.ret(cols)
        spy = ds.ret(["SPY"])["SPY"].reindex(R.index)
        braw = _fast_beta(R, spy, w=p["beta_window"])
        b = p["shrink"] * braw + (1 - p["shrink"])
        vol = R.rolling(p["beta_window"], min_periods=150).std()
        mvol = spy.rolling(p["beta_window"], min_periods=150).std()
        corr = braw * mvol.values[:, None] / vol
        me = period_ends(U.index, "M")
        W = {}
        for d in me:
            elig = U.loc[d]
            bb = b.loc[d][elig[elig].index].dropna()
            if len(bb) < 50:
                continue
            if p.get("mode") == "bac":
                cc = corr.loc[d, bb.index]
                vb = pd.qcut(vol.loc[d, bb.index].rank(method="first"), p["vol_buckets"], labels=False)
                z = cc.groupby(vb).rank(pct=True) - 0.5          # correlation rank within each vol bucket
                z = z.dropna()
                bb = bb[z.index]
            else:
                z = bb.rank()
                z = z - z.mean()
            lo, hi = -z.clip(upper=0), z.clip(lower=0)
            lo, hi = lo / lo.sum(), hi / hi.sum()
            beta_l, beta_h = (lo * bb).sum(), (hi * bb).sum()
            W[d] = lo / beta_l - hi / beta_h
        out = pd.DataFrame(W).T.reindex(columns=cols).fillna(0.0).reindex(U.index)
        return keep_rows(out, list(W))


STRATEGY = BettingAgainstBeta
