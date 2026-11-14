"""s08 - Walk-forward machine-learning stock ranker (ML / regularised-learning TYPE from the decks; NEW object).

Thesis: many weak price/volume signals (momentum at several horizons, reversal, volatility, beta, lottery-ness,
liquidity, attention, 52-week-high anchoring, seasonality) interact non-linearly; a shallow, heavily regularised
learner trained walk-forward on the cross-section can combine them better than any single one (Gu, Kelly & Xiu
2020). The Macro Momentum deck's lesson - prefer regularised models and walk-forward fits over deep nets - is kept.

v1: month-end panel of eligible S&P 500 members (ds.stock_universe), 15 features rank-normalised per date;
target = next-month return rank. Refit every December on all past month-ends whose label is fully observed
(expanding window, >= 3 years); prediction = average rank of a HistGradientBoosting model and a ridge model.
Long top decile, short bottom decile, equal weight, dollar-neutral, monthly.
"""
import numpy as np
import pandas as pd

from .base import Strategy
from .utils import period_ends, keep_rows


def _fast_beta(R: pd.DataFrame, m: pd.Series, w: int = 252, minp: int = 150):
    """rolling beta with pairwise-complete windows via cumulative sums (fast)"""
    ok = R.notna() & m.notna().values[:, None]
    x = R.where(ok, 0.0)
    mm = pd.DataFrame(np.where(ok, m.values[:, None], 0.0), index=R.index, columns=R.columns)
    n = ok.astype(float).rolling(w).sum()
    sx, sm = x.rolling(w).sum(), mm.rolling(w).sum()
    sxm, smm = (x * mm).rolling(w).sum(), (mm * mm).rolling(w).sum()
    cov = sxm / n - sx / n * sm / n
    var = smm / n - (sm / n) ** 2
    return (cov / var).where(n >= minp)


class MLStockRanker(Strategy):
    name = "s08_ml_stock_ranker"
    version = "v2"
    universe = "stocks"
    thesis = "A regularised learner combines many weak price/volume anomalies better than any single one."
    inspired_by = "ML / regularised walk-forward learning (Macro Momentum deck methodology)"

    def default_params(self):
        # v2: target = next-month RESIDUAL return (minus beta x SPY), rolling 10-year training window (adapts to
        # the post-2007 regime where v1 decayed), beta-neutral book (v1 had beta 0.13 and a -68% DD in 2009).
        return {"min_train_years": 3, "q": 0.1, "gbm_iter": 150, "gbm_depth": 3, "ridge_alpha": 10.0,
                "train_years": 10, "residual_target": True, "beta_neutral": True}

    def features(self, ds):
        U = ds.stock_universe()
        cols = list(U.columns)
        R = ds.ret(cols)
        spy = ds.ret(["SPY"])["SPY"].reindex(R.index)
        lr = np.log1p(R)
        cs = lambda w, s=0: lr.rolling(w, min_periods=int(w * 0.8)).sum().shift(s)
        dv = ds.panel("close", cols) * ds.panel("volume", cols)
        tri = ds.tri(cols)
        beta = _fast_beta(R, spy)
        resid = R - beta.shift(1) * spy.values[:, None]
        F = {"r1m": cs(21), "r3m": cs(63), "r6m": cs(126), "r12_1": cs(231, 21), "rev5": cs(5),
             "vol21": R.rolling(21, min_periods=15).std(), "vol252": R.rolling(252, min_periods=150).std(),
             "beta": beta, "ivol63": resid.rolling(63, min_periods=40).std(),
             "max21": R.rolling(21, min_periods=15).max(), "skew63": R.rolling(63, min_periods=40).skew(),
             "ldv": np.log(dv.rolling(63, min_periods=20).median()),
             "abvol": np.log(dv.rolling(21, min_periods=15).mean() / dv.rolling(252, min_periods=150).mean()),
             "d52h": tri / tri.rolling(252, min_periods=150).max() - 1}
        me = period_ends(R.index, "M")
        # same-calendar-month seasonality: mean of the returns in the upcoming calendar month over past 5 years
        mret = np.expm1(lr.groupby(np.searchsorted(me, R.index)).sum(min_count=15))
        mret = mret.iloc[: len(me)]
        mret.index = me[: len(mret)]
        seas = sum(mret.shift(12 * k - 1).fillna(0) for k in range(1, 6)) / \
            sum(mret.shift(12 * k - 1).notna().astype(float) for k in range(1, 6)).replace(0, np.nan)
        return U, R, F, me, mret, seas

    def target_weights(self, ds):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        p = self.params
        U, R, F, me, mret, seas = self.features(ds)
        names = list(F) + ["seas"]
        X_by_date, y_by_date, idx_by_date = {}, {}, {}
        nxt = mret.shift(-1)                                      # next month's return (label)
        spy_d = ds.ret(["SPY"])["SPY"].reindex(U.index)
        grp = np.searchsorted(me, spy_d.index)
        okg = grp < len(me)
        spym = np.expm1(np.log1p(spy_d[okg]).groupby(grp[okg]).sum(min_count=15))
        spym.index = me[: len(spym)]
        spy_next = spym.shift(-1)
        bshr = 0.6 * F["beta"] + 0.4
        for d in me:
            elig = U.loc[d]
            elig = elig[elig].index
            if len(elig) < 50:
                continue
            cols = {k: F[k].loc[d, elig] for k in F}
            cols["seas"] = seas.loc[d, elig] if d in seas.index else pd.Series(np.nan, index=elig)
            df = pd.DataFrame(cols)
            df = df.rank(pct=True) - 0.5
            df = df.fillna(0.0)
            X_by_date[d] = df[names].to_numpy()
            idx_by_date[d] = elig
            y = nxt.loc[d, elig] if d in nxt.index else pd.Series(np.nan, index=elig)
            if p.get("residual_target"):
                y = y - F["beta"].loc[d, elig].fillna(1.0) * (spy_next.loc[d] if d in spy_next.index else np.nan)
            y_by_date[d] = (y.rank(pct=True) - 0.5).to_numpy()
        dates = sorted(X_by_date)
        W = {}
        model = None
        for i, d in enumerate(dates):
            refit = model is None or d.month == 12
            train = [t for t in dates[:i]]                         # label of t (month after t) is observed at d
            if p.get("train_years"):
                train = train[-12 * p["train_years"]:]
            if refit and len(train) >= 12 * p["min_train_years"]:
                Xtr = np.vstack([X_by_date[t] for t in train])
                ytr = np.concatenate([y_by_date[t] for t in train])
                ok = ~np.isnan(ytr)
                gbm = HistGradientBoostingRegressor(max_iter=p["gbm_iter"], max_depth=p["gbm_depth"],
                                                    learning_rate=0.05, l2_regularization=1.0,
                                                    min_samples_leaf=200, random_state=0)
                gbm.fit(Xtr[ok], ytr[ok])
                rdg = Ridge(alpha=p["ridge_alpha"]).fit(Xtr[ok], ytr[ok])
                model = (gbm, rdg)
            if model is None:
                continue
            Xd = X_by_date[d]
            pr = pd.Series(model[0].predict(Xd), index=idx_by_date[d]).rank(pct=True) + \
                pd.Series(model[1].predict(Xd), index=idx_by_date[d]).rank(pct=True)
            pr = pr.rank(pct=True)
            w = pd.Series(0.0, index=idx_by_date[d])
            top, bot = pr > 1 - p["q"], pr <= p["q"]
            w[top] = 1.0 / top.sum()
            w[bot] = -1.0 / bot.sum()
            if p.get("beta_neutral"):
                bd = bshr.loc[d, idx_by_date[d]].fillna(1.0).clip(lower=0.3)
                bl, bh = (w[top] * bd[top]).sum(), (-w[bot] * bd[bot]).sum()
                w[top] = w[top] / bl
                w[bot] = w[bot] / bh
            W[d] = w
        out = pd.DataFrame(W).T.reindex(columns=U.columns).fillna(0.0)
        out = out.reindex(U.index)
        return keep_rows(out, list(W))


STRATEGY = MLStockRanker
