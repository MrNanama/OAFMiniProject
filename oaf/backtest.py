import numpy as np, pandas as pd
from scipy import stats

def run(W, F, cost_bps=5.0, hedge_cost_bps=1.0, target_vol=0.10, max_lev=6.0, hedge=True, risk_win=500):
    """W: target stock weights decided at close t (dates x stocks). Earns r[t+1]. Market-hedged with SPX using
    trailing beta known at t. Sized with EX-ANTE risk (sqrt(sum (w*sig_idio)^2) + hedge residual), normalised by its
    own trailing-2y mean (lagged) so that relative sizing is preserved and sparse books don't get levered up by a
    collapsing realised-vol estimate. Costs charged on turnover."""
    r, m, beta, sig = F["r"], F["m"], F["beta"], F["sig_e"]
    W = W.reindex(r.index).fillna(0.0)
    H = -(W * beta.fillna(1.0)).sum(axis=1) if hedge else pd.Series(0.0, index=W.index)
    ex = np.sqrt(((W * sig.fillna(sig.mean().mean())) ** 2).sum(axis=1)) * np.sqrt(252)
    base = ex.where(ex > 0).rolling(risk_win, min_periods=250).mean().shift(1)   # only days with a live book
    scale = (target_vol / base).clip(upper=max_lev).fillna(0.0)
    Ws, Hs = W.mul(scale, axis=0), H * scale
    pnl = (Ws.shift(1) * r).sum(axis=1) + Hs.shift(1) * m
    turn_s = Ws.diff().abs().sum(axis=1).fillna(0.0); turn_h = Hs.diff().abs().fillna(0.0)
    cost = turn_s * cost_bps / 1e4 + turn_h * hedge_cost_bps / 1e4
    net = pnl - cost
    return dict(net=net, gross=pnl, cost=cost, turnover=turn_s, scale=scale, W=Ws)

def metrics(x, start=None, ew=None, mkt=None, turn=None):
    x = x.dropna()
    x = x[x.index >= start] if start else x
    x = x[(x.index > x.index[(x != 0).values.argmax()])] if (x != 0).any() else x
    a = x.mean() * 252; v = x.std() * np.sqrt(252); sh = a / v if v > 0 else np.nan
    eq = (1 + x).cumprod(); dd = (eq / eq.cummax() - 1).min()
    out = dict(ann_ret=a, ann_vol=v, sharpe=sh, maxdd=dd, hit=(x > 0).mean(), skew=stats.skew(x), n_years=len(x) / 252)
    for nm, b in [("beta_EW", ew), ("beta_SPX", mkt)]:
        if b is not None:
            bb = b.reindex(x.index); out[nm] = np.cov(x, bb)[0, 1] / bb.var()
    if turn is not None: out["turnover_x/yr"] = turn.reindex(x.index).mean() * 252
    return out

def boot_sharpe(x, n=2000, block=20, seed=0):
    """stationary-ish block bootstrap CI for annualised Sharpe"""
    x = x.dropna().values; T = len(x); rng = np.random.default_rng(seed); res = []
    nb = int(np.ceil(T / block))
    for _ in range(n):
        idx = (rng.integers(0, T, nb)[:, None] + np.arange(block)[None]) % T
        s = x[idx.ravel()[:T]]; res.append(s.mean() / s.std() * np.sqrt(252))
    return np.percentile(res, [2.5, 50, 97.5])

def psr(x, sr_bench=0.0):
    """probabilistic Sharpe ratio (Bailey & Lopez de Prado) on daily returns"""
    x = x.dropna(); T = len(x); sr = x.mean() / x.std()
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    z = (sr - sr_bench) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return stats.norm.cdf(z)

def deflated_sr(x, n_trials, sr_var_trials=None):
    """expected-max Sharpe under n_trials of zero-skill strategies as the PSR benchmark"""
    x = x.dropna(); T = len(x)
    var = (1 / (T - 1)) if sr_var_trials is None else sr_var_trials     # var of daily SR estimates under null
    em = np.sqrt(var) * ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    return psr(x, em)
