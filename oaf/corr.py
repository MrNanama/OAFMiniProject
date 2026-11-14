"""Iteration 2: correlation-state market timing. Features use info up to close t only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from arch.data import frenchdata
from .data import load

def avg_corr(r, win):
    R = r.values; out = np.full(len(r), np.nan)
    for t in range(win, len(r)):
        C = np.nan_to_num(np.corrcoef(R[t-win+1:t+1].T)); n = C.shape[0]
        out[t] = (C.sum() - n) / (n * (n - 1))        # diag of a constant-return column is 0 after nan_to_num; negligible
    return pd.Series(out, r.index)

def rf_daily(idx):
    f = frenchdata.load()["RF"] / 100.0                # monthly, to 2018-11; zero afterwards (approximation, flagged)
    f.index = f.index.to_period("M"); per = idx.to_period("M")
    nd = pd.Series(1, idx).groupby(per).transform("sum")
    return pd.Series(f.reindex(per).fillna(0.0).values, idx) / nd.values

def build():
    px, mk = load(); m = mk.pct_change().iloc[1:]; r = px.pct_change().iloc[1:]
    d = pd.DataFrame({"m": m, "ac63": avg_corr(r, 63), "rv21": m.rolling(21).std() * np.sqrt(252),
                      "rv63": m.rolling(63).std() * np.sqrt(252)})
    d["rf"] = rf_daily(d.index)
    d["fwd"] = ((d.m - d.rf)[::-1].rolling(21).sum()[::-1]).shift(-1)    # excess return t+1..t+21
    return d

def strategy(d, kind, start="2000-01-01", reb=21, min_hist=2016, cap=1.5, cost_bps=1.0, norm_win=504):
    """kind: bh | voltarget | mv_const | corr. Weight decided at close t, earns day t+1."""
    n = len(d); sig = (0.5 * d.rv21 + 0.5 * d.rv63).values; ac = d.ac63.values; fwd = d.fwd.values
    raw = np.full(n, np.nan); idx = np.arange(n)
    for t in range(0, n, reb):
        if np.isnan(sig[t]): continue
        s = np.arange(0, t - 21)                          # purged: fwd of s ends by t
        s = s[~np.isnan(ac[s]) & ~np.isnan(fwd[s])]
        if len(s) < min_hist: continue
        if kind == "corr":
            X = np.c_[np.ones(len(s)), ac[s]]; b = np.linalg.lstsq(X, fwd[s], rcond=None)[0]
            mu = (b[0] + b[1] * ac[t]) * 12                       # annualised expected excess return
        else:
            mu = fwd[s].mean() * 12
        if kind == "voltarget": raw[t:t+reb] = 1 / sig[t]
        elif kind == "bh": raw[t:t+reb] = 1.0
        else: raw[t:t+reb] = max(mu, 0.0) / sig[t] ** 2
    raw = pd.Series(raw[:n], d.index)
    if kind != "bh":                                       # normalise to ~1x average exposure using lagged trailing mean
        base = raw.where(raw > 0).rolling(norm_win, min_periods=252).mean().shift(1)
        w = (raw / base).clip(0, cap)
    else: w = raw
    w = w.where(d.index >= start)
    pnl = w.shift(1) * (d.m - d.rf) + d.rf                 # cash earns rf; leverage financed at rf
    cost = w.diff().abs().fillna(0) * cost_bps / 1e4
    return (pnl - cost - d.rf).where(w.shift(1).notna()), w     # EXCESS return series
