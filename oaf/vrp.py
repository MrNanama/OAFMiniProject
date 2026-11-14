"""Iteration 6: implied-vol-based sizing + variance-risk-premium tilt. Walk-forward, 21d purge, monthly rebalance."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

def prep(path="results/vrp_X.pkl"):
    X = pd.read_pickle(path).copy(); X["vrpr"] = np.log(X.vix / X.rv21); X["lrv"] = np.log(X.rv21); X["lvix"] = np.log(X.vix)
    return X

def strategy(X, kind, feats=("vrpr", "lrv"), start="1998-01-01", reb=21, min_hist=2016, cap=1.5, cost_bps=1.0, norm_win=504, sig_src="vix"):
    n = len(X); sig = (X.vix if sig_src == "vix" else 0.5 * X.rv21 + 0.5 * X.rv63).values
    F = X[list(feats)].values; fwd = X.fwd.values; raw = np.full(n, np.nan)
    for t in range(0, n, reb):
        if np.isnan(sig[t]): continue
        s = np.arange(0, t - 21); s = s[~np.isnan(F[s]).any(1) & ~np.isnan(fwd[s])]
        if len(s) < min_hist: continue
        if kind == "bh": raw[t:t+reb] = 1.0; continue
        if kind == "voltarget": raw[t:t+reb] = 1 / sig[t]; continue
        if kind == "mv_const": mu = fwd[s].mean() * 12
        else:
            A = np.c_[np.ones(len(s)), F[s]]; b = np.linalg.lstsq(A, fwd[s], rcond=None)[0]; mu = (b[0] + F[t] @ b[1:]) * 12
        raw[t:t+reb] = max(mu, 0.0) / sig[t] ** 2
    raw = pd.Series(raw, X.index)
    if kind != "bh":
        base = raw.where(raw > 0).rolling(norm_win, min_periods=252).mean().shift(1); w = (raw / base).clip(0, cap)
    else: w = raw
    w = w.where(X.index >= start)
    pnl = w.shift(1) * (X.r - X.rf) + X.rf; cost = w.diff().abs().fillna(0) * cost_bps / 1e4
    return (pnl - cost - X.rf).where(w.shift(1).notna()), w
