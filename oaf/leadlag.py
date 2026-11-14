"""Iteration 4: learned cross-industry lead-lag network (ridge), walk-forward. Weekly, market-relative returns."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from .kf import industries

def weekly(I, k=5):
    n = (len(I) // k) * k; R = np.log1p(I.iloc[len(I) - n:].fillna(0.0)); ok = I.iloc[len(I) - n:].notna()
    Rw = R.groupby(np.arange(n) // k).sum(); Ow = ok.groupby(np.arange(n) // k).mean() > 0.99
    idx = I.index[len(I) - n:][k - 1::k]; Rw.index = idx; Ow.index = idx
    Rw = np.expm1(Rw).where(Ow)
    rel = Rw.sub(Rw.mean(axis=1), axis=0)             # market-relative (EW average of available industries)
    return rel, Rw

def fit_ridge(X, Y, lam):
    p = X.shape[1]; return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)

def walk_forward(rel, model="net", min_train_weeks=1040, refit_every=52, val_weeks=260, lams=(1e2, 1e3, 1e4, 1e5), perm=None, shift=0):
    A = rel.values; T, N = A.shape; mask = ~np.isnan(A); Z = np.nan_to_num(A)
    pred = np.full((T, N), np.nan)
    Xall = Z.copy()
    if perm is not None: Xall = Z[:, perm]
    if shift: Xall = np.roll(Xall, shift, axis=0)           # placebo: break lead-lag by circularly shifting predictors in time
    for t0 in range(min_train_weeks, T - 1, refit_every):
        tr = np.arange(0, t0 - 1)                              # x_s -> y=s+1 with s+1 <= t0-1  (target fully in the past)
        X, Y = Xall[tr], Z[tr + 1]; m = mask[tr + 1]
        mu = Y.sum(0) / np.maximum(m.sum(0), 1); Yc = (Y - mu) * m
        sd = X.std(0) + 1e-9
        Xs = X / sd
        if model == "own":
            beta = (Xs * Yc).sum(0) / (Xs ** 2).sum(0)       # per-industry AR(1) on own lag (own lag = same index, only valid w/o perm)
            B = np.diag(beta)
        else:
            nv = val_weeks; best = None
            for lam in lams:
                Bv = fit_ridge(Xs[:-nv], Yc[:-nv], lam); e = ((Yc[-nv:] - Xs[-nv:] @ Bv) ** 2 * m[-nv:]).sum()
                if best is None or e < best[0]: best = (e, lam)
            B = fit_ridge(Xs, Yc, best[1])
        te = np.arange(t0 - 1, min(t0 - 1 + refit_every, T - 1))
        pred[te] = (Xall[te] / sd) @ B                         # forecast of rel return for week te+1 made at end of week te
    return pd.DataFrame(pred, rel.index, rel.columns)

def backtest(pred, Rw, rel, q=10, cost_bps=5.0):
    P = pred.where(rel.notna().shift(-1).fillna(False))        # only industries that exist next week
    W = pd.DataFrame(0.0, P.index, P.columns)
    rk = P.rank(axis=1, method="first"); n = P.notna().sum(axis=1)
    long = rk.gt(n - q, axis=0) & P.notna(); short = rk.le(q, axis=0) & P.notna()
    W = long.astype(float) / q - short.astype(float) / q
    ret = Rw.shift(-1).fillna(0.0)                              # return of week t+1 earned by weights set at end of week t
    gross = (W * ret).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0); net = gross - turn * cost_bps / 1e4
    out = pd.DataFrame({"gross": gross, "net": net, "turn": turn}); out = out[pred.notna().any(axis=1)]
    return out.iloc[:-1], W
