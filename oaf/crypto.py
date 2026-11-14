import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

def load():
    o = pd.read_pickle("data/crypto.pkl"); return o["close"], o["vol"]

def universe(C, V, top=30, win=30, min_hist=90):
    dv = (C * V).rolling(win).mean().shift(1); ok = C.rolling(min_hist).count().shift(1) >= min_hist
    rk = dv.where(ok).rank(axis=1, ascending=False); return rk <= top

def run(C, U, k, kind, reb=7, q=5, cost_bps=10.0):
    r = C.pct_change(); lr = np.log(C)
    sig = (lr - lr.shift(k)) * (1 if kind == "mom" else -1)
    sig = sig.where(U)
    n = sig.notna().sum(axis=1); rk = sig.rank(axis=1, method="first")
    raw = (rk.gt(n - q, axis=0) & sig.notna()).astype(float) / q - (rk.le(q, axis=0) & sig.notna()).astype(float) / q
    raw = raw.where(n >= 2 * q)
    reb_mask = pd.Series(np.arange(len(C)) % reb == 0, C.index)
    W = raw.where(reb_mask, np.nan).ffill()
    gross = (W.shift(1) * r).sum(axis=1, min_count=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * cost_bps / 1e4, gross, turn
